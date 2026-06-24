# mypy: allow-untyped-defs
import dataclasses
import math
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet


def normalize_shape(shape: Any) -> Any:
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


def normalize_reduce_dims(dim: Any) -> tuple[Any, ...]:
    return tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSALayout:
    axis: int
    group_size: int

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    @property
    def fragment_group_size(self) -> int:
        return min(self.group_size, 32)

    @property
    def tensorssa_shape(self) -> str:
        if self.axis == 1:
            return f"((1, {self.fragment_group_size}, {32 // self.fragment_group_size}), 1, 1)"
        return (
            f"(({self.fragment_group_size}, 1, {32 // self.fragment_group_size}), 1, 1)"
        )

    @property
    def keepdim_shape(self) -> str:
        return f"((1, 1, {32 // self.fragment_group_size}), 1, 1)"

    @property
    def needs_physical_combine(self) -> bool:
        return self.axis == 0 or self.group_size > self.fragment_group_size

    @property
    def reduction_profile(self) -> str:
        if self.axis == 1:
            return "((None, 1, None), 1, 1)"
        return "((1, None, None), 1, 1)"


def grouped_tensor_layout(shape: Any) -> GroupedTensorSSALayout | None:
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple) or len(shape) not in (3, 4):
        return None
    if isinstance(shape[-1], int) and shape[-1] > 0 and shape[-2] == -1:
        return GroupedTensorSSALayout(axis=1, group_size=shape[-1])
    if shape[-3] == -1 and isinstance(shape[-2], int) and shape[-2] > 0:
        return GroupedTensorSSALayout(axis=0, group_size=shape[-2])
    return None


def _cute_op_name(target: Any) -> str | None:
    if isinstance(target, torch._ops.OpOverload):
        op_name = target.overloadpacket.__name__
    elif isinstance(target, str):
        op_name = target
    else:
        op_name = target.__name__ if callable(target) else None
    return "truediv" if op_name == "div" else op_name


def partial_vector_reduction_error() -> NotImplementedError:
    return NotImplementedError(
        "unsupported FlexGEMM epilogue partial-output contract: reductions "
        "over GEMM M/N dimensions require an explicit partial-output contract; "
        "QuACK row/column partial reductions produce CTA-tiled partial shapes, "
        "not final PyTorch reduction tensors"
    )


@dataclasses.dataclass(frozen=True)
class GroupedTensorSSAInfo:
    layout: GroupedTensorSSALayout
    nonnegative: bool = False

    @property
    def axis(self) -> int:
        return self.layout.axis

    @property
    def group_size(self) -> int:
        return self.layout.group_size


@dataclasses.dataclass(frozen=True)
class FlexGemmReductionSpec:
    reduction_type: str
    cute_op: str
    init_val: str
    combine_expr: str
    scale_by_group_size: bool = False
    finalize_expr: str = "value"


FlexGemmPhysicalReduction = FlexGemmReductionSpec


FLEX_GEMM_REDUCTIONS = {
    "sum": FlexGemmReductionSpec("sum", "cute.ReductionOp.ADD", "0.0", "lhs + rhs"),
    "mean": FlexGemmReductionSpec(
        "mean",
        "cute.ReductionOp.ADD",
        "0.0",
        "lhs + rhs",
        scale_by_group_size=True,
        finalize_expr="value / {group_size}.0",
    ),
    "prod": FlexGemmReductionSpec("prod", "cute.ReductionOp.MUL", "1.0", "lhs * rhs"),
    "max": FlexGemmReductionSpec(
        "max", "cute.ReductionOp.MAX", 'float("-inf")', "cute.arch.fmax(lhs, rhs)"
    ),
    "min": FlexGemmReductionSpec(
        "min", "cute.ReductionOp.MIN", 'float("inf")', "cute.arch.fmin(lhs, rhs)"
    ),
}


FLEX_GEMM_TENSORSSA_REDUCTIONS = FLEX_GEMM_REDUCTIONS


FLEX_GEMM_SHAPE_PRESERVING_POINTWISE_OPS = frozenset(
    OrderedSet(
        [
            "_to_copy",
            "abs",
            "add",
            "clamp",
            "clamp_max",
            "clamp_min",
            "convert_element_type",
            "cos",
            "div",
            "eq",
            "exp",
            "exp2",
            "ge",
            "gt",
            "le",
            "log",
            "log10",
            "log2",
            "logical_and",
            "logical_not",
            "logical_or",
            "logical_xor",
            "lt",
            "maximum",
            "minimum",
            "mul",
            "ne",
            "neg",
            "pow",
            "reciprocal",
            "relu",
            "rsqrt",
            "sigmoid",
            "sin",
            "sqrt",
            "square",
            "sub",
            "tanh",
            "to_dtype",
            "truediv",
            "where",
        ]
    )
)


def _cute_arg(value: Any, env: dict[torch.fx.Node, Any]) -> Any:
    """Translate FX node references and constants into CuTeDSL epilogue values."""
    if isinstance(value, torch.fx.Node):
        if value in env:
            return env[value]
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue dependency: {value.format_node()}"
        )
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return 'float("nan")'
        return 'float("inf")' if value > 0 else 'float("-inf")'
    if isinstance(
        value,
        (
            int,
            float,
            bool,
            torch.dtype,
            torch.device,
            torch.layout,
            torch.memory_format,
        ),
    ):
        return value
    if isinstance(value, (tuple, list)):
        return type(value)(_cute_arg(item, env) for item in value)
    raise NotImplementedError(f"unsupported FlexGEMM epilogue constant: {value!r}")


def _cute_call(target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    op_name = _cute_op_name(target)
    if op_name in {
        "all",
        "any",
        "argmax",
        "argmin",
        "mean",
        "amin",
        "prod",
        "std",
        "var",
    }:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction without a grouped "
            f"TensorSSA lowering: {target}"
        )
    if op_name is None:
        raise NotImplementedError(f"unsupported FlexGEMM epilogue op: {target}")
    try:
        op = getattr(V.get_ops_handler(), op_name)
    except AttributeError:
        raise NotImplementedError(
            f"unsupported FlexGEMM epilogue op: {target}"
        ) from None
    return op(*args, **kwargs)


def _local_reduce_store_arg(
    value: Any, env: dict[torch.fx.Node, Any], sources: dict[torch.fx.Node, Any]
) -> Any:
    if isinstance(value, torch.fx.Node) and value in sources:
        return sources[value]
    if isinstance(value, (tuple, list)):
        return type(value)(
            _local_reduce_store_arg(item, env, sources) for item in value
        )
    return _cute_arg(value, env)


def has_local_reduce_store_source(
    value: Any, sources: dict[torch.fx.Node, Any]
) -> bool:
    if isinstance(value, torch.fx.Node):
        return value in sources
    if isinstance(value, (tuple, list)):
        return any(has_local_reduce_store_source(item, sources) for item in value)
    return False


def is_shape_preserving_pointwise_node(node: torch.fx.Node) -> bool:
    return _cute_op_name(node.target) in FLEX_GEMM_SHAPE_PRESERVING_POINTWISE_OPS


def iter_fx_node_inputs(value: Any):
    """Yield FX node inputs nested in args/kwargs-style containers."""
    if isinstance(value, torch.fx.Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from iter_fx_node_inputs(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_fx_node_inputs(item)


def propagate_grouped_tensorssa_info(
    node: torch.fx.Node, grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo]
) -> GroupedTensorSSAInfo | None:
    input_infos = [
        grouped_tensors[arg]
        for arg in iter_fx_node_inputs((node.args, node.kwargs))
        if arg in grouped_tensors
    ]
    if not input_infos:
        return None
    layout = input_infos[0].layout
    if any(info.layout != layout for info in input_infos):
        raise NotImplementedError(
            "FlexGEMM local reductions do not support mixing grouped TensorSSA "
            "values with different grouped layouts"
        )
    return GroupedTensorSSAInfo(
        layout,
        (node.op == "call_method" and node.target == "abs")
        or (
            node.op == "call_function"
            and node.target in (torch.ops.aten.abs.default, torch.abs)
        ),
    )


def lower_view_or_reshape(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op == "call_method" and node.target in ("view", "reshape"):
        source_node = node.args[0]
        shape = node.args[1:]
    elif node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        source_node = node.args[0]
        shape = node.args[1]
    else:
        return None
    if (
        isinstance(source_node, torch.fx.Node)
        and source_node in local_reduce_store_sources
    ):
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    grouped_layout = grouped_tensor_layout(shape)
    if grouped_layout is None or not isinstance(source_node, torch.fx.Node):
        return None
    group_size = grouped_layout.group_size
    if group_size <= 1:
        raise NotImplementedError(
            "FlexGEMM local reductions require group size greater than 1"
        )
    if grouped_layout.axis == 1:
        if group_size > 32 and group_size % 32 != 0:
            raise NotImplementedError(
                "FlexGEMM local reductions larger than TensorSSA fragment width 32 require group size to be a multiple of 32"
            )
        if group_size <= 32 and 32 % group_size != 0:
            raise NotImplementedError(
                "FlexGEMM local reductions require group size to divide TensorSSA fragment width 32"
            )
    elif group_size > 32:
        if group_size % 32 != 0:
            raise NotImplementedError(
                "FlexGEMM local reductions larger than TensorSSA fragment width 32 require group size to be a multiple of 32"
            )
    elif 32 % group_size != 0:
        raise NotImplementedError(
            "FlexGEMM local reductions require group size to divide TensorSSA fragment width 32"
        )
    source = _cute_arg(source_node, env)
    result = kernel.cse.generate(
        kernel.body,
        f"{source}.reshape({grouped_layout.tensorssa_shape})",
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    grouped_tensors[node] = GroupedTensorSSAInfo(grouped_layout)
    return result


def reduction_from_node(node: torch.fx.Node) -> tuple[Any, Any, Any, Any, str] | None:
    if node.op == "call_method" and node.target in (
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        reduction_type = {"amax": "max", "amin": "min"}.get(node.target, node.target)
        return input_node, dim, keepdim, dtype, reduction_type
    if node.op == "call_function" and node.target in (
        torch.ops.aten.sum.dim_IntList,
        torch.ops.aten.mean.dim,
        torch.mean,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        reduction_type = (
            "mean" if node.target in (torch.ops.aten.mean.dim, torch.mean) else "sum"
        )
        return input_node, dim, keepdim, dtype, reduction_type
    if node.op == "call_function" and node.target == torch.ops.aten.prod.dim_int:
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return input_node, dim, keepdim, dtype, "prod"
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amax.default,
        torch.amax,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, keepdim, None, "max"
    if node.op == "call_function" and node.target in (
        torch.ops.aten.amin.default,
        torch.amin,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, keepdim, None, "min"
    return None


def moment_reduction_from_node(
    node: torch.fx.Node,
) -> tuple[Any, Any, Any, Any, str] | None:
    if node.op == "call_method" and node.target in ("var", "std"):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        if "correction" in node.kwargs:
            correction = node.kwargs["correction"]
        elif len(node.args) > 2:
            correction = 1 if node.args[2] else 0
        else:
            correction = 1 if node.kwargs.get("unbiased", True) else 0
        keepdim = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("keepdim", False)
        )
        return input_node, dim, correction, keepdim, node.target
    if node.op == "call_function" and node.target in (
        torch.ops.aten.var.dim,
        torch.ops.aten.std.dim,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        unbiased = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("unbiased", True)
        )
        keepdim = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("keepdim", False)
        )
        reduction_type = "std" if node.target is torch.ops.aten.std.dim else "var"
        return input_node, dim, 1 if unbiased else 0, keepdim, reduction_type
    if node.op == "call_function" and node.target in (
        torch.ops.aten.var.correction,
        torch.ops.aten.std.correction,
    ):
        input_node = node.args[0]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        correction = node.kwargs.get(
            "correction", node.args[2] if len(node.args) > 2 else 1
        )
        keepdim = node.kwargs.get(
            "keepdim", node.args[3] if len(node.args) > 3 else False
        )
        reduction_type = (
            "std" if node.target is torch.ops.aten.std.correction else "var"
        )
        return input_node, dim, correction, keepdim, reduction_type
    return None


def unsupported_reduction_from_node(node: torch.fx.Node) -> str | None:
    target = node.target
    if node.op == "call_method" and node.target in (
        "all",
        "any",
        "argmax",
        "argmin",
    ):
        return str(node.target)
    if node.op == "call_function" and target in (
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
        torch.ops.aten.all.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any.default,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
    ):
        return str(target)
    return None


def lower_full_scalar(node: torch.fx.Node) -> Any | None:
    if node.op != "call_function" or node.target is not torch.ops.aten.full.default:
        return None
    shape = normalize_shape(node.args[0])
    if shape != ():
        return None
    return node.args[1]


def lower_squeeze(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op == "call_method" and node.target == "squeeze":
        source_node = node.args[0]
    elif node.op == "call_function" and node.target in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        source_node = node.args[0]
    else:
        return None
    if (
        isinstance(source_node, torch.fx.Node)
        and source_node in local_reduce_store_sources
    ):
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node]
        return _cute_arg(source_node, env)
    return None


def lower_getitem(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if node.op != "call_function" or node.target is not operator.getitem:
        return None
    source_node, index = node.args
    if not isinstance(source_node, torch.fx.Node) or not isinstance(index, int):
        return None
    source = _cute_arg(source_node, env)
    if source_node in local_reduce_store_sources:
        local_reduce_store_sources[node] = local_reduce_store_sources[source_node][
            index
        ]
    return source[index]


def lower_prepare_softmax_online(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    if (
        node.op != "call_function"
        or node.target is not inductor_prims.prepare_softmax_online
    ):
        return None
    input_node = node.args[0]
    dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if info.layout.needs_physical_combine:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: prepare_softmax_online "
            "needs a multi-value generated physical reducer"
        )
    if not OrderedSet(normalize_reduce_dims(dim)) <= OrderedSet(
        info.layout.reduce_dims
    ):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: prepare_softmax_online "
            "currently supports only the grouped dimension"
        )
    source = _cute_arg(input_node, env)
    max_reduced = kernel.cse.generate(
        kernel.body,
        f'{source}.reduce(cute.ReductionOp.MAX, init_val=float("-inf"), reduction_profile={info.layout.reduction_profile})',
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    max_keepdim = kernel.cse.generate(
        kernel.body,
        f"{max_reduced}.reshape({info.layout.keepdim_shape})",
        dtype=getattr(max_reduced, "dtype", None),
        shape=getattr(max_reduced, "shape", None),
    )
    max_store = kernel.cse.generate(
        kernel.body,
        f"{max_keepdim}.broadcast_to({source}.shape)",
        dtype=getattr(max_keepdim, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    centered = kernel.cse.generate(
        kernel.body,
        f"({source} - {max_store})",
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    exp_centered = CuteDSLOpOverrides.exp(centered)
    sum_reduced = kernel.cse.generate(
        kernel.body,
        f"{exp_centered}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile})",
        dtype=getattr(exp_centered, "dtype", None),
        shape=getattr(exp_centered, "shape", None),
    )
    sum_keepdim = kernel.cse.generate(
        kernel.body,
        f"{sum_reduced}.reshape({info.layout.keepdim_shape})",
        dtype=getattr(sum_reduced, "dtype", None),
        shape=getattr(sum_reduced, "shape", None),
    )
    sum_store = kernel.cse.generate(
        kernel.body,
        f"{sum_keepdim}.broadcast_to({source}.shape)",
        dtype=getattr(sum_keepdim, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    local_reduce_store_sources[node] = (max_store, sum_store)
    return max_store, sum_store


def lower_tensorssa_moment_reduce(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
) -> Any | None:
    """Lower grouped var/std as generated mean plus squared residual reductions."""
    reduction = moment_reduction_from_node(node)
    if reduction is None:
        return None
    input_node, dim, correction, keepdim, reduction_type = reduction
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if info.layout.needs_physical_combine:
        raise NotImplementedError(
            "unsupported FlexGEMM physical local reduction: moment reductions "
            "need matching QuACK combine/finalize callbacks"
        )
    if not OrderedSet(normalize_reduce_dims(dim)) <= OrderedSet(
        info.layout.reduce_dims
    ):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: moment reductions "
            "currently support only the innermost grouped dimension"
        )
    if correction is None:
        correction = 1
    if not isinstance(correction, (int, float)):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: dynamic variance correction"
        )
    denominator = info.group_size - correction
    if denominator <= 0:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: variance correction "
            "must be smaller than the group size"
        )
    source = _cute_arg(input_node, env)
    sum_reduced = kernel.cse.generate(
        kernel.body,
        f"{source}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile})",
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    mean_reduced = kernel.cse.generate(
        kernel.body,
        f"{sum_reduced} / {float(info.group_size)!r}",
        dtype=getattr(sum_reduced, "dtype", None),
        shape=getattr(sum_reduced, "shape", None),
    )
    mean_keepdim = kernel.cse.generate(
        kernel.body,
        f"{mean_reduced}.reshape({info.layout.keepdim_shape})",
        dtype=getattr(mean_reduced, "dtype", None),
        shape=getattr(mean_reduced, "shape", None),
    )
    mean_store = kernel.cse.generate(
        kernel.body,
        f"{mean_keepdim}.broadcast_to({source}.shape)",
        dtype=getattr(mean_keepdim, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    centered = kernel.cse.generate(
        kernel.body,
        f"({source} - {mean_store})",
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    squared = kernel.cse.generate(
        kernel.body,
        f"({centered} * {centered})",
        dtype=getattr(centered, "dtype", None),
        shape=getattr(centered, "shape", None),
    )
    var_reduced = kernel.cse.generate(
        kernel.body,
        f"{squared}.reduce(cute.ReductionOp.ADD, init_val=0.0, reduction_profile={info.layout.reduction_profile}) / {float(denominator)!r}",
        dtype=getattr(squared, "dtype", None),
        shape=getattr(squared, "shape", None),
    )
    if reduction_type == "std":
        result = kernel.cse.generate(
            kernel.body,
            f"cute.math.sqrt({var_reduced})",
            dtype=getattr(var_reduced, "dtype", None),
            shape=getattr(var_reduced, "shape", None),
        )
    else:
        result = var_reduced
    keepdim_source = kernel.cse.generate(
        kernel.body,
        f"{result}.reshape({info.layout.keepdim_shape})",
        dtype=getattr(result, "dtype", None),
        shape=getattr(result, "shape", None),
    )
    local_reduce_store_sources[node] = kernel.cse.generate(
        kernel.body,
        f"{keepdim_source}.broadcast_to({source}.shape)",
        dtype=getattr(keepdim_source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    if keepdim:
        return keepdim_source
    return result


def lower_tensorssa_reduce(
    node: torch.fx.Node,
    env: dict[torch.fx.Node, Any],
    kernel: Any,
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo],
    local_reduce_store_sources: dict[torch.fx.Node, Any],
    local_reduce_physical_reductions: dict[torch.fx.Node, FlexGemmPhysicalReduction]
    | None = None,
) -> Any | None:
    """Lower value reductions while deferring cross-fragment finalization to QuACK."""
    reduction = reduction_from_node(node)
    if reduction is None:
        return None
    input_node, dim, keepdim, dtype, reduction_type = reduction
    if dtype is not None:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: explicit reduction dtype"
        )
    if not isinstance(input_node, torch.fx.Node):
        return None
    if input_node not in grouped_tensors:
        raise partial_vector_reduction_error()
    info = grouped_tensors[input_node]
    if not OrderedSet(normalize_reduce_dims(dim)) <= OrderedSet(
        info.layout.reduce_dims
    ):
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: currently support only "
            "the innermost grouped dimension"
        )
    desc = FLEX_GEMM_TENSORSSA_REDUCTIONS.get(reduction_type)
    if desc is None:
        raise NotImplementedError(
            "unsupported FlexGEMM epilogue local reduction: "
            f"reduction_type={reduction_type!r}"
        )
    source = _cute_arg(input_node, env)
    if info.layout.needs_physical_combine:
        physical_reduction = FLEX_GEMM_REDUCTIONS.get(reduction_type)
        if physical_reduction is None:
            raise NotImplementedError(
                "unsupported FlexGEMM local reduction: "
                f"reduction_type={reduction_type!r} needs a generated physical reducer"
            )
        if local_reduce_physical_reductions is not None:
            local_reduce_physical_reductions[node] = dataclasses.replace(
                physical_reduction,
                finalize_expr=physical_reduction.finalize_expr.format(
                    group_size=info.group_size
                ),
            )
        if info.axis == 0:
            local_reduce_store_sources[node] = source
            return source
    reduced = kernel.cse.generate(
        kernel.body,
        f"{source}.reduce({desc.cute_op}, init_val={desc.init_val}, reduction_profile={info.layout.reduction_profile})",
        dtype=getattr(source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    if desc.scale_by_group_size and not info.layout.needs_physical_combine:
        reduced = kernel.cse.generate(
            kernel.body,
            f"{reduced} / {float(info.group_size)!r}",
            dtype=getattr(reduced, "dtype", None),
            shape=getattr(reduced, "shape", None),
        )
    keepdim_source = kernel.cse.generate(
        kernel.body,
        f"{reduced}.reshape({info.layout.keepdim_shape})",
        dtype=getattr(reduced, "dtype", None),
        shape=getattr(reduced, "shape", None),
    )
    local_reduce_store_sources[node] = kernel.cse.generate(
        kernel.body,
        f"{keepdim_source}.broadcast_to({source}.shape)",
        dtype=getattr(keepdim_source, "dtype", None),
        shape=getattr(source, "shape", None),
    )
    if keepdim:
        return keepdim_source
    return reduced
