# mypy: allow-untyped-defs
import dataclasses
import hashlib
import operator
from typing import Any

import torch
from torch._inductor import inductor_prims
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
    upcast_compute_type,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    _cute_arg,
    _cute_call,
    _local_reduce_store_arg,
    FlexGemmPhysicalReduction,
    grouped_tensor_layout,
    GroupedTensorSSAInfo,
    has_local_reduce_store_source,
    is_shape_preserving_pointwise_node,
    iter_fx_node_inputs,
    lower_full_scalar,
    lower_getitem,
    lower_prepare_softmax_online,
    lower_squeeze,
    lower_tensorssa_moment_reduce,
    lower_tensorssa_reduce,
    lower_view_or_reshape,
    moment_reduction_from_node,
    normalize_reduce_dims,
    propagate_grouped_tensorssa_info,
    reduction_from_node,
    unsupported_reduction_from_node,
)
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges


class FlexGemmCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class FlexGemmCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class FlexGemmCuteDSLKernel:
    def __init__(self) -> None:
        self.body = FlexGemmCuteDSLBody()
        self.cse = FlexGemmCuteDSLCSE()


class FlexGemmCuteDSLOpOverrides(CuteDSLOpOverrides):
    # Aten add/sub carry alpha as schema sugar; CuTeDSL only needs the scaled RHS.
    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                "unsupported kwargs for FlexGEMM epilogue op _to_copy: "
                f"{unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)


@dataclasses.dataclass(frozen=True)
class FlexGemmOutputPlan:
    """Classify the FlexGEMM body output into a main result and aux returns."""

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce_aux: torch.fx.Node | None = None
    local_reduce_group: int | None = None
    local_reduce_axis: int | None = None


@dataclasses.dataclass(frozen=True)
class FlexGemmLocalReduceContract:
    aux: torch.fx.Node
    group: int
    axis: int


def view_or_reshape_shape(node: torch.fx.Node) -> tuple[Any, ...] | None:
    if node.op == "call_method" and node.target in ("view", "reshape"):
        return tuple(node.args[1:])
    if node.op == "call_function" and node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        shape = node.args[1]
        return tuple(shape) if isinstance(shape, (tuple, list, torch.Size)) else None
    return None


def squeeze_source_node(node: torch.fx.Node) -> torch.fx.Node | None:
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
    return source_node if isinstance(source_node, torch.fx.Node) else None


def local_reduce_contract_from_aux(
    graph_module: torch.fx.GraphModule, aux: torch.fx.Node
) -> FlexGemmLocalReduceContract | None:
    """Derive the local-reduce ABI from traced grouped-reduction provenance."""
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo] = {}
    local_reduce_nodes: dict[torch.fx.Node, FlexGemmLocalReduceContract] = {}
    for node in graph_module.graph.nodes:
        if node.op == "output":
            break
        if node.op not in ("call_function", "call_method"):
            continue
        shape = view_or_reshape_shape(node)
        if shape is not None:
            source_node = node.args[0]
            if (
                isinstance(source_node, torch.fx.Node)
                and source_node in local_reduce_nodes
            ):
                local_reduce_nodes[node] = local_reduce_nodes[source_node]
                continue
            layout = grouped_tensor_layout(shape)
            if layout is not None and isinstance(source_node, torch.fx.Node):
                grouped_tensors[node] = GroupedTensorSSAInfo(layout)
                continue
        reduction = reduction_from_node(node)
        if reduction is not None:
            input_node, dim, _, dtype, _ = reduction
            if isinstance(input_node, torch.fx.Node):
                info = grouped_tensors.get(input_node)
                if info is not None:
                    if dtype is not None:
                        raise NotImplementedError(
                            "unsupported FlexGEMM epilogue local reduction: "
                            "explicit reduction dtype"
                        )
                    if not OrderedSet(normalize_reduce_dims(dim)) <= OrderedSet(
                        info.layout.reduce_dims
                    ):
                        raise NotImplementedError(
                            "unsupported FlexGEMM epilogue local reduction: currently "
                            "support only the innermost grouped dimension"
                        )
                    local_reduce_nodes[node] = FlexGemmLocalReduceContract(
                        node, info.group_size, info.axis
                    )
                    continue
        if (
            node.op == "call_function"
            and node.target is inductor_prims.prepare_softmax_online
        ):
            input_node = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            if isinstance(input_node, torch.fx.Node):
                info = grouped_tensors.get(input_node)
                if info is not None and OrderedSet(
                    normalize_reduce_dims(dim)
                ) <= OrderedSet(info.layout.reduce_dims):
                    local_reduce_nodes[node] = FlexGemmLocalReduceContract(
                        node, info.group_size, info.axis
                    )
                    continue
        moment_reduction = moment_reduction_from_node(node)
        if moment_reduction is not None:
            input_node, dim, _, _, _ = moment_reduction
            if isinstance(input_node, torch.fx.Node):
                info = grouped_tensors.get(input_node)
                if info is not None:
                    if not OrderedSet(normalize_reduce_dims(dim)) <= OrderedSet(
                        info.layout.reduce_dims
                    ):
                        raise NotImplementedError(
                            "unsupported FlexGEMM epilogue local reduction: currently "
                            "support only the innermost grouped dimension"
                        )
                    local_reduce_nodes[node] = FlexGemmLocalReduceContract(
                        node, info.group_size, info.axis
                    )
                    continue
        unsupported_reduction = unsupported_reduction_from_node(node)
        if unsupported_reduction is not None:
            input_node = node.args[0]
            if isinstance(input_node, torch.fx.Node) and input_node in grouped_tensors:
                raise NotImplementedError(
                    "unsupported FlexGEMM epilogue local reduction: "
                    f"{unsupported_reduction} does not map to a CuTe TensorSSA"
                )
        squeeze_source = squeeze_source_node(node)
        if squeeze_source in local_reduce_nodes:
            local_reduce_nodes[node] = local_reduce_nodes[squeeze_source]
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            source_node = node.args[0]
            if (
                isinstance(source_node, torch.fx.Node)
                and source_node in local_reduce_nodes
            ):
                local_reduce_nodes[node] = local_reduce_nodes[source_node]
                continue
        if is_shape_preserving_pointwise_node(node):
            grouped_info = propagate_grouped_tensorssa_info(node, grouped_tensors)
            if grouped_info is not None:
                grouped_tensors[node] = grouped_info
            source_contracts = [
                local_reduce_nodes[arg]
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in local_reduce_nodes
            ]
            if source_contracts:
                contract = source_contracts[0]
                if any(
                    item.group != contract.group or item.axis != contract.axis
                    for item in source_contracts
                ):
                    raise NotImplementedError(
                        "FlexGEMM local reductions do not support mixing different "
                        "local-reduce contracts"
                    )
                local_reduce_nodes[node] = FlexGemmLocalReduceContract(
                    node, contract.group, contract.axis
                )
    return local_reduce_nodes.get(aux)


def output_plan(
    graph_module: torch.fx.GraphModule,
) -> FlexGemmOutputPlan:
    output_nodes = [node for node in graph_module.graph.nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one output node")
    output_value = output_nodes[0].args[0]
    if isinstance(output_value, (tuple, list)):
        if len(output_value) == 1:
            output_value = output_value[0]
        else:
            output, *aux_outputs = output_value
            if len(aux_outputs) == 1 and isinstance(aux_outputs[0], torch.fx.Node):
                aux = aux_outputs[0]
                contract = local_reduce_contract_from_aux(graph_module, aux)
                output_meta = (
                    output.meta.get("val")
                    if isinstance(output, torch.fx.Node)
                    else None
                )
                aux_meta = aux.meta.get("val")
                if (
                    contract is not None
                    and aux_meta is not None
                    and output_meta is not None
                ):
                    aux_shape = tuple(aux_meta.shape)
                    output_shape = tuple(output_meta.shape)
                    expected_aux_shape = list(output_shape)
                    expected_aux_shape[contract.axis - 2] //= contract.group
                    if tuple(expected_aux_shape) == aux_shape:
                        return FlexGemmOutputPlan(
                            output,
                            local_reduce_aux=aux,
                            local_reduce_group=contract.group,
                            local_reduce_axis=contract.axis,
                        )
            return FlexGemmOutputPlan(output, tuple(aux_outputs))
    if not isinstance(output_value, torch.fx.Node):
        raise NotImplementedError("FlexGEMM expects one tensor output")
    return FlexGemmOutputPlan(output_value)


def gemm_node(
    graph_module: torch.fx.GraphModule, gemm_op: torch._ops.OpOverload
) -> torch.fx.Node:
    gemm_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and node.target == gemm_op
    ]
    if len(gemm_nodes) != 1:
        raise NotImplementedError("FlexGEMM expects one GEMM body")
    return gemm_nodes[0]


def materialize_flex_gemm_epilogue(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    epilogue_arg_placeholders: tuple[torch.fx.Node, ...] = (),
) -> tuple[str, str]:
    """Build the generated CuTeDSL epilogue callable from the traced FX body."""
    gemm = gemm_node(graph_module, gemm_op)
    outputs = output_plan(graph_module)
    kernel = FlexGemmCuteDSLKernel()
    env: dict[torch.fx.Node, Any] = {
        gemm: CuteDSLCSEVariable(
            "acc", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )
    }
    grouped_tensors: dict[torch.fx.Node, GroupedTensorSSAInfo] = {}
    local_reduce_store_sources: dict[torch.fx.Node, Any] = {}
    local_reduce_physical_reductions: dict[
        torch.fx.Node, FlexGemmPhysicalReduction
    ] = {}
    with V.set_kernel_handler(kernel), V.set_ops_handler(FlexGemmCuteDSLOpOverrides()):
        for index, node in enumerate(epilogue_arg_placeholders):
            epilogue_arg_meta = node.meta["val"]
            physical_dtype = (
                torch.uint8
                if epilogue_arg_meta.dtype is torch.bool
                else epilogue_arg_meta.dtype
            )
            logical_dtype = upcast_compute_type(epilogue_arg_meta.dtype)
            env[node] = CuteDSLCSEVariable(
                f"aux{index}",
                ValueRanges.unknown(),
                dtype=physical_dtype,
                shape=(1,),
            )
            if logical_dtype != physical_dtype:
                env[node] = FlexGemmCuteDSLOpOverrides.to_dtype(
                    env[node], logical_dtype, use_compute_types=False
                )

        for node in graph_module.graph.nodes:
            if node is gemm or node.op in ("placeholder", "output"):
                continue
            with V.set_current_node(node):
                node_args = tuple(_cute_arg(arg, env) for arg in node.args)
                node_kwargs = {
                    key: _cute_arg(value, env) for key, value in node.kwargs.items()
                }
                if node.op in ("call_function", "call_method"):
                    lowered_full_scalar = lower_full_scalar(node)
                    if lowered_full_scalar is not None:
                        env[node] = lowered_full_scalar
                        continue
                    lowered_squeeze = lower_squeeze(
                        node, env, local_reduce_store_sources
                    )
                    if lowered_squeeze is not None:
                        env[node] = lowered_squeeze
                        continue
                    lowered_getitem = lower_getitem(
                        node, env, local_reduce_store_sources
                    )
                    if lowered_getitem is not None:
                        env[node] = lowered_getitem
                        continue
                    lowered_prepare_softmax = lower_prepare_softmax_online(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                    )
                    if lowered_prepare_softmax is not None:
                        env[node] = lowered_prepare_softmax
                        continue
                    lowered_view = lower_view_or_reshape(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                    )
                    if lowered_view is not None:
                        env[node] = lowered_view
                        continue
                    lowered_moment_reduce = lower_tensorssa_moment_reduce(
                        node, env, kernel, grouped_tensors, local_reduce_store_sources
                    )
                    if lowered_moment_reduce is not None:
                        env[node] = lowered_moment_reduce
                        continue
                    lowered_reduce = lower_tensorssa_reduce(
                        node,
                        env,
                        kernel,
                        grouped_tensors,
                        local_reduce_store_sources,
                        local_reduce_physical_reductions,
                    )
                    if lowered_reduce is not None:
                        env[node] = lowered_reduce
                        continue
                    unsupported_reduction = unsupported_reduction_from_node(node)
                    if unsupported_reduction is not None:
                        raise NotImplementedError(
                            "unsupported FlexGEMM epilogue local reduction: "
                            f"{unsupported_reduction} does not map to a CuTe TensorSSA "
                            "value-only reduction"
                        )
                    is_shape_preserving = is_shape_preserving_pointwise_node(node)
                    if is_shape_preserving and any(
                        isinstance(arg, torch.fx.Node)
                        and arg in local_reduce_physical_reductions
                        for arg in (*node.args, *node.kwargs.values())
                    ):
                        raise NotImplementedError(
                            "unsupported FlexGEMM physical local reduction: "
                            "post-reduction pointwise transforms require generated finalize code"
                        )
                    if is_shape_preserving and has_local_reduce_store_source(
                        (node.args, tuple(node.kwargs.values())),
                        local_reduce_store_sources,
                    ):
                        store_args = tuple(
                            _local_reduce_store_arg(
                                arg, env, local_reduce_store_sources
                            )
                            for arg in node.args
                        )
                        store_kwargs = {
                            key: _local_reduce_store_arg(
                                value, env, local_reduce_store_sources
                            )
                            for key, value in node.kwargs.items()
                        }
                        env[node] = _cute_call(node.target, store_args, store_kwargs)
                        local_reduce_store_sources[node] = env[node]
                    else:
                        env[node] = _cute_call(node.target, node_args, node_kwargs)
                    if is_shape_preserving:
                        grouped_info = propagate_grouped_tensorssa_info(
                            node, grouped_tensors
                        )
                        if grouped_info is not None:
                            grouped_tensors[node] = grouped_info
                    continue
                raise NotImplementedError(
                    f"unsupported FlexGEMM epilogue node: {node.format_node()}"
                )

    body = "\n".join(f"    {line}" for line in kernel.body.lines)
    if body:
        body += "\n"
    aux_args = [f"aux{index}" for index in range(len(epilogue_arg_placeholders))]
    epilogue_params = ", ".join(["acc", *aux_args])
    result = _cute_arg(outputs.output, env)
    if outputs.local_reduce_aux is not None:
        aux_result = local_reduce_store_sources.get(outputs.local_reduce_aux)
        if aux_result is None:
            raise NotImplementedError(
                "FlexGEMM local-reduce aux output must be produced by a grouped TensorSSA reduction"
            )
        result = f"({result}, {aux_result})"
    elif outputs.aux_outputs:
        aux_results = [_cute_arg(aux_output, env) for aux_output in outputs.aux_outputs]
        result = f"({', '.join(str(item) for item in (result, *aux_results))})"
    physical_reduction = None
    if outputs.local_reduce_aux is not None:
        physical_reduction = local_reduce_physical_reductions.get(
            outputs.local_reduce_aux
        )
    physical_reduction_payload = (
        ""
        if physical_reduction is None
        else f"\ncombine {physical_reduction.combine_expr}\nfinalize {physical_reduction.finalize_expr}"
    )
    key_payload = (
        f"{graph_module.code}\n{body}\nreturn {result}{physical_reduction_payload}"
    )
    key = hashlib.sha256(key_payload.encode()).hexdigest()[:16]
    name = f"flex_gemm_epilogue_{key}"
    local_reduce_source = ""
    if physical_reduction is not None:
        local_reduce_source = (
            f"@cute.jit\ndef {name}_local_reduce_combine_fn(lhs, rhs):\n"
            f"    return {physical_reduction.combine_expr}\n\n"
            f"@cute.jit\ndef {name}_local_reduce_finalize_fn(value):\n"
            f"    return {physical_reduction.finalize_expr}\n\n"
        )
    return (
        name,
        "import cutlass\n"
        "import cutlass.cute as cute\n"
        "import operator\n"
        "from cutlass._mlir.dialects import math as mlir_math\n\n"
        f"{local_reduce_source}"
        f"@cute.jit\ndef {name}({epilogue_params}):\n"
        f"{body}    return {result}\n",
    )
