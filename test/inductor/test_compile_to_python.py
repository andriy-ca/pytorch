# Owner(s): ["module: inductor"]
import ast
import textwrap

import torch
import torch.utils._pytree as pytree
from torch._inductor import compile_to_python
from torch._inductor.standalone_compile import NoRunnableInductorModuleError
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _capture(m, x):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), mirroring how ``torch.compiler.precompile`` feeds a post-AOTAutograd inner
    graph to ``torch._inductor.compile_to_python``. Tracing runs ``m(x)`` once, so pass
    a throwaway module."""
    pnames = [n for n, _ in m.named_parameters()]
    bnames = [n for n, _ in m.named_buffers()]
    pb = [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()]
    k = len(pnames)

    def flat_fn(flat):
        params = dict(zip(pnames, flat[:k]))
        buffers = dict(zip(bnames, flat[k : k + len(bnames)]))
        with stateless._reparametrize_module(
            m, {**params, **buffers}, tie_weights=True
        ):
            out = m(flat[-1])
        return pytree.tree_flatten(out)[0]

    with torch.enable_grad():
        return make_fx(flat_fn)(pb + [x])


def _flat_inputs(m, x):
    return (
        [p for _, p in m.named_parameters()] + [b for _, b in m.named_buffers()] + [x]
    )


def _exec(src):
    ns = {"__name__": "_compiled"}
    exec(compile(src, "<compiled>", "exec"), ns)
    return ns["call"]


def _extract_call(src):
    """Return the dedented source of the module-level ``call`` entry point. The expect
    goldens lock this runtime entry point only: the rest of the emitted module (imports,
    the inert compile-time auto-tuning docstring, the ``# AOT ID`` global-counter
    comment) carries build- and ordering-dependent noise that should not be goldened."""
    mod = ast.parse(src)
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            body = "\n".join(src.split("\n")[node.lineno - 1 : node.end_lineno])
            return textwrap.dedent(body)
    raise AssertionError("generated module has no module-level def call")


def _normalize_device(src):
    """Rewrite the baked-in CUDA device ordinal in a captured wrapper to 0.

    Inductor bakes the input tensor's device index into the emitted wrapper
    (``_DeviceGuard(N)``, ``set_device(N)``, ``get_raw_stream(N)``, ``raw_streamN``).
    The CUDA goldens run on whatever single GPU the test process is pinned to, so
    normalize that ordinal to 0 to keep the golden independent of which visible device
    the test lands on. A no-op when the current device is already 0."""
    idx = torch.cuda.current_device()
    if idx == 0:
        return src
    return (
        src.replace(f"_DeviceGuard({idx})", "_DeviceGuard(0)")
        .replace(f"set_device({idx})", "set_device(0)")
        .replace(f"get_raw_stream({idx})", "get_raw_stream(0)")
        .replace(f"raw_stream{idx}", "raw_stream0")
    )


class _Pointwise(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 2.0 + 1.0)


class _SumDim1(torch.nn.Module):
    def forward(self, x):
        return x.sum(dim=1)


class _Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class TestInductorCompileToPythonCodegen(TestCase):
    # graph_partition is pinned off so the entry point is the stable top-level
    # ``def call(args)`` rather than the default ``Runner.call`` method wrapper. The
    # extern-kernel / CPU codegen the goldens lock is identical either way and does not
    # depend on whether torch was built with CUDA (these are CPU tensors).
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"graph_partition": False}
        )
        return src, _extract_call(src)

    def test_addmm_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (3, ), (1, ), 'input')
    assert_size_stride(arg2_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, addmm], Original ATen: [aten.t, aten.addmm]
    extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg1_1
    del arg2_1
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_matmul_extern_kernel_codegen(self):
        m = torch.nn.Linear(4, 3, bias=False).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg1_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, mm], Original ATen: [aten.t, aten.mm]
    extern_kernels.mm(arg1_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), out=buf0)
    del arg0_1
    del arg1_1
    return (buf0, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_pointwise_codegen(self):
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
        x = torch.randn(5, 4)
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (3, ), (1, ), 'input')
    assert_size_stride(arg2_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    buf0 = empty_strided_cpu((5, 3), (3, 1), torch.float32)
    # Topologically Sorted Source Nodes: [t, addmm], Original ATen: [aten.t, aten.addmm]
    extern_kernels.addmm(arg1_1, arg2_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), alpha=1, beta=1, out=buf0)
    del arg0_1
    del arg1_1
    del arg2_1
    buf1 = buf0; del buf0  # reuse
    cpp_fused_relu_0(buf1)
    return (buf1, )""",
        )
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_benchmark_harness_suppressed(self):
        # #187858 pins benchmark_harness=False, so the emitted module is runnable rather
        # than an Inductor profiling harness: none of these debug entry points appear.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _call_src = self._inner_call(m, x)
        for marker in (
            "benchmark_compiled_module",
            "def get_args(",
            "compiled_module_main",
            "print_performance",
        ):
            self.assertNotIn(marker, src)

    def test_no_runnable_module_for_no_compute(self):
        # A graph with no compute (returns its input unchanged) lowers to no module-level
        # ``call``, so there is nothing runnable to inline.
        m = torch.nn.Identity().eval()
        x = torch.randn(5, 4)
        gm = _capture(m, x)
        with self.assertRaises(NoRunnableInductorModuleError):
            compile_to_python(gm, _flat_inputs(m, x))


@requires_cuda_and_triton
class TestInductorCompileToPythonCudaCodegen(TestCase):
    # On CUDA, compile_to_python emits actual @triton.jit kernels rather than the CPU
    # extern_kernels / cpp_fused path the sibling class goldens. The expect goldens
    # lock the device-independent ``call()`` body only: its grid launch, the
    # empty_strided_cuda allocations, and the stream plumbing carry NO autotuning
    # artifacts -- XBLOCK / num_warps are chosen inside ``.run`` at launch time and the
    # arch-specific DeviceProperties live in the kernel decorator, not in ``call``. The
    # hardware- and Triton-version-dependent kernel body is therefore checked
    # structurally (assertIn), matching the inductor-suite convention. graph_partition
    # is pinned off so the entry point is the stable top-level ``def call(args)``.
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm, _flat_inputs(m, x), options={"graph_partition": False}
        )
        return src, _normalize_device(_extract_call(src))

    def test_pointwise_triton_kernel_codegen(self):
        m = _Pointwise().eval().cuda()
        x = torch.randn(128, 64, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 64), (64, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((128, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul, add, relu], Original ATen: [aten.mul, aten.add, aten.relu]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_relu_0.run(arg0_1, buf0, 8192, stream=raw_stream0)
        del arg0_1
    return (buf0, )""",
        )
        # The pointwise fusion lowers to a single @triton.jit pointwise kernel; the
        # call drives it directly with no extern (BLAS/cuDNN) kernel.
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.pointwise", src)
        self.assertIn("def triton_poi_fused_add_mul_relu_0(", src)
        self.assertIn("tl.load", src)
        self.assertIn("tl.store", src)
        self.assertIn("tl.maximum", src)  # the fused relu
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_reduction_triton_kernel_codegen(self):
        m = _SumDim1().eval().cuda()
        x = torch.randn(64, 256, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (64, 256), (256, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [sum_1], Original ATen: [aten.sum]
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_sum_0.run(arg0_1, buf0, 64, 256, stream=raw_stream0)
        del arg0_1
    return (buf0, )""",
        )
        # A row reduction lowers to a (persistent) reduction Triton kernel doing the
        # cross-row ``tl.sum``; there is no extern kernel.
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        self.assertIn("def triton_per_fused_sum_0(", src)
        self.assertIn("tl.sum", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_addmm_relu_fused_triton_epilogue_codegen(self):
        # CUDA counterpart of test_addmm_relu_fused_pointwise_codegen: the matmul is an
        # extern BLAS call but the relu epilogue fuses into a @triton.jit kernel
        # (triton_poi_fused_addmm_relu_0) rather than the CPU cpp_fused_relu_0.
        m = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval().cuda()
        x = torch.randn(5, 4, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertExpectedInline(
            call_src,
            """\
def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg2_1, (5, 4), (4, 1), 'input')
    assert_size_stride(arg0_1, (3, 4), (4, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((5, 3), (3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [t, addmm], Original ATen: [aten.t, aten.addmm]
        extern_kernels.mm(arg2_1, reinterpret_tensor(arg0_1, (4, 3), (1, 4), 0), out=buf0)
        del arg0_1
        del arg2_1
        assert_size_stride(arg1_1, (3, ), (1, ), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [addmm, relu], Original ATen: [aten.addmm, aten.relu]
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_0.run(buf1, arg1_1, 15, stream=raw_stream0)
        del arg1_1
    return (buf1, )""",
        )
        self.assertIn("extern_kernels.mm", call_src)
        self.assertIn("@triton.jit", src)
        self.assertIn("def triton_poi_fused_addmm_relu_0(", src)
        self.assertIn("tl.maximum", src)  # the fused relu epilogue
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_softmax_fused_reduction_triton_kernel(self):
        # softmax is the canonical multi-stage fusion: max, subtract, exp, sum, divide
        # all collapse into ONE persistent-reduction Triton kernel. Its exact name
        # (the decomposition route) varies, so this checks structure + numerics rather
        # than goldening the call body.
        m = _Softmax().eval().cuda()
        x = torch.randn(32, 128, device="cuda")
        src, call_src = self._inner_call(m, x)
        self.assertIn("@triton.jit", src)
        self.assertIn("@triton_heuristics.persistent_reduction", src)
        self.assertIn("tl.sum", src)  # the denominator reduction
        self.assertIn("libdevice.exp", src)  # the numerator
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))


if __name__ == "__main__":
    run_tests()
