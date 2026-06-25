# Owner(s): ["module: inductor"]
import ast
import re
import textwrap

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._inductor import compile_to_python, config
from torch._inductor.standalone_compile import (
    _defines_module_level_call,
    _extract_runnable_module,
    NoRunnableInductorModuleError,
)
from torch._inductor.utils import fresh_cache
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.triton_utils import requires_cuda_and_triton


def _capture(m, x, tracing_mode="real"):
    """Trace ``m(x)`` into a flat-input ATen graph (params+buffers then ``x`` lifted to
    inputs), mirroring how ``torch.compiler.precompile`` feeds a post-AOTAutograd inner
    graph to ``torch._inductor.compile_to_python``. Tracing runs ``m(x)`` once, so pass
    a throwaway module. ``tracing_mode="symbolic"`` produces a graph with dynamic dims."""
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
        return make_fx(flat_fn, tracing_mode=tracing_mode)(pb + [x])


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
    normalize whatever ordinal actually appears to 0 (regex on the emitted ordinal, not
    on current_device(), so an explicitly non-current device would still normalize)."""
    src = re.sub(r"_DeviceGuard\(\d+\)", "_DeviceGuard(0)", src)
    src = re.sub(r"set_device\(\d+\)", "set_device(0)", src)
    src = re.sub(r"get_raw_stream\(\d+\)", "get_raw_stream(0)", src)
    src = re.sub(r"raw_stream\d+", "raw_stream0", src)
    return src


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
    # depend on whether torch was built with CUDA (these are CPU tensors). size_asserts /
    # nan_asserts are pinned so the goldened assert_size_stride lines do not drift with the
    # ambient TORCHINDUCTOR_SIZE_ASSERTS / TORCHINDUCTOR_NAN_ASSERTS env.
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm,
            _flat_inputs(m, x),
            options={
                "graph_partition": False,
                "size_asserts": True,
                "nan_asserts": False,
            },
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
    # structurally (assertIn), matching the inductor-suite convention. graph_partition is
    # pinned off for a stable top-level ``def call(args)``, and size_asserts / nan_asserts
    # are pinned so the goldened assert_size_stride lines do not drift with the ambient env.
    def _inner_call(self, m, x):
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm,
            _flat_inputs(m, x),
            options={
                "graph_partition": False,
                "size_asserts": True,
                "nan_asserts": False,
            },
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
        # the exp numerator: libdevice.exp / tl_math.exp / tl.exp depending on fast-math
        self.assertIn(".exp(", src)
        self.assertNotIn("extern_kernels", call_src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_max_autotune_single_runnable_module(self):
        # Integration check that compile_to_python works under max_autotune, which drives
        # the GEMM autotuning machinery (Triton templates plus the SubgraphChoiceCaller
        # decompose_k path, a candidate when k >= 32*m and k >= 32*n). The contract under
        # test is that exactly ONE runnable module is still captured and runs correctly
        # under that machinery; a regression that let a nested benchmark lowering leak into
        # the capture would trip the >1-module RuntimeError here.
        m = (
            torch.nn.Sequential(torch.nn.Linear(2048, 8, bias=False), torch.nn.ReLU())
            .eval()
            .cuda()
        )
        x = torch.randn(8, 2048, device="cuda")
        gm = _capture(m, x)
        src, _cache = compile_to_python(
            gm,
            _flat_inputs(m, x),
            options={
                "graph_partition": False,
                "max_autotune": True,
                "max_autotune_gemm_backends": "TRITON",
            },
        )
        self.assertTrue(_defines_module_level_call(src))
        self.assertNotIn("benchmark_", src)  # no nested benchmark lowering leaked
        with torch.no_grad():
            self.assertEqual(
                _exec(src)(_flat_inputs(m, x))[0], m(x), atol=1e-2, rtol=1e-2
            )


class TestInductorCompileToPythonContract(TestCase):
    # Contract + branch coverage the codegen-golden classes above do not exercise: the
    # option pin-override, the cache return value, the structural module detector's two
    # forms, the error paths, and warm-cache capture. All CPU, so no GPU is required.
    def test_rejects_non_graphmodule(self):
        with self.assertRaises(TypeError):
            compile_to_python("not a graph module", [])

    def test_defines_module_level_call_both_forms(self):
        # graph_partition off emits a top-level ``def call``; on emits ``call = runner.call``.
        # Both must be recognized; a ``def call`` inside a string literal and unparsable
        # source must not be.
        self.assertTrue(
            _defines_module_level_call("def call(args):\n    return args\n")
        )
        self.assertTrue(
            _defines_module_level_call(
                "class Runner:\n    def call(self):\n        pass\ncall = runner.call\n"
            )
        )
        self.assertFalse(_defines_module_level_call("x = 1\ndef other():\n    pass\n"))
        self.assertFalse(
            _defines_module_level_call('s = """\ndef call(a):\n    pass\n"""\n')
        )
        self.assertFalse(_defines_module_level_call("def (:\n"))

    def test_extract_runnable_module_zero_and_multiple(self):
        # 0 runnable modules is the expected no-compute case (typed exception callers
        # special-case); >1 is an unexpected invariant violation (plain RuntimeError).
        with self.assertRaises(NoRunnableInductorModuleError):
            _extract_runnable_module([])
        # >1 raises a plain RuntimeError, NOT the NoRunnableInductorModuleError subclass,
        # so distinguish them (NoRunnableInductorModuleError is itself a RuntimeError).
        with self.assertRaises(RuntimeError) as cm:
            _extract_runnable_module(
                ["def call(a):\n    pass", "def call(b):\n    pass"]
            )
        self.assertNotIsInstance(cm.exception, NoRunnableInductorModuleError)

    def test_pins_override_conflicting_user_options(self):
        # The benchmark_harness/cpp_wrapper pins must beat conflicting user options so the
        # captured module stays the runnable python wrapper rather than a C++ wrapper or a
        # profiling harness.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={
                "graph_partition": False,
                "cpp_wrapper": True,
                "benchmark_harness": True,
            },
        )
        self.assertIn("def call(args):", src)
        self.assertNotIn('extern "C"', src)
        self.assertNotIn("AOTInductorModel", src)
        self.assertNotIn("benchmark_compiled_module", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_cache_bytes_returned_for_cacheable_graph(self):
        # The bytes come from the AOTAutograd cache artifact, so they require both
        # force_disable_caches off AND enable_autograd_cache on. These flags are
        # env-authoritative on PyTorch CI's cache-disabled shards (cannot be patched back
        # on), where compile_to_python correctly returns None; skip there --
        # test_no_cache_when_caches_disabled covers the None path.
        if (
            config.force_disable_caches
            or not config.fx_graph_cache
            or not functorch_config.enable_autograd_cache
        ):
            self.skipTest("requires inductor + autograd caches enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        _src, cache = compile_to_python(
            _capture(m, x), _flat_inputs(m, x), options={"graph_partition": False}
        )
        self.assertIsInstance(cache, bytes)
        self.assertGreater(len(cache), 0)

    def test_no_cache_when_caches_disabled(self):
        # With caches disabled there is no saveable artifact, so cache is None; the source
        # still runs (the kernels JIT-compile from it on first call).
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, cache = compile_to_python(
            _capture(m, x),
            _flat_inputs(m, x),
            options={"graph_partition": False, "force_disable_caches": True},
        )
        self.assertIsNone(cache)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_graph_partition_runner_call_form(self):
        # graph_partition=True emits the Runner form (``call = runner.call``) instead of a
        # top-level ``def call``; _extract_runnable_module must recognize it and the emitted
        # module must still run.
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        src, _cache = compile_to_python(
            _capture(m, x), _flat_inputs(m, x), options={"graph_partition": True}
        )
        self.assertIn("call = runner.call", src)
        with torch.no_grad():
            self.assertEqual(_exec(src)(_flat_inputs(m, x))[0], m(x))

    def test_dynamic_shapes_emits_symbolic_codegen(self):
        # dynamic_shapes="from_graph" on a symbolically-traced graph emits a call() keyed
        # on symbolic sizes (sN) rather than baked constants, and the single module runs at
        # multiple shapes. (The default "from_example_inputs" specializes instead -- the
        # other tests exercise that static path.) Symbol names are non-deterministic, so
        # assert structure + multi-shape numerics rather than goldening.
        m = _Pointwise().eval()
        x = torch.randn(8, 4)
        gm = _capture(m, x, tracing_mode="symbolic")
        src, _cache = compile_to_python(
            gm,
            _flat_inputs(m, x),
            dynamic_shapes="from_graph",
            options={"graph_partition": False},
        )
        call_src = _extract_call(src)
        self.assertRegex(call_src, r"\bs\d+\b")  # a symbolic size symbol is present
        self.assertNotIn("(8, 4)", call_src)  # the input shape is not baked in
        fn = _exec(src)
        for n in (8, 16, 5):
            xi = torch.randn(n, 4)
            with torch.no_grad():
                self.assertEqual(fn(_flat_inputs(m, xi))[0], m(xi))

    @config.patch({"compile_threads": 1})
    def test_warm_cache_still_captures_source(self):
        # The capture hook also fires on cache restore, so a second compile of the same
        # graph (warm FxGraphCache) still yields a runnable module even though no fresh
        # codegen happens. fresh_cache isolates an empty cache dir so the first compile is a
        # guaranteed miss and the second a guaranteed hit; compile_threads=1 keeps codegen
        # in-process so the hit counter is deterministic. The warm path needs caching, which
        # is env-disabled on some CI shards (force_disable_caches cannot be patched back on),
        # so skip there.
        if config.force_disable_caches or not config.fx_graph_cache:
            self.skipTest("requires inductor FxGraphCache enabled")
        m = torch.nn.Linear(4, 3).eval()
        x = torch.randn(5, 4)
        with fresh_cache():
            counters.clear()
            src1, _ = compile_to_python(
                _capture(m, x), _flat_inputs(m, x), options={"graph_partition": False}
            )
            hits_after_first = counters["inductor"]["fxgraph_cache_hit"]
            src2, _ = compile_to_python(
                _capture(m, x), _flat_inputs(m, x), options={"graph_partition": False}
            )
            hits_after_second = counters["inductor"]["fxgraph_cache_hit"]
        self.assertEqual(hits_after_first, 0)
        self.assertEqual(hits_after_second, 1)
        self.assertIn("def call(args):", src1)
        self.assertIn("def call(args):", src2)
        with torch.no_grad():
            self.assertEqual(_exec(src2)(_flat_inputs(m, x))[0], m(x))


if __name__ == "__main__":
    run_tests()
