# Owner(s): ["module: inductor"]
import logging
import os
import re
import unittest


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, try_import_ck_lib
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    _quantize_rowwise,
    _quantize_tensorwise,
    HAS_CPU,
    HAS_CUDA_AND_TRITON,
)


if HAS_CUDA_AND_TRITON:
    torch.cuda.memory._set_allocator_settings("expandable_segments:False")

log = logging.getLogger(__name__)


# patch env for tests if needed
_test_env = {}


# How many CK instances the conv autotune tests sample. The production default is
# None (uncapped); these tests cap it to keep compile time down.
_CONV_PROFILING_CONFIGS = 6


# A selected CK kernel is emitted into the generated wrapper as an
# `async_compile.rocm(...)` block defining a `rocm_fused_*` kernel that is then
# called (see torch/_inductor/codegen/rocm/rocm_kernel.py). A green assert_close
# alone does not prove CK ran -- autotune silently falls back to ATen/Triton when
# no CK candidate wins. Matching `async_compile.rocm(` in the code captured by
# run_and_get_code is the reliable signal (verified on gfx942/MI325X: the CK path
# emits `rocm_fused_mm_0 = async_compile.rocm(r'''...CKGemmTemplate...''')`).
_CK_KERNEL_RE = re.compile(r"async_compile\.rocm\(")


def _assert_ck_selected(codes):
    if not _CK_KERNEL_RE.search("\n".join(codes)):
        raise AssertionError(
            "Expected a CK (async_compile.rocm) kernel in the generated code; "
            "CK was not selected (fell back to ATen/Triton)."
        )


# A selected CK-Tile kernel embeds the CK-Tile instance name
# `ck_tile_gemm_universal_*` (see CKTileGemmOperation.name()) inside its
# async_compile.rocm block. This is more specific than _CK_KERNEL_RE (which also
# matches the classic CKGemmTemplate), so it distinguishes CK-Tile from classic CK.
_CKTILE_KERNEL_RE = re.compile(r"ck_tile_gemm_universal_")


def _assert_cktile_selected(codes):
    if not _CKTILE_KERNEL_RE.search("\n".join(codes)):
        raise AssertionError(
            "Expected a CK-Tile (ck_tile_gemm_universal_) kernel in the generated "
            "code; CK-Tile was not selected (fell back to ATen/Triton/classic-CK)."
        )


# A selected CK WMMA kernel embeds the WMMA instance alias
# `ck_devicegemm_multid_wmma_shuffle_v3_*` (see CKGemmOperation.name() with
# is_wmma=True). More specific than _CK_KERNEL_RE (which also matches the classic
# XDL CKGemmTemplate), so it distinguishes WMMA from classic-XDL CK.
_CKWMMA_KERNEL_RE = re.compile(r"ck_devicegemm_multid_wmma_shuffle_v3_")


def _assert_ckwmma_selected(codes):
    if not _CKWMMA_KERNEL_RE.search("\n".join(codes)):
        raise AssertionError(
            "Expected a CK WMMA (ck_devicegemm_multid_wmma_shuffle_v3_) kernel in "
            "the generated code; WMMA was not selected (fell back to "
            "ATen/Triton/classic-XDL-CK/CK-Tile)."
        )


# The batched alias is a *different* string: CKBatchedGemmOperation.name() emits
# `ck_device_batched_gemm_multi_d_wmma_c_shuffle_v3_*`, which _CKWMMA_KERNEL_RE
# above does not match. Asserting the non-batched pattern on a bmm test would
# therefore never fire, and the test would pass without exercising WMMA at all.
_CKWMMA_BATCHED_KERNEL_RE = re.compile(
    r"ck_device_batched_gemm_multi_d_wmma_c_shuffle_v3_"
)


def _assert_ckwmma_batched_selected(codes):
    if not _CKWMMA_BATCHED_KERNEL_RE.search("\n".join(codes)):
        raise AssertionError(
            "Expected a CK batched WMMA "
            "(ck_device_batched_gemm_multi_d_wmma_c_shuffle_v3_) kernel in the "
            "generated code; batched WMMA was not selected (fell back to "
            "ATen/Triton/classic-XDL-CK)."
        )


# And conv is a *third* distinct alias: CKGroupedConvFwdOp.name() with
# is_wmma=True emits `ck_device_grouped_convolution_fwd_multiple_abd_wmma_*`.
# Neither GEMM pattern above matches it, and _CK_KERNEL_RE only matches
# `async_compile.rocm(` which any ROCm kernel emits -- including plain XDL. So
# without this regex a WMMA conv assertion passes even when no WMMA kernel
# ran at all.
_CKWMMA_CONV_KERNEL_RE = re.compile(
    r"ck_device_grouped_convolution_fwd_multiple_abd_wmma_c_shuffle_v3_"
)


def _assert_ckwmma_conv_selected(codes):
    if not _CKWMMA_CONV_KERNEL_RE.search("\n".join(codes)):
        raise AssertionError(
            "Expected a CK WMMA conv "
            "(ck_device_grouped_convolution_fwd_multiple_abd_wmma_c_shuffle_v3_) "
            "kernel in the generated code; WMMA conv was not selected (fell back "
            "to ATen/Triton/classic-XDL-CK)."
        )


def _is_gfx1250_runtime():
    """True when the *running* device is gfx1250.

    Distinct from config.rocm.arch (the compile target): used to decide whether a
    WMMA kernel could actually have won autotune in this process.
    """
    return "gfx1250" in torch.cuda.get_device_properties(0).gcnArchName


@instantiate_parametrized_tests
class TestCKBackend(TestCase):
    def setUp(self):
        # The new inductor cache refresh mechanism
        # introduced with https://github.com/pytorch/pytorch/pull/122661
        # interacts badly with persistent subprocesses during
        # autotuning. So we need to disable automatic cache refresh
        # before calling setUp() on the parent class.
        old_disable_fresh_cache_envvar = os.environ.get(
            "INDUCTOR_TEST_DISABLE_FRESH_CACHE", ""
        )

        torch.random.manual_seed(1234)

        self.ck_dir, _, _, _ = try_import_ck_lib()
        if not self.ck_dir:
            # Fall back to an explicit CK source tree
            # (e.g. pointing at a checkout that ships the ck_tile headers).
            self.ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR")
        if not self.ck_dir:
            raise unittest.SkipTest("Composable Kernel library is not installed")

        try:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = "1"
            super().setUp()
        finally:
            os.environ["INDUCTOR_TEST_DISABLE_FRESH_CACHE"] = (
                old_disable_fresh_cache_envvar
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "CKTILE", "CKWMMA", "ATen,CK"),
        name_fn=lambda b: {
            "CK": "standalone_ck",
            "CKTILE": "standalone_cktile",
            "CKWMMA": "standalone_ckwmma",
            "ATen,CK": "fallback",
        }[b],
    )
    @parametrize("autotune_in_subproc", (True, False))
    @parametrize("use_aoti", (True, False))
    def test_max_autotune_precompile_matmul(
        self, max_autotune_gemm_backends, autotune_in_subproc, use_aoti
    ):
        """
        Make sure autotuning mm doesn't crash.
        """
        # CKWMMA is a single-token value: on non-gfx1250 the WMMA gate yields zero
        # choices and there is no ATen fallback in the list, so there is nothing to
        # select. Skip it off gfx1250 (the two/three-token values still work).
        if max_autotune_gemm_backends == "CKWMMA":
            runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx1250" not in runtime_arch:
                self.skipTest(f"CKWMMA requires gfx1250, got {runtime_arch}")

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_wmma_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):
            if use_aoti:
                Y_compiled = AOTIRunnerUtil.run(
                    model=mm,
                    example_inputs=(a, b),
                )
            else:

                @torch.compile(dynamic=False)
                def compiled_mm(x, w):
                    return mm(x, w)

                if max_autotune_gemm_backends == "CK":
                    Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
                    _assert_ck_selected(codes)
                elif max_autotune_gemm_backends == "CKWMMA":
                    Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
                    _assert_ckwmma_selected(codes)
                else:
                    Y_compiled = compiled_mm(a, b)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    def test_ck_selected_smoke_mm_bf16(self):
        """
        Tier-1 smoke: on the runtime GPU, force the standalone CK backend for a
        small bf16 mm and assert a CK kernel is actually selected (not an ATen or
        Triton fallback) and produces correct numerics. Cheapest unambiguous
        signal that the CK path is reachable and a matrix-core kernel runs.
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}
        a = torch.randn(512, 256, **tensor_options)
        b = torch.randn(256, 512, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CK",
                    "compile_threads": 4,
                    "rocm.ck_max_profiling_configs": 4,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_mm(x, w):
                return mm(x, w)

            Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
            _assert_ck_selected(codes)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    def test_cktile_selected_smoke_mm(self):
        """
        Tier-1 smoke for the CK-Tile backend: on the runtime GPU, force the
        standalone CK-Tile backend for a small float16 mm and assert a CK-Tile
        kernel is actually selected (not ATen/Triton/classic-CK) and produces
        correct numerics. The shape (M=N=512, K=256, Row/Row/Row) is served by a
        CK-Tile instance on both gfx9 (MFMA 32x32x16) and gfx1250 (WMMA 16x16x32):
        the block tile 256x256 divides M/N and K is a multiple of both warp-tile K
        values, so a CK-Tile candidate must win when the backend works.

        Deliberately float16 (F16), not bf16: the CK-Tile ops() product previously
        tagged fp16 as "FP16" which did not match the "F16" dtype key, silently
        filtering out every fp16 CK-Tile instance. This test is the CI regression
        guard for that mismatch class -- it fails if the dtype tokens drift again.
        A bf16 test would have stayed green through the entire bug.
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.float16}
        a = torch.randn(512, 256, **tensor_options)
        b = torch.randn(256, 512, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CKTILE",
                    "compile_threads": 4,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_mm(x, w):
                return mm(x, w)

            Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
            _assert_cktile_selected(codes)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "CKTILE", "CKWMMA", "ATen,CK"),
        name_fn=lambda b: {
            "CK": "standalone_ck",
            "CKTILE": "standalone_cktile",
            "CKWMMA": "standalone_ckwmma",
            "ATen,CK": "fallback",
        }[b],
    )
    @parametrize("autotune_in_subproc", (True,))
    def test_max_autotune_precompile_matmul_dynamic(
        self, max_autotune_gemm_backends, autotune_in_subproc
    ):
        """
        Test matmul with dynamic shapes
        """
        if max_autotune_gemm_backends == "CKWMMA":
            runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
            if "gfx1250" not in runtime_arch:
                self.skipTest(f"CKWMMA requires gfx1250, got {runtime_arch}")

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(256, 2048, **tensor_options)

        torch._dynamo.mark_dynamic(a, 0)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": autotune_in_subproc,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                    "rocm.ck_wmma_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=True)
            def compiled_mm(a, b):
                return a @ b

            if max_autotune_gemm_backends == "CKWMMA":
                Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
                _assert_ckwmma_selected(codes)
            else:
                Y_compiled = compiled_mm(a, b)
            Y = a @ b
            torch.testing.assert_close(Y_compiled, Y)

            a1 = torch.randn(1024, 256, **tensor_options)
            Y1_compiled = compiled_mm(a1, b)
            Y1 = a1 @ b
            torch.testing.assert_close(Y1_compiled, Y1)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    def test_max_autotune_precompile_preselected(self, max_autotune_gemm_backends):
        """
        End to end test for picking preselected ck instances
        """

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.randn(2240, 256, **tensor_options)
        b = torch.randn(2048, 256, **tensor_options).transpose(0, 1)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 12,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.use_preselected_instances": True,
                }
            ),
            tf32_off(),
        ):
            Y_compiled = torch.compile(mm, dynamic=False)(a, b)
            Y = mm(a, b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("max_autotune_gemm_backends", ("Aten,CK",))
    def test_max_autotune_precompile_non_contiguous(self, max_autotune_gemm_backends):
        """
        Make sure the matmul with non-contiguous inputs can fallback
        """

        tensor_options = {"device": "cuda", "dtype": torch.float16}

        a = torch.empty_strided((50257, 32768), (1, 50304), **tensor_options)
        b = torch.empty_strided((32768, 768), (768, 1), **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 16,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 8,
                    "rocm.ck_tile_max_profiling_configs": 8,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def mm(a, b):
                return a @ b

            Y_compiled = mm(a, b)
            Y_eager = a @ b
            torch.testing.assert_close(Y_compiled, Y_eager, equal_nan=True)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK,CKWMMA", "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK,CKWMMA" else "fallback",
    )
    @parametrize(
        "x_shape",
        ([4096, 2048], [2048], [4096, 1]),
        name_fn=lambda x_shape: f"x_shape_{'x'.join(map(str, x_shape))}",
    )
    @parametrize(
        "dtype",
        (torch.float16, torch.bfloat16),
        name_fn=lambda d: {torch.float16: "float16", torch.bfloat16: "bfloat16"}[d],
    )
    def test_max_autotune_addmm(self, max_autotune_gemm_backends, x_shape, dtype):
        m, k, n = 4096, 224, 2048
        alpha, beta = 1.0, 1.0

        tensor_options = {"device": "cuda", "dtype": dtype}
        x = torch.ones(x_shape, **tensor_options)
        a = torch.randn(m, k, **tensor_options)
        b = torch.randn(k, n, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_wmma_max_profiling_configs": 2,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def addmm(x, a, b, alpha, beta):
                return torch.addmm(x, a, b, alpha=alpha, beta=beta)

            # CK-forced (no ATen fallback), so either token winning is a pass: CK's
            # gfx1250 gate still accepts 16x16 2-byte XDL instances, so an XDL win
            # there is legitimate. WMMA-only selection is covered by the smoke test.
            if max_autotune_gemm_backends == "CK,CKWMMA":
                Y_compiled, codes = run_and_get_code(addmm, x, a, b, alpha, beta)
                _assert_ck_selected(codes)
            else:
                Y_compiled = addmm(x, a, b, alpha, beta)
            Y_eager = torch.addmm(x, a, b, alpha=alpha, beta=beta)

            torch.testing.assert_close(Y_compiled, Y_eager)

    @unittest.skip(
        "FIXME(tenpercent): kernel compilation errors on gfx942 as of 09/01/25"
    )
    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK", "ATen,CK"),
        name_fn=lambda b: "standalone" if b == "CK" else "fallback",
    )
    @parametrize("quantize_type", ("tensorwise", "rowwise"))
    @parametrize("has_bias", (True, False))
    def test_max_autotune_scaled_mm(
        self, max_autotune_gemm_backends, quantize_type, has_bias
    ):
        use_fast_accum = False
        runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx94" not in runtime_arch and "gfx95" not in runtime_arch:
            self.skipTest(f"Unsupported arch {runtime_arch}")
        # output dtype
        dtype = torch.bfloat16
        tensor_options = {"device": "cuda", "dtype": dtype}

        M = 2240
        N = 2048
        K = 256

        x = torch.randn(M, K, **tensor_options)
        w = torch.randn(N, K, **tensor_options)

        bias = None
        if has_bias:
            bias = torch.randn(N, **tensor_options)

        dtype_float8 = (
            torch.float8_e4m3fnuz if "gfx94" in runtime_arch else torch.float8_e4m3fn
        )

        f_quantize = (
            _quantize_tensorwise if quantize_type == "tensorwise" else _quantize_rowwise
        )

        # quantize weight (prior to inference)
        w_fp8, w_inverse_scale = f_quantize(w, dtype_float8)
        w_t_fp8 = w_fp8.t()
        w_inverse_scale_t = w_inverse_scale.t()

        # quantize input x
        x_fp8, x_inverse_scale = f_quantize(x, dtype_float8)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        def linear(x_fp8, x_inverse_scale, w_t_fp8, w_inverse_scale, bias):
            y = torch._scaled_mm(
                x_fp8,
                w_t_fp8,
                x_inverse_scale,
                w_inverse_scale,
                bias,
                out_dtype=dtype,
                use_fast_accum=use_fast_accum,
            )
            return y

        y_eager = linear(
            x_fp8,
            x_inverse_scale,
            w_t_fp8,
            w_inverse_scale_t,
            bias,
        )

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": max_autotune_gemm_backends,
                "compile_threads": 24,
                "rocm.ck_max_profiling_configs": 24,
                "rocm.ck_dir": self.ck_dir,
            }
        ):
            linear_compiled = torch.compile(
                linear, backend="inductor", mode="max-autotune"
            )
            y_compiled = linear_compiled(
                x_fp8,
                x_inverse_scale,
                w_t_fp8,
                w_inverse_scale_t,
                bias,
            )
            self.assertEqual(y_eager.dtype, dtype)
            self.assertEqual(y_compiled.dtype, dtype)

            torch.testing.assert_close(y_eager, y_compiled, rtol=1e-2, atol=0.05)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize(
        "max_autotune_conv_backends",
        ("CK", "ATEN,CK"),
        # Not the GEMM tests' standalone/fallback labels. Conv appends ATen
        # whenever the choice list comes back empty (the `if not choices` branch
        # in conv.py), so the "CK" case is never standalone; and ATEN in the
        # token list makes ATen a competitor up front, not a fallback.
        name_fn=lambda b: "ck_only" if b == "CK" else "vs_aten",
    )
    def test_max_autotune_conv2d_float32(self, max_autotune_conv_backends):
        """Float32 conv2d. Dtype is in the name because f32 and f16/bf16 take
        materially different CK paths on gfx1250: Wave32Force16MNPerXDL requires
        2-byte compute types, so f32 never gets the 16x16 warp-tile remap."""
        tensor_options = {"device": "cuda", "dtype": torch.float32}

        x = torch.randn(1, 8, 224, 224, **tensor_options)
        w = torch.randn(64, 8, 7, 7, **tensor_options)
        x_cl = x.to(memory_format=torch.channels_last)
        w_cl = w.to(memory_format=torch.channels_last)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": False,
                    "max_autotune_conv_backends": max_autotune_conv_backends,
                    "compile_threads": 4,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": 4,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def conv2d(x, w):
                return torch.conv2d(x, w)

            Y_eager = torch.conv2d(x_cl, w_cl)
            if max_autotune_conv_backends == "CK":
                Y_compiled, codes = run_and_get_code(conv2d, x_cl, w_cl)
                _assert_ck_selected(codes)
            else:
                Y_compiled = conv2d(x_cl, w_cl)

            torch.testing.assert_close(Y_compiled, Y_eager, atol=2e-4, rtol=2e-4)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("arch", ("gfx1250", "gfx950"))
    def test_ck_conv_gfx1250_filter(self, arch):
        """
        Arch-gated instance pruning for the CK grouped-conv backend.

        On gfx1250 (wave32) most enumerated conv instances cannot emit a
        matrix-core kernel: CK decides this at compile time and then rejects them
        with a bare `return false` and no diagnostic. `filter_op` used to be
        architecture-blind, so it handed autotune a pool that is ~92% dead on
        that target; with a small `ck_max_profiling_configs` the sampler could
        draw zero working candidates and raise NoValidChoicesError.

        Patches `config.rocm.arch` rather than reading the physical device, so
        this runs anywhere -- including gfx9 CI -- and pins both directions:
        pruning happens on gfx1250, and changes nothing off it.
        """
        from torch._inductor.codegen.rocm.ck_conv_template import (
            CKGroupedConvFwdTemplate,
        )
        from torch._inductor.graph import GraphLowering
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR") or self.ck_dir
        device = torch.device("cuda")
        dtype = torch.float32

        # filter_op resolves layouts through V.graph.sizevars, so it needs a
        # graph context even though nothing is lowered or compiled here.
        gm = make_fx(lambda: torch.zeros(1))()
        graph = GraphLowering(gm)

        def gen_ops_for(target_arch):
            # The shape from test_max_autotune_conv2d_float32: conv2d(x[1,8,224,224],
            # w[64,8,7,7]) in channels-last.
            x = Buffer(
                name="X",
                layout=FixedLayout(
                    device, dtype, [1, 8, 224, 224], [401408, 1, 1792, 8]
                ),
            )
            w = Buffer(
                name="W",
                layout=FixedLayout(device, dtype, [64, 8, 7, 7], [392, 1, 56, 8]),
            )
            out_layout = FixedLayout(
                device, dtype, [1, 64, 218, 218], [3041536, 1, 13952, 64]
            )
            template = CKGroupedConvFwdTemplate(
                [x, w],
                out_layout,
                stride=[1, 1],
                padding=[0, 0],
                dilation=[1, 1],
                groups=1,
                n_spatial_dimensions=2,
            )
            # No cap: compare the whole filtered pool, not a random sample of it.
            with (
                config.patch(
                    {
                        "max_autotune": True,
                        "rocm.arch": [target_arch],
                        "rocm.ck_dir": ck_dir,
                        "rocm.ck_max_profiling_configs": None,
                    }
                ),
                tf32_off(),
                V.set_graph_handler(graph),
            ):
                return template, template.gen_ops()

        template, ops = gen_ops_for(arch)
        self.assertGreater(
            len(ops), 0, f"No CK conv instances survived filter_op for {arch}"
        )

        if arch == "gfx1250":
            # Every surviving instance must be one that can actually emit a
            # kernel. Assert the invariant rather than a count, so this stays
            # stable as CK adds or removes instances.
            not_viable = [op for op in ops if not template._wave32_viable(op)]
            self.assertEqual(
                not_viable,
                [],
                f"{len(not_viable)}/{len(ops)} instances kept for gfx1250 cannot "
                f"emit a matrix-core kernel; filter_op is letting dead instances "
                f"through to autotune.",
            )
            # The pool must shrink materially. If it ever stops doing so the
            # predicate has silently become a no-op.
            _, unpruned = gen_ops_for("gfx950")
            self.assertLess(
                len(ops),
                len(unpruned),
                "gfx1250 pruning kept as many instances as the unpruned gfx950 "
                "pool; the viability predicate is not doing anything.",
            )
            # Golden values, independent of the predicate. The assertions above
            # are self-referential -- `ops` came out of filter_op, which calls
            # _wave32_viable, so "every kept op is viable" holds even if the
            # predicate is gutted. These two numbers come from per-instance
            # compile-and-disassemble of every f32 conv instance on gfx1250, and
            # are the only thing here that pins the arithmetic itself. If CK ships or
            # removes conv instances they will need updating -- do that by
            # re-measuring, not by relaxing the assertion.
            self.assertEqual(
                (len(ops), len(unpruned)),
                (4, 48),
                "Pruned/unpruned f32 pool for the conv2d shape changed. Expected "
                "4 of 48 (measured on gfx1250 silicon). If CK's instance set moved, "
                "re-measure; if not, _wave32_viable no longer matches CK.",
            )
            # Direct unit check of the wave mapping, so a gutted _xdl_per_wave
            # cannot pass. 64/16/16 with 4 waves does not divide; 128/128/16 does.
            self.assertEqual(template._xdl_per_wave(64, 16, 16, 16, 16, 1), 0)
            self.assertNotEqual(template._xdl_per_wave(128, 16, 128, 16, 16, 1), 0)
            # The float32 pool above cannot exercise Wave32Force16MNPerXDL,
            # which needs a 2-byte compute type -- so check the 16x16 remap
            # directly.
            from ck4inductor.grouped_conv_fwd.gen_instances import (
                gen_conv_ops_library,
            )

            remapped = [
                op
                for op in gen_conv_ops_library()
                if op.a_element_dtype in ("F16", "BF16")
                and (op.m_per_xdl, op.n_per_xdl) == (32, 32)
                and template._wave32_viable(op)
            ]
            self.assertGreater(
                len(remapped),
                0,
                "No declared-32x32 f16/bf16 instance passed _wave32_viable, so "
                "Wave32Force16MNPerXDL is never firing. The 16x16 remap is "
                "disabled and gfx1250 loses its f16/bf16 conv coverage.",
            )
        else:
            # Off-target the filter must change nothing: everything that
            # passes the dtype/layout/spec checks is still offered, including
            # instances gfx1250 would reject. This pins the regression class
            # where an arch gate keys on the physical device rather than the
            # compile target.
            self.assertTrue(
                any(not template._wave32_viable(op) for op in ops),
                "Expected the un-gated gfx950 pool to contain instances that "
                "gfx1250 would prune; if not, this assertion proves nothing.",
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize(
        "max_autotune_conv_backends",
        ("CK,CKWMMA", "CK,CKWMMA,ATEN"),
        # Same labels, same reasoning as test_max_autotune_conv2d_float32.
        name_fn=lambda b: "ck_only" if b == "CK,CKWMMA" else "vs_aten",
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_max_autotune_conv2d_wmma(self, max_autotune_conv_backends, dtype):
        """
        f16/bf16 sibling of test_max_autotune_conv2d_float32, exercising the
        CKWMMA token.

        Meaningful on every arch, unlike a CKWMMA case bolted onto the f32 test:
        f16/bf16 conv is a genuinely different CK code path (Wave32Force16MNPerXDL
        applies), there is no other f16/bf16 conv coverage in this suite, and on
        gfx9 the CKWMMA gate is simply False so the case falls back to plain CK.

        The two backend values differ exactly when CK works: `ck_only` proves CK
        produces a correct kernel unopposed; `vs_aten` additionally requires CK to
        beat ATen on measured time, which is the only competitiveness signal here.
        """
        tensor_options = {"device": "cuda", "dtype": dtype}

        x = torch.randn(1, 8, 56, 56, **tensor_options)
        w = torch.randn(64, 8, 3, 3, **tensor_options)
        x_cl = x.to(memory_format=torch.channels_last)
        w_cl = w.to(memory_format=torch.channels_last)

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": False,
                    "max_autotune_conv_backends": max_autotune_conv_backends,
                    "compile_threads": 4,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_max_profiling_configs": _CONV_PROFILING_CONFIGS,
                    "rocm.ck_wmma_max_profiling_configs": _CONV_PROFILING_CONFIGS,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def conv2d(x, w):
                return torch.conv2d(x, w)

            Y_eager = torch.conv2d(x_cl, w_cl)
            if "ATEN" not in max_autotune_conv_backends:
                # CK is unopposed, so it must produce the winning kernel. On gfx9
                # that is legitimately an XDL kernel (the CKWMMA gate is False
                # there); asserting the WMMA alias here would make this a perf
                # assertion. test_ckwmma_selected_smoke_conv is where "WMMA
                # actually wins" is pinned, under a forced single backend.
                Y_compiled, codes = run_and_get_code(conv2d, x_cl, w_cl)
                _assert_ck_selected(codes)
            else:
                # ATen competes on measured time. Which kernel wins is a
                # performance question and varies with node load, so assert
                # numerics only -- matching test_max_autotune_conv2d_float32's treatment
                # of its own ATen case.
                Y_compiled = conv2d(x_cl, w_cl)

            # bf16 has ~8 mantissa bits; f16 has 10, so it can be held tighter.
            tol = 2e-2 if dtype is torch.bfloat16 else 2e-3
            torch.testing.assert_close(Y_compiled, Y_eager, atol=tol, rtol=tol)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(
        os.environ,
        {**_test_env, "PYTORCH_MIOPEN_SUGGEST_NHWC": "1"},
    )
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_ckwmma_selected_smoke_conv(self, dtype):
        """
        Proves a WMMA *conv* kernel wins autotune and is numerically correct.

        Forces CKWMMA alone -- excluding the CK token removes the XDL competition,
        so a WMMA kernel must win rather than merely being allowed to. Requires
        gfx1250 silicon, hence the explicit skip: conv auto-appends ATen when the
        choice list is empty (the `if not choices` branch in conv.py), so
        off-target this would quietly pass via ATen without exercising WMMA at all.
        The skip, not the token string, is what stops the test from passing without
        running a WMMA kernel.

        Parametrized over both dtypes to catch a renamed dtype token -- the class of
        bug where fp16 is tagged "FP16" vs "F16" and every fp16 instance is
        silently filtered out.
        """
        if not _is_gfx1250_runtime():
            runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
            self.skipTest(f"CKWMMA conv requires gfx1250, got {runtime_arch}")

        tensor_options = {"device": "cuda", "dtype": dtype}
        x = torch.randn(1, 8, 56, 56, **tensor_options).to(
            memory_format=torch.channels_last
        )
        w = torch.randn(64, 8, 3, 3, **tensor_options).to(
            memory_format=torch.channels_last
        )

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "autotune_in_subproc": False,
                    "max_autotune_conv_backends": "CKWMMA",
                    "compile_threads": 4,
                    "rocm.ck_dir": self.ck_dir,
                    "rocm.ck_wmma_max_profiling_configs": 8,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def conv2d(x, w):
                return torch.conv2d(x, w)

            Y_eager = torch.conv2d(x, w)
            Y_compiled, codes = run_and_get_code(conv2d, x, w)
            _assert_ck_selected(codes)
            _assert_ckwmma_conv_selected(codes)

            torch.testing.assert_close(Y_compiled, Y_eager, atol=2e-2, rtol=2e-2)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "max_autotune_gemm_backends",
        ("CK,CKWMMA", "ATen,CK"),
        # Explicit map: the standalone slot now requests two tokens, which the
        # previous `"standalone" if b == "CK"` lambda would have mislabelled.
        # Both test ids are unchanged.
        name_fn=lambda b: {"CK,CKWMMA": "standalone", "ATen,CK": "fallback"}[b],
    )
    def test_max_autotune_precompile_bmm(
        self,
        max_autotune_gemm_backends,
    ):
        """
        Test gemm-max-autotune torch.bmm with CK backend

        The standalone case is CK-forced (no ATen fallback). On gfx1250 the XDL
        instances that survive filtering are all pipeline v2, which the K-loop
        prefetch check rejects at runtime, so every choice scores +inf; CKWMMA
        supplies working candidates. Off gfx1250 the CKWMMA gate is False and the
        two-token value falls back to plain CK.
        """

        def bmm(a, b):
            return torch.bmm(a, b)

        tensor_options = {"device": "cuda", "dtype": torch.bfloat16}

        a = torch.randn(16, 2240, 256, **tensor_options)
        b = torch.randn(16, 2048, 256, **tensor_options).transpose(1, 2)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": max_autotune_gemm_backends,
                    "compile_threads": 2,
                    "rocm.ck_max_profiling_configs": 2,
                    "rocm.ck_wmma_max_profiling_configs": 2,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_bmm(x, w):
                return bmm(x, w)

            if max_autotune_gemm_backends == "CK,CKWMMA":
                Y_compiled, codes = run_and_get_code(compiled_bmm, a, b)
                # A CK-family kernel must win. Which of XDL/WMMA is faster is not
                # the contract, so this deliberately accepts either -- on gfx9 the
                # winner is legitimately XDL.
                _assert_ck_selected(codes)
            else:
                Y_compiled = compiled_bmm(a, b)

            Y_eager = bmm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y_eager)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("arch", ("gfx950",))
    def test_ck_tile_gemm_compiles(self, arch):
        """
        Compile-only regression test for the CK-Tile universal GEMM backend.

        Renders the HIP source for a representative CK-Tile GEMM instance from each
        (pipeline, epilogue) stratum and cross-compiles each with hipcc (object
        only, no device execution), asserting that *every* stratum compiles for the
        target arch.

        This catches the silent-breakage class where a CK API change makes CK-Tile
        instances fail to compile -- in which case the autotuner prunes them and
        falls back to ATen/Triton ("CKTILE ignored") instead of erroring. Covering
        each (pipeline, epilogue) ensures a break confined to a single variant --
        e.g. only the CShuffle epilogue -- is still caught.
        """
        import subprocess
        import tempfile
        from collections import defaultdict

        from torch._inductor.codegen.rocm.ck_tile_universal_gemm_template import (
            CKTileGemmTemplate,
            ops as ck_tile_ops,
        )
        from torch._inductor.codegen.rocm.compile_command import rocm_compile_command
        from torch._inductor.graph import GraphLowering
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        # Prefer an explicit CK source tree (local dev) over the installed wheel.
        ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR") or self.ck_dir

        dtype = torch.bfloat16
        M, N, K = 2240, 2048, 256
        device = torch.device("cuda")
        compile_timeout_s = 600

        gm = make_fx(lambda: torch.zeros(1))()
        graph = GraphLowering(gm)

        # Render one representative instance per (pipeline, epilogue) stratum within
        # a minimal graph context.
        sources = []
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "rocm.arch": [arch],
                    "rocm.ck_dir": ck_dir,
                }
            ),
            V.set_graph_handler(graph),
        ):
            x = Buffer(name="X", layout=FixedLayout(device, dtype, [M, K], [K, 1]))
            w = Buffer(name="W", layout=FixedLayout(device, dtype, [K, N], [N, 1]))
            out_layout = FixedLayout(device, dtype, [M, N], [N, 1])

            template = CKTileGemmTemplate([x, w], out_layout)

            by_stratum = defaultdict(list)
            for op in ck_tile_ops():
                if template.filter_op(op) is not None:
                    by_stratum[(op.pipeline, op.epilogue)].append(op)
            self.assertGreater(
                len(by_stratum), 0, f"No CK-Tile instances were generated for {arch}"
            )

            # The synthetic GraphLowering doesn't know our input buffers, so teach
            # V.graph.get_dtype about them. generate() wraps this with its own fake
            # for the output node, delegating unknown names back to this function.
            dtype_lookup = {
                "X": dtype,
                "W": dtype,
                template.output_node.get_name(): dtype,
            }

            with unittest.mock.patch.object(
                V.graph, "get_dtype", lambda name: dtype_lookup[name]
            ):
                for stratum, op_list in sorted(by_stratum.items()):
                    op = op_list[0]
                    k_batch = template.k_batch_choices(op)[0]
                    caller = template.generate(op=op, kBatch=k_batch)
                    sources.append((stratum, op.name(), caller.bmreq.source_code))

        def compile_object(source):
            with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
                f.write(source)
                src_path = f.name
            obj_path = src_path + ".o"
            command = rocm_compile_command([src_path], obj_path, "o")
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=compile_timeout_s,
                )
                rc, out = proc.returncode, proc.stderr or proc.stdout
            except subprocess.TimeoutExpired:
                rc, out = 1, f"timed out after {compile_timeout_s}s"
            finally:
                for p in (src_path, obj_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return rc, command, out

        failures = []
        with config.patch({"rocm.arch": [arch], "rocm.ck_dir": ck_dir}):
            for stratum, name, source in sources:
                rc, command, out = compile_object(source)
                if rc != 0:
                    failures.append((stratum, name, command, out))

        if failures:
            stratum, name, command, out = failures[0]
            self.fail(
                f"{len(failures)}/{len(sources)} CK-Tile (pipeline, epilogue) strata "
                f"failed to compile for {arch} "
                f"(failed strata: {[f[0] for f in failures]}); the CK-Tile backend "
                f"is silently disabled for those.\nFirst failing instance: {name}\n"
                f"Reproduce: {command}\n--- compiler output (truncated) ---\n"
                f"{out[-4000:]}"
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "dtype",
        (torch.float16, torch.bfloat16),
        name_fn=lambda d: {torch.float16: "float16", torch.bfloat16: "bfloat16"}[d],
    )
    def test_ckwmma_selected_smoke_mm(self, dtype):
        """
        Tier-1 smoke for the gfx1250 CKWMMA backend: force the standalone CKWMMA
        backend for a small mm and assert a WMMA kernel is actually selected (not
        ATen/Triton/classic-XDL-CK/CK-Tile) and produces correct numerics. The
        shape (M=N=512, K=256, Row/Row/Row = mk_kn_mn) is served by shipped WMMA
        instances: block tiles 128/256 divide M/N and K=256 is a multiple of both
        kpb 32 and 64, so a WMMA candidate must win when the backend works.

        Parametrized over float16 AND bfloat16: both ship as WMMA instances, and
        this is the CI regression guard for dtype-token drift (the class of bug
        where fp16 was tagged "FP16" not "F16" and silently filtered out). A
        bf16-only test would stay green through such a bug.
        """
        runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "gfx1250" not in runtime_arch:
            self.skipTest(f"CKWMMA requires gfx1250, got {runtime_arch}")

        def mm(a, b):
            return a @ b

        tensor_options = {"device": "cuda", "dtype": dtype}
        a = torch.randn(512, 256, **tensor_options)
        b = torch.randn(256, 512, **tensor_options)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CKWMMA",
                    "compile_threads": 4,
                    "rocm.ck_wmma_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_mm(x, w):
                return mm(x, w)

            Y_compiled, codes = run_and_get_code(compiled_mm, a, b)
            _assert_ckwmma_selected(codes)

            Y = mm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize(
        "dtype",
        (torch.float16, torch.bfloat16),
        name_fn=lambda d: {torch.float16: "float16", torch.bfloat16: "bfloat16"}[d],
    )
    def test_ckwmma_selected_smoke_bmm(self, dtype):
        """
        Batched counterpart of test_ckwmma_selected_smoke_mm: force the standalone
        CKWMMA backend for a small bmm and assert a *batched* WMMA kernel is
        selected and numerically correct.

        B is transposed, giving Row/Col/Row (gmk_gnk_gmn) -- the layout shipped for
        batched WMMA in both dtypes. M=N=512, K=256 divide the shipped block tiles,
        so a candidate must win when the backend works.
        """
        if not _is_gfx1250_runtime():
            runtime_arch = torch.cuda.get_device_properties(0).gcnArchName
            self.skipTest(f"CKWMMA requires gfx1250, got {runtime_arch}")

        def bmm(a, b):
            return torch.bmm(a, b)

        tensor_options = {"device": "cuda", "dtype": dtype}
        a = torch.randn(4, 512, 256, **tensor_options)
        b = torch.randn(4, 512, 256, **tensor_options).transpose(1, 2)

        if "rocm" not in dir(config):
            raise AssertionError("'rocm' not found in dir(config)")

        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CKWMMA",
                    "compile_threads": 4,
                    "rocm.ck_wmma_max_profiling_configs": 8,
                    "rocm.ck_dir": self.ck_dir,
                }
            ),
            tf32_off(),
        ):

            @torch.compile(dynamic=False)
            def compiled_bmm(x, w):
                return bmm(x, w)

            Y_compiled, codes = run_and_get_code(compiled_bmm, a, b)
            _assert_ckwmma_batched_selected(codes)

            Y = bmm(a=a, b=b)
            torch.testing.assert_close(Y_compiled, Y)

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("arch", ("gfx1250",))
    @parametrize(
        "batched", (False, True), name_fn=lambda b: "batched" if b else "classic"
    )
    def test_ck_wmma_gemm_compiles(self, arch, batched):
        """
        Compile-only regression test for the gfx1250 CKWMMA universal GEMM backend.

        Renders the HIP source for a representative WMMA GEMM instance from each
        (pipeline_version, scheduler) stratum and cross-compiles each with hipcc
        (object only, no device execution), asserting that *every* stratum compiles
        for gfx1250. Runs on any GPU (gfx9 CI included) since it patches
        config.rocm.arch rather than executing on device.

        This catches the silent-breakage class where a CK API change makes WMMA
        instances fail to compile -- the autotuner would then prune them and fall
        back, hiding the break. Covering each stratum ensures a break confined to a
        single variant is still caught.

        The batched variant covers the 3-D path, which resolves to a different
        device op (DeviceBatchedGemmMultiD_Wmma_CShuffleV3) and header than the
        classic one, and so can break independently.
        """
        import subprocess
        import tempfile
        from collections import defaultdict

        from torch._inductor.codegen.rocm.ck_universal_gemm_template import (
            CKWMMAGemmTemplate,
        )
        from torch._inductor.codegen.rocm.compile_command import rocm_compile_command
        from torch._inductor.graph import GraphLowering
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR") or self.ck_dir

        dtype = torch.bfloat16
        # mk_kn_mn shape (Row/Row/Row) served by shipped WMMA instances.
        M, N, K = 512, 512, 256
        B = 4
        device = torch.device("cuda")
        compile_timeout_s = 600

        gm = make_fx(lambda: torch.zeros(1))()
        graph = GraphLowering(gm)

        sources = []
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "rocm.arch": [arch],
                    "rocm.ck_dir": ck_dir,
                }
            ),
            V.set_graph_handler(graph),
        ):
            if batched:
                # B is column-major (as torch.bmm produces after a transpose),
                # giving Row/Col/Row -- the layout shipped for batched WMMA.
                x = Buffer(
                    name="X",
                    layout=FixedLayout(device, dtype, [B, M, K], [M * K, K, 1]),
                )
                w = Buffer(
                    name="W",
                    layout=FixedLayout(device, dtype, [B, K, N], [K * N, 1, K]),
                )
                out_layout = FixedLayout(device, dtype, [B, M, N], [M * N, N, 1])
            else:
                x = Buffer(name="X", layout=FixedLayout(device, dtype, [M, K], [K, 1]))
                w = Buffer(name="W", layout=FixedLayout(device, dtype, [K, N], [N, 1]))
                out_layout = FixedLayout(device, dtype, [M, N], [N, 1])

            template = CKWMMAGemmTemplate([x, w], out_layout, alpha=1, beta=0)

            by_stratum = defaultdict(list)
            for op_info in template.gen_ops():
                op = op_info.op
                by_stratum[
                    (op.block_gemm_pipeline_version, op.block_gemm_pipeline_scheduler)
                ].append(op_info)
            self.assertGreater(
                len(by_stratum), 0, f"No CKWMMA instances were generated for {arch}"
            )

            dtype_lookup = {
                "X": dtype,
                "W": dtype,
                template.output_node.get_name(): dtype,
            }

            with unittest.mock.patch.object(
                V.graph, "get_dtype", lambda name: dtype_lookup[name]
            ):
                for stratum, op_list in sorted(by_stratum.items()):
                    op_info = op_list[0]
                    caller = template.generate(op=op_info.op, kBatch=op_info.kBatch)
                    sources.append(
                        (stratum, op_info.op.name(), caller.bmreq.source_code)
                    )

        def compile_object(source):
            with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
                f.write(source)
                src_path = f.name
            obj_path = src_path + ".o"
            command = rocm_compile_command([src_path], obj_path, "o")
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=compile_timeout_s,
                )
                rc, out = proc.returncode, proc.stderr or proc.stdout
            except subprocess.TimeoutExpired:
                rc, out = 1, f"timed out after {compile_timeout_s}s"
            finally:
                for p in (src_path, obj_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return rc, command, out

        failures = []
        with config.patch({"rocm.arch": [arch], "rocm.ck_dir": ck_dir}):
            for stratum, name, source in sources:
                rc, command, out = compile_object(source)
                if rc != 0:
                    failures.append((stratum, name, command, out))

        if failures:
            stratum, name, command, out = failures[0]
            # Surface the actual diagnostics: hipcc prints the `error:` lines early,
            # so a plain tail-truncation hides the root cause behind template spew.
            diag = "\n".join(
                line for line in out.splitlines() if "error:" in line.lower()
            )[:3000]
            self.fail(
                f"{len(failures)}/{len(sources)} CKWMMA "
                f"{'batched' if batched else 'classic'} (pipeline_version, scheduler) "
                f"strata failed to compile for {arch} "
                f"(failed strata: {[f[0] for f in failures]}); the CKWMMA backend "
                f"is silently disabled for those.\nFirst failing instance: {name}\n"
                f"Reproduce: {command}\n"
                f"--- compiler error lines ---\n{diag or '(none matched)'}\n"
                f"--- compiler output (tail) ---\n{out[-2000:]}"
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    @parametrize("arch", ("gfx1250",))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_ck_wmma_conv_compiles(self, arch, dtype):
        """
        Compile-only regression test for the gfx1250 CKWMMA grouped-conv backend.

        Renders one representative WMMA conv instance per (pipeline_version,
        scheduler) stratum and cross-compiles each with hipcc (object only, no
        device execution), asserting every stratum builds for gfx1250. Patches
        config.rocm.arch, so it runs on any GPU including gfx9 CI.

        This is the CI guard for the backend: if a CK API change breaks WMMA conv
        compilation, autotune silently prunes the broken instances and falls back,
        so nothing else here would notice. The conv device op and header differ
        from both GEMM ones, so it can break independently of them.
        """
        import subprocess
        import tempfile
        from collections import defaultdict

        from torch._inductor.codegen.rocm.ck_conv_template import (
            CKWMMAGroupedConvFwdTemplate,
        )
        from torch._inductor.codegen.rocm.compile_command import rocm_compile_command
        from torch._inductor.graph import GraphLowering
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR") or self.ck_dir
        device = torch.device("cuda")
        compile_timeout_s = 600

        gm = make_fx(lambda: torch.zeros(1))()
        graph = GraphLowering(gm)

        sources = []
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "rocm.arch": [arch],
                    "rocm.ck_dir": ck_dir,
                    "rocm.ck_wmma_max_profiling_configs": None,
                }
            ),
            V.set_graph_handler(graph),
        ):
            # Channels-last conv2d: NHWGC/GKYXC/NHWGK, the layout WMMA conv ships.
            x = Buffer(
                name="X",
                layout=FixedLayout(device, dtype, [1, 8, 56, 56], [25088, 1, 448, 8]),
            )
            w = Buffer(
                name="W",
                layout=FixedLayout(device, dtype, [64, 8, 3, 3], [72, 1, 24, 8]),
            )
            out_layout = FixedLayout(
                device, dtype, [1, 64, 54, 54], [186624, 1, 3456, 64]
            )

            template = CKWMMAGroupedConvFwdTemplate(
                [x, w],
                out_layout,
                stride=[1, 1],
                padding=[0, 0],
                dilation=[1, 1],
                groups=1,
                n_spatial_dimensions=2,
            )

            by_stratum = defaultdict(list)
            for op in template.gen_ops():
                by_stratum[
                    (op.block_gemm_pipeline_version, op.block_gemm_pipeline_scheduler)
                ].append(op)
            # A silent enumeration failure (bad packaging, renamed header) would
            # otherwise make this pass with nothing compiled.
            self.assertGreater(
                len(by_stratum),
                0,
                f"No CKWMMA conv instances were generated for {arch}/{dtype}",
            )

            dtype_lookup = {
                "X": dtype,
                "W": dtype,
                template.output_node.get_name(): dtype,
            }

            with unittest.mock.patch.object(
                V.graph, "get_dtype", lambda name: dtype_lookup[name]
            ):
                for stratum, op_list in sorted(by_stratum.items()):
                    op = op_list[0]
                    caller = template.generate(op=op)
                    sources.append((stratum, op.name(), caller.bmreq.source_code))

        def compile_object(source):
            with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
                f.write(source)
                src_path = f.name
            obj_path = src_path + ".o"
            command = rocm_compile_command([src_path], obj_path, "o")
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=compile_timeout_s,
                )
                rc, out = proc.returncode, proc.stderr or proc.stdout
            except subprocess.TimeoutExpired:
                rc, out = 1, f"timed out after {compile_timeout_s}s"
            finally:
                for p in (src_path, obj_path):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            return rc, command, out

        failures = []
        with config.patch({"rocm.arch": [arch], "rocm.ck_dir": ck_dir}):
            for stratum, name, source in sources:
                rc, command, out = compile_object(source)
                if rc != 0:
                    failures.append((stratum, name, command, out))

        if failures:
            stratum, name, command, out = failures[0]
            self.fail(
                f"{len(failures)}/{len(sources)} CKWMMA conv "
                f"(pipeline_version, scheduler) strata failed to compile for "
                f"{arch}/{dtype} (failed strata: {[f[0] for f in failures]}); the "
                f"CKWMMA conv backend is silently disabled for those.\n"
                f"First failing instance: {name}\nReproduce: {command}\n"
                f"--- compiler output (tail) ---\n{out[-2000:]}"
            )

    @unittest.skipIf(not torch.version.hip, "ROCM only")
    @unittest.mock.patch.dict(os.environ, _test_env)
    def test_ck_wmma_gate(self):
        """
        Arch-gate + discoverability-warning behavior for the CKWMMA backend. Pure
        gate-level checks (no compilation), so it runs on any GPU.

        (a) The WMMA gate is True only for the gfx1250 compile target; on gfx950 it
            is False, CKWMMA yields zero choices, and the classic CK enumeration is
            unchanged (no WMMA leakage into the CK token). This pins that the gate
            keys on the compile target, so WMMA source is never emitted for a
            non-WMMA arch.
        (b) The one-time CK-on-gfx1250 warning fires only for plain CK on gfx1250
            and is suppressed when CKWMMA or CKTILE is also requested, or off
            gfx1250 -- and never changes the gate's verdict.
        """
        from torch._inductor.codegen.rocm.ck_universal_gemm_template import (
            CKWMMAGemmTemplate,
        )
        from torch._inductor.graph import GraphLowering
        from torch._inductor.ir import Buffer, FixedLayout
        from torch._inductor.utils import (
            _warn_ck_xdl_on_gfx1250,
            use_ck_gemm_template,
            use_ck_wmma_gemm_template,
        )
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        device = torch.device("cuda")
        M, N, K = 512, 512, 256
        layout = FixedLayout(device, torch.bfloat16, [M, N], [N, 1])

        # The gates consult V.graph.sizevars, so every call must run inside a graph
        # context; without one V.graph is a NullHandler and the gate raises.
        gm = make_fx(lambda: torch.zeros(1))()
        graph = GraphLowering(gm)

        with V.set_graph_handler(graph):
            # (a) Arch gating.
            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CK,CKWMMA",
                    "rocm.arch": ["gfx1250"],
                    "rocm.ck_dir": self.ck_dir,
                }
            ):
                self.assertTrue(use_ck_wmma_gemm_template(layout, M, N, K))

            with config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "CK,CKWMMA",
                    "rocm.arch": ["gfx950"],
                    "rocm.ck_dir": self.ck_dir,
                }
            ):
                self.assertFalse(use_ck_wmma_gemm_template(layout, M, N, K))
                # No WMMA leakage: off gfx1250 the WMMA template yields no choices.
                x = Buffer(
                    name="X",
                    layout=FixedLayout(device, torch.bfloat16, [M, K], [K, 1]),
                )
                w = Buffer(
                    name="W",
                    layout=FixedLayout(device, torch.bfloat16, [K, N], [N, 1]),
                )
                wmma_template = CKWMMAGemmTemplate([x, w], layout, alpha=1, beta=0)
                self.assertEqual(len(wmma_template.gen_ops()), 0)

            # The gate and the enumerator must behave the same way for the 3-D
            # (bmm) path: enabled and non-empty on gfx1250, silent on gfx950.
            B = 4
            batched_layout = FixedLayout(
                device, torch.bfloat16, [B, M, N], [M * N, N, 1]
            )
            batched_x = Buffer(
                name="X",
                layout=FixedLayout(device, torch.bfloat16, [B, M, K], [M * K, K, 1]),
            )
            batched_w = Buffer(
                name="W",
                layout=FixedLayout(device, torch.bfloat16, [B, K, N], [K * N, 1, K]),
            )

            for arch, expected in (("gfx1250", True), ("gfx950", False)):
                with config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": "CK,CKWMMA",
                        "rocm.arch": [arch],
                        "rocm.ck_dir": self.ck_dir,
                    }
                ):
                    self.assertEqual(
                        use_ck_wmma_gemm_template(batched_layout, M, N, K), expected
                    )
                    batched_template = CKWMMAGemmTemplate(
                        [batched_x, batched_w], batched_layout, alpha=1, beta=0
                    )
                    self.assertTrue(batched_template.is_batched)
                    ops = batched_template.gen_ops()
                    if expected:
                        self.assertTrue(ops, f"no batched WMMA instances for {arch}")
                    else:
                        self.assertEqual(len(ops), 0)

        # (a2) The same for the CONV gate, which reads a different config key
        # (max_autotune_conv_backends via _use_conv_autotune_backend). Wiring it
        # to the GEMM helper would make the CKWMMA token permanently unreachable
        # for conv -- and nothing else here would notice, since the conv smoke
        # test skips off gfx1250 and the two-token conv test passes via plain CK.
        from torch._inductor.codegen.rocm.ck_conv_template import (
            CKWMMAGroupedConvFwdTemplate,
        )
        from torch._inductor.utils import use_ck_wmma_conv_template

        conv_layout = FixedLayout(
            device, torch.bfloat16, [1, 64, 54, 54], [186624, 1, 3456, 64]
        )
        with V.set_graph_handler(graph):
            for arch, backends, expected in (
                ("gfx1250", "CK,CKWMMA", True),
                ("gfx950", "CK,CKWMMA", False),
                # Opt-in: the CK token alone must never imply WMMA.
                ("gfx1250", "CK", False),
            ):
                with config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_conv_backends": backends,
                        "rocm.arch": [arch],
                        "rocm.ck_dir": self.ck_dir,
                    }
                ):
                    self.assertEqual(
                        use_ck_wmma_conv_template(conv_layout),
                        expected,
                        f"conv WMMA gate wrong for arch={arch} backends={backends}",
                    )
                    conv_template = CKWMMAGroupedConvFwdTemplate(
                        [
                            Buffer(
                                name="X",
                                layout=FixedLayout(
                                    device,
                                    torch.bfloat16,
                                    [1, 8, 56, 56],
                                    [25088, 1, 448, 8],
                                ),
                            ),
                            Buffer(
                                name="W",
                                layout=FixedLayout(
                                    device,
                                    torch.bfloat16,
                                    [64, 8, 3, 3],
                                    [72, 1, 24, 8],
                                ),
                            ),
                        ],
                        conv_layout,
                        stride=[1, 1],
                        padding=[0, 0],
                        dilation=[1, 1],
                        groups=1,
                        n_spatial_dimensions=2,
                    )
                    conv_ops = conv_template.gen_ops()
                    if arch == "gfx1250":
                        # gen_ops is gated on the compile target, not the token,
                        # so it yields instances whenever the arch matches.
                        self.assertTrue(conv_ops, f"no WMMA conv instances for {arch}")
                        self.assertTrue(
                            all(op.is_wmma for op in conv_ops),
                            "non-WMMA op in the WMMA conv pool",
                        )
                    else:
                        self.assertEqual(
                            len(conv_ops),
                            0,
                            "WMMA conv instances leaked onto a non-gfx1250 target",
                        )

        # (b) Warning suppression table. Also inside the graph context: the CK gate
        # consults V.graph.sizevars too.
        import logging

        def _warns(backends, arch):
            _warn_ck_xdl_on_gfx1250.cache_clear()
            with (
                V.set_graph_handler(graph),
                config.patch(
                    {
                        "max_autotune": True,
                        "max_autotune_gemm_backends": backends,
                        "rocm.arch": [arch],
                        "rocm.ck_dir": self.ck_dir,
                    }
                ),
                self.assertLogs("torch._inductor.utils", level="WARNING") as captured,
            ):
                verdict = use_ck_gemm_template(layout, M, N, K)
                # Sentinel so assertLogs never fails for "no logs"; we inspect the
                # captured records ourselves.
                logging.getLogger("torch._inductor.utils").warning("sentinel")
            fired = any("Add 'CKWMMA'" in m for m in captured.output)
            return verdict, fired

        v_ck_1250, warn_ck_1250 = _warns("CK", "gfx1250")
        self.assertTrue(warn_ck_1250)
        _, warn_ckwmma = _warns("CK,CKWMMA", "gfx1250")
        self.assertFalse(warn_ckwmma)
        _, warn_cktile = _warns("CK,CKTILE", "gfx1250")
        self.assertFalse(warn_cktile)
        _, warn_gfx950 = _warns("CK", "gfx950")
        self.assertFalse(warn_gfx950)

        # Diagnostic-only: the warning must not change the gate verdict.
        self.assertTrue(v_ck_1250)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI.
    if HAS_CUDA_AND_TRITON and HAS_CPU and is_big_gpu():
        run_tests()
