import os
import re
from pathlib import Path

import jax
import numpy
import pytest
from jax import dtypes, random
from jax import numpy as jnp

from .configs import (
    OperationTestConfig,
    make_binary_op_configs,
    make_control_flow_op_configs,
    make_conv_op_configs,
    make_conversion_op_configs,
    make_flax_op_configs,
    make_linalg_op_configs,
    make_matmul_op_configs,
    make_misc_op_configs,
    make_numpyro_op_configs,
    make_random_op_configs,
    make_reduction_op_configs,
    make_shape_op_configs,
    make_slice_op_configs,
    make_sort_op_configs,
    make_unary_op_configs,
)

# Test mode configuration via environment variable:
# - "compare" (default): Run on both CPU and MPS, compare results
# - "mps": Run only on MPS
# - "cpu": Run only on CPU
TEST_MODE = os.environ.get("JAX_TEST_MODE", "compare").lower()
if TEST_MODE not in ("compare", "mps", "cpu"):
    raise ValueError(
        f"Invalid JAX_TEST_MODE: {TEST_MODE}. Must be 'compare', 'mps', or 'cpu'."
    )


def get_test_platforms() -> list[str]:
    """Return the platforms to test based on JAX_TEST_MODE environment variable."""
    if TEST_MODE == "compare":
        return ["cpu", "mps"]
    else:
        return [TEST_MODE]


OPERATION_TEST_CONFIGS = [
    *make_binary_op_configs(),
    *make_control_flow_op_configs(),
    *make_conv_op_configs(),
    *make_conversion_op_configs(),
    *make_flax_op_configs(),
    *make_linalg_op_configs(),
    *make_matmul_op_configs(),
    *make_misc_op_configs(),
    *make_numpyro_op_configs(),
    *make_random_op_configs(),
    *make_reduction_op_configs(),
    *make_shape_op_configs(),
    *make_slice_op_configs(),
    *make_sort_op_configs(),
    *make_unary_op_configs(),
]


@pytest.fixture(params=OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name)
def op_config(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["jit", "eager"])
def jit(request: pytest.FixtureRequest):
    return request.param


def fassert(cond: bool, message: str) -> None:
    """Functional assertion."""
    assert cond, message


def assert_allclose_with_path(path, actual, desired):
    # Extract key data if these are random keys rather than regular data.
    is_prng_key = dtypes.issubdtype(actual.dtype, dtypes.prng_key)  # pyright: ignore[reportPrivateImportUsage]
    if is_prng_key:
        actual = random.key_data(actual)
        desired = random.key_data(desired)

    try:
        # Use exact comparison for exact dtypes, tolerance-based for inexact.
        if jnp.issubdtype(actual.dtype, jnp.inexact):
            numpy.testing.assert_allclose(actual, desired, atol=1e-5, rtol=1e-5)
        else:
            numpy.testing.assert_array_equal(actual, desired)
    except AssertionError as ex:
        raise AssertionError(f"Values are not close at path '{path}'.") from ex


def test_op_value(op_config: OperationTestConfig, jit: bool) -> None:
    platforms = get_test_platforms()
    results = []
    for platform in platforms:
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            result = op_config.evaluate_value(jit)
            jax.tree.map_with_path(
                lambda path, value: fassert(
                    value.device == device,
                    f"Value at '{path}' is on device {value.device}; expected {device}.",
                ),
                result,
            )
            results.append(result)

    if len(results) == 2:
        jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_op_grad(
    op_config: OperationTestConfig, jit: bool, request: pytest.FixtureRequest
) -> None:
    argnums = op_config.get_differentiable_argnums()
    if not argnums:
        pytest.skip(f"No differentiable arguments for operation '{op_config.func}'.")

    if op_config.grad_xfail:
        from .configs.util import MPS_DEVICE

        # Only apply xfail when actually testing MPS, not just when MPS is available
        if MPS_DEVICE is not None and "mps" in get_test_platforms():
            request.applymarker(
                pytest.mark.xfail(  # type: ignore[call-overload]
                    reason=op_config.grad_xfail,
                    match=op_config.grad_xfail,
                    strict=True,
                )
            )

    platforms = get_test_platforms()
    for argnum in argnums:
        results = []
        for platform in platforms:
            device = jax.devices(platform)[0]
            with jax.default_device(device):
                result = op_config.evaluate_grad(argnum, jit)
                jax.tree.map_with_path(
                    lambda path, value: fassert(
                        value.device == device,
                        f"Value at '{path}' is on device {value.device}; expected {device}.",
                    ),
                    result,
                )
                results.append(result)

        if len(results) == 2:
            jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_unsupported_op_error_message(jit: bool) -> None:
    """Check that unsupported-op errors link to the issue template and CONTRIBUTING.md."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    device = jax.devices("mps")[0]
    with jax.default_device(device):
        try:
            # This is an obscure op. It's unlikely to be implemented, but this test may
            # break if `clz` gets implemented.
            func = jax.lax.clz
            if jit:
                func = jax.jit(func)
            func(numpy.int32(7))
        except Exception as exc:
            message = str(exc)
            assert "issues/new?template=missing-op.yml" in message
            assert "CONTRIBUTING.md" in message
        else:
            pytest.skip("clz is now supported; test needs a new unregistered op")


def test_boolean_constant_in_jit() -> None:
    """Regression test: boolean constants (JIT-captured bool arrays) must be unpacked
    from MLIR's bit-packed i1 format to byte-per-element for MPS Graph."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    mps = jax.devices("mps")[0]

    # Small boolean array
    mask = jax.device_put(jnp.array([True, False, True, True, False, True]), mps)

    @jax.jit
    def sum_captured():
        return jnp.sum(mask.astype(jnp.float32))

    assert float(sum_captured()) == 4.0

    # Larger boolean array (exercises multi-byte bit-packing)
    rng = numpy.random.RandomState(42)
    big_mask_np = numpy.zeros(2000, dtype=bool)
    big_mask_np[rng.permutation(2000)[:140]] = True
    big_mask = jax.device_put(jnp.array(big_mask_np), mps)

    @jax.jit
    def sum_captured_large():
        return jnp.sum(big_mask.astype(jnp.float32))

    assert float(sum_captured_large()) == 140.0


def test_dot_general_scalar_vector_gradient() -> None:
    """Regression test: dot_general with scalar * vector operands must not expand
    the vector to rank-2 when the other operand is rank-0 (scalar broadcast)."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    A = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    x = jnp.array([0.5, -0.5])

    # grad of x^T A x w.r.t. x (generates scalar * vector dot_general in backward)
    g_cpu = numpy.asarray(jax.jit(jax.grad(lambda x, A: jnp.dot(x, jnp.dot(A, x))),
                                   device=cpu)(x, A))
    g_mps = numpy.asarray(jax.jit(jax.grad(lambda x, A: jnp.dot(x, jnp.dot(A, x))),
                                   device=mps)(jax.device_put(x, mps),
                                               jax.device_put(A, mps)))
    numpy.testing.assert_allclose(g_mps, g_cpu, atol=1e-5)

    # grad of x^T A x w.r.t. A (generates outer product via scalar broadcast)
    gA_cpu = numpy.asarray(jax.jit(jax.grad(lambda A, x: jnp.dot(x, jnp.dot(A, x)),
                                             argnums=0), device=cpu)(A, x))
    gA_mps = numpy.asarray(jax.jit(jax.grad(lambda A, x: jnp.dot(x, jnp.dot(A, x)),
                                             argnums=0), device=mps)(
                                    jax.device_put(A, mps), jax.device_put(x, mps)))
    numpy.testing.assert_allclose(gA_mps, gA_cpu, atol=1e-5)


def test_dot_general_3d_1d_matmul() -> None:
    """Regression test: 3D @ 1D matmul must not trigger rank-1 expansion
    (designed for 2D only). The general fallback handles batched cases correctly."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    # 3D @ 1D: batched matrix-vector product
    A = jnp.ones((4, 3, 5))
    v = jnp.ones((5,))
    cpu_result = numpy.asarray(jax.jit(jnp.matmul, device=cpu)(A, v))
    mps_result = numpy.asarray(jax.jit(jnp.matmul, device=mps)(
        jax.device_put(A, mps), jax.device_put(v, mps)))
    numpy.testing.assert_allclose(mps_result, cpu_result, atol=1e-5)

    # 3D @ 1D inside lax.scan (exercises JIT + control flow + matmul interaction)
    def scan_body(carry, x_row):
        return carry + x_row @ v, None

    def run_scan(A, v):
        init = jnp.zeros(A.shape[1])
        result, _ = jax.lax.scan(scan_body, init, A)
        return result

    cpu_scan = numpy.asarray(jax.jit(run_scan, device=cpu)(A, v))
    mps_scan = numpy.asarray(jax.jit(run_scan, device=mps)(
        jax.device_put(A, mps), jax.device_put(v, mps)))
    numpy.testing.assert_allclose(mps_scan, cpu_scan, atol=1e-5)

    # Gradient through 3D @ 1D
    def loss_fn(v):
        return jnp.sum(A @ v)

    g_cpu = numpy.asarray(jax.jit(jax.grad(loss_fn), device=cpu)(v))
    g_mps = numpy.asarray(jax.jit(jax.grad(loss_fn), device=mps)(
        jax.device_put(v, mps)))
    numpy.testing.assert_allclose(g_mps, g_cpu, atol=1e-5)


def test_dot_general_2d_outer_product() -> None:
    """Regression test: dot_general with two 2D operands and no contracting/batch
    dims (pure outer product) must produce the correct 4D result."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    # einsum 'ij,kl->ikjl' is a 2D outer product (no contraction, no batch)
    cpu_result = numpy.asarray(jax.jit(
        lambda a, b: jnp.einsum('ij,kl->ikjl', a, b), device=cpu)(a, b))
    mps_result = numpy.asarray(jax.jit(
        lambda a, b: jnp.einsum('ij,kl->ikjl', a, b), device=mps)(
        jax.device_put(a, mps), jax.device_put(b, mps)))
    numpy.testing.assert_allclose(mps_result, cpu_result, atol=1e-5)


def test_reduce_window_identity() -> None:
    """Regression test: reduce_window with all-ones window sizes (identity op)
    must return the input unchanged rather than erroring."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    x = jnp.arange(16, dtype=jnp.float32).reshape(1, 4, 4, 1)

    @jax.jit
    def identity_pool(x):
        return jax.lax.reduce_window(x, 0.0, jax.lax.add, (1, 1, 1, 1),
                                     (1, 1, 1, 1), 'VALID')

    cpu_result = numpy.asarray(identity_pool(jax.device_put(x, cpu)))
    mps_result = numpy.asarray(identity_pool(jax.device_put(x, mps)))
    numpy.testing.assert_allclose(mps_result, cpu_result, atol=1e-5)


def test_depthwise_conv_gradient() -> None:
    """Regression test: gradient through depthwise convolution generates a convolution
    with batch_group_count > 1 (for the kernel gradient). This must be converted to
    an equivalent feature_group_count convolution."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    from jax import lax

    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    key = jax.random.PRNGKey(55)
    k1, k2 = jax.random.split(key)
    channels = 4

    x = jax.random.normal(k1, (2, 8, 8, channels), dtype=jnp.float32)
    # Depthwise kernel: (kH, kW, 1, channels) with feature_group_count=channels
    dw_kernel = jax.random.normal(k2, (3, 3, 1, channels), dtype=jnp.float32) * 0.1

    def depthwise_conv_loss(x, kernel):
        dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
        out = lax.conv_general_dilated(
            x, kernel, window_strides=(1, 1), padding='SAME',
            dimension_numbers=dn, feature_group_count=channels
        )
        return jnp.sum(out ** 2)

    # Compare forward
    cpu_fwd = numpy.asarray(jax.jit(depthwise_conv_loss, device=cpu)(x, dw_kernel))
    mps_fwd = numpy.asarray(jax.jit(depthwise_conv_loss, device=mps)(
        jax.device_put(x, mps), jax.device_put(dw_kernel, mps)))
    numpy.testing.assert_allclose(mps_fwd, cpu_fwd, atol=1e-4)

    # Compare gradients (kernel gradient uses batch_group_count internally)
    grad_fn = jax.grad(depthwise_conv_loss, argnums=(0, 1))
    g_cpu_x, g_cpu_k = jax.jit(grad_fn, device=cpu)(x, dw_kernel)
    g_mps_x, g_mps_k = jax.jit(grad_fn, device=mps)(
        jax.device_put(x, mps), jax.device_put(dw_kernel, mps))
    numpy.testing.assert_allclose(numpy.asarray(g_mps_x), numpy.asarray(g_cpu_x), atol=1e-4)
    numpy.testing.assert_allclose(numpy.asarray(g_mps_k), numpy.asarray(g_cpu_k), atol=1e-4)

    # Also test with different group count (2 groups instead of fully depthwise)
    k3 = jax.random.PRNGKey(99)
    kernel_2g = jax.random.normal(k3, (3, 3, 2, channels), dtype=jnp.float32) * 0.1

    def grouped_conv_loss(x, kernel):
        dn = lax.conv_dimension_numbers(x.shape, kernel.shape, ('NHWC', 'HWIO', 'NHWC'))
        out = lax.conv_general_dilated(
            x, kernel, window_strides=(1, 1), padding='SAME',
            dimension_numbers=dn, feature_group_count=2
        )
        return jnp.sum(out ** 2)

    g2_cpu_x, g2_cpu_k = jax.jit(jax.grad(grouped_conv_loss, argnums=(0, 1)), device=cpu)(
        x, kernel_2g)
    g2_mps_x, g2_mps_k = jax.jit(jax.grad(grouped_conv_loss, argnums=(0, 1)), device=mps)(
        jax.device_put(x, mps), jax.device_put(kernel_2g, mps))
    numpy.testing.assert_allclose(numpy.asarray(g2_mps_x), numpy.asarray(g2_cpu_x), atol=1e-4)
    numpy.testing.assert_allclose(numpy.asarray(g2_mps_k), numpy.asarray(g2_cpu_k), atol=1e-4)


def test_vmap_dynamic_slice() -> None:
    """Regression test: vmap over dynamic_slice produces a batched gather with
    slice_sizes > 1. The gather handler must build iota offsets from the start
    index rather than treating it as a point gather."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    from jax import lax

    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    key = jax.random.PRNGKey(42)
    xs = jax.random.normal(key, (5, 10), dtype=jnp.float32)
    starts = jnp.array([0, 1, 2, 3, 4])

    def slice_one(x, s):
        return lax.dynamic_slice(x, (s,), (4,))

    cpu_result = numpy.asarray(jax.jit(jax.vmap(slice_one), device=cpu)(xs, starts))
    mps_result = numpy.asarray(jax.jit(jax.vmap(slice_one), device=mps)(
        jax.device_put(xs, mps), jax.device_put(starts, mps)))
    numpy.testing.assert_allclose(mps_result, cpu_result, atol=1e-5)


def test_outer_product_then_batched_dot() -> None:
    """Regression test: 3-way outer product (i,j,k->ijk) via dot_general should not
    corrupt subsequent dot_general operations with batch/contracting dims.
    The rank-1 expansion for outer products must only apply to 1D×1D cases,
    not 2D×1D which goes through the general outer product path."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    mps = jax.devices("mps")[0]
    cpu = jax.devices("cpu")[0]

    key = jax.random.PRNGKey(235)

    # 3-way outer product via einsum
    a = jax.random.normal(key, (8,), dtype=jnp.float32)
    b = jax.random.normal(jax.random.PRNGKey(1), (6,), dtype=jnp.float32)
    c = jax.random.normal(jax.random.PRNGKey(2), (4,), dtype=jnp.float32)

    cpu_outer = numpy.asarray(jnp.einsum('i,j,k->ijk', *[jax.device_put(x, cpu) for x in [a, b, c]]))
    mps_outer = numpy.asarray(jnp.einsum('i,j,k->ijk', *[jax.device_put(x, mps) for x in [a, b, c]]))
    numpy.testing.assert_allclose(mps_outer, cpu_outer, atol=1e-5)

    # Gradient through 3-tensor einsum (exercises dot_general with batch+contract dims)
    A = jax.random.normal(key, (4, 5, 6), dtype=jnp.float32)
    B = jax.random.normal(jax.random.PRNGKey(3), (5, 6, 7), dtype=jnp.float32)
    C = jax.random.normal(jax.random.PRNGKey(4), (6, 7, 8), dtype=jnp.float32)

    def loss(A, B, C):
        return jnp.sum(jnp.einsum('ijk,jkl,klm->im', A, B, C) ** 2)

    with jax.default_device(cpu):
        ga_cpu = numpy.asarray(jax.grad(loss)(A, B, C))
    with jax.default_device(mps):
        ga_mps = numpy.asarray(jax.grad(loss)(
            jax.device_put(A, mps), jax.device_put(B, mps), jax.device_put(C, mps)))
    numpy.testing.assert_allclose(ga_mps, ga_cpu, atol=1e-3, rtol=1e-3)


@pytest.fixture(autouse=True, scope="module")
def assert_all_ops_tested():
    yield

    if "CI" not in os.environ:
        return

    # Skip op coverage check in CPU-only mode since EXERCISED_STABLEHLO_OPS is only
    # populated when running on MPS.
    if TEST_MODE == "cpu":
        return

    ops_dir = Path(__file__).parent.parent / "src/pjrt_plugin/ops"
    assert ops_dir.is_dir()

    # Patterns matching op registration calls
    patterns = [
        re.compile(r'REGISTER_MPS_OP\("([^"]+)"'),
        re.compile(r'REGISTER_NATIVE_MPS_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_BINARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_UNARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_LOGICAL_BITWISE_OP\("([^"]+)"'),
    ]

    # Ops that appear in StableHLO IR but get lowered by MLIR before reaching
    # our handlers. They work but don't need explicit registration.
    mlir_lowered_ops = {"chlo.lgamma", "chlo.digamma", "chlo.bessel_i1e"}

    op_names = set()
    for mm_file in ops_dir.glob("*.mm"):
        with mm_file.open() as fp:
            content = fp.read()
            for pattern in patterns:
                op_names.update(pattern.findall(content))

    assert op_names, "Failed to discover any ops."
    exercised = OperationTestConfig.EXERCISED_STABLEHLO_OPS - mlir_lowered_ops
    unsupported = exercised - op_names
    assert not unsupported, (
        f"Discovered {len(unsupported)} unsupported ops: {', '.join(sorted(unsupported))}"
    )
    missing = op_names - OperationTestConfig.EXERCISED_STABLEHLO_OPS
    assert not missing, (
        f"Discovered {len(missing)} untested ops: {', '.join(sorted(missing))}"
    )
