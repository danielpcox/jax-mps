# jax-mps

A JAX backend for Apple Metal Performance Shaders (MPS), enabling GPU-accelerated JAX computations on Apple Silicon.

## Quick Start

```bash
pip install jax-mps
```

The plugin registers itself with JAX automatically. Set `JAX_PLATFORMS=mps` to select it explicitly.

Requires macOS on Apple Silicon, Python >= 3.11, and jaxlib 0.9.x.

## Performance

jax-mps achieves a modest 3x speed-up over the CPU backend when training a simple ResNet18 model on CIFAR-10 using an M4 MacBook Air.

```bash
$ JAX_PLATFORMS=cpu uv run examples/resnet/main.py --steps=30
Time per step (second half): 3.041

$ JAX_PLATFORMS=mps uv run examples/resnet/main.py --steps=30
Time per step (second half): 0.991
```

## What Works

**All JAX operations are supported**, verified across 1974 tests covering all major categories:

| Category | Status | Notes |
|----------|--------|-------|
| Element-wise math (unary/binary) | Full | sin, cos, exp, log, erf, gelu, etc. |
| Reductions | Full | sum, prod, max, min, argmax, cumsum, cummax, logsumexp |
| Matmul / dot products | Full | float16, bfloat16, float32, int, complex64 |
| Convolution | Full | 1D, 2D, depthwise, transposed, dilated |
| Pooling | Full | max, avg, min pool with gradients |
| FFT | Full | fft, ifft, rfft, irfft, fft2, ifft2 |
| Sorting | Full | sort, argsort, top_k, unique, searchsorted |
| Shape ops | Full | reshape, transpose, pad, gather, scatter, concatenate |
| Bitwise ops | Full | and, or, xor, not, shifts, population_count, clz |
| Random | Full | normal, uniform, bernoulli, categorical, poisson, gamma, beta, etc. |
| Type conversions | Full | float16/bfloat16/float32/int8-64/bool/complex64, reduce_precision |
| Control flow | Full | cond, switch, while_loop, fori_loop, scan, associative_scan |
| Autodiff | Full | grad, jacobian, hessian, HVP, checkpoint, custom_jvp/vjp |
| Transforms | Full | jit, vmap, pmap (single device) |
| **Linear algebra** | **Full** | See below |
| Complex numbers | Nearly full | Arithmetic, matmul, FFT, all linalg. No complex sort/conv |
| scipy.special | Full | erf, gammaln, digamma, betaln, logit, expit, etc. |
| scipy.signal | Full | convolve, correlate, fftconvolve |
| scipy.stats | Full | norm.logpdf, norm.cdf, norm.ppf |
| scipy.ndimage | Full | map_coordinates |

### Linear Algebra

All operations work for both real (float32) and complex (complex64) inputs:

| Operation | Function | Backend |
|-----------|----------|---------|
| Solve | `jnp.linalg.solve` | MPS native |
| Inverse | `jnp.linalg.inv` | MPS native |
| Cholesky | `jnp.linalg.cholesky` | MPS native (real), Accelerate cpotrf_ (complex) |
| Triangular solve | `scipy.linalg.solve_triangular` | MPS native (real), Accelerate ctrsm_ (complex) |
| QR | `jnp.linalg.qr` | Accelerate sgeqrf_/cgeqrf_ |
| SVD | `jnp.linalg.svd` | Accelerate sgesdd_/cgesdd_ |
| Eigendecomposition (symmetric) | `jnp.linalg.eigh` | Accelerate ssyevd_/cheevd_ |
| Eigendecomposition (general) | `jnp.linalg.eig` | Accelerate sgeev_/cgeev_ |
| Schur | `scipy.linalg.schur` | Accelerate sgees_/cgees_ |
| Matrix square root | `scipy.linalg.sqrtm` | Via Schur |
| Matrix exponential | `scipy.linalg.expm` | Via Schur + solve |
| LU | `scipy.linalg.lu` | Via JAX primitives |
| Determinant, norm, cond, rank, pinv, lstsq | All | Via SVD/QR/solve |

### ML Framework Compatibility

Tested successfully with:
- **Flax NNX** — training loops with optimizers
- **NumPyro** — MCMC inference (NUTS sampler)
- **Optax** — all standard optimizers
- **Equinox** — neural network modules

Transformer components work end-to-end: multi-head attention, RoPE, RMSNorm, SwiGLU, causal masking.

## Known Limitations

These are **Metal/MPS hardware constraints**, not bugs in jax-mps:

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No float64 | Metal GPUs only support 32-bit floats | Use float32 (default). `jax.config.update("jax_enable_x64", True)` will not work. |
| No complex sort | `jnp.sort` on complex arrays crashes MPS | Sort real/imag parts separately |
| No complex convolution | MPS conv ops don't support complex types | Decompose into real/imag convolutions manually |
| No `jax.debug.print` | Debug printing inside JIT not supported | Use `jax.debug.callback` or print outside JIT |
| No buffer donation | Memory optimization hint is ignored (warning only) | No impact on correctness, minor memory overhead |
| `scipy.linalg.polar(method='qdwh')` crashes | QDWH algorithm promotes to float64 internally | Use `polar(method='svd')` instead |
| Zero-size arrays | MPS doesn't support empty tensors | Avoid zero-dimension operations |

## Architecture

This project implements a [PJRT plugin](https://openxla.org/xla/pjrt) to offload evaluation of JAX expressions to a [Metal Performance Shaders Graph](https://developer.apple.com/documentation/metalperformanceshadersgraph). The evaluation proceeds in several stages:

1. JAX lowers the program to [StableHLO](https://openxla.org/stablehlo), a set of high-level operations for ML.
2. The plugin parses the StableHLO representation and builds the corresponding MPS graph. The graph is cached to avoid re-construction on repeated invocations.
3. The MPS graph is executed on the GPU. Operations not natively supported by MPS (e.g., linear algebra decompositions) run on CPU via Apple's Accelerate framework using a "native handler" mechanism.

### Operation Implementations

| Layer | Examples | Count |
|-------|----------|-------|
| MPS Graph ops | add, matmul, conv, reduce, sort, FFT | ~60 |
| CHLO ops | erf, top_k, acos, sinh | ~12 |
| Native handlers (CPU via Metal command buffer) | cholesky, triangular_solve, QR, SVD, eigh, eig, Schur | ~10 |
| Custom calls (LAPACK via Accelerate) | QR, SVD, eigh, eig, Schur | 5 |
| Python lowering rules | eigh, svd, eig, schur → custom_call | 4 |

## Building from Source

1. Install build tools and build LLVM/MLIR & StableHLO (one-time, ~30 minutes):

```bash
brew install cmake ninja
./scripts/setup_deps.sh
```

2. Build and install:

```bash
uv pip install -e .
```

3. Run tests:

```bash
uv sync --all-groups
uv run pytest
```

### Version Pinning

jax-mps is built against the StableHLO bytecode format matching jaxlib 0.9.x. The `setup_deps.sh` script pins LLVM and StableHLO to specific commits from the XLA version used by jaxlib 0.9.0.

**Runtime compatibility:** Any jaxlib 0.9.x release should work with a built binary — the bytecode format is stable within a minor version series. The plugin will warn (not error) if the minor version doesn't match.

**Updating for a new jaxlib release:** Trace the dependency chain:

```bash
# 1. Find XLA commit used by jaxlib
curl -s https://raw.githubusercontent.com/jax-ml/jax/jax-v0.9.0/third_party/xla/revision.bzl

# 2. Find LLVM and StableHLO commits used by that XLA version
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/llvm/workspace.bzl
curl -s https://raw.githubusercontent.com/openxla/xla/<XLA_COMMIT>/third_party/stablehlo/workspace.bzl
```

Then update `STABLEHLO_COMMIT` and `LLVM_COMMIT_OVERRIDE` in `setup_deps.sh`.

## Project Structure

```
jax-mps/
├── CMakeLists.txt
├── src/
│   ├── jax_plugins/mps/         # Python plugin: registration + lowering rules
│   ├── pjrt_plugin/             # C++ PJRT implementation
│   │   ├── pjrt_api.cc          # PJRT C API entry point
│   │   ├── mps_client.h/mm      # Metal client management
│   │   ├── mps_executable.h/mm  # StableHLO compilation & execution
│   │   └── ops/                 # Operation implementations
│   │       ├── unary_ops.mm     # Element-wise unary operations
│   │       ├── binary_ops.mm    # Element-wise binary operations
│   │       ├── bitwise_ops.mm   # Bitwise operations
│   │       ├── shape_ops.mm     # Gather, scatter, reshape, pad, etc.
│   │       ├── reduction_ops.mm # Reduce, reduce_window, scan
│   │       ├── linalg_ops.mm    # Cholesky, QR, SVD, eigh, eig, Schur
│   │       ├── dot_ops.mm       # Dot products and matmul
│   │       ├── fft_ops.mm       # FFT operations
│   │       ├── sort_ops.mm      # Sort and top-k
│   │       └── registry.h       # Op registration macros
│   └── proto/                   # Protobuf definitions
├── tests/
│   ├── test_ops.py              # Main test file (parameterized)
│   └── configs/                 # Test configurations by category
└── mps_ops/                     # Reference docs for MPS Graph methods
```

## Benchmarks

```bash
uv run pytest -m benchmark --benchmark-only
```
