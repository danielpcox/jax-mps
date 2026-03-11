# Changelog

## Unreleased

### Added
- **Window scatter support** — scatter ops with empty `insertedWindowDims` (e.g., `x.at[1:-1, 1:-1].set(val)`) now work correctly. (#32)
- **Partial-index scatter** — `x.at[i, j].add(val)` on 3D+ tensors now works for single scatter points. (#31)
- **OOB index clamping** — `dynamic_slice`, `dynamic_update_slice`, and `gather` now clamp out-of-bounds indices to match CPU/XLA behavior instead of producing incorrect results. (#29, #30)
- Bug report issue template.

### Fixed
- `dot_general` outer product with rank-1 expansion no longer corrupts subsequent operations. (#28)
- Batched `dynamic_slice` via `vmap` no longer crashes. (#27)
- `batch_group_count` in convolution (depthwise conv gradients) now handled correctly. (#26)
- `dot_general` 2D outer product and `reduce_window` identity edge cases. (#25)
- `dot_general` crash on 3D+ @ 1D matmul. (#24)
- `dot_general` crash when scalar broadcasts to vector in gradient. (#23)
- Boolean constant bit-packing bug in `HandleConstant`. (#22)
- `reduce_precision` NaN passthrough. (#21)

### Changed
- **Renamed to `applejax`** on PyPI. Install with `pip install applejax`. Fork of [tillahoffmann/jax-mps](https://github.com/tillahoffmann/jax-mps).
- `equinox` and `optax` moved from hard dependencies to dev dependencies. Users only need `jax` and `jaxlib`.
- PLAN.md renamed to CUDA_PARITY.md and rewritten as a public-facing status document.
- README updated with accurate op counts, known limitations, and project structure.
- Copilot review instructions updated to match current API signatures.

---

## 0.9.7 — 2026-02-15

### Added
- **Full linear algebra** — Cholesky, triangular solve, QR, SVD, eigh, eig, Schur decomposition, all via Apple Accelerate LAPACK. Both real (float32) and complex (complex64) inputs supported.
- **Complex number support for linalg** — complex Cholesky (cpotrf_), QR (cgeqrf_), SVD (cgesdd_), eigh (cheevd_), eig (cgeev_), Schur (cgees_).
- **`count_leading_zeros`** — via float32 log2 + exponent extraction.
- **`reduce_precision`** — bitwise float32 mantissa rounding and exponent clamping with NaN passthrough.
- **Buffer sharing** — pass-through buffers share the underlying MTLBuffer instead of copying.
- **`optimization_barrier`** — enables `jax.checkpoint` (remat).
- **Native ops inside control flow** — `expm`, `solve`, `inv` work inside `lax.cond`.
- **Error hardening** — clean error messages for float64, complex sort/conv, zero-size tensors.
- Benchmarking framework (`uv run pytest -m benchmark --benchmark-only`).
- Comprehensive README rewrite with full op coverage table.
- PLAN.md documenting remaining work for CUDA parity.

### Fixed
- Batched scatter for embedding gradients (updates rank > operand rank).
- Boolean scatter and slice scatter patterns.
- Scatter/gather patterns for `searchsorted` and cross-entropy gradients.
- Zero-sized tensors for `associative_scan` and zero-batch ops.
- Integer/boolean matmul segfault.
- Batched gather and scatter for LU decomposition.
- Multi-index gather dimension ordering for offset dims.
- `dot_general` with batch dims and no contracting dims.

---

## 0.9.6 — 2026-02-11

### Added
- **Sorting and top-k** — `jnp.sort`, `jnp.argsort`, `jax.lax.top_k`.
- **FFT** — `jnp.fft.fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`.
- **Control flow** — `lax.while_loop`, `lax.cond`/`lax.switch` (case).
- **1D convolution** — via 2D lift (contributed by @dlwh).
- **Bitwise ops** — `bitwise_and`, `or`, `xor`, `not`, shifts, `population_count`.
- **Trigonometric/hyperbolic ops** — `asin`, `acos`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh`, `cbrt`.
- **Linalg ops** — Cholesky, triangular solve with native MPS kernel execution.
- **NumPyro distribution tests** — Poisson, Binomial, VonMises, etc.
- clang-tidy in pre-commit hooks and CI.
- Dev releases on main branch pushes.
- Transposed convolution fix using DataGradient API.

---

## 0.9.5 — 2026-02-08

### Added
- **Batched linalg** — batch dimension support for Cholesky and triangular solve.
- **Batched dot_general** — batch dimensions for matmul gradients.
- `HandlerContext` struct unifying handler signatures.
- GitHub Copilot code review instructions.
- `JAX_TEST_MODE` env var for test platform selection.

### Fixed
- Scatter rank mismatch for scalar-index updates.
- dot_general crash on mixed-rank dot products.
- Memory leaks and shader cache pressure.
- `log1p` numerical stability for small inputs.

---

## 0.9.4 — 2026-01-30

### Added
- **Gather expansion** — multi-index point and batched gather patterns.
- **Interior padding** in pad op.
- **Pooling** — `reduce_window` for max/avg/min pool.
- **General ScatterND fallback** for multi-dim index vectors.
- `stablehlo.round_nearest_even` op.
- Missing op issue template.
- CONTRIBUTING.md.
- CPU-only tests in CI (MPS not available on GitHub runners).

### Fixed
- `dot_general` crash on mixed-rank dot products.

---

## 0.9.3 — 2026-01-28

### Added
- **Complex number support** — `complex64` arithmetic, matmul, FFT.
- Gradient checks for complex-valued operations.

---

## 0.9.2 — 2026-01-27

### Added
- `ceil`, `cos`, `sin`, `tan` ops.
- `chlo.square` op.
- IOHW/CNHW conv layouts for NCHW backward pass.
- Comprehensive test coverage for all registered StableHLO/CHLO ops.
- Apache License 2.0.
- ResNet example README.

### Fixed
- Scatter with scalar indices and updates.
- Scatter modes and `.at[]` indexing.
- Protoc discovery for static protobuf build.

---

## 0.9.1 — 2026-01-26

### Added
- **PyPI release** — `pip install jax-mps` (later renamed to `applejax`).
- ResNet18 CIFAR-10 example with 3x MPS speedup.
- Microbenchmark script.
- pyright type checking.

### Fixed
- Strided convolution gradient divergence on MPS.
- Non-contiguous array transfer to MPS.
- Asymmetric stride gradient in transposed convolution.

---

## 0.9.0 — 2026-01-20

Initial release by [@tillahoffmann](https://github.com/tillahoffmann).

### Added
- PJRT plugin for Apple Metal Performance Shaders.
- StableHLO → MPS Graph compilation and execution.
- Element-wise unary and binary operations.
- Reductions (sum, prod, max, min).
- Matmul and dot products.
- 2D convolution (forward and backward).
- Reshape, transpose, pad, slice, concatenate.
- Random number generation.
- Gradient support via autodiff.
- `jit`, `vmap` transforms.
- Scatter and gather operations.
- Dynamic slice and update.
- CI with GitHub Actions (build, lint, test).
- Pre-commit hooks (clang-format, ruff, pyright, pytest).

### Contributors
- [@tillahoffmann](https://github.com/tillahoffmann) — creator and primary author through 0.9.7
- [@dlwh](https://github.com/dlwh) — 1D convolution lowering
- [@dominik3141](https://github.com/dominik3141) — scatter rank mismatch fixes
- [@jkitchin](https://github.com/jkitchin) — expanded gather, select_and_scatter, control flow, dot_general
- [@danielpcox](https://github.com/danielpcox) — full linalg suite, complex support, scatter/gather hardening, stress testing
