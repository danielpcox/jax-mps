# CUDA Parity Status

Feature parity with `jaxlib[cuda12]` for single-GPU workloads.

## What's Implemented

- **All StableHLO ops** that JAX generates (71 graph ops, 12 CHLO ops)
- **All linalg decompositions** via Apple Accelerate LAPACK: Cholesky, QR, SVD, eigh, eig, Schur, sqrtm, expm
- **Complex64 support** for arithmetic, matmul, FFT, and all linalg ops
- **All JAX transforms**: jit, vmap, grad, jacobian, hessian, checkpoint, custom_jvp/vjp, scan, associative_scan
- **Control flow**: cond, switch, while_loop, fori_loop, scan
- **Error hardening**: clean error messages for unsupported operations (float64, complex sort/conv)
- **Buffer sharing**: pass-through buffers share the underlying MTLBuffer instead of copying

## Remaining Gaps

### `jax.debug.print` / `jax.debug.callback`
- **Impact:** Medium — useful for debugging JIT-compiled code.
- **Status:** Blocked. JAX's `emit_python_callback` explicitly excludes MPS (`platform not in {"cpu", "cuda", "rocm", "tpu"}`). Would require either patching JAX upstream or reimplementing the host callback infrastructure.
- **Workaround:** Print outside JIT, or use `jax.debug.callback` with `jax.effects_barrier()`.

### Buffer donation
- **Impact:** Low — Apple Silicon unified memory reduces the benefit.
- **Approach:** Implement `PJRT_Buffer_DonateWithControlDependency` to reuse input buffers.
- **Files:** `src/pjrt_plugin/pjrt_api.cc`, `src/pjrt_plugin/mps_executable.mm`

### Async execution
- **Impact:** Potentially high for workloads with heavy CPU tracing.
- **Status:** Attempted and reverted. For GPU-bound workloads (ResNet18/CIFAR-10), async execution adds overhead without benefit because the CPU dispatch time is negligible compared to GPU execution. Would benefit workloads with lots of small dispatches or heavy CPU-side tracing.
- **Approach:** Make `PJRT_Event` track Metal command buffer completion; return from Execute before GPU finishes.

## Apple MPS Framework Limitations

These are hardware or framework constraints that cannot be fixed in applejax:

| Issue | Workaround |
|-------|------------|
| No float64 — Metal GPUs only support 32-bit floats | Use float32 (caught with clean error) |
| Complex sort crashes MPS framework | Sort real/imag parts separately (caught with clean error) |
| Complex convolution crashes MPS framework | Manual real/imag decomposition (caught with clean error) |
| `scipy.linalg.polar(method='qdwh')` promotes to float64 internally | Use `polar(method='svd')` |
| Zero-size tensors not supported by Metal | Avoid zero-dimension operations |
| Linalg inside control flow crashes (native ops can't run inside MPS Graph) | Call linalg ops outside scan/fori_loop/while_loop |

## Performance

Benchmarked on M4 MacBook Air (CPU = Accelerate BLAS, MPS = Metal GPU):

| Workload | CPU | MPS | Speedup |
|----------|-----|-----|---------|
| ResNet18 CIFAR-10 train step | 3.0s | 1.0s | 3x |
| matmul 2048x2048 | 27ms | 9.6ms | 2.8x |
| matmul 1024x1024 | 2.0ms | 1.7ms | 1.2x |
| conv2d 128ch 32x32 | 5.0ms | 4.1ms | 1.2x |
| layernorm 1024 | 0.95ms | 3.6ms | 0.26x |

MPS excels at compute-bound workloads (large matmul, batched attention). Dispatch overhead makes small/elementwise ops slower than CPU.

## Potential Improvements

- **Performance:** Graph fusion, Metal shader cache tuning, profiling
- **Broader dtypes:** float8/FP8 (needs Metal support), int4 (needs custom shaders)
- **Multi-device:** Collective ops (all_reduce, etc.) — N/A for single-GPU Macs
- **Upstream:** Report MPS bugs to Apple
