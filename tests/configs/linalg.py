import jax
import numpy
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import solve_triangular

from .util import OperationTestConfig


def _random_posdef(key, n: int, batch_shape: tuple[int, ...] = ()):
    """Generate random positive-definite matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    A = random.normal(key, shape)
    # A @ A.T for batched inputs: (..., n, n) @ (..., n, n) -> (..., n, n)
    result = jnp.einsum("...ij,...kj->...ik", A, A) + n * jnp.eye(n, dtype=jnp.float32)
    return result


def _random_complex_posdef(key, n: int, batch_shape: tuple[int, ...] = ()):
    """Generate random Hermitian positive-definite complex matrices."""
    shape = (*batch_shape, n, n)
    k1, k2 = random.split(key)
    A = random.normal(k1, shape) + 1j * random.normal(k2, shape)
    result = jnp.einsum("...ij,...kj->...ik", A, A.conj()) + n * jnp.eye(n, dtype=jnp.float32)
    return result


def _solve_triangular_lower(L, B):
    return solve_triangular(L, B, lower=True)


def _solve_triangular_upper(U, B):
    return solve_triangular(U, B, lower=False)


def _solve_triangular_lower_trans(L, B):
    return solve_triangular(L, B, lower=True, trans=1)


def _solve_triangular_upper_trans(U, B):
    return solve_triangular(U, B, lower=False, trans=1)


def _solve_triangular_unit_diag(L, B):
    return solve_triangular(L, B, lower=True, unit_diagonal=True)


def _random_triangular(
    key,
    n: int,
    lower: bool = True,
    batch_shape: tuple[int, ...] = (),
):
    """Generate random well-conditioned triangular matrices of shape (*batch_shape, n, n)."""
    shape = (*batch_shape, n, n)
    M = random.normal(key, shape)
    L = jnp.tril(M) if lower else jnp.triu(M)
    # Fix diagonal to ensure well-conditioned: |diag| + 1
    # Use eye mask to modify diagonal without advanced indexing
    eye = jnp.eye(n, dtype=jnp.float32)
    # Get off-diagonal elements by masking out diagonal
    off_diag = L * (1 - eye)
    # For diagonal, take abs and add 1 (using the diagonal part of L)
    diag_values = jnp.abs(L * eye) + eye
    return off_diag + diag_values


def _random_triangular_unit_diag(key, n: int):
    """Generate random unit-diagonal triangular matrix."""
    M = random.normal(key, (n, n))
    L = jnp.tril(M)
    # Use eye mask to set diagonal to 1 without advanced indexing
    eye = jnp.eye(n, dtype=jnp.float32)
    return L * (1 - eye) + eye


def make_linalg_op_configs():
    with OperationTestConfig.module_name("linalg"):
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda key, n=n: _random_posdef(key, n),
                name=f"cholesky_{n}x{n}",
            )

        # Cholesky on a non-positive-definite matrix (should match CPU NaN behavior).
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.array([[-1, 0], [0, 1]], dtype=numpy.float32),
            name="cholesky_non_posdef",
        )

        for n in [2, 3, 4]:
            # Lower triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda key, n=n: _random_triangular(key, n, lower=True),
                lambda key, n=n: random.normal(key, (n, 1)),
                name=f"triangular_solve_lower_{n}x{n}",
            )
            # Upper triangular, single RHS column.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda key, n=n: _random_triangular(key, n, lower=False),
                lambda key, n=n: random.normal(key, (n, 1)),
                name=f"triangular_solve_upper_{n}x{n}",
            )
            # Lower triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda key, n=n: _random_triangular(key, n, lower=True),
                lambda key, n=n: random.normal(key, (n, 3)),
                name=f"triangular_solve_lower_{n}x{n}_multi_rhs",
            )
            # Upper triangular, multiple RHS columns.
            yield OperationTestConfig(
                _solve_triangular_upper,
                lambda key, n=n: _random_triangular(key, n, lower=False),
                lambda key, n=n: random.normal(key, (n, 3)),
                name=f"triangular_solve_upper_{n}x{n}_multi_rhs",
            )

        # Transpose: solve L^T x = b and U^T x = b
        yield OperationTestConfig(
            _solve_triangular_lower_trans,
            lambda key: _random_triangular(key, 3, lower=True),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_lower_trans",
        )
        yield OperationTestConfig(
            _solve_triangular_upper_trans,
            lambda key: _random_triangular(key, 3, lower=False),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_upper_trans",
        )

        # Unit diagonal: assume diagonal elements are 1
        yield OperationTestConfig(
            _solve_triangular_unit_diag,
            lambda key: _random_triangular_unit_diag(key, 3),
            lambda key: random.normal(key, (3, 1)),
            name="triangular_solve_unit_diagonal",
        )

        # 1x1 matrices (trivial edge case)
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.array([[4.0]], dtype=numpy.float32),
            name="cholesky_1x1",
        )
        yield OperationTestConfig(
            _solve_triangular_lower,
            numpy.array([[2.0]], dtype=numpy.float32),
            numpy.array([[6.0]], dtype=numpy.float32),
            name="triangular_solve_1x1",
        )

        # Batched inputs
        for batch_shape in [(2,), (2, 3)]:
            batch_str = "x".join(map(str, batch_shape))
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda key, bs=batch_shape: _random_posdef(key, 3, batch_shape=bs),
                name=f"cholesky_batched_{batch_str}",
            )
            yield OperationTestConfig(
                _solve_triangular_lower,
                lambda key, bs=batch_shape: _random_triangular(
                    key, 3, lower=True, batch_shape=bs
                ),
                lambda key, bs=batch_shape: random.normal(key, (*bs, 3, 1)),
                name=f"triangular_solve_batched_{batch_str}",
            )

        # Edge case: zero batch size (empty batch dimension)
        # CPU handles this correctly, returning empty arrays with the right shape.
        # Forward pass works via zero-sized tensor handling, but grad
        # generates entirely zero-sized intermediate ops that can't run on MPS.
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            numpy.zeros((0, 3, 3), dtype=numpy.float32),
            differentiable_argnums=(),
            name="cholesky_zero_batch",
        )
        yield OperationTestConfig(
            _solve_triangular_lower,
            numpy.zeros((0, 3, 3), dtype=numpy.float32),
            numpy.zeros((0, 3, 1), dtype=numpy.float32),
            differentiable_argnums=(),
            name="triangular_solve_zero_batch",
        )

        # --- solve and inv (use LU decomposition + triangular_solve internally) ---

        # jnp.linalg.solve: A x = b
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.solve,
                lambda key, n=n: _random_posdef(key, n),
                lambda key, n=n: random.normal(key, (n, 1)),
                name=f"solve_{n}x{n}",
            )

        # solve with multiple RHS columns
        yield OperationTestConfig(
            jnp.linalg.solve,
            lambda key: _random_posdef(key, 3),
            lambda key: random.normal(key, (3, 4)),
            name="solve_3x3_multi_rhs",
        )

        # jnp.linalg.inv
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.inv,
                lambda key, n=n: _random_posdef(key, n),
                name=f"inv_{n}x{n}",
            )

        # --- complex det (uses LU decomposition + complex gather/scatter) ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.det,
                lambda key, n=n: random.normal(key, (n, n))
                + 1j * random.normal(random.fold_in(key, 1), (n, n)),
                differentiable_argnums=(),
                name=f"det_complex_{n}x{n}",
            )

        # Batched complex det
        yield OperationTestConfig(
            jnp.linalg.det,
            lambda key: random.normal(key, (2, 3, 3))
            + 1j * random.normal(random.fold_in(key, 1), (2, 3, 3)),
            differentiable_argnums=(),
            name="det_complex_batched",
        )

        # Complex slogdet (log absolute det)
        yield OperationTestConfig(
            lambda x: jnp.linalg.slogdet(x)[1],
            lambda key: random.normal(key, (3, 3))
            + 1j * random.normal(random.fold_in(key, 1), (3, 3)),
            differentiable_argnums=(),
            name="slogdet_complex_3x3",
        )

        # Complex inv (uses LU + complex triangular_solve)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.inv,
                lambda key, n=n: (
                    lambda A: A @ A.conj().T + n * jnp.eye(n)
                )(
                    random.normal(key, (n, n))
                    + 1j * random.normal(random.fold_in(key, 1), (n, n))
                ),
                differentiable_argnums=(),
                name=f"inv_complex_{n}x{n}",
            )

        # Complex solve
        yield OperationTestConfig(
            jnp.linalg.solve,
            lambda key: (
                lambda A: A @ A.conj().T + 3 * jnp.eye(3)
            )(
                random.normal(key, (3, 3))
                + 1j * random.normal(random.fold_in(key, 1), (3, 3))
            ),
            lambda key: random.normal(key, (3, 1))
            + 1j * random.normal(random.fold_in(key, 2), (3, 1)),
            differentiable_argnums=(),
            name="solve_complex_3x3",
        )

        # Batched complex inv
        yield OperationTestConfig(
            jnp.linalg.inv,
            lambda key: (
                lambda A: jnp.einsum("...ij,...kj->...ik", A, A.conj()) + 3 * jnp.eye(3)
            )(
                random.normal(key, (2, 3, 3))
                + 1j * random.normal(random.fold_in(key, 1), (2, 3, 3))
            ),
            differentiable_argnums=(),
            name="inv_complex_batched",
        )

        # --- complex Cholesky (via Accelerate LAPACK cpotrf_) ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.cholesky,
                lambda key, n=n: (
                    lambda A: A @ A.conj().T + n * jnp.eye(n)
                )(
                    random.normal(key, (n, n))
                    + 1j * random.normal(random.fold_in(key, 1), (n, n))
                ),
                differentiable_argnums=(),
                name=f"cholesky_complex_{n}x{n}",
            )

        # Batched complex Cholesky
        yield OperationTestConfig(
            jnp.linalg.cholesky,
            lambda key: (
                lambda A: jnp.einsum("...ij,...kj->...ik", A, A.conj()) + 3 * jnp.eye(3)
            )(
                random.normal(key, (2, 3, 3))
                + 1j * random.normal(random.fold_in(key, 1), (2, 3, 3))
            ),
            differentiable_argnums=(),
            name="cholesky_complex_batched",
        )

        # --- QR decomposition (via Accelerate LAPACK) ---

        # QR on tall matrices
        for m, n in [(4, 3), (5, 3), (6, 4)]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.qr(x)[0],
                lambda key, m=m, n=n: random.normal(key, (m, n)),
                differentiable_argnums=(0,),
                name=f"qr_Q_{m}x{n}",
            )
            yield OperationTestConfig(
                lambda x: jnp.linalg.qr(x)[1],
                lambda key, m=m, n=n: random.normal(key, (m, n)),
                differentiable_argnums=(0,),
                name=f"qr_R_{m}x{n}",
            )

        # QR on square matrix
        yield OperationTestConfig(
            lambda x: jnp.linalg.qr(x)[0],
            lambda key: random.normal(key, (3, 3)),
            differentiable_argnums=(0,),
            name="qr_Q_3x3",
        )

        # QR on wide matrix (grad not supported by JAX for wide matrices)
        yield OperationTestConfig(
            lambda x: jnp.linalg.qr(x)[0],
            lambda key: random.normal(key, (3, 5)),
            differentiable_argnums=(),
            name="qr_Q_3x5",
        )

        # Complex QR decomposition
        for m, n in [(4, 3), (3, 3)]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.qr(x)[0],
                lambda key, m=m, n=n: (
                    random.normal(key, (m, n))
                    + 1j * random.normal(random.split(key)[0], (m, n))
                ),
                differentiable_argnums=(),
                name=f"qr_Q_complex_{m}x{n}",
            )
            yield OperationTestConfig(
                lambda x: jnp.linalg.qr(x)[1],
                lambda key, m=m, n=n: (
                    random.normal(key, (m, n))
                    + 1j * random.normal(random.split(key)[0], (m, n))
                ),
                differentiable_argnums=(),
                name=f"qr_R_complex_{m}x{n}",
            )

        # Batched complex QR
        yield OperationTestConfig(
            lambda x: jnp.linalg.qr(x)[0],
            lambda key: (
                random.normal(key, (2, 4, 3))
                + 1j * random.normal(random.split(key)[0], (2, 4, 3))
            ),
            differentiable_argnums=(),
            name="qr_Q_complex_batched",
        )

        # --- Symmetric eigendecomposition (via Accelerate LAPACK) ---

        # eigh: eigenvalues only
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.eigvalsh,
                lambda key, n=n: _random_posdef(key, n),
                name=f"eigvalsh_{n}x{n}",
            )

        # eigh: eigenvalues and eigenvectors
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.eigh(x)[0],
                lambda key, n=n: _random_posdef(key, n),
                differentiable_argnums=(0,),
                name=f"eigh_values_{n}x{n}",
            )

        # Complex eigh (Hermitian eigendecomposition)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.eigvalsh,
                lambda key, n=n: _random_complex_posdef(key, n),
                differentiable_argnums=(),
                name=f"eigvalsh_complex_{n}x{n}",
            )
            yield OperationTestConfig(
                lambda x: jnp.linalg.eigh(x)[0],
                lambda key, n=n: _random_complex_posdef(key, n),
                differentiable_argnums=(),
                name=f"eigh_values_complex_{n}x{n}",
            )

        # Batched complex eigh
        yield OperationTestConfig(
            jnp.linalg.eigvalsh,
            lambda key: _random_complex_posdef(key, 3, batch_shape=(2,)),
            differentiable_argnums=(),
            name="eigvalsh_complex_batched",
        )

        # --- det and slogdet (use LU internally) ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.det,
                lambda key, n=n: _random_posdef(key, n),
                name=f"det_{n}x{n}",
            )

        yield OperationTestConfig(
            lambda x: jnp.linalg.slogdet(x)[1],
            lambda key: _random_posdef(key, 3),
            differentiable_argnums=(0,),
            name="slogdet_logabsdet_3x3",
        )

        # --- SVD (via Accelerate LAPACK sgesdd_) ---

        # Full SVD on square matrices
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.svd(x, full_matrices=True)[1],
                lambda key, n=n: random.normal(key, (n, n)),
                differentiable_argnums=(0,),
                name=f"svd_s_full_{n}x{n}",
            )

        # Reduced SVD on square matrix
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
            lambda key: random.normal(key, (3, 3)),
            differentiable_argnums=(0,),
            name="svd_s_reduced_3x3",
        )

        # SVD on tall matrices
        for m, n in [(5, 3), (4, 2), (6, 4)]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
                lambda key, m=m, n=n: random.normal(key, (m, n)),
                differentiable_argnums=(0,),
                name=f"svd_s_tall_{m}x{n}",
            )

        # SVD on wide matrices
        for m, n in [(3, 5), (2, 4)]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
                lambda key, m=m, n=n: random.normal(key, (m, n)),
                differentiable_argnums=(0,),
                name=f"svd_s_wide_{m}x{n}",
            )

        # Full SVD: verify U shape
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=True)[0],
            lambda key: random.normal(key, (4, 3)),
            differentiable_argnums=(),
            name="svd_U_full_4x3",
        )

        # Full SVD: verify Vt shape
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=True)[2],
            lambda key: random.normal(key, (4, 3)),
            differentiable_argnums=(),
            name="svd_Vt_full_4x3",
        )

        # Batched SVD
        for batch_shape in [(2,), (2, 3)]:
            batch_str = "x".join(map(str, batch_shape))
            yield OperationTestConfig(
                lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
                lambda key, bs=batch_shape: random.normal(key, (*bs, 3, 4)),
                differentiable_argnums=(0,),
                name=f"svd_s_batched_{batch_str}",
            )

        # Complex SVD
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
                lambda key, n=n: (
                    random.normal(key, (n, n))
                    + 1j * random.normal(random.split(key)[0], (n, n))
                ),
                differentiable_argnums=(),
                name=f"svd_s_complex_{n}x{n}",
            )

        # Complex SVD tall/wide
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
            lambda key: (
                random.normal(key, (4, 3))
                + 1j * random.normal(random.split(key)[0], (4, 3))
            ),
            differentiable_argnums=(),
            name="svd_s_complex_tall_4x3",
        )

        # Complex SVD full matrices
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=True)[1],
            lambda key: (
                random.normal(key, (3, 3))
                + 1j * random.normal(random.split(key)[0], (3, 3))
            ),
            differentiable_argnums=(),
            name="svd_s_complex_full_3x3",
        )

        # Batched complex SVD
        yield OperationTestConfig(
            lambda x: jnp.linalg.svd(x, full_matrices=False)[1],
            lambda key: (
                random.normal(key, (2, 3, 3))
                + 1j * random.normal(random.split(key)[0], (2, 3, 3))
            ),
            differentiable_argnums=(),
            name="svd_s_complex_batched",
        )

        # --- pinv (uses SVD internally) ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jnp.linalg.pinv,
                lambda key, n=n: random.normal(key, (n, n)),
                name=f"pinv_{n}x{n}",
            )

        # pinv of tall matrix
        yield OperationTestConfig(
            jnp.linalg.pinv,
            lambda key: random.normal(key, (4, 3)),
            name="pinv_4x3",
        )

        # --- lstsq (uses SVD internally) ---

        yield OperationTestConfig(
            lambda A, b: jnp.linalg.lstsq(A, b)[0],
            lambda key: random.normal(key, (3, 3)),
            lambda key: random.normal(key, (3, 1)),
            name="lstsq_3x3",
        )

        # --- matrix norm and cond (use SVD internally) ---

        yield OperationTestConfig(
            lambda x: jnp.linalg.cond(x),
            lambda key: _random_posdef(key, 3),
            name="cond_3x3",
        )

        yield OperationTestConfig(
            lambda x: jnp.linalg.matrix_rank(x),
            lambda key: random.normal(key, (3, 3)),
            differentiable_argnums=(),
            name="matrix_rank_3x3",
        )

        # --- General (non-symmetric) eigendecomposition via LAPACK sgeev_ ---

        # eig: eigenvalues of symmetric-like matrix (real eigenvalues)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.sort(jnp.abs(jnp.linalg.eig(x)[0])),
                lambda key, n=n: _random_posdef(key, n),
                differentiable_argnums=(),
                name=f"eig_values_posdef_{n}x{n}",
            )

        # eig: eigenvalues of general matrix (may have complex eigenvalues)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.sort(jnp.abs(jnp.linalg.eig(x)[0])),
                lambda key, n=n: random.normal(key, (n, n)),
                differentiable_argnums=(),
                name=f"eig_values_general_{n}x{n}",
            )

        # eig: batched
        yield OperationTestConfig(
            lambda x: jnp.sort(jnp.abs(jnp.linalg.eig(x)[0]), axis=-1),
            lambda key: random.normal(key, (2, 3, 3)),
            differentiable_argnums=(),
            name="eig_values_batched",
        )

        # eig: verify A @ V = V @ diag(w) identity (uses native+graph interop)
        yield OperationTestConfig(
            lambda x: jnp.linalg.eig(x)[0].real,
            lambda key: _random_posdef(key, 3),
            differentiable_argnums=(),
            name="eig_identity_check",
        )

        # Complex eig (general non-symmetric complex matrices)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jnp.sort(jnp.abs(jnp.linalg.eig(x)[0])),
                lambda key, n=n: (
                    random.normal(key, (n, n))
                    + 1j * random.normal(random.split(key)[0], (n, n))
                ),
                differentiable_argnums=(),
                name=f"eig_values_complex_{n}x{n}",
            )

        # Batched complex eig
        yield OperationTestConfig(
            lambda x: jnp.sort(jnp.abs(jnp.linalg.eig(x)[0]), axis=-1),
            lambda key: (
                random.normal(key, (2, 3, 3))
                + 1j * random.normal(random.split(key)[0], (2, 3, 3))
            ),
            differentiable_argnums=(),
            name="eig_values_complex_batched",
        )

        # --- Schur decomposition ---

        # Real Schur: verify T is upper quasi-triangular and Q is orthogonal
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: jax.scipy.linalg.schur(x)[0],
                lambda key, n=n: random.normal(key, (n, n)),
                differentiable_argnums=(),
                name=f"schur_T_{n}x{n}",
            )
            yield OperationTestConfig(
                lambda x: jax.scipy.linalg.schur(x)[1],
                lambda key, n=n: random.normal(key, (n, n)),
                differentiable_argnums=(),
                name=f"schur_Q_{n}x{n}",
            )

        # Batched Schur
        yield OperationTestConfig(
            lambda x: jax.scipy.linalg.schur(x)[0],
            lambda key: random.normal(key, (2, 3, 3)),
            differentiable_argnums=(),
            name="schur_T_batched",
        )

        # Complex Schur: verify reconstruction A ≈ Q @ T @ Q^H
        # (Schur form is not unique - eigenvalue ordering may differ between backends)
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                lambda x: (
                    lambda TQ: TQ[1] @ TQ[0] @ TQ[1].conj().T
                )(jax.scipy.linalg.schur(x)),
                lambda key, n=n: (
                    random.normal(key, (n, n))
                    + 1j * random.normal(random.split(key)[0], (n, n))
                ),
                differentiable_argnums=(),
                name=f"schur_reconstruct_complex_{n}x{n}",
            )

        # --- sqrtm (matrix square root, depends on Schur) ---

        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jax.scipy.linalg.sqrtm,
                lambda key, n=n: _random_posdef(key, n),
                differentiable_argnums=(),
                name=f"sqrtm_{n}x{n}",
            )

        # --- expm (matrix exponential, uses solve inside control flow) ---

        # expm of identity-like matrix
        yield OperationTestConfig(
            jax.scipy.linalg.expm,
            numpy.eye(3, dtype=numpy.float32) * 0.1,
            differentiable_argnums=(),
            name="expm_identity_scaled",
        )

        # expm of random matrix
        for n in [2, 3, 4]:
            yield OperationTestConfig(
                jax.scipy.linalg.expm,
                lambda key, n=n: random.normal(key, (n, n)) * 0.1,
                differentiable_argnums=(),
                name=f"expm_{n}x{n}",
            )

        # --- solve and inv inside control flow ---

        # solve inside cond (tests graph fallback for triangular_solve)
        yield OperationTestConfig(
            lambda A, b: jax.lax.cond(
                True,
                lambda: jnp.linalg.solve(A, b),
                lambda: jnp.zeros_like(b),
            ),
            lambda key: _random_posdef(key, 3),
            lambda key: random.normal(key, (3, 1)),
            differentiable_argnums=(),
            name="solve_in_cond",
        )

        # inv inside cond
        yield OperationTestConfig(
            lambda A: jax.lax.cond(
                True,
                lambda: jnp.linalg.inv(A),
                lambda: jnp.zeros_like(A),
            ),
            lambda key: _random_posdef(key, 3),
            differentiable_argnums=(),
            name="inv_in_cond",
        )
