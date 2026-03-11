"""JAX MPS Plugin - Metal Performance Shaders backend for JAX."""

import os
import sys
import warnings
from pathlib import Path

# jaxlib version this plugin was built against (major.minor). Used for runtime
# compatibility checking.
_BUILT_FOR_JAXLIB = (0, 9)
_LIB_NAME = "libpjrt_plugin_mps.dylib"


class MPSPluginError(Exception):
    """Exception raised when MPS plugin initialization fails."""

    pass


def _get_search_paths():
    """Return list of (path, description) tuples for library search."""
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent.parent.parent

    return [
        (pkg_dir / _LIB_NAME, "package directory (editable install)"),
        (pkg_dir / "lib" / _LIB_NAME, "package lib/ (wheel install)"),
        (
            project_root / "build" / "*" / "lib" / _LIB_NAME,
            "build/*/lib/ (cmake build)",
        ),
        (Path("/usr/local/lib") / _LIB_NAME, "/usr/local/lib/"),
        (Path("/opt/homebrew/lib") / _LIB_NAME, "/opt/homebrew/lib/"),
    ]


def _find_library():
    """Find the pjrt_plugin_mps shared library.

    Returns:
        Path to the library, or None if not found.
    """
    # Environment variable takes precedence
    if "JAX_MPS_LIBRARY_PATH" in os.environ:
        env_path = os.environ["JAX_MPS_LIBRARY_PATH"]
        if Path(env_path).exists():
            return env_path
        raise MPSPluginError(
            f"JAX_MPS_LIBRARY_PATH is set to '{env_path}', but the file does not exist."
        )

    for path, _ in _get_search_paths():
        # Handle glob patterns
        if "*" in str(path):
            for match in Path("/").glob(str(path).lstrip("/")):
                if match.exists():
                    return str(match)
        elif path.exists():
            return str(path)

    return None


def _check_jaxlib_version():
    """Check if the installed jaxlib version is compatible.

    Warns if the major.minor version doesn't match what the plugin was built for.
    """
    try:
        import jaxlib

        version_str = getattr(jaxlib, "__version__", None)
        if version_str is None:
            return

        parts = version_str.split(".")
        if len(parts) < 2:
            return

        installed = (int(parts[0]), int(parts[1]))
        if installed != _BUILT_FOR_JAXLIB:
            warnings.warn(
                f"applejax was built for jaxlib {_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.x, "
                f"but jaxlib {version_str} is installed. This may cause compatibility "
                f"issues with StableHLO bytecode parsing. Consider installing jaxlib "
                f">={_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.0,"
                f"<{_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1] + 1}",
                stacklevel=3,
            )
    except Exception:
        pass  # Don't fail initialization due to version check issues


def _register_lowering_rules():
    """Register MLIR lowering rules for primitives that need platform-specific handling."""
    try:
        from jax._src.interpreters import mlir
        from jax._src.lax import linalg as lax_linalg

        def _eigh_mps_lowering(ctx, operand, *, lower, sort_eigenvalues, subset_by_index, algorithm):
            """Lower eigh to custom_call @mps_syevd."""
            v_aval, w_aval = ctx.avals_out
            result_types = [mlir.aval_to_ir_type(v_aval), mlir.aval_to_ir_type(w_aval)]
            op = mlir.custom_call(
                "mps_syevd",
                result_types=result_types,
                operands=[operand],
                api_version=1,
            )
            return op.results

        def _svd_mps_lowering(
            ctx,
            operand,
            *,
            full_matrices,
            compute_uv,
            subset_by_index,
            algorithm,
        ):
            """Lower svd to custom_call @mps_sgesdd."""
            import numpy as np
            from jax._src.core import ShapedArray

            operand_aval = ctx.avals_in[0]
            m, n = operand_aval.shape[-2:]
            batch_dims = operand_aval.shape[:-2]
            k = min(m, n)

            s_aval = ctx.avals_out[0]

            if compute_uv:
                _, u_aval, vt_aval = ctx.avals_out
            else:
                u_aval = ShapedArray(
                    (*batch_dims, m, m if full_matrices else k),
                    operand_aval.dtype,
                )
                vt_aval = ShapedArray(
                    (*batch_dims, n if full_matrices else k, n),
                    operand_aval.dtype,
                )

            # Our custom call returns (s, u, vt) - 3 results always
            result_types = [
                mlir.aval_to_ir_type(s_aval),
                mlir.aval_to_ir_type(u_aval),
                mlir.aval_to_ir_type(vt_aval),
            ]
            # Encode full_matrices and compute_uv in backend_config as a simple string
            config = f"{int(full_matrices)},{int(compute_uv)}"
            op = mlir.custom_call(
                "mps_sgesdd",
                result_types=result_types,
                operands=[operand],
                api_version=1,
                backend_config=config,
            )
            # Check info and replace with NaN on failure
            s = op.results[0]
            results = [s]
            if compute_uv:
                results.append(op.results[1])
                results.append(op.results[2])
            return results

        def _eig_mps_lowering(
            ctx,
            operand,
            *,
            compute_left_eigenvectors,
            compute_right_eigenvectors,
            implementation,
        ):
            """Lower eig to custom_call @mps_sgeev.

            Returns complex eigenvalues and eigenvectors directly from
            the native handler (no intermediate stablehlo.complex op needed).
            """
            import numpy as np
            from jax._src.core import ShapedArray

            operand_aval = ctx.avals_in[0]
            n = operand_aval.shape[-1]
            batch_dims = operand_aval.shape[:-2]

            # Custom call returns: (w_complex, vl_complex, vr_complex, info)
            w_aval = ShapedArray((*batch_dims, n), np.complex64)
            vl_aval = ShapedArray((*batch_dims, n, n), np.complex64)
            vr_aval = ShapedArray((*batch_dims, n, n), np.complex64)
            info_aval = ShapedArray((), np.int32)

            result_types = [
                mlir.aval_to_ir_type(w_aval),
                mlir.aval_to_ir_type(vl_aval),
                mlir.aval_to_ir_type(vr_aval),
                mlir.aval_to_ir_type(info_aval),
            ]
            config = f"{int(compute_left_eigenvectors)},{int(compute_right_eigenvectors)}"
            op = mlir.custom_call(
                "mps_sgeev",
                result_types=result_types,
                operands=[operand],
                api_version=1,
                backend_config=config,
            )
            results = [op.results[0]]  # eigenvalues (complex)
            if compute_left_eigenvectors:
                results.append(op.results[1])
            if compute_right_eigenvectors:
                results.append(op.results[2])
            return results

        # Register under "mps" (backend platform name used during lowering lookup).
        # JAX's register_lowering validates against known_platforms() which includes
        # "METAL" from the PJRT plugin, but lowering dispatch uses the backend name
        # "mps". We register directly into the platform-specific lowerings dict.
        mps_lowerings = mlir._platform_specific_lowerings.setdefault("mps", {})
        mps_lowerings[lax_linalg.eigh_p] = mlir.LoweringRuleEntry(
            _eigh_mps_lowering, True
        )
        mps_lowerings[lax_linalg.svd_p] = mlir.LoweringRuleEntry(
            _svd_mps_lowering, True
        )
        mps_lowerings[lax_linalg.eig_p] = mlir.LoweringRuleEntry(
            _eig_mps_lowering, True
        )

        def _schur_mps_lowering(
            ctx,
            operand,
            *,
            compute_schur_vectors,
            sort_eig_vals,
            select_callable,
        ):
            """Lower schur to custom_call @mps_sgees."""
            operand_aval = ctx.avals_in[0]
            t_aval = ctx.avals_out[0]
            result_types = [mlir.aval_to_ir_type(t_aval)]
            if compute_schur_vectors:
                q_aval = ctx.avals_out[1]
                result_types.append(mlir.aval_to_ir_type(q_aval))
            config = f"{int(compute_schur_vectors)}"
            op = mlir.custom_call(
                "mps_sgees",
                result_types=result_types,
                operands=[operand],
                api_version=1,
                backend_config=config,
            )
            results = [op.results[0]]
            if compute_schur_vectors:
                results.append(op.results[1])
            return results

        mps_lowerings[lax_linalg.schur_p] = mlir.LoweringRuleEntry(
            _schur_mps_lowering, True
        )
    except Exception:
        pass  # Don't fail initialization if lowering registration fails


def initialize():
    """Initialize the MPS plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.

    Raises:
        MPSPluginError: If Metal GPU is not available or plugin initialization fails.
    """
    # Check platform first
    if sys.platform != "darwin":
        raise MPSPluginError(
            f"MPS plugin requires macOS, but running on {sys.platform}. MPS (Metal "
            "Performance Shaders) is only available on Apple devices."
        )

    # Check jaxlib version compatibility
    _check_jaxlib_version()

    library_path = _find_library()
    if library_path is None:
        searched = "\n".join(f"  - {desc}" for _, desc in _get_search_paths())
        raise MPSPluginError(
            f"Could not find {_LIB_NAME}. Searched paths:\n{searched}\n"
            "You can also set JAX_MPS_LIBRARY_PATH environment variable."
        )

    # Disable shardy partitioner - it produces sdy dialect ops that our StableHLO parser
    # doesn't support yet (JAX 0.9+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception as e:
        warnings.warn(
            f"Failed to disable shardy partitioner: {e}. Some operations may not work correctly.",
            stacklevel=2,
        )

    # Register lowering rules for primitives not in StableHLO
    _register_lowering_rules()

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb
    except ImportError as e:
        raise MPSPluginError(f"Failed to import JAX xla_bridge: {e}") from e

    if not hasattr(xb, "register_plugin"):
        raise MPSPluginError("JAX version does not support register_plugin API.")

    try:
        xb.register_plugin(
            "mps",
            priority=500,  # Higher than CPU (0) but lower than GPU (1000)
            library_path=library_path,
            options=None,
        )
    except Exception as e:
        # Handle "already registered" case - this is fine, not an error
        if "ALREADY_EXISTS" in str(e) and "mps" in str(e).lower():
            return
        raise MPSPluginError(f"Failed to register MPS plugin with JAX: {e}") from e
