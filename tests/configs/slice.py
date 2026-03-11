import jax
import numpy
from jax import lax, random
from jax import numpy as jnp

from .util import OperationTestConfig


def make_slice_op_configs():
    with OperationTestConfig.module_name("slice"):
        return [
            OperationTestConfig(
                lambda x, idx: x[idx],
                lambda key: random.normal(key, (4, 5)),
                lambda key: (
                    random.randint(random.split(key)[0], (), 0, 4),
                    random.randint(random.split(key)[1], (), 0, 5),
                ),
            ),
            OperationTestConfig(
                lambda x, idx, y: x[idx],
                lambda key: random.normal(key, (4, 5)),
                lambda key: (
                    random.randint(random.split(key)[0], (), 0, 4),
                    random.randint(random.split(key)[1], (), 0, 5),
                ),
                lambda key: numpy.asarray(7.0),
            ),
            OperationTestConfig(
                lambda x: lax.dynamic_slice(x, (2,), (4,)),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x, idx: jnp.take(x, idx, axis=0),
                lambda key: random.normal(key, (5, 3)),
                numpy.array([0, 2, 4]),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: random.normal(key, (5, 3)),
                numpy.array([0, 2]),
                lambda key: random.normal(key, (2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: jnp.zeros((10, 1, 4), dtype=jnp.float32),
                lambda key: numpy.int32(0),
                lambda key: jnp.ones((1, 4), dtype=jnp.float32),
                name="scalar_index_set_rank_squeezed_update",
            ),
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0].set(val),
                lambda key: jnp.zeros((2, 2, 2), dtype=jnp.float32),
                lambda key: jnp.array(3.14, dtype=jnp.float32),
                name="scalar_update_rank_mismatch_gt_1",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                lambda key: jnp.zeros((2, 2, 2), dtype=jnp.float32),
                lambda key: numpy.int32(0),
                lambda key: jnp.array(5.0, dtype=jnp.float32),
                name="slice_update_scalar_broadcast_rank3",
            ),
            # Full-index gather: x[i, j, k] on rank-3 tensor returns scalar
            OperationTestConfig(
                lambda x: x[1, 2, 0],
                lambda key: random.normal(key, (3, 4, 2)),
                name="full_index_gather_rank3",
            ),
            # ScatterND with add mode (not just set)
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0].add(val),
                lambda key: jnp.ones((2, 2, 2), dtype=jnp.float32),
                lambda key: jnp.array(5.0, dtype=jnp.float32),
                name="scatternd_add_mode",
            ),
            # Higher rank tensor (rank 4)
            OperationTestConfig(
                lambda x, val: x.at[0, 0, 0, 0].set(val),
                lambda key: jnp.zeros((2, 3, 4, 5), dtype=jnp.float32),
                lambda key: jnp.array(1.0, dtype=jnp.float32),
                name="full_index_scatter_rank4",
            ),
            # Non-zero indices
            OperationTestConfig(
                lambda x, val: x.at[1, 1, 1].set(val),
                lambda key: jnp.zeros((3, 3, 3), dtype=jnp.float32),
                lambda key: jnp.array(7.0, dtype=jnp.float32),
                name="full_index_scatter_nonzero",
            ),
            # Mixed index pattern
            OperationTestConfig(
                lambda x, val: x.at[2, 0, 1].set(val),
                lambda key: jnp.zeros((4, 3, 2), dtype=jnp.float32),
                lambda key: jnp.array(9.0, dtype=jnp.float32),
                name="full_index_scatter_mixed",
            ),
            # Partial-index scatter: x.at[i,j].add(1d_val) on 3D tensor
            # Regression test: scatter with N=1 and trailing window dims
            # differentiable_argnums=() because gradient generates scatter
            # patterns that crash MPS (hardware limitation)
            OperationTestConfig(
                lambda x, val: x.at[1, 2].add(val),
                lambda key: jnp.zeros((4, 4, 4), dtype=jnp.float32),
                lambda key: jnp.ones(4, dtype=jnp.float32),
                name="partial_index_scatter_add_3d",
                differentiable_argnums=(),
            ),
            # Partial-index scatter: x.at[i,j].set(1d_val) on 3D tensor
            OperationTestConfig(
                lambda x, val: x.at[1, 2].set(val),
                lambda key: jnp.zeros((4, 4, 4), dtype=jnp.float32),
                lambda key: jnp.ones(4, dtype=jnp.float32) * 3.0,
                name="partial_index_scatter_set_3d",
                differentiable_argnums=(),
            ),
            # Partial-index scatter on 4D: x.at[i,j].add(2d_val)
            OperationTestConfig(
                lambda x, val: x.at[1, 2].add(val),
                lambda key: jnp.zeros((3, 4, 5, 6), dtype=jnp.float32),
                lambda key: jnp.ones((5, 6), dtype=jnp.float32),
                name="partial_index_scatter_add_4d",
                differentiable_argnums=(),
            ),
            # Single partial index on 3D: x.at[i].add(2d_val)
            OperationTestConfig(
                lambda x, val: x.at[2].add(val),
                lambda key: jnp.zeros((4, 3, 5), dtype=jnp.float32),
                lambda key: jnp.ones((3, 5), dtype=jnp.float32),
                name="partial_index_scatter_add_3d_single",
                differentiable_argnums=(),
            ),
            # Window scatter (dynamic_update_slice-style): x.at[1:-1, 1:-1].set(val)
            # These generate scatter with insertedWindowDims=[] where the update
            # is a window placed at a start position given by the indices.
            OperationTestConfig(
                lambda x, val: x.at[1:-1, 1:-1].set(val),
                lambda key: jnp.zeros((5, 5), dtype=jnp.float32),
                lambda key: jnp.ones((3, 3), dtype=jnp.float32),
                name="window_scatter_set_2d",
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x, val: x.at[1:-1, 1:-1].add(val),
                lambda key: jnp.ones((5, 5), dtype=jnp.float32),
                lambda key: jnp.ones((3, 3), dtype=jnp.float32) * 2.0,
                name="window_scatter_add_2d",
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x, val: x.at[1:-1, 1:-1, 1:-1].set(val),
                lambda key: jnp.zeros((6, 6, 6), dtype=jnp.float32),
                lambda key: jnp.ones((4, 4, 4), dtype=jnp.float32) * 3.0,
                name="window_scatter_set_3d",
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x, val: x.at[1:4, :].set(val),
                lambda key: jnp.zeros((5, 8), dtype=jnp.float32),
                lambda key: jnp.ones((3, 8), dtype=jnp.float32) * 5.0,
                name="window_scatter_partial_dim",
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lambda x: x.at[0].set(1.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].add(1.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x: x.at[0].divide(2.0),
                lambda key: random.normal(key, (10,)),
            ),
            OperationTestConfig(
                lambda x, update: lax.dynamic_update_slice(x, update, (1, 0)),
                lambda key: random.normal(key, (5, 3)),
                lambda key: random.normal(key, (2, 3)),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].add(updates),
                numpy.zeros((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.ones((3, 4), dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].subtract(updates),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 0.1, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].mul(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].divide(updates, unique_indices=True),
                numpy.ones((10, 4), dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 2.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].power(updates, unique_indices=True),
                numpy.full((10, 4), 2.0, dtype=numpy.float32),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                numpy.full((3, 4), 3.0, dtype=numpy.float32),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].min(updates),
                lambda key: random.normal(key, (10, 4)),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda key: random.normal(key, (3, 4)),
            ),
            OperationTestConfig(
                lambda x, idx, updates: x.at[idx].max(updates),
                lambda key: random.normal(key, (10, 4)),
                numpy.array([0, 2, 5], dtype=numpy.int32),
                lambda key: random.normal(key, (3, 4)),
            ),
            # Multi-index point gather: x[arange(n), arange(n)] extracts diagonal
            OperationTestConfig(
                lambda x: x[jnp.arange(4), jnp.arange(4)],
                lambda key: random.normal(key, (4, 4)),
                name="multi_index_point_gather",
            ),
            # Multi-dim indexing with offset: x[i, j] on rank-3 tensor
            # Produces gather with collapsed_slice_dims=(0,1), offset_dims=(1,)
            OperationTestConfig(
                lambda x: x[jnp.array([0, 1, 2]), jnp.array([1, 2, 0])],
                lambda key: random.normal(key, (5, 4, 3)),
                name="multi_dim_indexing_with_offset",
            ),
            # Diagonal gather with offset dims: jnp.diagonal on batched tensor
            OperationTestConfig(
                lambda x: jnp.diagonal(x, axis1=-1, axis2=-2),
                lambda key: random.normal(key, (3, 4, 4)),
                name="diagonal_gather_batched",
            ),
            # Diagonal gather on 2D tensor
            OperationTestConfig(
                lambda x: jnp.diagonal(x),
                lambda key: random.normal(key, (5, 5)),
                name="diagonal_gather_2d",
            ),
            # Batched gather via vmap
            OperationTestConfig(
                lambda x, idx: jax.vmap(
                    lambda xi, ii: lax.dynamic_index_in_dim(xi, ii, 0, False)
                )(x, idx),
                lambda key: random.normal(key, (3, 10)),
                lambda key: random.randint(key, (3,), 0, 10),
                name="batched_single_axis_gather",
            ),
            # Large integer gather tests: verify integers > 2^24 are preserved
            # These test the bitcast workaround for MPS gather operations
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([16777217, 2**30, 2**31 - 1], dtype=numpy.uint32),
                numpy.int32(1),
                name="large_uint32_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([16777217, 2**30, 2**31 - 1], dtype=numpy.int32),
                numpy.int32(0),
                name="large_int32_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([2**40, 2**50, 2**62], dtype=numpy.uint64),
                numpy.int32(1),
                name="large_uint64_gather",
            ),
            OperationTestConfig(
                lambda x, idx: x[idx],
                numpy.array([2**40, 2**50, 2**62], dtype=numpy.int64),
                numpy.int32(2),
                name="large_int64_gather",
            ),
            # Large integer scatter tests: verify integers > 2^24 are preserved in scatter
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.uint32),
                numpy.int32(2),
                numpy.uint32(16777217),
                name="large_uint32_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.int32),
                numpy.int32(1),
                numpy.int32(2**30),
                name="large_int32_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.uint64),
                numpy.int32(3),
                numpy.uint64(2**50),
                name="large_uint64_scatter",
            ),
            OperationTestConfig(
                lambda x, idx, val: x.at[idx].set(val),
                numpy.zeros((5,), dtype=numpy.int64),
                numpy.int32(0),
                numpy.int64(2**62),
                name="large_int64_scatter",
            ),
            # Integer scatter-add regression test: verifies int32 scatter-add
            # produces correct results (not zeros from bitcast float arithmetic)
            OperationTestConfig(
                lambda x, idx: x.at[idx].add(1),
                numpy.zeros(5, dtype=numpy.int32),
                numpy.array([0, 2, 4], dtype=numpy.int32),
                name="int32_scatter_add",
            ),
            # Large integer scatter-add: verify values > 2^24 aren't corrupted
            OperationTestConfig(
                lambda x, idx: x.at[idx].add(numpy.int32(16777217)),
                numpy.zeros(5, dtype=numpy.int32),
                numpy.int32(2),
                name="large_int32_scatter_add",
            ),
            # Multi-dim index scatter: scatter values along diagonal of a matrix
            # Exercises general ScatterND with multi-point, multi-dim index vectors
            # indices shape [N, K] where N=4 scatter points, K=2 index dims
            OperationTestConfig(
                lambda x, vals: x.at[numpy.arange(4), numpy.arange(4)].add(vals),
                lambda key: jnp.zeros((4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (4,)),
                differentiable_argnums=(0,),
                name="scatter_multi_dim_diagonal_add",
            ),
            # Grad of multi-dim scatter with respect to updates
            OperationTestConfig(
                lambda x, vals: x.at[numpy.arange(4), numpy.arange(4)].add(vals),
                lambda key: jnp.zeros((4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (4,)),
                differentiable_argnums=(1,),
                name="scatter_multi_dim_diagonal_add_grad_updates",
            ),
            # Batched scatter using vmap - tests numStableHLOBatch > 0
            OperationTestConfig(
                lambda x, idx, val: jax.vmap(lambda a, i, v: a.at[i].set(v))(
                    x, idx, val
                ),
                lambda key: random.normal(key, (3, 5)),
                lambda key: random.randint(key, (3,), 0, 5),
                lambda key: random.normal(key, (3,)),
                differentiable_argnums=(0, 2),
                name="scatter_vmap_simple",
            ),
            OperationTestConfig(
                lambda x, idx, val: jax.vmap(lambda a, i, v: a.at[i].add(v))(
                    x, idx, val
                ),
                lambda key: jnp.zeros((3, 5), dtype=jnp.float32),
                lambda key: jnp.array([[0, 2], [1, 3], [2, 4]]),
                lambda key: random.normal(key, (3, 2)),
                differentiable_argnums=(0, 2),
                name="scatter_vmap_multi_point",
            ),
            OperationTestConfig(
                lambda x, vals: jax.vmap(
                    lambda a, v: a.at[numpy.arange(2), numpy.arange(2)].add(v)
                )(x, vals),
                lambda key: jnp.zeros((3, 4, 4), dtype=jnp.float32),
                lambda key: random.normal(key, (3, 2)),
                differentiable_argnums=(0,),
                name="scatter_vmap_2d_diagonal",
            ),
            # Single-index slice gather: x[:, :-1] via gather (stablehlo.gather with
            # single start index, no collapsed dims, full-size offset dims)
            OperationTestConfig(
                lambda x: x[:, :-1],
                lambda key: random.normal(key, (4, 16)),
                name="slice_gather_trailing",
            ),
            # Slice gather with reshape: x[:, :-1, None]
            OperationTestConfig(
                lambda x: x[:, :-1, None],
                lambda key: random.normal(key, (4, 16)),
                differentiable_argnums=(),
                name="slice_gather_unsqueeze",
            ),
            # Embedding lookup via vmap (slice-gather pattern):
            # operand [V, D], indices [N, 2], offset_dims=[1,2], collapsed=[]
            OperationTestConfig(
                lambda w, idx: jax.vmap(lambda i: w[i])(idx),
                lambda key: random.normal(key, (10, 4)),
                lambda key: random.randint(key, (3,), 0, 10),
                name="vmap_embedding_gather",
            ),
            # Embedding gradient via vmap (slice-scatter pattern):
            # scatter-add with multi-index, no inserted_window_dims
            OperationTestConfig(
                lambda w, idx: jnp.sum(jax.vmap(lambda i: w[i])(idx) ** 2),
                lambda key: random.normal(key, (10, 4)),
                lambda key: random.randint(key, (3,), 0, 10),
                differentiable_argnums=(0,),
                name="vmap_embedding_grad",
            ),
            # Direct embedding gradient: embed[tokens] with 2D token indices
            # Exercises batched scatter-add where updates rank > operand rank
            OperationTestConfig(
                lambda w, idx: jnp.sum(w[idx] ** 2),
                lambda key: random.normal(key, (50, 8)),
                lambda key: random.randint(key, (4, 3), 0, 50),
                differentiable_argnums=(0,),
                name="embedding_grad_2d_indices",
            ),
            # Batched embedding via double vmap
            OperationTestConfig(
                lambda w, idx: jax.vmap(jax.vmap(lambda i: w[i]))(idx),
                lambda key: random.normal(key, (100, 8)),
                lambda key: random.randint(key, (4, 16), 0, 100),
                name="double_vmap_embedding_gather",
            ),
            # Batched embedding gradient via double vmap
            OperationTestConfig(
                lambda w, idx: jnp.sum(jax.vmap(jax.vmap(lambda i: w[i]))(idx) ** 2),
                lambda key: random.normal(key, (100, 8)),
                lambda key: random.randint(key, (4, 16), 0, 100),
                differentiable_argnums=(0,),
                name="double_vmap_embedding_grad",
            ),
            # Uncollapsed point gather (searchsorted pattern):
            # operand [5], indices [6,1], offset_dims=[1], collapsed=[], slice_sizes=[1]
            OperationTestConfig(
                lambda x, vals: jnp.searchsorted(x, vals),
                numpy.array([1.0, 3.0, 5.0, 7.0, 9.0], dtype=numpy.float32),
                numpy.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=numpy.float32),
                differentiable_argnums=(),
                name="searchsorted_basic",
            ),
            # Gradient of last-token cross-entropy (dynamic-update-slice scatter):
            # Tests scatter with update_window_dims covering all dims, N=1 scatter point
            OperationTestConfig(
                lambda logits, labels: jnp.mean(
                    jax.vmap(lambda l, lab: -jax.nn.log_softmax(l[-1])[lab])(
                        logits, labels
                    )
                ),
                lambda key: random.normal(key, (4, 3, 20)),
                lambda key: random.randint(key, (4,), 0, 20),
                differentiable_argnums=(0,),
                name="last_token_ce_grad",
            ),
            # Boolean point scatter: .at[idx].set(value) for bool arrays
            OperationTestConfig(
                lambda x: x.at[2].set(False),
                numpy.array([True, False, True, False, True]),
                differentiable_argnums=(),
                name="bool_scatter_point",
            ),
            # Boolean slice scatter: .at[1:3].set(value) for bool arrays
            OperationTestConfig(
                lambda x: x.at[1:3].set(False),
                numpy.array([True, True, True, True, True]),
                differentiable_argnums=(),
                name="bool_scatter_slice",
            ),
            # Slice scatter (DUS pattern): .at[start:end].set(value)
            OperationTestConfig(
                lambda x: x.at[1:3].set(0),
                numpy.array([1, 1, 1, 1, 1], dtype=numpy.int32),
                differentiable_argnums=(),
                name="int_scatter_slice",
            ),
            # Float slice scatter
            OperationTestConfig(
                lambda x: x.at[2:7].set(0.0),
                lambda key: random.normal(key, (10,)),
                name="float_scatter_slice",
            ),
            # 2D slice scatter
            OperationTestConfig(
                lambda x: x.at[1:3, :].set(0.0),
                lambda key: random.normal(key, (4, 4)),
                name="float_scatter_slice_2d",
            ),
            # unique (depends on boolean scatter internally)
            OperationTestConfig(
                lambda x: jnp.unique(x, size=4, fill_value=0),
                numpy.array([3, 1, 2, 1, 3, 2, 4], dtype=numpy.int32),
                differentiable_argnums=(),
                name="unique_basic",
            ),
            # Non-contiguous scatter: x.at[h, :, s].set(val) on 3D tensor
            # Produces scatter with scatterDimsToOperandDims=(0, 2) (non-contiguous)
            OperationTestConfig(
                lambda x: x.at[1, :, 2].set(0.0),
                lambda key: random.normal(key, (4, 6, 6)),
                differentiable_argnums=(),
                name="scatter_noncontig_dims",
            ),
            # Slice-scatter: x.at[i, :n].set(val) on 2D tensor
            # Produces scatter with insertedWindowDims=(0,) and K=2
            OperationTestConfig(
                lambda x, val: x.at[0, :3].set(val),
                lambda key: random.normal(key, (6, 6)),
                lambda key: random.normal(key, (3,)),
                differentiable_argnums=(),
                name="slice_scatter_2d",
            ),
            # Non-contiguous gather: x[oob, :, i] on 3D tensor
            # Out-of-bounds index triggers gather with non-contiguous indexed dims
            OperationTestConfig(
                lambda x: x[5, :, 1],
                lambda key: random.normal(key, (4, 6, 6)),
                differentiable_argnums=(),
                name="gather_noncontig_dims",
            ),
            # Multi-dimensional index arrays producing higher-rank output:
            # x[:, y_idx, x_idx] where y_idx and x_idx have shape (5, 3)
            # Result shape (3, 5, 3) has higher rank than source (3, 10, 10)
            OperationTestConfig(
                lambda x: x[
                    :,
                    jnp.arange(5)[:, None] + jnp.arange(3)[None, :],
                    jnp.arange(5)[:, None] + jnp.arange(3)[None, :],
                ],
                lambda key: random.normal(key, (3, 10, 10)),
                name="gather_multidim_index_arrays",
            ),
        ]
