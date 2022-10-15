# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest  # noqa: F401
import torch
import numpy as np

import theseus as th

from theseus.utils import random_sparse_binary_matrix, split_into_param_sizes

try:
    import theseus.extlib.baspacho_solver  # noqa: F401

    BASPACHO_EXT_NOT_AVAILABLE = False
except ModuleNotFoundError:
    BASPACHO_EXT_NOT_AVAILABLE = True

requires_baspacho = pytest.mark.skipif(
    BASPACHO_EXT_NOT_AVAILABLE,
    reason="Baspacho solver not in theseus extension library",
)


def _build_sparse_mat(batch_size):
    all_cols = list(range(10))
    col_ind = []
    row_ptr = [0]
    for i in range(12):
        start = max(0, i - 2)
        end = min(i + 1, 10)
        col_ind += all_cols[start:end]
        row_ptr.append(len(col_ind))
    data = torch.randn((batch_size, len(col_ind)), dtype=torch.double)
    return 12, 10, data, col_ind, row_ptr


def check_sparse_solver(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill, dev="cpu"
):
    torch.manual_seed(hash(str([batch_size, rows_to_cols_ratio, num_cols, fill])))

    # this is necessary assumption, so that the hessian can be full rank. actually we
    # add some damping to At*A's diagonal, so not really necessary
    assert rows_to_cols_ratio >= 1.0
    num_rows = round(rows_to_cols_ratio * num_cols)

    if isinstance(param_size_range, str):
        param_size_range = [int(x) for x in param_size_range.split(":")]

    void_objective = th.Objective()
    void_ordering = th.VariableOrdering(void_objective, default_order=False)
    solver = th.BaspachoSparseSolver(
        void_objective,
        linearization_kwargs={"ordering": void_ordering},
    )
    linearization = solver.linearization

    A_skel = random_sparse_binary_matrix(
        num_rows, num_cols, fill, min_entries_per_col=1
    )
    void_objective._batch_size = batch_size
    num_rows, num_cols = A_skel.shape
    linearization.num_rows = num_rows
    linearization.num_cols = num_cols
    linearization.A_col_ind = A_skel.indices
    linearization.A_row_ptr = A_skel.indptr
    linearization.A_val = torch.rand((batch_size, A_skel.nnz), dtype=torch.double).to(
        dev
    )
    linearization.b = torch.randn((batch_size, num_rows), dtype=torch.double).to(dev)

    # also need: var dims and var_start_cols (because baspacho is blockwise)
    linearization.var_dims = split_into_param_sizes(num_cols, *param_size_range)
    linearization.var_start_cols = np.cumsum([0, *linearization.var_dims[:-1]])

    # Only need this line for the test since the objective is a mock
    solver.reset(dev=dev)

    damping = 1e-4
    solved_x = solver.solve(damping=damping, ellipsoidal_damping=False)

    for i in range(batch_size):
        csrAi = linearization.structure().csr_straight(linearization.A_val[i, :].cpu())
        Ai = torch.tensor(csrAi.todense(), dtype=torch.double)
        ata = Ai.T @ Ai
        b = linearization.b[i].cpu()
        atb = torch.Tensor(csrAi.transpose() @ b)

        # the linear system solved is with matrix AtA
        solved_xi_cpu = solved_x[i].cpu()
        atb_check = ata @ solved_xi_cpu + damping * solved_xi_cpu

        max_offset = torch.norm(atb - atb_check, p=float("inf"))
        assert max_offset < 1e-4


@requires_baspacho
@pytest.mark.baspacho
@pytest.mark.parametrize("batch_size", [2, 8, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13", "3:9"])
@pytest.mark.parametrize("fill", [0.02, 0.05, 0.1])
def test_baspacho_solver_cpu(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_solver(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cpu",
    )


@requires_baspacho
@pytest.mark.cudaext
@pytest.mark.baspacho
@pytest.mark.parametrize("batch_size", [2, 8, 32])
@pytest.mark.parametrize("rows_to_cols_ratio", [1.1, 1.7])
@pytest.mark.parametrize("num_cols", [30, 70])
@pytest.mark.parametrize("param_size_range", ["2:6", "1:13", "3:9"])
@pytest.mark.parametrize("fill", [0.02, 0.05, 0.1])
def test_baspacho_solver_cuda(
    batch_size, rows_to_cols_ratio, num_cols, param_size_range, fill
):
    check_sparse_solver(
        batch_size=batch_size,
        rows_to_cols_ratio=rows_to_cols_ratio,
        num_cols=num_cols,
        param_size_range=param_size_range,
        fill=fill,
        dev="cuda",
    )
