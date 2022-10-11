import numpy as np
import torch

import theseus as th
from theseus.utils import numeric_jacobian

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rng = torch.Generator(device=device)
rng.manual_seed(0)

log = []

np.set_printoptions(precision=3)


# ---------------------------------
# Basic Lie Groups usage
# ---------------------------------
batch_size = 10
v1 = torch.randn(batch_size, 3)
v2 = torch.randn_like(v1)

# Running operators
x = th.SO3.exp_map(v1).log_map()
y = th.SO3.exp_map(v1).compose(th.SO3.exp_map(v2))

# If need jacobians for optimization,
# need to either use AutodiffCostFunction as illustrated below
# or compose analytic jacobians, for example
Jlog = []
Jexp = []
x = th.SO3.exp_map(v1, jacobians=Jexp).log_map(jacobians=Jlog)
J = Jlog[0].matmul(Jexp[0])

# Side note: our jacobians follow the right trivialized
# tangent deriviative and its inverse.
# See https://asco.lcsr.jhu.edu/papers/KoMa2010.pdf

# Note that by default our operations run checks to see
# that new group objects satisfy det(R)=1, R^T@R=I. These
# are time consuming and can be turned off as follows
from theseus.geometry.lie_group_check import no_lie_group_check

with no_lie_group_check():
    x = th.SO3.exp_map(v1)

# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #
# BELOW THIS POINT IS THE ORIGINAL SCRIPT WE SENT
# -------------------------------------------------------------------- #
# -------------------------------------------------------------------- #


def test_theseus_jacobian(batch_size, method="analytic"):
    torch.cuda.empty_cache()
    try:  # this fails for CPU, just ignore
        torch.cuda.get_device_capability(device)
    except Exception:
        pass

    rot = th.SO3.rand(batch_size, device=device, dtype=torch.float32)
    trans = th.Point3.rand(batch_size, device=device, dtype=torch.float32)
    # ------ original ------
    # input = th.Vector(
    #     tensor=torch.rand(1, batch_size * 3)
    # )  # to fit theseus jacobian input requirement
    # ----------------------
    # If the idea is to do R * x + T for a batch of B elements,
    # then x must be shape (B, 3) instead of shape (1, 3B)
    input = th.Point3.rand(batch_size, device=device, dtype=torch.float32)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    if method == "numeric":

        def func(vars):
            # ------ original ------
            # tmp_tensor = torch.reshape(vars[0].tensor, (batch_size, 3))
            # return rot.rotate(tmp_tensor) + trans
            # ----------------------
            # No need to reshape since batch size is the same for all elements
            return rot.rotate(vars[0]) + trans

        # No need to modify numeric jacobian with the aligned batch sizes
        # Original version had batch size B for R and T, but batch size 1 for x
        J = numeric_jacobian(func, [input])[0].to(device)
    elif method == "analytic":
        Jrot = []  # will store drot/dR and drot/dx
        y = rot.rotate(input, jacobians=Jrot) + trans
        J = Jrot[1]
    elif "autograd" in method:

        def cost_fn(optim_vars, aux_vars):
            x_ = optim_vars[0]
            rot_, trans_ = aux_vars
            return (rot_.rotate(x_) + trans_).tensor

        autograd_mode = "dense" if "dense" in method else "vmap"
        cost = th.AutoDiffCostFunction(
            [input],
            cost_fn,
            3,
            aux_vars=[rot, trans],
            autograd_mode=autograd_mode,
            autograd_vectorize=True,
            autograd_strict=False,
        )
        J = cost.jacobians()[0][0]  # jacobians returns ([jac], cost)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event)


if __name__ == "__main__":
    # Can make batch size even larger (100,000 akes <400MBs of GPU memory)
    batch_sizes = [1, 10, 100, 1000, 9000, 100000]

    for method in ["numeric", "analytic", "autograd_vmap", "autograd_dense"]:
        log = []
        for i in batch_sizes:
            log.append([])

        itrs = 100

        for itr in range(itrs):
            for i in range(len(batch_sizes)):
                if method == "autograd_dense" and batch_sizes[i] > 10000:
                    log[i].append(-1)
                else:
                    log[i].append(test_theseus_jacobian(batch_sizes[i], method=method))

        for i in log:
            i.sort()

        np_log = np.array(log, dtype=np.float64)
        # remove top 20% and last 20%, to filt noise
        np_log = np_log[:, int(itrs * 0.2) : int(itrs * 0.8)]

        mean = np.mean(np_log, axis=1)
        std = np.std(np_log, axis=1)

        print("----------------------------------------------")
        jac_type = "Dense" if "dense" in method else "Sparse"
        print(f"{jac_type} jacobian calculation method: {method}")
        if method == "autograd_dense":
            print(f"(Up to batch size {batch_sizes[-2]})")
        print(f"Mean = {mean}")
        print(f"Std = {std}")
        print("----------------------------------------------")
