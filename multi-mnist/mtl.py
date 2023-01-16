import random
import numpy as np
import copy
from typing import List, Tuple
from scipy.optimize import minimize
import torch


def PCGrad(grads: List[Tuple[torch.Tensor]], reduction: str = "sum") -> torch.Tensor:
    pc_grad = copy.deepcopy(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = sum(
                [
                    torch.dot(torch.flatten(grad_i), torch.flatten(grad_j))
                    for grad_i, grad_j in zip(g_i, g_j)
                ]
            )
            if g_i_g_j < 0:
                g_j_norm_square = (
                    torch.norm(torch.cat([torch.flatten(g) for g in g_j])) ** 2
                )
                for grad_i, grad_j in zip(g_i, g_j):
                    grad_i -= g_i_g_j * grad_j / g_j_norm_square

    merged_grad = [sum(g) for g in zip(*pc_grad)]
    if reduction == "mean":
        merged_grad = [g / len(grads) for g in merged_grad]

    return merged_grad

def CAGrad(grads, alpha=0.5, rescale=1):
    n_tasks = len(grads)
    grads =  grads.t()
    
    GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
    # GG = torch.zeros((n_tasks, n_tasks))
    # for i in range(n_tasks):
    #     for j in range(n_tasks):
    #         GG[i][j] = torch.dot(grads[i], grads[j]).cpu()
    g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

    x_start = np.ones(n_tasks) / n_tasks
    bnds = tuple((0, 1) for x in x_start)
    cons = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    A = GG.numpy()
    b = x_start.copy()
    c = (alpha * g0_norm + 1e-8).item()

    def objfn(x):
        return (
            x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1))
            + c
            * np.sqrt(
                x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1))
                + 1e-8
            )
        ).sum()

    res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
    w_cpu = res.x
    ww = torch.Tensor(w_cpu).to(grads.device)
    gw = (grads * ww.view(1, -1)).sum(1)
    gw_norm = gw.norm()
    lmbda = c / (gw_norm + 1e-8)
    g = grads.mean(1) + lmbda * gw
    if rescale == 0:
        return g
    elif rescale == 1:
        return g / (1 + alpha**2)
    else:
        return g / (1 + alpha)