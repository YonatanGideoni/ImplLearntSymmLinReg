from itertools import permutations

import torch
from torch import nn

from action_matching_analysis.repr_matching_utils import get_kernelised_elem
from symm_data_gen import PermXYSymmGenerator, SymmDataGenerator
from utils import MLP


def fill_grad_mat(model, data, grad_mat):
    for i, x in enumerate(data):
        model.zero_grad()
        res = model(x.reshape(1, -1))
        res.backward()

        grad_mat[i] = torch.cat([p.grad.flatten() for p in model.parameters()])


def get_grad_mat(model: nn.Module, data_gen: SymmDataGenerator, sym: callable, overparam_fctr: int) -> tuple:
    n_params = sum(p.numel() for p in model.parameters())
    n_samples = overparam_fctr * n_params

    data = torch.tensor(data_gen.gen_asymm_data(n_samples=n_samples)).float()
    sym_data = sym(data)

    grad_mat = torch.zeros((n_samples, n_params)).float()
    sym_grad_mat = torch.zeros((n_samples, n_params)).float()

    fill_grad_mat(model, data, grad_mat)
    fill_grad_mat(model, sym_data, sym_grad_mat)

    return grad_mat, sym_grad_mat


def ewk_repr_matching(data_gen: SymmDataGenerator, n_hidden: int, hidden_l_size: int, overparam_fctr: int,
                      sym: callable):
    model = MLP(input_size=data_gen.n_features, output_size=1, n_hidden=n_hidden,
                hidden_l_size=hidden_l_size)

    grad_mat, sym_grad_mat = get_grad_mat(model, data_gen, sym, overparam_fctr)
    test_grad_mat, test_sym_grad_mat = get_grad_mat(model, data_gen, sym, overparam_fctr)

    g = get_kernelised_elem(grad_mat, sym_grad_mat)

    n_params = sum(p.numel() for p in model.parameters())
    n_samples = overparam_fctr * n_params

    print(torch.linalg.norm(grad_mat @ g - sym_grad_mat) / n_samples,
          torch.linalg.norm(grad_mat - sym_grad_mat) / n_samples)
    print(torch.linalg.norm(test_grad_mat @ g - test_sym_grad_mat) / n_samples,
          torch.linalg.norm(test_grad_mat - test_sym_grad_mat) / n_samples)


if __name__ == '__main__':
    n_features = 3
    symm_data_gen = SymmDataGenerator(n_features,
                                      [lambda x: x[:, list(perm)] for perm in permutations(range(n_features))],
                                      lambda x: x.sum(axis=-1))

    ewk_repr_matching(symm_data_gen, n_hidden=1, hidden_l_size=10 ** 2, overparam_fctr=10 ** 2,
                      sym=lambda x: x[:, [1, 0, 2]])
    ewk_repr_matching(symm_data_gen, n_hidden=1, hidden_l_size=10 ** 2, overparam_fctr=10 ** 2,
                      sym=lambda x: x[:, [2, 1, 0]])
