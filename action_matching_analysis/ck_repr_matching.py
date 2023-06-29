from itertools import permutations

import torch

from action_matching_analysis.repr_matching_utils import get_kernelised_elem
from symm_data_gen import PermXYSymmGenerator, SymmDataGenerator
from utils import MLP


def ck_repr_matching(data_gen: SymmDataGenerator, output_size: int, n_hidden: int, hidden_l_size: int,
                     overparam_fctr: int, sym: callable):
    model = MLP(input_size=data_gen.n_features, output_size=output_size, n_hidden=n_hidden,
                hidden_l_size=hidden_l_size)

    n_samples = overparam_fctr * output_size
    data = torch.tensor(data_gen.gen_asymm_data(n_samples=n_samples)).float()
    test_data = torch.tensor(data_gen.gen_asymm_data(n_samples=n_samples)).float()

    with torch.no_grad():
        norm_output = model(data)
        transf_output = model(sym(data))

        test_norm_output = model(test_data)
        test_transf_output = model(sym(test_data))

    g = get_kernelised_elem(norm_output, transf_output)

    print(torch.linalg.norm(norm_output @ g - transf_output) / n_samples,
          torch.linalg.norm(norm_output - transf_output) / n_samples)
    print(torch.linalg.norm(test_norm_output @ g - test_transf_output) / n_samples,
          torch.linalg.norm(test_norm_output - test_transf_output) / n_samples)


if __name__ == '__main__':
    n_features = 3
    symm_data_gen = SymmDataGenerator(n_features,
                                      [lambda x: x[:, list(perm)] for perm in permutations(range(n_features))],
                                      lambda x: x.sum(axis=-1))

    ck_repr_matching(symm_data_gen, output_size=10 ** 2, n_hidden=1, hidden_l_size=10 ** 2, overparam_fctr=10 ** 2,
                     sym=lambda x: x[:, [1, 0, 2]])
    ck_repr_matching(symm_data_gen, output_size=10 ** 2, n_hidden=1, hidden_l_size=10 ** 2, overparam_fctr=10 ** 2,
                     sym=lambda x: x[:, [2, 1, 0]])
