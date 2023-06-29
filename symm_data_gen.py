from dataclasses import dataclass
from typing import Union

import numpy as np

Id = lambda x: x


@dataclass
class SymmDataGenerator:
    n_features: int
    group_elems: list
    labeller: callable
    domain: str = 'hypercube'
    noise: float = None
    classification: bool = False
    cutoff: float = 1.
    # whether to explicitly input all the elements in an orbit or have the symmetry be in the data only on average
    symmetrise_expl: bool = True
    postprocess_func: callable = Id
    # required in cases where the postprocessing adds/removes features
    n_base_features: int = None
    equiv: bool = False

    def __call__(self, n_samples: int, *args, **kwargs) -> tuple:
        n_base_samples = n_samples // (len(self.group_elems) if self.symmetrise_expl else 1)

        # need to be finnicky about what gets labelled when because the postprocessed data might not have a group
        # representation - the symmetrisation works on the base data
        base_data = self.gen_asymm_data(n_base_samples, postprocess=False)

        labels = self.labeller(self.postprocess_func(base_data))

        return self.symmetrise(base_data, labels) if self.symmetrise_expl else (base_data, labels.reshape(-1, 1))

    def gen_asymm_data(self, n_samples: int, postprocess: bool = True) -> np.ndarray:
        n_features = self.n_base_features if self.n_base_features is not None else self.n_features
        if self.domain == 'hypercube':
            data = 2 * (np.random.rand(n_samples, n_features) - 0.5)

            return data if not postprocess else self.postprocess_func(data)

        raise NotImplementedError('Still need to implement more general domains!')

    def symmetrise(self, base_data: np.ndarray, labels: np.ndarray = None, noise: float = 0.,
                   postprocess: bool = True) -> Union[tuple, np.ndarray]:
        if self.noise is not None and noise == 0.:
            noise = self.noise
        data = np.concatenate([group_elem(base_data) for group_elem in self.group_elems])
        data = data if not postprocess else self.postprocess_func(data)
        if labels is not None:
            labels = np.concatenate([labels] * len(self.group_elems) if not self.equiv else
                                    [group_elem(labels) for group_elem in self.group_elems])

            if len(labels.shape) == 1:
                labels = labels.reshape(-1, 1)

            labels += np.random.randn(*labels.shape) * noise

            # TODO - make sure this makes sense in noisy cases
            if self.classification:
                labels = labels > self.cutoff

            return data, labels

        return data


AbsXSymmGenerator = SymmDataGenerator(1, [Id, lambda x: -x], lambda x: abs(x))
XSquareSymmGenerator = SymmDataGenerator(1, [Id, lambda x: -x], lambda x: x ** 2)
PermXYSymmGenerator = SymmDataGenerator(2, [Id, lambda x: x[:, [1, 0]]], lambda x: x.sum(axis=-1))


def poly_kernel(features, order):
    # returns 1,x,y,x**2,y**2,xy,etc.
    n_samples, n_features = features.shape
    kernelised_features = np.empty((n_samples, (order + 1) * (order + 2) // 2))

    for i in range(order + 1):
        for j in range(i + 1):
            kernelised_features[:, i * (i + 1) // 2 + j] = features[:, 0] ** j * features[:, 1] ** (i - j)

    return kernelised_features


PolyXYSymmGenerator = lambda order: SymmDataGenerator(n_features=(order + 1) * (order + 2) // 2, n_base_features=2,
                                                      group_elems=[Id, lambda x: x[:, [1, 0]]],
                                                      labeller=lambda x: x.sum(axis=-1),
                                                      postprocess_func=lambda features: poly_kernel(features, order))
NoisyPermXYSymmGenerator = SymmDataGenerator(2, [Id, lambda x: x[:, [1, 0]]], lambda x: x.sum(axis=-1), noise=5e-0)


def sq_inputs(data) -> np.ndarray:
    return data ** 2


def mod_inputs(data) -> np.ndarray:
    # x,y -> xy,xy^2
    return np.array([data[:, 0] * data[:, 1], data[:, 0] * data[:, 1] ** 2]).reshape(-1, 2)


def get_data_rotator(angle) -> callable:
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])

    def rot_data(data):
        return data @ rot_mat.T

    return rot_data


RotXYSqSymmGenerator = SymmDataGenerator(2, [get_data_rotator(angle)
                                             for angle in np.linspace(0, 2 * np.pi, num=16, endpoint=False)],
                                         lambda x: x.sum(axis=-1), postprocess_func=sq_inputs)

RotWeirdXYSqSymmGenerator = SymmDataGenerator(2, [get_data_rotator(angle)
                                                  for angle in np.linspace(0, 2 * np.pi, num=16, endpoint=False)],
                                              lambda x: x.sum(axis=-1), postprocess_func=mod_inputs)

if __name__ == '__main__':
    RotWeirdXYSqSymmGenerator(100)
