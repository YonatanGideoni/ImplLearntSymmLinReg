import numpy
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from scipy.optimize import curve_fit
from torch import nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from consts import DEVICE
from symm_data_gen import SymmDataGenerator


# from https://github.com/ajacot/NTK_utils/blob/master/network.py
class NTKLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, beta=0.1):
        super(NTKLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(DEVICE))

        self.stdv = np.sqrt(1 - beta ** 2) / np.sqrt(in_features * 1.0)
        self.beta = beta

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(DEVICE))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.normal_(0.0, 1.0)  # .uniform_(-alpha, alpha)
        if self.bias is not None:
            self.bias.data.normal_(0.0, 1.0)  # .uniform_(-alpha, alpha)

    def forward(self, input):
        if self.bias is not None:
            return F.linear(input, self.weight) * self.stdv + self.bias * self.beta
        else:
            return F.linear(input, self.weight) * self.stdv

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Perceptron(nn.Module):
    def __init__(self, input_dim, output_dim, activation: nn.modules.activation = nn.ReLU(), bias: bool = True,
                 ntk: bool = False):
        super(Perceptron, self).__init__()

        self.layer = nn.Linear(input_dim, output_dim, bias=bias) if not ntk else NTKLinear(input_dim, output_dim,
                                                                                           bias=bias)
        self.act = activation

    def forward(self, x):
        x = self.act(self.layer(x))

        return x


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_l_size: int, n_hidden: int, output_size: int,
                 activation: nn.modules.activation = nn.ReLU(), bias: bool = True, ntk: bool = False):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()

        assert n_hidden >= 1, 'Error - you need some hidden layers for this to be an MLP!'

        layers = [Perceptron(input_size, hidden_l_size, activation=activation, bias=bias, ntk=ntk)]
        layers += [Perceptron(hidden_l_size, hidden_l_size, activation=activation, bias=bias, ntk=ntk)
                   for _ in range(n_hidden - 1)]
        layers += [nn.Linear(hidden_l_size, output_size, bias=bias)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        y = self.layers(x)
        return y


def train_loop(dataloader, model: nn, loss_fn: callable, optimizer: Optimizer):
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn: callable, verbose: bool = False) -> float:
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            assert pred.shape == y.shape, 'Error - prediction/label shape mismatch!'
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches

    if verbose:
        print(f"Avg test loss: {test_loss:>8f} \n")

    return test_loss


def get_concat_weights(model: nn.Module, grad: bool = False):
    params = []
    for param in model.parameters():
        if grad:
            param = param.grad
        params.append(param.view(-1))
    return torch.cat(params)


def set_concat_weights(model: nn.Module, params):
    with torch.no_grad():
        n_set_params = 0
        for model_param in model.parameters():
            n_params = model_param.numel()
            model_param.copy_(torch.tensor(params[n_set_params:n_set_params + n_params].reshape(*model_param.shape))
                              .type(model_param.dtype))
            n_set_params += n_params


def get_data_loader(data_generator, dataset_size, batch_size) -> DataLoader:
    X, y = data_generator(dataset_size)

    X = torch.Tensor(X).to(DEVICE)
    X = torch.flatten(X, start_dim=1, end_dim=-1)
    y = torch.Tensor(y).to(DEVICE)
    dataset = TensorDataset(X, y)

    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def calc_asymm_loss(model: nn.Module, data_gen: SymmDataGenerator, n_points: int = 100) -> float:
    with torch.no_grad():
        asymm_loss = 0
        for _ in range(n_points):
            x = data_gen.gen_asymm_data(1, postprocess=False)
            symm_x = data_gen.symmetrise(x, postprocess=True)

            x = torch.Tensor(data_gen.postprocess_func(x)).to(DEVICE)
            symm_x = torch.Tensor(symm_x).to(DEVICE)

            pred = model(x)
            symm_pred = model(symm_x)

            if data_gen.classification:
                pred = torch.sigmoid(pred)
                symm_pred = torch.sigmoid(symm_pred)

            if data_gen.equiv:
                symm_orig_pred = torch.tensor(data_gen.symmetrise(pred))
                asymm_loss += torch.linalg.norm(symm_orig_pred - symm_pred).item()
            else:
                asymm_loss += torch.linalg.norm(pred - symm_pred.mean(dim=0)).item()

    return asymm_loss / n_points


def calc_decay_rate(asymm_loss: pd.Series, lr: float, data_batch_ratio: float) -> float:
    n_epochs = len(asymm_loss)
    t = np.arange(n_epochs) * lr * data_batch_ratio
    [decay_rate, _], _ = curve_fit(lambda t, decay_rate, l0: l0 * np.exp(-decay_rate * t), t, asymm_loss)

    return decay_rate


def train_model(model: nn.Module, data_gen: SymmDataGenerator, lr: float = 1e-3, n_epochs: int = 500,
                n_samples: int = 10 ** 3, batch_size: int = 100, loss: nn.modules.loss = nn.MSELoss(),
                calc_decay: bool = True, calc_asymm: bool = True):
    model = model.to(DEVICE)

    optimiser = optim.SGD(model.parameters(), lr=lr)

    train_loader = get_data_loader(data_gen, n_samples, batch_size)
    test_loader = get_data_loader(data_gen, n_samples, batch_size)

    training_res = []
    for epoch in range(n_epochs):
        train_loop(train_loader, model, loss, optimiser)
        test_loss = test_loop(test_loader, model, loss)

        epoch_res = {'epoch': epoch,
                     'test_loss': test_loss,
                     'params': [p.cpu().clone().detach().numpy() for p in model.parameters()],
                     'lr': lr}

        if calc_asymm:
            asymm_loss = calc_asymm_loss(model, data_gen)
            epoch_res['asymm_loss'] = asymm_loss

        training_res.append(epoch_res)
    training_res = pd.DataFrame.from_records(training_res)

    if calc_decay:
        decay_rate = calc_decay_rate(training_res.asymm_loss, lr, n_samples / batch_size)
        training_res['decay_rate'] = decay_rate

    return training_res
