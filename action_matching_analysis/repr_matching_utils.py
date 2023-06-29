import torch


def get_kernelised_elem(norm_output, transf_output):
    return torch.linalg.lstsq(norm_output.T @ norm_output, norm_output.T @ transf_output).solution
