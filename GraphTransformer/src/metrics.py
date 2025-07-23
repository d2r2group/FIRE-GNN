import torch

def mean_absolute_percentage_error(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - input) / torch.abs(target))