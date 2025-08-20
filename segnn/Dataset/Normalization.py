import torch
from typing import Dict

class Normalizer:
    """
    Holds data relating to normalization and the functions that will perform normalization and unnormalization.
    """
    def __init__(self, coef_stats: dict, device: torch.DeviceObjType) -> None:
        """
        Initializes the normalizer with a set of coefficient stats related to the 
        """
        self.coef_stats = {
            k: torch.tensor(v, dtype=torch.float32, device=device) for k, v in coef_stats.items()
        }

    def normalize(self, coefs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Normalizes coefficients based on the coef_stats data
        using standard normalization (x-\mu)/\sigma
        
        Args:
            coefs: The coefficients in a dictionary from type to coefficient to normalize.

        Returns:
            The normalized coefs.
        """
        normalized_coefs = {}
        for coef_type, coef_value in coefs.items():
            normalized_coefs[coef_type] = (coef_value - self.coef_stats[f"{coef_type}_mean"])/self.coef_stats[f"{coef_type}_std"]
        return normalized_coefs

    def unnormalize(self, coefs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Unnormalizes coefficients based on the coef_stats data
        using standard normalization (x*\sigma)+\mu
        
        Args:
            coefs: The coefficients in a dictionary from type to coefficient to unnormalize.

        Returns:
            The unnormalized coefs.
        """
        unnormalized_coefs = {}
        for coef_type, coef_value in coefs.items():
            unnormalized_coefs[coef_type] = (coef_value * self.coef_stats[f"{coef_type}_std"]) + self.coef_stats[f"{coef_type}_mean"]
        return unnormalized_coefs