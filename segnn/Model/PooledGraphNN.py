from abc import ABCMeta
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from typing import Literal
import torch

class PooledGraphNN(torch.nn.Module, metaclass=ABCMeta):
    """
    Acts as an abstract class for graph neural networks that
    scatter together their node values
    """

    def __init__(self, pooling: Literal["mean", "add", "max"]) -> None:
        """
        Initializes a scatter graph neural network with a self.pool function
        with the given mode

        Args:
            pooling: The mode of how to combine the nodes.
        """
        super().__init__()
        self.pooling = pooling
        if pooling == "mean":
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Illegal pooling mode.")
