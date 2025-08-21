import torch.nn as nn
from e3nn.o3 import Irreps
from typing import List, Literal
from abc import ABCMeta, abstractmethod
from .PooledGraphNN import PooledGraphNN

from .Layers.SO3BuildingBlocks import O3TensorProduct, O3TensorProductSwishGate

class SEGNN(nn.Module, metaclass=ABCMeta):
    """Steerable E(3) equivariant network"""

    def __init__(
        self,
        input_irreps: Irreps,
        hidden_irreps: Irreps,
        output_irreps_lst: List[Irreps],
        edge_attr_irreps: Irreps,
        node_attr_irreps: Irreps,
        num_graph_layers: int,
        num_prepool_layers: int,
        num_postpool_layers: int,
        norm: str,
        pooling: str,
        additional_message_irreps: Irreps
    ):
        super().__init__()
        self.input_irreps = input_irreps
        self.hidden_irreps = hidden_irreps
        self.output_irreps_lst = output_irreps_lst
        self.edge_attr_irreps = edge_attr_irreps
        self.node_attr_irreps = node_attr_irreps
        self.num_graph_layers = num_graph_layers
        self.num_prepool_layers = num_prepool_layers
        self.num_postpool_layers = num_postpool_layers
        self.norm = norm
        self.additional_message_irreps = additional_message_irreps
        self.pooling = pooling

        embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        graph_layers = nn.ModuleList()
        for _ in range(num_graph_layers):
            graph_layers.append(
                self._make_layer()
            )

        self.graph_nn = SEGNNGraphLayers(embedding_layer, graph_layers)

        # Prepool setup
        pre_pool_layers = nn.ModuleList()
        for _ in range(num_prepool_layers - 1):
            pre_pool_layers.append(O3TensorProductSwishGate(hidden_irreps, hidden_irreps, node_attr_irreps))
        pre_pool_layers.append(O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        ))
        
        # All outputs
        outputs = nn.ModuleList()
        for output_irrep in output_irreps_lst:
            postpool_layers = []
            for _ in range(num_postpool_layers - 1):
                postpool_layers.append(O3TensorProductSwishGate(hidden_irreps, hidden_irreps))
            outputs.append(nn.Sequential(*postpool_layers, O3TensorProduct(hidden_irreps, output_irrep)))

        # Store in one precombined module
        self.mlp = SEGNNMLPLayers(pooling, pre_pool_layers, outputs)

    @abstractmethod
    def _make_layer(self):
        raise NotImplementedError("Must be implemented in inherited classes.")


    def forward(self, graph):
        """SEGNN forward pass"""
        x, edge_index, edge_attr, node_attr, batch, additional_message_features = (
            graph.x,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
            graph.edge_dis
        )
        # Graph interaction
        x = self.graph_nn(x, edge_index, edge_attr, node_attr, batch, additional_message_features)
        # Predict
        return self.mlp(x, node_attr, batch)
    

class SEGNNGraphLayers(nn.Module):
    """
    Holds all the Graph layers for a SEGNN model.
    """
    def __init__(self, embedding_layer: O3TensorProduct, graph_layers: nn.ModuleList) -> None:
        """
        Sets up holder for the graph layers and embedding layer.

        Args:
            embedding_layer: The embedding layer from atom information to hidden irreps
            graph_layers: The graph interaction layers.
        """
        super().__init__()
        self.embedding_layer = embedding_layer
        self.graph_layers = graph_layers

    def forward(self, x, edge_index, edge_attr, node_attr, batch, additional_message_features):
        """
        Graph interaction forward pass
        """
        x = self.embedding_layer(x, node_attr)

        # Graph Interactions
        for layer in self.graph_layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        return x

class SEGNNMLPLayers(PooledGraphNN):
    """
    Holds all the MLP layers for a SEGNN model. Also does pooling.
    """
    def __init__(self, pooling: Literal['mean', 'add', 'max'], pre_pool_layers: nn.ModuleList, outputs: nn.ModuleList) -> None:
        """
        Sets up the holder for the MLP layers, prepool, pooling, and postpool.

        Args:
            pooling: What kind of pooling to use
            pre_pool_layers: Per node prepool mlp layers
            outputs: Every set of output MLPs
        """
        super().__init__(pooling)
        self.pre_pool_layers = pre_pool_layers
        self.outputs = outputs

    def forward(self, x, node_attr, batch):
        """
        MLP Forward pass
        """
        # Pre pool
        for layer in self.pre_pool_layers:
            x = layer(x, node_attr)

        # Pool over nodes
        x = self.pool(x, batch)

        # Predict
        return [output(x) for output in self.outputs]