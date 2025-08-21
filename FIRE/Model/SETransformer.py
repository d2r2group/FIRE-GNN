from .SEGNN import SEGNN
from .Layers import SETransformerLayer
from .Layers import SEConvLayer

class SETransformer(SEGNN):
    """Steerable E(3) Graph Transformer"""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps_lst,
        edge_attr_irreps,
        node_attr_irreps,
        num_graph_layers,
        num_prepool_layers=2,
        num_postpool_layers=2,
        norm=None,
        pooling="mean",
        additional_message_irreps=None,
        max_radius: float = 8.0
    ):
        self.max_radius = max_radius
        super().__init__(input_irreps, hidden_irreps, output_irreps_lst, edge_attr_irreps, node_attr_irreps, 
                         num_graph_layers, num_prepool_layers, num_postpool_layers, norm, pooling, additional_message_irreps)
        # They finish graph layers with a convolutional layer w/out attention
        self.graph_nn.graph_layers.append(SEConvLayer(self.hidden_irreps, self.hidden_irreps, self.hidden_irreps, 
                                            self.edge_attr_irreps, self.node_attr_irreps, norm=self.norm, 
                                            additional_message_irreps=self.additional_message_irreps,
                                            conv_type="nonlinear"))
        
    def _make_layer(self):
        return SETransformerLayer(self.hidden_irreps, self.hidden_irreps, self.hidden_irreps, 
                          self.edge_attr_irreps, norm=self.norm, 
                          additional_message_irreps=self.additional_message_irreps,
                          max_radius=self.max_radius)
