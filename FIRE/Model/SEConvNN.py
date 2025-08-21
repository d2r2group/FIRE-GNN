from .SEGNN import SEGNN
from .Layers import SEConvLayer

class SEConv(SEGNN):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

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
        conv_type="linear",
    ):
        self.conv_type = conv_type
        super().__init__(input_irreps, hidden_irreps, output_irreps_lst, edge_attr_irreps, node_attr_irreps, 
                         num_graph_layers, num_prepool_layers, num_postpool_layers, norm, pooling, additional_message_irreps)

    def _make_layer(self):
        return SEConvLayer(
                    self.hidden_irreps,
                    self.hidden_irreps,
                    self.hidden_irreps,
                    self.edge_attr_irreps,
                    self.node_attr_irreps,
                    norm=self.norm,
                    additional_message_irreps=self.additional_message_irreps,
                    conv_type=self.conv_type,
                )