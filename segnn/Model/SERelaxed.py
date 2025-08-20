from .SEGNN import SEGNN
from .Layers.relaxedMPlayer import MessagePassing

class SEMPRelaxed(SEGNN):
    """Steerable E(3) equivariant message passing network"""

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
    ):
        super().__init__(input_irreps, hidden_irreps, output_irreps_lst, edge_attr_irreps, node_attr_irreps, 
                         num_graph_layers, num_prepool_layers, num_postpool_layers, norm, pooling, additional_message_irreps)
        # irreps_node_input,
        # irreps_node_hidden,
        # irreps_node_output,
        # irreps_node_attr,
        # irreps_edge_attr,
        # layers,
        # fc_neurons,
        # num_neighbors,
        # relaxed,
        
    def _make_layer(self):
        return MessagePassing(self.hidden_irreps, self.hidden_irreps, self.hidden_irreps, 
                          self.node_attr_irreps,self.edge_attr_irreps, layers=1, 
                          fc_neurons = [64], num_neighbors = 1, relaxed = True)