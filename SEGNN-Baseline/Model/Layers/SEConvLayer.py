import torch
from torch_geometric.nn import MessagePassing
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps

from .SO3BuildingBlocks import O3TensorProduct, O3TensorProductSwishGate, O3SwishGate

class SEConvLayer(MessagePassing):
    """E(3) equivariant (non-linear) convolutional layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
        conv_type="linear",
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps
        self.conv_type = conv_type

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        self.setup_gate(hidden_irreps)
        if self.conv_type == "linear":
            self.message_layer = O3TensorProduct(
                message_input_irreps, self.irreps_g, edge_attr_irreps
            )
        elif self.conv_type == "nonlinear":
            self.message_layer_1 = O3TensorProductSwishGate(
                message_input_irreps, hidden_irreps, edge_attr_irreps
            )
            self.message_layer_2 = O3TensorProduct(
                hidden_irreps, self.irreps_g, edge_attr_irreps
            )
        else:
            raise Exception("Invalid convolution type for SEConvLayer")

        self.setup_normalisation(norm)

    def setup_gate(self, hidden_irreps):
        """Add necessary scalar irreps for gate to output_irreps, similar to O3TensorProductSwishGate"""
        irreps_g_scalars = Irreps(str(hidden_irreps[0]))
        irreps_g_gate = Irreps(
            "{}x0e".format(hidden_irreps.num_irreps - irreps_g_scalars.num_irreps)
        )
        irreps_g_gated = Irreps(str(hidden_irreps[1:]))
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()
        self.gate = O3SwishGate(irreps_g_scalars, irreps_g_gate, irreps_g_gated)
        self.irreps_g = irreps_g

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.irreps_g)
        elif norm == "instance":
            self.feature_norm = BatchNorm(self.hidden_irreps, instance=True)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            x = self.feature_norm(x)
        
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        if self.conv_type == "linear":
            message = self.message_layer(input, edge_attr)
        elif self.conv_type == "nonlinear":
            message = self.message_layer_1(input, edge_attr)
            message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update node features"""
        update = self.gate(message)
        x += update  # Residual connection
        return x