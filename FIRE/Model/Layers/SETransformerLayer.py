import torch
from e3nn.nn import BatchNorm, FullyConnectedNet, NormActivation
from e3nn.o3 import Irreps, FullyConnectedTensorProduct, Linear
from e3nn.math import soft_one_hot_linspace, soft_unit_step

from torch_scatter import scatter

class SETransformerLayer(torch.nn.Module):
    """E(3) Graph Transformer layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        additional_message_irreps,
        norm=None,
        number_of_basis=16,
        max_radius=8.0
    ):
        super().__init__()
        assert additional_message_irreps == Irreps("1x0e"), "Additional message features must be a distance length."
        self.hidden_irreps = hidden_irreps

        # Conv with Attention
        self.h_q = Linear(input_irreps, hidden_irreps)

        self.tp_k = FullyConnectedTensorProduct(input_irreps, edge_attr_irreps, hidden_irreps, shared_weights=False)
        self.fc_k = FullyConnectedNet([number_of_basis, 16, self.tp_k.weight_numel], act=torch.nn.functional.silu)

        self.tp_v = FullyConnectedTensorProduct(input_irreps, edge_attr_irreps, output_irreps, shared_weights=False)
        self.fc_v = FullyConnectedNet([number_of_basis, 16, self.tp_v.weight_numel], act=torch.nn.functional.silu)

        self.dot = FullyConnectedTensorProduct(hidden_irreps, hidden_irreps, "0e")
        
        self.number_of_basis = number_of_basis
        self.max_radius = max_radius

        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = BatchNorm(self.hidden_irreps, instance=True)

    def transform(self, x, edge_index, edge_sh, edge_length):
        # Need to compress to single dim, expanded by per edge in embedding.
        assert len(edge_length.shape) < 3, "Edge length has too many dimensions."
        if len(edge_length.shape) == 2:
            assert edge_length.shape[1] == 1, "Edge length is not a scalar."
            edge_length = edge_length.reshape(-1)
        edge_length_embedded = soft_one_hot_linspace(
            edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis='smooth_finite',
            cutoff=True
        )
        edge_length_embedded = edge_length_embedded.mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))

        # compute the queries (per node), keys (per edge) and values (per edge)
        q = self.h_q(x)
        k = self.tp_k(x[edge_index[0]], edge_sh, self.fc_k(edge_length_embedded))
        v = self.tp_v(x[edge_index[0]], edge_sh, self.fc_v(edge_length_embedded))

        # compute the softmax (per edge)
        exp = edge_weight_cutoff[:, None] * self.dot(q[edge_index[1]], k).exp() # compute the numerator
        z = scatter(exp, edge_index[1], dim=0, dim_size=len(x)) # compute the denominator (per nodes)
        # Avoids 0/0 when neighbors are at cutoff exactly
        z[z == 0] = 1
        alpha = exp / z[edge_index[1]]

        return scatter(alpha.relu().sqrt() * v, edge_index[1], dim=0, dim_size=len(x))

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features
    ):
        # Run attention
        x = self.transform(x, edge_index, edge_attr, additional_message_features)
        # Normalise features
        if self.feature_norm:
            x = self.feature_norm(x)
        return x