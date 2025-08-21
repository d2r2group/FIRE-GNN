from e3nn import o3
from Model.Layers import SEConvLayer, SETransformerLayer, SEMPLayer
import torch
from torch_geometric.nn import knn_graph
from torch_scatter import scatter
from argparse import ArgumentParser

irreps_input = o3.Irreps.spherical_harmonics(3)
irreps_hidden = o3.Irreps.spherical_harmonics(3)
irreps_output = o3.Irreps.spherical_harmonics(3)
irreps_sh = o3.Irreps.spherical_harmonics(2)

def run_layer(layer, f, pos):

    edge_src, edge_dst = knn_graph(pos, 4)
    edge_index = torch.vstack((edge_src, edge_dst))

    edge_vec = pos[edge_src] - pos[edge_dst]
    edge_length = edge_vec.norm(dim=1, keepdim=True)

    edge_sh = o3.spherical_harmonics(irreps_sh, edge_vec, True, normalization='component')
    node_sh = scatter(edge_sh, edge_dst, dim=0, reduce="mean")

    return layer(f, edge_index, edge_sh, node_sh, torch.zeros(len(f)), edge_length)

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("layer", choices=["transformer", "conv-linear", "conv-nonlinear", "mp"])
    args = args.parse_args()

    max_radius = 8.0

    if args.layer == "transformer":
        layer = SETransformerLayer(irreps_input, irreps_hidden, irreps_output, irreps_sh, 
                                        additional_message_irreps=o3.Irreps("1x0e"), norm="batch",
                                        max_radius=max_radius)
    elif "conv" in args.layer:
        layer = SEConvLayer(irreps_input, irreps_hidden, irreps_output, irreps_sh, irreps_sh,
                                        additional_message_irreps=o3.Irreps("1x0e"), norm="batch",
                                        conv_type=args.layer.split("-")[-1])
    elif args.layer == "mp":
        layer = SEMPLayer(irreps_input, irreps_hidden, irreps_output, irreps_sh,  irreps_sh,
                                        additional_message_irreps=o3.Irreps("1x0e"), norm="batch")
    f = irreps_input.randn(10, -1)
    pos = torch.randn(10, 3)

    rot = o3.rand_matrix()
    D_in = irreps_input.D_from_matrix(rot)
    D_out = irreps_output.D_from_matrix(rot)

    f_before = run_layer(layer, f @ D_in.T, pos @ rot.T)
    f_after = run_layer(layer, f, pos) @ D_out.T

    if torch.allclose(f_before, f_after, atol=1e-3, rtol=1e-3):
        print(f"{args.layer} is equivariant")
    else:
        print(f"{args.layer} is not equivariant")