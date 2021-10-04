import torch

from torch_geometric.datasets.molecule_net import x_map as x_map_default
from torch_geometric.datasets.molecule_net import e_map as e_map_default


def get_atom_feature_dims():
    allowable_features = x_map_default
    return list(map(len, [
        allowable_features['atomic_num'],
        allowable_features['chirality'],
        allowable_features['degree'],
        allowable_features['formal_charge'],
        allowable_features['num_hs'],
        allowable_features['num_radical_electrons'],
        allowable_features['hybridization'],
        allowable_features['is_aromatic'],
        allowable_features['is_in_ring']
        ]))


def get_bond_feature_dims():
    allowable_features = e_map_default
    return list(map(len, [
        allowable_features['bond_type'],
        allowable_features['stereo'],
        allowable_features['is_conjugated']
        ]))


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        full_atom_feature_dims = get_atom_feature_dims()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        full_bond_feature_dims = get_bond_feature_dims()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
