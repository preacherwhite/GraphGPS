import torch
from torch_geometric.data import Data


# The original function with loop-based computation
def pre_transform_NeuroGraphDataset_original(data):
    if data.x is None:
        raise ValueError("Data object does not have node features, which are required to create edge features.")

    edge_features = torch.zeros(data.edge_index.size(1), dtype=torch.float32)
    for i, (node1, node2) in enumerate(data.edge_index.t()):
        node1_features = data.x[node1].unsqueeze(0)
        node2_features = data.x[node2].unsqueeze(0)
        features = torch.cat((node1_features, node2_features), dim=0)
        correlation_matrix = torch.corrcoef(features)
        edge_features[i] = correlation_matrix[0, 1]

    return edge_features


# The vectorized function
def pre_transform_NeuroGraphDataset_vectorized(data):
    if data.x is None:
        raise ValueError("Data object does not have node features, which are required to create edge features.")

    node_features_1 = data.x[data.edge_index[0]]
    node_features_2 = data.x[data.edge_index[1]]

    node_features_1 = (node_features_1 - node_features_1.mean(dim=1, keepdim=True)) / node_features_1.std(dim=1,
                                                                                                          keepdim=True)
    node_features_2 = (node_features_2 - node_features_2.mean(dim=1, keepdim=True)) / node_features_2.std(dim=1,
                                                                                                          keepdim=True)

    correlation = (node_features_1 * node_features_2).sum(dim=1) / (node_features_1.size(1) - 1)

    return correlation


# Create a sample Data object
num_nodes = 10
num_node_features = 5
num_edges = 15
edge_index = torch.randint(0, num_nodes, (2, num_edges))
x = torch.randn(num_nodes, num_node_features)
data = Data(x=x, edge_index=edge_index)

# Compute edge features using both functions
edge_features_original = pre_transform_NeuroGraphDataset_original(data)
edge_features_vectorized = pre_transform_NeuroGraphDataset_vectorized(data)

# Compare the outputs
are_equal = torch.allclose(edge_features_original, edge_features_vectorized, atol=1e-6)
print(are_equal, edge_features_original, edge_features_vectorized)

