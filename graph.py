import torch
from torch_geometric.data import Data

def build_graph(parsed):
    functions = parsed["functions"]
    calls = parsed["calls"]

    # Map function name → integer index
    node_index = {name: i for i, name in enumerate(functions)}
    num_nodes = len(functions)

    # One-hot node features
    x = torch.eye(num_nodes, dtype=torch.float)

    # Build edge_index from calls that reference known functions
    edges = [
        (node_index[caller], node_index[callee])
        for caller, callee in calls
        if caller in node_index and callee in node_index
    ]

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index), node_index