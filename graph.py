import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

# Load a lightweight, fast model suitable for a 1-week sprint
# This model turns text into a 384-dimensional semantic vector
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def build_graph(parsed):
    """
    Constructs a PyTorch Geometric Data object with Semantic Node Features.
    """
    functions = parsed["functions"]
    calls = parsed["calls"]

    # 1. Map function name → integer index
    node_index = {name: i for i, name in enumerate(functions)}
    num_nodes = len(functions)

    # 2. SEMANTIC NOVELTY: Generate Node Features (x)
    # Instead of an Identity Matrix, we use the "Meaning" of the function names.
    # This allows the GNN to learn patterns based on what the code DOES.
    print(f"Generating semantic embeddings for {num_nodes} functions...")
    with torch.no_grad():
        embeddings = embedder.encode(functions)
        x = torch.tensor(embeddings, dtype=torch.float)

    # 3. Build edge_index from calls
    edges = [
        (node_index[caller], node_index[callee])
        for caller, callee in calls
        if caller in node_index and callee in node_index
    ]

    if edges:
        # Transfer to tensor [2, num_edges]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Handle case with no internal calls
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4. Create the PyG Data object
    # We also store the function names in the object for easier visualization later
    data = Data(x=x, edge_index=edge_index)
    data.node_names = functions 

    return data, node_index