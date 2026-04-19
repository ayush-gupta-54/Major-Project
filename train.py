import torch
import torch.nn.functional as F

def train(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pos_edge = data.edge_index  # shape: [2, num_edges]
    num_nodes = data.num_nodes
    num_pos = pos_edge.size(1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        embeddings = model(data.x, data.edge_index)  # [num_nodes, out_channels]

        # Positive scores: dot product of connected node pairs
        src, dst = pos_edge[0], pos_edge[1]
        pos_scores = (embeddings[src] * embeddings[dst]).sum(dim=1)

        # Negative sampling: random node pairs
        neg_src = torch.randint(0, num_nodes, (num_pos,))
        neg_dst = torch.randint(0, num_nodes, (num_pos,))
        neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones(num_pos), torch.zeros(num_pos)])

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

    return model