import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from parser import parse_file
from graph import build_graph
from model import GNN
from train import train


def visualize_graph(parsed):
    G = nx.DiGraph()
    G.add_nodes_from(parsed["functions"])
    G.add_edges_from(parsed["calls"])

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="skyblue",
            node_size=2000, arrows=True, font_size=10)
    plt.title("Function Call Graph")
    plt.savefig("call_graph.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: call_graph.png")


def visualize_embeddings(model, data, node_index):
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).numpy()

    reduced = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(embeddings)

    names = list(node_index.keys())
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], s=200, c="coral")
    for i, name in enumerate(names):
        plt.annotate(name, (reduced[i, 0], reduced[i, 1]),
                     fontsize=9, ha="center", va="bottom")
    plt.title("Node Embeddings (t-SNE)")
    plt.savefig("embeddings.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: embeddings.png")


def visualize_predictions(model, data, node_index, top_k=5):
    model.eval()
    with torch.no_grad():
        emb = model(data.x, data.edge_index)

    names = list(node_index.keys())
    n = len(names)
    scores = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                score = torch.sigmoid((emb[i] * emb[j]).sum()).item()
                scores[(names[i], names[j])] = score

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print("\nTop predicted links:")
    for (src, dst), score in top:
        print(f"  {src} → {dst}  (score: {score:.3f})")


def main():
    # 1. Parse
    parsed = parse_file("data/sample.py")
    print("Functions:", parsed["functions"])
    print("Calls:    ", parsed["calls"])

    # 2. Visualize call graph
    visualize_graph(parsed)

    # 3. Build graph
    data, node_index = build_graph(parsed)
    print(f"\nNodes: {data.num_nodes} | Edges: {data.edge_index.size(1)}")

    if data.edge_index.size(1) == 0:
        print("No edges found — nothing to train on.")
        return

    # 4. Init model
    model = GNN(in_channels=data.num_nodes, hidden_channels=16, out_channels=8)

    # 5. Train
    print("\nTraining...\n")
    train(model, data, epochs=100)

    # 6. Visualize embeddings
    visualize_embeddings(model, data, node_index)

    # 7. Visualize predictions
    visualize_predictions(model, data, node_index, top_k=5)

    print("\nDone.")


if __name__ == "__main__":
    main()