# Semantic Codebase Cartography using GNNs

Parses a Python file, builds a function call graph, trains a GNN on it, and predicts missing dependencies between functions.

---

## Project Structure

```
.
├── data/sample.py       # Example input file
├── parser.py            # AST parser — extracts functions and calls
├── graph.py             # Builds PyG graph from parsed output
├── model.py             # 2-layer GCN
├── train.py             # Link prediction training loop
├── main.py              # Runs full pipeline + visualizations
└── requirements.txt
```

---

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

---

## What It Does

1. Parses `data/sample.py` using `ast` — finds functions and who calls whom
2. Builds a directed graph — functions are nodes, calls are edges
3. Trains a 2-layer GCN to learn embeddings for each function
4. Predicts likely missing links using dot product scoring + BCE loss
5. Outputs `call_graph.png`, `embeddings.png`, and top predicted links in terminal

---

## Requirements

```
torch
torch-geometric
networkx
matplotlib
scikit-learn
```

---

## Limitations

- Single file only
- No semantic understanding (one-hot features only)
- Only call edges — no imports, inheritance, or variable usage
- No evaluation split — trains and scores on the same graph
- Not tested on real repositories

---

## Roadmap

| Stage | Goal |
|---|---|
| 1 | Fix AST coverage (method calls, aliased imports) ✅ |
| 2 | Multi-file parsing |
| 3 | Heterogeneous graph (imports, inheritance, usage edges) |
| 4 | R-GCN for relation-aware message passing |
| 5 | Code embeddings (CodeBERT) instead of one-hot |
| 6 | Scale to real repositories |

---

## License

MIT
