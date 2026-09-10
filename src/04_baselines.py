"""Classical topological link-prediction baselines.

Every score is computed on the *training* graph only, then evaluated on the
held-out test edges (same split as the GNN). This is the bar the GNN has to clear.
"""
import json

import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score

from config import SPLIT_FILE, RESULTS_DIR


def build_train_graph():
    d = np.load(SPLIT_FILE)
    g = nx.Graph()
    g.add_edges_from(d["train_pos"].T.tolist())
    test = np.concatenate([d["test_pos"], d["test_neg"]], axis=1).T
    y = np.concatenate([np.ones(d["test_pos"].shape[1]), np.zeros(d["test_neg"].shape[1])])
    return g, test, y


def _valid(g, e):
    return [(u, v) for u, v in e if g.has_node(u) and g.has_node(v)]


def _score(fn, g, e):
    lookup = {}
    for u, v, p in fn(g, _valid(g, e)):
        lookup[(u, v)] = p
    return [lookup.get((u, v), lookup.get((v, u), 0.0)) for u, v in e]


def _common_neighbors(g, e):
    return [len(list(nx.common_neighbors(g, u, v))) if g.has_node(u) and g.has_node(v) else 0
            for u, v in e]


PREDICTORS = {
    "Common Neighbors": _common_neighbors,
    "Jaccard": lambda g, e: _score(nx.jaccard_coefficient, g, e),
    "Adamic-Adar": lambda g, e: _score(nx.adamic_adar_index, g, e),
    "Resource Allocation": lambda g, e: _score(nx.resource_allocation_index, g, e),
    "Preferential Attachment": lambda g, e: _score(nx.preferential_attachment, g, e),
}


def main() -> None:
    g, test_edges, y = build_train_graph()
    edge_list = [tuple(x) for x in test_edges.tolist()]
    rows = {}
    for name, fn in PREDICTORS.items():
        scores = np.array(fn(g, edge_list), dtype=float)
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        rows[name] = {"roc_auc": round(auc, 4), "ap": round(ap, 4)}
        print(f"{name:26s}  ROC-AUC {auc:.4f}   AP {ap:.4f}")

    out = RESULTS_DIR / "baseline_metrics.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved {out.name}")


if __name__ == "__main__":
    main()
