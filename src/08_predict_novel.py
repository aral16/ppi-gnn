"""Rank currently-unrecorded protein pairs with the trained hybrid model.

We restrict candidates to the best-connected proteins (enough neighbourhood
context for the topological features to be meaningful), score every non-edge
among them, and report the highest-confidence predictions with gene symbols.
These are hypotheses for interactions that STRING v12.0 does not yet record at
high confidence -- not validated findings.
"""
import numpy as np
import pandas as pd
import networkx as nx
import joblib

from config import SPLIT_FILE, NODE_MAP_FILE, RESULTS_DIR
from pair_features import pair_matrix

TOP_HUBS = 800          # candidate proteins (by degree)
TOP_K = 50              # predictions to report


def main() -> None:
    s = np.load(SPLIT_FILE)
    z = np.load(RESULTS_DIR / "embeddings.npy")
    nmap = pd.read_csv(NODE_MAP_FILE)
    gb = joblib.load(RESULTS_DIR / "hybrid_gb.joblib")

    g = nx.Graph()
    g.add_edges_from(s["train_pos"].T.tolist())

    known = set()
    for key in ["train_pos", "val_pos", "test_pos"]:
        for u, v in s[key].T.tolist():
            known.add((min(u, v), max(u, v)))

    hubs = nmap.sort_values("degree", ascending=False)["node_idx"].to_numpy()[:TOP_HUBS]
    pairs = [(int(a), int(b)) for i, a in enumerate(hubs) for b in hubs[i + 1:]
             if (min(int(a), int(b)), max(int(a), int(b))) not in known]
    print(f"scoring {len(pairs):,} candidate non-edges among {TOP_HUBS} hub proteins ...")

    X = pair_matrix(g, z, pairs)
    scores = gb.predict_proba(X)[:, 1]
    order = np.argsort(-scores)[:TOP_K]

    gene = dict(zip(nmap["node_idx"], nmap["gene"]))
    deg = dict(zip(nmap["node_idx"], nmap["degree"]))
    rows = []
    for j in order:
        a, b = pairs[j]
        cn = len(list(nx.common_neighbors(g, a, b)))
        rows.append({
            "protein_A": gene[a], "protein_B": gene[b],
            "degree_A": deg[a], "degree_B": deg[b],
            "shared_partners": cn,
            "hybrid_score": round(float(scores[j]), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "top_novel_predictions.csv", index=False)
    print(f"\nTop {TOP_K} predicted novel interactions -> top_novel_predictions.csv")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
