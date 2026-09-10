"""Hybrid link predictor: GNN embedding features + topological features -> gradient boosting.

Motivation: on STRING, classical neighbourhood heuristics (Adamic-Adar, Resource
Allocation) are already very strong because the STRING combined score aggregates
evidence channels that correlate with network topology. A pure GNN captures
complementary signal but does not beat them alone. Feeding both the learned GNN
embeddings and the topological scores into one classifier combines the two.
"""
import json

import numpy as np
import pandas as pd
import networkx as nx
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import SPLIT_FILE, RESULTS_DIR
from pair_features import pair_matrix, FEATURE_NAMES


def main() -> None:
    s = np.load(SPLIT_FILE)
    z = np.load(RESULTS_DIR / "embeddings.npy")
    g = nx.Graph()
    g.add_edges_from(s["train_pos"].T.tolist())

    def build(pos, neg):
        edges = np.concatenate([pos, neg], axis=1).T.tolist()
        X = pair_matrix(g, z, edges)
        y = np.concatenate([np.ones(pos.shape[1]), np.zeros(neg.shape[1])])
        return X, y

    print("Building features ...")
    Xtr, ytr = build(s["train_pos"], s["train_neg"])
    Xte, yte = build(s["test_pos"], s["test_neg"])

    results = {}
    gb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                        max_depth=4, random_state=42)
    gb.fit(Xtr, ytr)
    p = gb.predict_proba(Xte)[:, 1]
    results["Hybrid (GradientBoosting)"] = {
        "roc_auc": round(roc_auc_score(yte, p), 4),
        "ap": round(average_precision_score(yte, p), 4),
    }

    lr = LogisticRegression(max_iter=1000, C=1.0)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    lr.fit((Xtr - mu) / sd, ytr)
    p_lr = lr.predict_proba((Xte - mu) / sd)[:, 1]
    results["Hybrid (LogisticRegression)"] = {
        "roc_auc": round(roc_auc_score(yte, p_lr), 4),
        "ap": round(average_precision_score(yte, p_lr), 4),
    }

    for tag, cols in [("GNN features only", slice(7, None)),
                      ("Topological features only", slice(0, 7))]:
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           max_depth=4, random_state=42)
        m.fit(Xtr[:, cols], ytr)
        pp = m.predict_proba(Xte[:, cols])[:, 1]
        results[tag] = {"roc_auc": round(roc_auc_score(yte, pp), 4),
                        "ap": round(average_precision_score(yte, pp), 4)}

    imp = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "logreg_coef": np.round(lr.coef_.ravel(), 3),
    }).sort_values("logreg_coef", key=np.abs, ascending=False)

    joblib.dump(gb, RESULTS_DIR / "hybrid_gb.joblib")
    (RESULTS_DIR / "hybrid_metrics.json").write_text(json.dumps(results, indent=2))
    imp.to_csv(RESULTS_DIR / "hybrid_feature_importance.csv", index=False)

    for k, v in results.items():
        print(f"{k:32s}  ROC-AUC {v['roc_auc']:.4f}   AP {v['ap']:.4f}")
    print("\nLogistic-regression coefficients (standardised):")
    print(imp.to_string(index=False))
    print("\nSaved hybrid_gb.joblib, hybrid_metrics.json, hybrid_feature_importance.csv")


if __name__ == "__main__":
    main()
