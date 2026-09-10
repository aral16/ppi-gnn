"""Combine baseline, GNN and hybrid metrics into one comparison figure + table."""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RESULTS_DIR


def main():
    base = json.loads((RESULTS_DIR / "baseline_metrics.json").read_text())
    gnn = json.loads((RESULTS_DIR / "gnn_metrics.json").read_text())
    rows = [(k, v["roc_auc"], v["ap"], "topological baseline") for k, v in base.items()]
    rows.append(("GraphSAGE (GNN)", gnn["test"]["roc_auc"], gnn["test"]["ap"], "GNN"))

    hybrid_path = RESULTS_DIR / "hybrid_metrics.json"
    if hybrid_path.exists():
        for k, v in json.loads(hybrid_path.read_text()).items():
            grp = "hybrid" if k.startswith("Hybrid") else "ablation"
            rows.append((k, v["roc_auc"], v["ap"], grp))

    df = pd.DataFrame(rows, columns=["method", "roc_auc", "ap", "group"]).sort_values("roc_auc")
    df.to_csv(RESULTS_DIR / "all_metrics.csv", index=False)

    colors = {"topological baseline": "tab:blue", "GNN": "tab:green",
              "hybrid": "tab:red", "ablation": "tab:grey"}
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(df))
    ax.barh(y, df["roc_auc"], color=[colors[g] for g in df["group"]])
    ax.set_yticks(y); ax.set_yticklabels(df["method"])
    ax.set_xlim(0.80, 1.06); ax.set_xlabel("ROC-AUC (held-out test edges)")
    ax.set_title("PPI link prediction — method comparison")
    ax.axvline(0.80, color="black", lw=0.8)
    for i, (a, p) in enumerate(zip(df["roc_auc"], df["ap"])):
        ax.text(min(a, 1.0) + 0.004, i, f"AUC {a:.3f} · AP {p:.3f}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved baseline_comparison.png, all_metrics.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
