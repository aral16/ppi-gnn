"""Build a clean PPI graph from the raw STRING download.

Steps
-----
1. Load edges, keep combined_score >= CONFIDENCE_THRESHOLD.
2. Make the graph undirected and restrict to its largest connected component.
3. Re-index proteins to contiguous integer node ids.
4. Compute structural node features (degree, clustering, k-core, ...).
5. Save edge_index + features to data/graph.npz and the id/symbol map.
"""
import gzip

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla

LAP_PE_DIM = 32

from config import (
    LINKS_FILE, INFO_FILE, CONFIDENCE_THRESHOLD,
    GRAPH_FILE, NODE_MAP_FILE,
)


def load_edges() -> pd.DataFrame:
    print(f"Reading {LINKS_FILE.name} ...")
    df = pd.read_csv(LINKS_FILE, sep=" ")
    df.columns = [c.strip() for c in df.columns]
    n_raw = len(df)
    df = df[df["combined_score"] >= CONFIDENCE_THRESHOLD]
    print(f"  {n_raw:,} raw edges -> {len(df):,} at combined_score >= {CONFIDENCE_THRESHOLD}")
    return df[["protein1", "protein2", "combined_score"]]


def load_symbols() -> dict:
    with gzip.open(INFO_FILE, "rt") as fh:
        info = pd.read_csv(fh, sep="\t")
    col = "#string_protein_id" if "#string_protein_id" in info.columns else info.columns[0]
    return dict(zip(info[col], info["preferred_name"]))


def main() -> None:
    edges = load_edges()
    symbols = load_symbols()

    g = nx.from_pandas_edgelist(edges, "protein1", "protein2")
    g.remove_edges_from(nx.selfloop_edges(g))
    print(f"Graph: {g.number_of_nodes():,} nodes / {g.number_of_edges():,} edges")

    lcc = max(nx.connected_components(g), key=len)
    g = g.subgraph(lcc).copy()
    print(f"Largest connected component: {g.number_of_nodes():,} nodes / "
          f"{g.number_of_edges():,} edges")

    nodes = sorted(g.nodes())
    idx = {p: i for i, p in enumerate(nodes)}
    g = nx.relabel_nodes(g, idx)
    n = g.number_of_nodes()

    # --- structural node features -------------------------------------------
    print("Computing structural features ...")
    deg = np.array([d for _, d in g.degree()], dtype=np.float32)
    clustering = np.array(list(nx.clustering(g).values()), dtype=np.float32)
    core = np.array(list(nx.core_number(g).values()), dtype=np.float32)
    avg_nbr_deg = np.array(list(nx.average_neighbor_degree(g).values()), dtype=np.float32)
    triangles = np.array(list(nx.triangles(g).values()), dtype=np.float32)
    pagerank = np.array(list(nx.pagerank(g, alpha=0.85).values()), dtype=np.float32)

    struct = np.stack([
        np.log1p(deg),
        clustering,
        np.log1p(core),
        np.log1p(avg_nbr_deg),
        np.log1p(triangles),
        np.log1p(pagerank * n),
    ], axis=1)
    struct = (struct - struct.mean(0)) / (struct.std(0) + 1e-8)

    # --- Laplacian positional encoding -------------------------------------
    print(f"Computing {LAP_PE_DIM} Laplacian eigenvectors ...")
    A = nx.to_scipy_sparse_array(g, nodelist=range(n), format="csr", dtype=np.float64)
    d = np.asarray(A.sum(1)).ravel()
    dinv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-12)))
    L = sp.eye(n) - dinv_sqrt @ A @ dinv_sqrt
    vals, vecs = spla.eigsh(L, k=LAP_PE_DIM + 1, which="SM", tol=1e-3)
    order = np.argsort(vals)
    lap_pe = vecs[:, order[1:LAP_PE_DIM + 1]].astype(np.float32)   # drop trivial eigvec
    lap_pe /= (np.linalg.norm(lap_pe, axis=0, keepdims=True) + 1e-8)

    feats = np.concatenate([struct, lap_pe], axis=1)

    edge_index = np.array(g.edges(), dtype=np.int64).T          # (2, E) directed one way
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)  # undirected

    np.savez_compressed(GRAPH_FILE, edge_index=edge_index, features=feats.astype(np.float32),
                        struct_dim=struct.shape[1], lap_pe_dim=LAP_PE_DIM, num_nodes=n)
    print(f"Saved {GRAPH_FILE.name}: features {feats.shape}, edge_index {edge_index.shape}")

    pd.DataFrame({
        "node_idx": range(n),
        "string_id": nodes,
        "gene": [symbols.get(p, p.split(".")[-1]) for p in nodes],
        "degree": deg.astype(int),
    }).to_csv(NODE_MAP_FILE, index=False)
    print(f"Saved {NODE_MAP_FILE.name}")


if __name__ == "__main__":
    main()
