"""Feature construction for a candidate protein pair (shared by the hybrid model
and the novel-interaction ranker)."""
import numpy as np
import networkx as nx

FEATURE_NAMES = [
    "common_neighbors", "adamic_adar", "resource_alloc", "jaccard",
    "log_pref_attach", "log_min_degree", "log_max_degree",
    "gnn_cosine", "gnn_dot", "gnn_l2", "gnn_had_mean", "gnn_had_max", "gnn_had_min",
]


def topo_features(g, edges):
    valid = [(u, v) for u, v in edges if g.has_node(u) and g.has_node(v)]
    aa = {(u, v): p for u, v, p in nx.adamic_adar_index(g, valid)}
    ra = {(u, v): p for u, v, p in nx.resource_allocation_index(g, valid)}
    jc = {(u, v): p for u, v, p in nx.jaccard_coefficient(g, valid)}
    rows = []
    for u, v in edges:
        if g.has_node(u) and g.has_node(v):
            cn = len(list(nx.common_neighbors(g, u, v)))
            du, dv = g.degree(u), g.degree(v)
        else:
            cn = du = dv = 0
        rows.append([
            cn, aa.get((u, v), 0.0), ra.get((u, v), 0.0), jc.get((u, v), 0.0),
            np.log1p(du) + np.log1p(dv), np.log1p(min(du, dv)), np.log1p(max(du, dv)),
        ])
    return np.array(rows, dtype=np.float32)


def gnn_features(z, edges):
    idx = np.asarray(edges)
    zi, zj = z[idx[:, 0]], z[idx[:, 1]]
    cos = (zi * zj).sum(1) / (np.linalg.norm(zi, axis=1) * np.linalg.norm(zj, axis=1) + 1e-8)
    had = zi * zj
    return np.column_stack([
        cos, (zi * zj).sum(1), np.linalg.norm(zi - zj, axis=1),
        had.mean(1), had.max(1), had.min(1),
    ]).astype(np.float32)


def pair_matrix(g, z, edges):
    return np.hstack([topo_features(g, edges), gnn_features(z, edges)])
