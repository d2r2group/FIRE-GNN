import torch, numpy as np, collections as C
from pathlib import Path

def describe(split_path: Path, name: str):
    ds = torch.load(split_path, map_location='cpu')   # list[Data]

    # ---- 1) basic sizes ----------------------------------------------------
    n_graphs = len(ds)
    n_nodes   = [d.natoms.item()            for d in ds]
    y         = torch.stack([d.y  for d in ds])  # (N,2)  *normalised* targets
    pos_miss  = [i for i,d in enumerate(ds) if d.pos.shape[0] != d.natoms]

    print(f"\n{name} ({n_graphs:>5,d} graphs)")
    print(f"  nodes/graph   : min={min(n_nodes):3d}  p50={int(np.median(n_nodes)):3d}  "
          f"p95={int(np.percentile(n_nodes,95))}  max={max(n_nodes):3d}")
    print(f"  graphs w/ inconsistent pos <-> natoms : {len(pos_miss)}")

    # ---- 2) target statistics ---------------------------------------------
    mu,  std  = y.mean(0).tolist(), y.std(0, unbiased=False).tolist()
    y_min, y_max = y.min(0)[0].tolist(), y.max(0)[0].tolist()

    print("  y  (normalised) :")
    for i, label in enumerate(["WF_top", "WF_bottom"]):
        print(f"     {label:<10}  μ={mu[i]:6.3f}  σ={std[i]:6.3f}  "
              f"min={y_min[i]:6.3f}  max={y_max[i]:6.3f}")

    # ---- 3) quick outlier scan --------------------------------------------
    big = [i for i,n in enumerate(n_nodes)
        if n > np.percentile(n_nodes, 99)]

    # mask of outliers per target
    z = y.abs()                       # (N,2)   already normalised → σ≈1
    out_top = (z[:, 0] > 8)           # True where WF_top is an outlier
    out_bot = (z[:, 1] > 8)           # True where WF_bottom is an outlier

    weird_any = (out_top | out_bot)   # original criterion “either target”

    print(f"  >99-percentile graphs (size outliers) : {len(big)}")
    print(f"  |y| > 8 σ  (target outliers)          : "
          f"{weird_any.sum()}  "
          f"(top={out_top.sum()} , bot={out_bot.sum()})")

# ---------------------------------------------------------------------
def describe1(split_path: Path, name: str, target_labels=None, bins=30):
    """
    General‐purpose dataset summary.
    • target_labels – list/tuple with one label per target column (optional)
    • bins          – #bins for a quick percentile histogram print-out
    """
    ds = torch.load(split_path, map_location='cpu')      # list[Data]
    n_graphs = len(ds)

    # ---------- 1) graph-size stats -----------------------------------
    n_nodes = np.array([d.natoms.item() for d in ds])
    print(f"\n{name} ({n_graphs:>5,d} graphs)")
    p50, p95 = np.percentile(n_nodes, [50, 95]).astype(int)
    print(f"  nodes/graph   : min={n_nodes.min():3d}  p50={p50:3d}  "
          f"p95={p95:3d}  max={n_nodes.max():3d}")

    bad_pos   = sum(d.pos.shape[0] != d.natoms for d in ds)
    print(f"  graphs w/ inconsistent pos <-> natoms : {bad_pos}")

    # ---------- 2) target statistics ----------------------------------
    y = torch.stack([d.y.view(-1)           # flatten even for (1,1)
                     for d in ds])          # shape (N, T)
    T = y.shape[1]
    if target_labels is None or len(target_labels) != T:
        # fall back to generic labels
        target_labels = [f"t{i}" for i in range(T)]

    print("  y  (raw units) :")
    mu, std = y.mean(0), y.std(0, unbiased=False)
    y_min, y_max = y.min(0)[0], y.max(0)[0]

    for i, lbl in enumerate(target_labels):
        print(f"     {lbl:<12} μ={mu[i]:6.3f}  σ={std[i]:6.3f}  "
              f"min={y_min[i]:6.3f}  max={y_max[i]:6.3f}")

    # ---------- 3) quick “shape” visual – percentile histogram --------
    edges = np.linspace(y_min.min(), y_max.max(), bins+1)
    hist  = np.histogram(y.numpy().ravel(), bins=edges)[0]
    cdf   = np.cumsum(hist) / hist.sum()
    mid   = 0.5*(edges[1:] + edges[:-1])
    print("  percentile bins:")
    for m, c in zip(mid, cdf):
        if c in {0.25,0.5,0.75,0.9,0.95,0.99}:          # print a few key ones
            print(f"     p{int(c*100):02d} ≈ {m:6.3f} eV")

    # ---------- 4) simple outlier scan --------------------------------
    size_outliers  = (n_nodes > np.percentile(n_nodes, 99)).sum()
    value_outliers = ( (y - mu) / (std + 1e-9) ).abs().gt(8).any(1).sum()
    print(f"  >99-percentile graphs (size outliers) : {size_outliers}")
    print(f"  |y| > 8 σ  (target outliers)          : {value_outliers}")
# ---------------------------------------------------------------------



if __name__ == "__main__":
    for p,lbl in [
        (Path("datasets/wfstructureid_train.pt"), "TRAIN"),
        (Path("datasets/wfstructureid_validation.pt"),   "VAL"),
        (Path("datasets/wfstructureid_test.pt"),  "TEST"),
    ]:
        describe(p, lbl)
