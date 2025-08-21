import numpy as np
import pandas as pd
import json

def compute_target_stats(csv_file, out_path="./datasets/wf_bot_structureid_stats.npz"):
    df        = pd.read_csv(csv_file)
    mu    = float(df["WF_bottom"].mean())
    std   = float(df["WF_bottom"].std())
    np.savez(out_path, mu=mu, std=std)   # both scalars

    return mu, std

"""
def compute_target_stats(csv_file, out_path="./datasets/wf_element_stats.npz):
    df   = pd.read_csv(csv_file)
    mu   = df[["WF_bottom", "WF_top"]].mean().values.astype("float32")   # (2,)
    std  = df[["WF_bottom", "WF_top"]].std().values.astype("float32")    # (2,)

    np.savez(out_path, mu=mu, std=std)
    print(f"saved normalisation stats to {out_path}")
    print("μ :", mu)
    print("σ :", std)
    return mu, std
"""


    
if __name__ == "__main__":
    compute_target_stats("./datasets/WF-CE-splits.structureid-val_structureid-test_0.7-0.2-0.1.train.csv")


