from pathlib import Path
from typing import List, Union
import json
import numpy as np
from sklearn.model_selection import train_test_split
import warnings

ROOT = Path("Multi")
ROOT.mkdir(exist_ok=True)
DATA = ROOT.joinpath("../../../datasets/WF_CE_database_with_forces.json")
TRAIN = ROOT.joinpath("train")
VALIDATE = ROOT.joinpath("validate")
TEST = ROOT.joinpath("test")

def collect_stats(data: List[dict]):
    """
    Computes mean and stdev of the data spherical harmonic coefficients
    and saves them to a file

    Args:
        data: Data to compute mean and stdev of.
    """
    WF_top = [] ; WF_bottom = [] ; cleavage_energy = []

    for d in data:
        WF_top.append(d["WF_top"])
        WF_bottom.append(d["WF_bottom"])
        cleavage_energy.append(d["cleavage_energy"])

    WF = np.vstack((WF_top, WF_bottom))

    WF_mean = np.mean(WF)
    WF_std = np.std(WF)
    cleavage_energy_mean = np.mean(cleavage_energy)
    cleavage_energy_std = np.std(cleavage_energy)
    

    coef_stats = {
        "WF_std": np.mean(WF).tolist(),
        "WF_mean": np.std(WF).tolist(),
        "WF_top_std": np.mean(WF_top).tolist(),
        "WF_top_mean": np.std(WF_top).tolist(),
        "WF_bottom_std": np.mean(WF_bottom).tolist(),
        "WF_bottom_mean": np.std(WF_bottom).tolist(),
        "cleavage_energy_std": cleavage_energy_mean.tolist(),
        "cleavage_energy_mean": cleavage_energy_std.tolist(),
    }

    with ROOT.joinpath("coef_stats.json").open("wt+") as f:
        json.dump(coef_stats, f, indent=2)

def save_data(data: List[dict], dest: Union[str, Path]):
    """
    Saves the data to a certain destination writing down the material_ids

    Args:
        data: The data to save.
        dest: The folder to save the data to.
    """
    dest = Path(dest)
    dest.mkdir(exist_ok=True)
    material_ids = [d["material_id"] for d in data]
    useful_data = [
        {
            "material": d["material"], 
            "forces": d["forces"], 
            "WF_top": d["WF_top"],
            "WF_bottom": d["WF_bottom"],
            "cleavage_energy": d["cleavage_energy"]
        } 
    for d in data]
    with open(dest.joinpath("material_ids.csv"), "wt+") as f:
        f.write("\n".join(material_ids))
    with open(dest.joinpath("data.json"), "wt+") as f:
        json.dump(useful_data, f)

def get_by_ids(data: List[dict]):
    """
    Sorts the data into train, test, and validate by the IDs.

    Args:
        data: The data dictionary to sort

    Returns:
        The train, validation, and test sets by the pregotten ids.
    """
    train, validate, test = [], [], []
    train_ids = TRAIN.joinpath("material_ids.csv").read_text().split("\n") 
    validate_ids = VALIDATE.joinpath("material_ids.csv").read_text().split("\n") 
    test_ids = TEST.joinpath("material_ids.csv").read_text().split("\n") 
    for d in data:
        if d["material_id"] in train_ids:
            train.append(d)
        elif d["material_id"] in validate_ids:
            validate.append(d)
        elif d["material_id"] in test_ids:
            test.append(d)
        else:
            warnings.warn("Task ID not in used IDs found. Left unused.")
    return train, validate, test

def main():
    with open(DATA) as f:
        data = json.load(f)
    
    # Process to my fields
    data = [{
        "material_id": data[d]["mpid"],
        "material": eval(data[d]["slab"]),
        "forces": data[d]["forces"], 
        "WF_top": data[d]["WF_top"],
        "WF_bottom": data[d]["WF_bottom"],
        "cleavage_energy": data[d]["cleavage_energy"]
    } for d in data]
    

    print(f"Number of total usable datapoints: {len(data)}")

    if all([p.joinpath("material_ids.csv").exists() for p in [TRAIN, VALIDATE, TEST]]):
        print("Using preset IDs.")
        train, validate, test = get_by_ids(data)
    else:
        train, test_and_validate = train_test_split(data, test_size=0.3)
        test, validate = train_test_split(test_and_validate, test_size=0.5) 

    collect_stats(train + validate + test)

    print(f"Number Training Samples: {len(train)}\nNumber Validation Samples: {len(validate)}\nNumber Testing Samples: {len(test)}")

    save_data(train, TRAIN)
    save_data(validate, VALIDATE)
    save_data(test, TEST)

if __name__ == "__main__":
    main()