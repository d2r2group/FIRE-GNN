import pandas as pd
import json
from argparse import ArgumentParser

def get_important(df):
    data = [{
        "material": eval(d["slab"]),
        "cleavage_energy": d["cleavage_energy"],
        # "WF_bottom": d["WF_bottom"],
        # "WF_top": d["WF_top"]
    } for idx, d in df.iterrows()]
    return data

def save_data(data, dest):
    with open(f"{dest}/data.json", "wt+") as f:
        json.dump(data, f)

def main(path):
    test = pd.read_csv(f'WF-CE-Splits_v2/{path}/test.csv')
    train = pd.read_csv(f'WF-CE-Splits_v2/{path}/train.csv')
    validation = pd.read_csv(f'WF-CE-Splits_v2/{path}/validation.csv')
    test = get_important(test)
    train = get_important(train)
    validation = get_important(validation)
    save_data(test, 'cleavage_energy/test')
    save_data(train, 'cleavage_energy/train')
    save_data(validation, 'cleavage_energy/validate')

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('path', help='Path to the dataset')
    args = args.parse_args()
    main(args.path)