from pathlib import Path
import pandas as pd
import json
from argparse import ArgumentParser

def get_important(df):
    data = [{
        "material": eval(d["slab"]),
        "cleavage_energy": d["cleavage_energy"],
        "WF_top": d["WF_top"],
        "WF_bottom": d["WF_bottom"]
    } for idx, d in df.iterrows()]
    return data

def save_data(data, dest):
    dp = Path(dest)
    dp.mkdir(exist_ok=True)
    with open(f"{dest}/data.json", "wt+") as f:
        json.dump(data, f)

def main(path):
    test = pd.read_csv(f'../../datasets/{path}/test.csv')
    train = pd.read_csv(f'../../datasets/{path}/train.csv')
    validation = pd.read_csv(f'../../datasets/{path}/validation.csv')
    test = get_important(test)
    train = get_important(train)
    validation = get_important(validation)
    
    dp = Path(path)
    dp.mkdir(exist_ok=True)
    save_data(test, f'{path}/test')
    save_data(train, f'{path}/train')
    save_data(validation, f'{path}/validate')

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('path', help='Path to the dataset')
    args = args.parse_args()
    main(args.path)