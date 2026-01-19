from Runner import SEED, NUM_WORKERS
from ModelExecutor import ModelExecutor
from argparse import ArgumentParser
import json
import torch
from torch_geometric.loader import DataLoader
from Dataset.CustomMultiDataset import CustomMultiDataset
from e3nn.o3 import Irreps
from tqdm import tqdm 
from pathlib import Path
from typing import List
import json

class CustomEvaluator(ModelExecutor):
    """
    Represents a class which evaluates a model on test data to 
    get specific graphical results.
    """
    def __init__(self) -> None:
        """
        Initializes an evaluator.
        """
        args = ArgumentParser()
        args.add_argument("model", help="The model weight to evaluate", type=str)
        args.add_argument("config", help="The configuration file that model was built on")
        args.add_argument("slabs", help="The file where all the slabs are stored as cifs, can be a cif file, json list of cifs, or folder of cifs", type=Path)
        args = args.parse_args()

        with open(args.config) as f:
            config = json.load(f)

        self.config = {
            **config,
            "seed": SEED,
            "validation": False
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        with open(f"./Dataset/{config["atom_init"]}.json") as atom_init:
            atom_init = json.load(atom_init)
        dataset = CustomMultiDataset(args.slabs)
        num_atom_feats = dataset[0].x.shape[-1] 

        self.dataloader = DataLoader(
            dataset, 
            batch_size=config["batch_size"], 
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False
        )

        self.setup_model(self.config, num_atom_feats,config["lmax_h"], radius=config["radius"], 
                        max_neighbors=config["max_neighbors"], normalize=config["normalize"])
        self.model.load_state_dict(torch.load(args.model))
        self.model = self.model.to(self.device)
        self.model.eval()

        # with open(f"{self.default_data_path}/coef_stats.json") as f:
        #     self.normalizer = Normalizer(json.load(f), self.device)

    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("3x0e")]
    
    @property
    def default_data_path(self) -> str:
        return "Dataset/data"

    def main(self):
        """
        Evaluates a model over the dataset
        """
        predictions = []
        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"Batch", leave=False):
                data = data.to(self.device)
                
                pred = self.model(data)[0]
                predictions.append(pred)
        predictions = torch.cat(predictions, dim=0)
        with open("results.json", "wt+") as f:
            json.dump({
                "wf_top": predictions[:, 0].cpu().numpy().tolist(),
                "wf_bot": predictions[:, 1].cpu().numpy().tolist(),
                "ce": predictions[:, 2].cpu().numpy().tolist()
            })
                

if __name__ == "__main__":
    evaluator = CustomEvaluator()
    evaluator.main()