import torch.utils
import torch.utils.data
from Runner import SEED, NUM_WORKERS
from ModelExecutor import ModelExecutor
from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
import json
import torch
from torch_geometric.loader import DataLoader
from Dataset.Normalization import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator(ModelExecutor, metaclass=ABCMeta):
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
        args.add_argument("--all", action="store_true", help="Use all the data rather than the test set.")
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

        train_dataset, test_dataset = self.setup_datasets(self.config)
        num_atom_feats = test_dataset[0].x.shape[-1]

        if args.all:
            test_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=NUM_WORKERS,
                drop_last=False,
                pin_memory=False
            )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False
        )

        self.setup_model(self.config, num_atom_feats)
        self.model.load_state_dict(torch.load(args.model))
        self.model = self.model.to(self.device)
        self.model.eval()

        with open(f"{self.default_data_path}/coef_stats.json") as f:
            self.normalizer = Normalizer(json.load(f), self.device)

    @abstractmethod
    def main(self):
        """
        Evaluates a model over the test dataset
        """
        raise NotImplementedError("Fill this in with what you want to evaluate")