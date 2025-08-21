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
        args.add_argument("save", help="Where to save the heatmap to.")
        args = args.parse_args()

        with open(args.config) as f:
            config = json.load(f)

        self.config = {
            **config,
            "seed": SEED,
            "validation": False,
            "save": args.save
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        _, test_dataset = self.setup_datasets(self.config)
        num_atom_feats = test_dataset[0].x.shape[-1] 

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

        with open(f"{self.default_data_path}/coef_stats.json") as f:
            self.normalizer = Normalizer(json.load(f), self.device)

    def main(self):
        """
        Evaluates a model over the test dataset
        """
        mae = self.evaluate_results()
        self.make_mae_heatmap(mae)

    @abstractmethod
    def evaluate_results(self):
        """
        Evaluate one run of the test data. 

        Returns:
            A tensor of errorwise components
        """
        raise NotImplementedError("Each subclass must implement evaluation on its own.")

    def make_mae_heatmap(self, mae: torch.Tensor):
        """
        Produces a heatmap of mae to show component wise error.

        Args:
            mae: MAE in the shape of the tensor.
        """
        assert len(mae.shape) == 2
        mae = mae.cpu().numpy()
        sns.heatmap(mae, annot=True, vmin=0, fmt=".2f")
        plt.savefig(self.config['save'])