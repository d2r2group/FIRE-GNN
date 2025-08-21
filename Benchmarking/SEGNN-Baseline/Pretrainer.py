from Runner import Runner, SEED, PROJECT, NUM_WORKERS
from abc import ABCMeta
from argparse import ArgumentParser
import json, wandb, torch
from tqdm import tqdm
from Dataset.CrystalSO3Dataset import CrystalSO3Dataset
from Dataset.Normalization import Normalizer
from torch_geometric.loader import DataLoader
from pathlib import Path

class Pretrainer(Runner, metaclass=ABCMeta):
    """
    Acts as a generic class for pretraining a model and saving the graph_nn.
    """
    def __init__(self) -> None:
        """
        Initializes a runner with a given config. Takes arguments from commandline.

        Args:
            name: The name of the run for wandb
            config: The config to set up the model and run training with
        """
        args = ArgumentParser()
        args.add_argument("name", help="The name of the run", type=str)
        args.add_argument("config", help="The path to the config to run.", type=str)
        args.add_argument("--resume", help="The path to the model file to resume from", type=str)
        args = args.parse_args()

        with open(args.config) as f:
            config = json.load(f)

        self.config = {
            **config,
            "seed": SEED
        }
        
        wandb.login()
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        print("loading data")
        dataset = self.setup_datasets()
        num_atom_feats = dataset[0].x.shape[-1] 

        self.data_loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False,
            generator=torch.Generator().manual_seed(SEED)
        )
        print("loaded data")

        print("loading model")
        self.setup_model(self.config, num_atom_feats)
        if args.resume:
            print("Resuming")
            self.model.load_state_dict(torch.load(args.resume))
        print("loaded model")

        self.model = self.model.to(self.device)
        self.setup_optimizer()
        self.setup_scheduler()

        # Setup Normalizer for unnorming later
        with open(f"{self.default_data_path}/coef_stats.json") as f:
            self.normalizer = Normalizer(json.load(f), self.device)

        self.name = args.name
        self.best_logs = None
        self.best_epoch = None
        self.best_model_state_dict = None

    def main(self):
        """
        The main file which runs the training for a model.
        """
        tags = [self.config["model"], "pretraining", 
                "normalized" if self.config["normalize"] else "unnormalized"]
        with wandb.init(config=self.config, name=self.name, project=PROJECT, tags=tags):
            try:
                for epoch in tqdm(range(self.config["epochs"]), "Epoch"):
                    logs = {}
                    logs.update({"learning_rate": self.scheduler.get_last_lr()[0]})
                    logs.update(self.run_epoch(self.data_loader, train=True))
                    wandb.log(logs)
                    self.scheduler.step()
                    self.record_best(epoch, logs)
                    if epoch - self.best_epoch  > (1/8) * self.config["epochs"]:
                        # Early stopping if it doesn't improve after 1/8 of the number of epochs
                        break
            except KeyboardInterrupt as e:
                print(e)
            finally:
                # Save results
                self.log_best()
                Path("trained_models").mkdir(exist_ok=True)
                torch.save(self.best_model_state_dict, f"trained_models/{self.name.replace(' ', '_')}")
                self.model.load_state_dict(self.best_model_state_dict)
                torch.save(self.model.graph_nn.state_dict(), f"trained_models/{self.name.replace(' ', '_')}_graphnn")

    def setup_datasets(self) -> CrystalSO3Dataset:
        """
        Sets up the dataset for training

        Returns:
            The training dataset and the testing dataset
        """
        with open(f"{self.default_data_path}/train/data.json") as train_data:
            data = json.load(train_data)
        with open(f"{self.default_data_path}/validate/data.json") as validation_data:
            data.extend(json.load(validation_data))
        with open(f"{self.default_data_path}/test/data.json") as test_data:
            data.extend(json.load(test_data))
        with open(f"./Dataset/atom_init.json") as atom_init:
            atom_init = json.load(atom_init)
        return self.dataset_class(data, atom_init, self.config["lmax_h"], radius=self.config["radius"], 
                                   max_neighbors=self.config["max_neighbors"], normalize=self.config["normalize"])