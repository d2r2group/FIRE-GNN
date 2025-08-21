import torch
from abc import ABCMeta, abstractmethod
import torch
from multiprocessing import cpu_count
import wandb
import torch.optim as optim
from torch_geometric.loader import DataLoader
from ModelExecutor import ModelExecutor
import warnings
from tqdm import tqdm
from argparse import ArgumentParser
import json
from Dataset.Normalization import Normalizer
from pathlib import Path

SEED = 13
NUM_WORKERS = min(cpu_count() - 1, 16)
PROJECT = "Work Function Prediction"

class Runner(ModelExecutor, metaclass=ABCMeta):
    """
    Acts as a generic class for training and evaluating SEGNN models.
    """
    def __init__(self) -> None:
        """
        Initializes a runner with a given config. Takes arguments from commandline.

        Args:
            name: The name of the run for wandb
            config: The config to set up the model and run training with
            validation: Whether or not to do a validation run.
            graph_weights_from_pretrained: Graph weights to load the model with.
            lock_weights: Whether or not to freeze the graph weights from pretrained.
        """
        args = ArgumentParser()
        args.add_argument("name", help="The name of the run", type=str)
        args.add_argument("config", help="The path to the config to run.", type=str)
        args.add_argument("--pretrained", help="The path to the pretrained weights", type=str)
        args.add_argument("--validation", help="If this is a validation run and not a testing run", action="store_true")
        args.add_argument("--lock_weights", help="If the pretrained graph layers should not be retrained", action="store_true")
        args.add_argument("--resume", help="Model to resume training from.", type=str)
        args = args.parse_args()

        with open(args.config) as f:
            config = json.load(f)

        self.config = {
            **config,
            "seed": SEED,
            "pretrained": args.pretrained is not None,
            "validation": args.validation,
            "lock_weights": args.lock_weights
        }

        if args.pretrained is None and args.lock_weights:
            warnings.warn("Can't restrict finetuning if not providing a pretrained model, defaulting to training graph_nn.")

        if args.resume and args.pretrained:
            raise Exception("Can't get model weights from two locations, only pass in resume or pretrain, not both.")
        
        wandb.login()
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True

        print("loading data")
        train_dataset, validation_dataset = self.setup_datasets(self.config)
        num_atom_feats = train_dataset[0].x.shape[-1] 

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False,
            generator=torch.Generator().manual_seed(SEED)
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=NUM_WORKERS,
            drop_last=False,
            pin_memory=False
        )
        print("loaded data")

        print("loading model")
        self.setup_model(self.config, num_atom_feats)
        if args.pretrained:
            print("Loading pretrained weights")
            self.model.graph_nn.load_state_dict(torch.load(args.pretrained))
            if args.lock_weights:
                print("Locking weights")
                for param in self.model.graph_nn.parameters():
                    param.requires_grad = False
        elif args.resume:
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
        tags = [self.config["model"], "validation" if self.config["validation"] else "testing", 
                "normalized" if self.config["normalize"] else "unnormalized"]
        with wandb.init(config=self.config, name=self.name, project=PROJECT, tags=tags):
            try:
                for epoch in tqdm(range(self.config["epochs"]), "Epoch"):
                    logs = {}
                    logs.update({"learning_rate": self.scheduler.get_last_lr()[0]})
                    logs.update(self.run_epoch(self.train_loader, train=True))
                    logs.update(self.run_epoch(self.validation_loader, train=False))
                    wandb.log(logs)
                    self.scheduler.step()
                    self.record_best(epoch, logs)
            except KeyboardInterrupt as e:
                print(e)
            finally:    
                # Save results
                self.log_best()
                Path("trained_models").mkdir(exist_ok=True)
                torch.save(self.best_model_state_dict, f"trained_models/{self.name.replace(' ', '_')}")
                self.model.load_state_dict(self.best_model_state_dict)
                torch.save(self.model.graph_nn.state_dict(), f"trained_models/{self.name.replace(' ', '_')}_graphnn")

    @abstractmethod
    def run_epoch(self, dataloader: DataLoader, train: bool) -> dict:
        """
        Runs a single epoch 

        Args:
            dataloader: The dataloader to use
            train: Whether or not to run this as a training loop.
        """
        raise NotImplementedError("run_epoch must be implemented in subclasses.")
    
    @abstractmethod
    def record_best(self, epoch: int, logs: dict):
        """
        Records best result and model to be logged later

        Args:
            epoch: The epoch number that these logs are associated with
            logs: The logs to check if they are the best
        """
        raise NotImplementedError("record_best must be set up in subclasses.")
    
    def log_best(self):
        """
        Logs best results to wandb
        """
        for k, v in self.best_logs.items():
            wandb.run.summary[f"Best/{k}"] = v
        wandb.run.summary["Best/Epoch"] = self.best_epoch
    
    def setup_optimizer(self):
        """
        Sets up an optimizer for a model based on the config
        """
        if self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"], 
                                       momentum=self.config["momentum"], weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], 
                                        weight_decay=self.config["weight_decay"])
        elif self.config["optimizer"] == "adamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], 
                                        weight_decay=self.config["weight_decay"])
        else:
            raise Exception("Not a legal optimizer")

    def setup_scheduler(self):
        """
        Sets up an optimizer scheduler based on the config
        """
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.5, milestones=[0.5*self.config["epochs"], 0.8*self.config["epochs"]])