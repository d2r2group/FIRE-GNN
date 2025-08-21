from argparse import ArgumentParser
from pathlib import Path
from Models.CGCNN import CGCNN
from Models.CGCNNB import CGCNNB
from Datasets.SlabDataset import SlabDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb
import torch
import json
from ProjectParameters import PROJECT, SEED
from metrics import mean_absolute_percentage_error

def main(config: Path, save: Path, data: str, validation: bool, breaking: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    wandb.login()

    with config.open("r") as f:
        config = json.load(f)

    save.mkdir(exist_ok=True, parents=True)

    model = CGCNNB(
        1,
        h_dim=config["model"]["h_dim"],
        num_graph_layers=config["model"]["graph_layers"],
        num_fc_layers=config["model"]["fc_layers"],
    ) if breaking else CGCNN(
        1,
        h_dim=config["model"]["h_dim"],
        num_graph_layers=config["model"]["graph_layers"],
        num_fc_layers=config["model"]["fc_layers"],
    )
    model.to(device)
    if save.joinpath("model").exists():
        model.load_state_dict(torch.load(save.joinpath("model"), weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), config["lr"], weight_decay=config["weight_decay"])

    print("Processing Data")
    train_dataset = SlabDataset(data, "train")
    train_dataloader = DataLoader(train_dataset, config["batch_size"], shuffle=True)
    train_batch_size = len(train_dataloader)

    test_dataset = SlabDataset(data, "validation" if validation else "test")
    test_dataloader = DataLoader(test_dataset, config["batch_size"], shuffle=False)
    test_batch_size = len(test_dataloader)

    with wandb.init(config=config, name=f"{data}-ce" + ("-b" if breaking else ""), project=PROJECT):
        for _ in tqdm(range(config["epochs"]), desc="Epoch"):
            train_total_ce_mse_loss = 0
            train_total_ce_mae_loss = 0
            train_total_ce_mape_loss = 0
            model.train()
            for batch in tqdm(train_dataloader, leave=False, desc="Train Batch"):
                batch = batch.to(device)

                optimizer.zero_grad()

                cleavage_energy = model(batch)

                loss_ce = torch.nn.functional.mse_loss(cleavage_energy, batch.cleavage_energy)

                total_loss = loss_ce

                total_loss.backward()
                optimizer.step()

                # Record stats
                train_total_ce_mse_loss += loss_ce.item()
                train_total_ce_mae_loss += torch.nn.functional.l1_loss(cleavage_energy, batch.cleavage_energy).item()
                train_total_ce_mape_loss += mean_absolute_percentage_error(cleavage_energy, batch.cleavage_energy).item()

            test_total_ce_mse_loss = 0
            test_total_ce_mae_loss = 0
            test_total_ce_mape_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dataloader, leave=False, desc="Test Batch"):
                    batch = batch.to(device)

                    cleavage_energy = model(batch)

                    test_total_ce_mse_loss += torch.nn.functional.mse_loss(cleavage_energy, batch.cleavage_energy).item()
                    test_total_ce_mae_loss += torch.nn.functional.l1_loss(cleavage_energy, batch.cleavage_energy).item()
                    test_total_ce_mape_loss += mean_absolute_percentage_error(cleavage_energy, batch.cleavage_energy).item()

            torch.save(model.state_dict(), save.joinpath("model"))

            wandb.log({
                "Train/Cleavage Energy MAE": train_total_ce_mae_loss / train_batch_size,
                "Train/Cleavage Energy MSE": train_total_ce_mse_loss / train_batch_size,
                "Train/Cleavage Energy MAPE": train_total_ce_mape_loss / train_batch_size,
                "Test/Cleavage Energy MAE": test_total_ce_mae_loss / test_batch_size,
                "Test/Cleavage Energy MSE": test_total_ce_mse_loss / test_batch_size,
                "Test/Cleavage Energy MAPE": test_total_ce_mape_loss / test_batch_size
            })


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("config", help="The experiment config to run.", type=Path)
    args.add_argument("save", help="Where to save the model to", type=Path)
    args.add_argument("data", help="The data to use when running the model", choices=["elemental_test_v2", "ptgroup_test", "spacegroup_test", "structureid_test"])
    args.add_argument("--validation", help="Whether or not to run a validation instance", action="store_true")
    args.add_argument("--breaking", help="Whether or not to run a symmetry breaking instance", action="store_true")
    args = args.parse_args()
    main(args.config, args.save, args.data, args.validation, args.breaking)