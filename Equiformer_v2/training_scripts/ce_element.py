import torch
import yaml
import logging
import wandb
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
# Load Preprocessed Data
train_data = torch.load('./datasets/elements_train1.pt')
val_data = torch.load('./datasets/elements_validation1.pt')
test_data = torch.load('./datasets/elements_test1.pt')
wandb.init(project="equiformer_v2_fine_tune",id="ceelement1",resume="allow")

# Add 'natoms' attribute to data samples
for dataset in [train_data, val_data, test_data]:
    for data in dataset:
        data.natoms = torch.tensor([data.x.shape[0]])

print(f"Data keys: {train_data[0].keys}")
print(f"Number of atoms: {train_data[0].natoms.item()}")
# Configure logging to output to both console and a file named 'evaluation.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),         # Console output
        logging.FileHandler("evaluation.log", mode='a')  # File output; 'w' mode to overwrite each run
    ]
)
logger = logging.getLogger(__name__)

# Define the custom collate function
def custom_collate_fn(batch):
    batch = Batch.from_data_list(batch)
    batch.cell = torch.stack([data.cell for data in batch.to_data_list()], dim=0)
    return batch

# Create DataLoaders
train_loader = DataLoader(
    train_data,
    batch_size=8,
    shuffle=True,
    collate_fn=custom_collate_fn
)
val_loader = DataLoader(
    val_data,
    batch_size=8,
    collate_fn=custom_collate_fn
)
test_loader = DataLoader(
    test_data,
    batch_size=8,
    collate_fn=custom_collate_fn
)

# Load Config File
config_path = './configs/equiformer_v2_N@8_L@4_M@2_31M.yml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EquiformerV2_OC20(
    num_atoms=None,
    bond_feat_dim=None,
    num_targets=1,
    max_radius=config['model']['max_radius'],
    regress_forces=False,
    num_layers=config['model']['num_layers'],
    sphere_channels=config['model']['sphere_channels'],
    attn_hidden_channels=config['model']['attn_hidden_channels'],
    num_heads=config['model']['num_heads'],
    attn_alpha_channels=config['model']['attn_alpha_channels'],
    attn_value_channels=config['model']['attn_value_channels'],
    ffn_hidden_channels=config['model']['ffn_hidden_channels'],
    norm_type=config['model']['norm_type'],
    lmax_list=config['model']['lmax_list'],
    mmax_list=config['model']['mmax_list'],
    edge_channels=config['model']['edge_channels'],
    alpha_drop=config['model']['alpha_drop'],
    drop_path_rate=config['model']['drop_path_rate'],
    proj_drop=config['model']['proj_drop']
).to(device)

# Load Checkpoint
checkpoint_path = "./checkpoints/element1.pt"
start_epoch = 1
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
previous_max_step=0

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logger.info("No optimizer state in checkpoint; starting fresh optimizer state.")
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint.get('epoch', 1) + 1
    logger.info("Resuming training from checkpoint...")
except FileNotFoundError:
    logging.info("Checkpoint not found. Starting fresh.")


# Loss function
loss_fn = torch.nn.L1Loss()  # Mean Absolute Error (L1 Loss)
gradient_accumulation_steps = 1

# Metric: Mean Absolute Error (MAE)
def calculate_mae(predictions, targets):
    """
    Compute Mean Absolute Error (MAE) between predictions and targets.
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()

# Metric: Mean Absolute Percentage Error (MAPE)
def calculate_mape(predictions, targets):
    """
    Compute Mean Absolute Percentage Error (MAPE) between predictions and targets.
    Avoid division by zero by adding a small epsilon.
    """
    epsilon = 1e-8  # To avoid division by zero
    mape = torch.mean(torch.abs((predictions - targets) / (targets + epsilon))) * 100
    return mape.item()

# Training Loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)

        # Mixed precision forward pass
        with autocast():
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)
            loss = loss_fn(outputs, target) / gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    avg_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch}, step=previous_max_step + epoch)
    logger.info(f"Epoch {epoch}: Training Loss = {avg_loss:.4f}")
    return avg_loss

# Validation Loop
def validate(epoch):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)

            all_predictions.append(outputs)
            all_targets.append(target)

            loss = torch.nn.L1Loss()(outputs, target)  # Use MAE loss instead of MSE
            total_loss += loss.item()

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    mae = calculate_mae(all_predictions, all_targets)
    mape = calculate_mape(all_predictions, all_targets)

    avg_loss = total_loss / len(val_loader)
    wandb.log({"val_loss": avg_loss, "val_mae": mae, "val_mape": mape}, step=previous_max_step + epoch)
    logger.info(f"Validation Loss = {avg_loss:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2f}%")
    return avg_loss, mae, mape


# Evaluate on Test Set
def test(epoch):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch).view(-1)
            target = batch.y.to(device).view(-1)

            all_predictions.append(outputs)
            all_targets.append(target)

            loss = torch.nn.L1Loss()(outputs, target)
            total_loss += loss.item()

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    mae = calculate_mae(all_predictions, all_targets)
    mape = calculate_mape(all_predictions, all_targets)

    avg_loss = total_loss / len(test_loader)
    wandb.log({"test_loss": avg_loss, "test_mae": mae, "test_mape": mape}, step=previous_max_step + epoch)
    logger.info(f"Test Loss = {avg_loss:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2f}%")
    return avg_loss, mae, mape

# Train and Validate
num_epochs = 600
best_val_loss = float('inf')

for epoch in range(start_epoch, num_epochs + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, val_mae, val_mape = validate(epoch)
    test_loss, test_mae, test_mape = test(epoch)
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch} — learning rate: {current_lr:.2e}")
    # Save checkpoint if validation improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch
        }, './checkpoints/element1.pt')
        logger.info(f"Saved best model at epoch {epoch} with validation loss {best_val_loss:.4f}")


