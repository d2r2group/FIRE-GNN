import torch
import numpy as np
import pandas as pd
import yaml
import logging
import wandb
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from nets.equiformer_v2.wf_equiformer_v2_oc20 import EquiformerV2_OC20
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
# Load Preprocessed Data
train_data = torch.load('./datasets/wfstructureid_train.pt')
val_data = torch.load('./datasets/wfstructureid_validation.pt')
test_data = torch.load('./datasets/wfstructureid_test.pt')
wandb.init(project="equiformer_v2_fine_tune",id="wfstructureid",resume="allow")
wandb.define_metric("epoch")
wandb.define_metric("*", step_metric="epoch")
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

def custom_collate_fn(data_list):
    # Create a Batch object from the original list of Data objects
    batch_obj = Batch.from_data_list(data_list)
    # Explicitly stack the 'y' attribute from each original Data object
    batch_obj.y = torch.stack([data.y for data in data_list], dim=0)
    # Similarly, stack the 'cell' attribute if needed
    batch_obj.cell = torch.stack([data.cell for data in data_list], dim=0)
    return batch_obj


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

stats = np.load("./datasets/wf_structureid_stats.npz")
mu  = torch.tensor(stats["mu"],  device=device)
std = torch.tensor(stats["std"], device=device)
model = EquiformerV2_OC20(
    num_atoms=None,
    bond_feat_dim=None,
    num_targets=2,
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
checkpoint_path = "./checkpoints/wfstructureid.pt"
start_epoch = 1
scaler = GradScaler()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
previous_max_step=0

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()

    # 3) Filter out any weights in the checkpoint that
    #    either don’t exist in your model or whose shape doesn’t match
    filtered_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_dict[k] = v
        # else: we skip it

    # 4) Overwrite those keys in your model’s state_dict
    model_dict.update(filtered_dict)

    # 5) Load back—in strict mode now, since we've already dropped the bad keys
    model.load_state_dict(model_dict) 
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
       
    logger.info("Resuming training from checkpoint...")
except (FileNotFoundError, KeyError):
    logging.info("Checkpoint not found. Starting fresh.")
# 3) inject initial_lr into each param_group
for group in optimizer.param_groups:
    # set initial_lr to whatever lr is currently in the group
    group.setdefault('initial_lr', group['lr'])
scheduler = StepLR(
    optimizer,
    step_size=50,
    gamma=0.5,
)
# Loss function
loss_fn = torch.nn.L1Loss()  # Mean Absolute Error (L1 Loss)
gradient_accumulation_steps = 1

# Metric: Mean Absolute Error (MAE)
def calculate_mae(predictions, targets):
    # Transpose predictions if necessary to match target shape
    if predictions.dim() > 1 and predictions.shape != targets.shape:
        predictions = predictions.transpose(0, 1)
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()

# Metric: Mean Absolute Percentage Error (MAPE)
def calculate_mape(predictions, targets):
    """
    Compute Mean Absolute Percentage Error (MAPE) between predictions and targets.
    Avoid division by zero by adding a small epsilon.
    """
    # Ensure predictions and targets have matching shapes
    if predictions.shape != targets.shape:
        predictions = predictions.transpose(0, 1)
    epsilon = 1e-8  # To avoid division by zero
    mape = torch.mean(torch.abs((predictions - targets) / (targets + epsilon))) * 100
    return mape.item()
# Training Loop
def train_one_epoch(epoch):
    model.train()
    total_loss = 0
    all_preds, all_targs = [], []
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        batch = batch.to(device)

        # Mixed precision forward pass
        with autocast():
            # Remove .view(-1) so outputs remain shape [batch_size, 2]
            outputs = model(batch)  
            # Keep target shape as [batch_size, 2]
            target = batch.y.to(device)
            if step % 20 == 0:
                # 1) compute L1 *for this batch only*
                loss_bot = (outputs[:,0] - target[:,0]).abs().mean()
                loss_top = (outputs[:,1] - target[:,1]).abs().mean()

                # 2) get gradient norm from *bot* head
                model.zero_grad(set_to_none=True)
                loss_bot.backward(retain_graph=True)
                g_bot = torch.stack([p.grad.abs().mean()
                                     for p in model.parameters()
                                     if p.grad is not None]).mean()

                # 3) now the *top* head
                model.zero_grad(set_to_none=True)
                loss_top.backward(retain_graph=True)
                g_top = torch.stack([p.grad.abs().mean()
                                     for p in model.parameters()
                                     if p.grad is not None]).mean()

                model.zero_grad(set_to_none=True)  # clean before main loss

                ratio = (g_top / g_bot).item()
                print(f"[epoch {epoch}  step {step}] "
                      f"grad‖ top / bot = {ratio:6.2f}")
                wandb.log({"grad_ratio_top_bot": ratio, "epoch": epoch})
            # ─── quick sanity-print ──────────────────────────────────────────────
            pred, targ = outputs.detach().cpu(), target.cpu()      # move to CPU so .item() etc. are safe
            print("─ batch debug ─")
            for i in range(min(5, pred.shape[0])):                 # show first 5 rows
                print(f" bottom  pred/targ : {pred[i,0]:6.3f} / {targ[i,0]:6.3f}   "
                    f"top pred/targ : {pred[i,1]:6.3f} / {targ[i,1]:6.3f}")
            print("──────────────────────────────────────────────────────────────")
 
            loss = loss_fn(outputs, target) / gradient_accumulation_steps
        all_preds.append(outputs.detach().cpu())
        all_targs.append(targets.detach().cpu())
        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps

    all_preds = torch.cat(all_preds, dim=0)  # (N,2) in norm space
    all_targs = torch.cat(all_targs, dim=0)  # (N,2) in norm space
    mu_b  = mu.view(1,2)                     # (1,2)
    std_b = std.view(1,2)

    pred_eV = all_preds * std_b + mu_b
    targ_eV = all_targs * std_b + mu_b

    # ——— compute MAE / MAPE per head + average ———
    mae_per_target  = (pred_eV - targ_eV).abs().mean(0)          # (2,)
    mae_bot, mae_top = mae_per_target.tolist()
    mae_avg = 0.5 * (mae_bot + mae_top)

    eps = 1e-8
    mape_per_target = ((pred_eV - targ_eV).abs() / (targ_eV.abs() + eps)).mean(0) * 100
    mape_bot, mape_top = mape_per_target.tolist()
    mape_avg = 0.5 * (mape_bot + mape_top)

    # ——— log everything ———
    avg_loss = total_loss / len(train_loader)
    wandb.log({
        "epoch":       epoch,
        "train_loss":  avg_loss,
        "mae_top":     mae_top,
        "mae_bottom":  mae_bot,
        "mae_avg":     mae_avg,
        "mape_top":    mape_top,
        "mape_bottom": mape_bot,
        "mape_avg":    mape_avg,
    })

    logger.info(
        f"Train Loss={avg_loss:.4f} | "
        f"MAE(top/bot/avg)={mae_top:.4f}/{mae_bot:.4f}/{mae_avg:.4f} | "
        f"MAPE(top/bot/avg)={mape_top:.2f}%/{mape_bot:.2f}%/{mape_avg:.2f}%"
    )

    return avg_loss, mae_avg, mape_avg

# Validation loop with per target MAE/MAPE + their average
def validate(epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_targs = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch)           # [B,2]
            targs = batch.y.to(device)     # [B,2]

            # Standard L1 loss over both targets
            loss = loss_fn(preds, targs)  
            total_loss += loss.item()

            all_preds.append(preds)
            all_targs.append(targs)

    
        # ------------------------------------------------------------
    # concatenate and *de-normalise* for human-readable metrics
    # ------------------------------------------------------------
    all_preds  = torch.cat(all_preds,  dim=0)            # (N,2) norm
    all_targs  = torch.cat(all_targs,  dim=0)            # (N,2) norm

    mu_b  = mu.view(1, 2)                                # (1,2)  broadcast
    std_b = std.view(1, 2)

    pred_eV = all_preds * std_b + mu_b                   # back to eV
    targ_eV = all_targs * std_b + mu_b

    # ---------- MAE / MAPE in physical units ----------
    mae_per_target  = (pred_eV - targ_eV).abs().mean(0)          # (2,)
    mae_bot, mae_top = mae_per_target.tolist()
    mae_avg = 0.5 * (mae_top + mae_bot)

    eps = 1e-8
    mape_per_target = ((pred_eV - targ_eV).abs() / (targ_eV.abs() + eps)).mean(0) * 100
    mape_bot, mape_top = mape_per_target.tolist()
    mape_avg = 0.5 * (mape_top + mape_bot)

    avg_loss = total_loss / len(val_loader)
    wandb.log({
        "epoch": epoch,
        "val_loss":    avg_loss,
        "mae_top":     mae_top,
        "mae_bottom":  mae_bot,
        "mae_avg":     mae_avg,
        "mape_top":    mape_top,
        "mape_bottom": mape_bot,
        "mape_avg":    mape_avg,
    })

    logger.info(
        f"Val Loss={avg_loss:.4f} | "
        f"MAE(top/bot/avg)={mae_top:.4f}/{mae_bot:.4f}/{mae_avg:.4f} | "
        f"MAPE(top/bot/avg)={mape_top:.2f}%/{mape_bot:.2f}%/{mape_avg:.2f}%"
    )
    return avg_loss, mae_avg, mape_avg

# 3) Test loop
def test(epoch):
    model.eval()
    total_loss = 0.0
    all_preds, all_targs = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds = model(batch)        # [B,2]
            targs = batch.y.to(device)  # [B,2]

            loss = loss_fn(preds, targs)
            total_loss += loss.item()

            all_preds.append(preds)
            all_targs.append(targs)

    all_preds  = torch.cat(all_preds,  dim=0)            # (N,2) norm
    all_targs  = torch.cat(all_targs,  dim=0)            # (N,2) norm

    mu_b  = mu.view(1, 2)                                # (1,2)  broadcast
    std_b = std.view(1, 2)

    pred_eV = all_preds * std_b + mu_b                   # back to eV
    targ_eV = all_targs * std_b + mu_b

    # ---------- MAE / MAPE in physical units ----------
    mae_per_target  = (pred_eV - targ_eV).abs().mean(0)          # (2,)
    mae_bot, mae_top = mae_per_target.tolist()
    mae_avg = 0.5 * (mae_top + mae_bot)

    eps = 1e-8
    mape_per_target = ((pred_eV - targ_eV).abs() / (targ_eV.abs() + eps)).mean(0) * 100
    mape_bot, mape_top = mape_per_target.tolist()
    mape_avg = 0.5 * (mape_top + mape_bot)

    # ---------- log ----------
    avg_loss = total_loss / len(test_loader)
    wandb.log({
        "epoch": epoch,
        "test_loss":    avg_loss,
        "test_mae_top": mae_top,
        "test_mae_bot": mae_bot,
        "test_mae_avg": mae_avg,
        "test_mape_top": mape_top,
        "test_mape_bot": mape_bot,
        "test_mape_avg": mape_avg,
    })

    logger.info(
        f"Test Loss={avg_loss:.4f} | "
        f"MAE={mae_top:.4f}/{mae_bot:.4f}/{mae_avg:.4f} | "
        f"MAPE={mape_top:.2f}%/{mape_bot:.2f}%/{mape_avg:.2f}%"
    )
    return avg_loss, mae_avg, mape_avg

# Train and Validate Loop remains the same
num_epochs = 600
best_val_loss = float('inf')
for epoch in range(start_epoch, num_epochs + 1):
    train_loss = train_one_epoch(epoch)
    val_loss, val_mae, val_mape = validate(epoch)
    test_loss, test_mae, test_mape = test(epoch)
    scheduler.step()
    # fetch and print/log the current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch} — learning rate: {current_lr:.2e}")     # goes to .out
    logger.info(f"Epoch {epoch} — learning rate: {current_lr:.2e}")  # goes to .err
    # Save checkpoint if validation improves
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'epoch': epoch
        }, './checkpoints/wfstructureid.pt')
        logger.info(f"Saved best model at epoch {epoch} with validation loss {best_val_loss:.4f}")

