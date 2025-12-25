from Evaluator import Evaluator
from typing import List
from Dataset.MultiDataset import MultiDataset
from e3nn.o3 import Irreps
import torch
from tqdm import tqdm
from torch.nn.functional import l1_loss, mse_loss
from metrics import mean_absolute_percentage_error

class WFEvaluator(Evaluator):
    @property
    def default_data_path(self) -> str:
        return "Dataset/data"
    
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> MultiDataset:
        return MultiDataset(data, atom_init, lmax_h, radius, max_neighbors, normalize)

    def main(self):
        total_wf_top_mae = 0
        total_wf_top_mse = 0
        total_wf_top_mape = 0
        
        total_wf_bot_mae = 0
        total_wf_bot_mse = 0
        total_wf_bot_mape = 0
        
        total_ce_mae = 0
        total_ce_mse = 0
        total_ce_mape = 0
        
        num_batches = len(self.test_loader)

        with torch.inference_mode():
            for data in tqdm(self.test_loader, f"test Batch", leave=False):
                data = data.to(self.device)
                
                pred = self.model(data)[0]
                wf_top_pred = pred[:,0]
                wf_bot_pred = pred[:,1]
                ce_pred = pred[:,2]
                
                wf_top_true = data.y[:,0]
                wf_bot_true = data.y[:,1]
                ce_true = data.y[:,2]
                
                
                # # Scale loss if normalized
                # if self.config["normalize"]:
                #     wf_top_pred, wf_bot_pred = self.unnormalize(wf_top_pred, wf_bot_pred)
                #     wf_top_true, wf_bot_true = self.unnormalize(data.wf_top, data.wf_bot)
                # else:
                #     wf_top_true = data.wf_top
                #     wf_bot_true = data.wf_bot

                # Deal with accuracy
                total_wf_top_mae += l1_loss(wf_top_pred, wf_top_true).item()
                total_wf_top_mse += mse_loss(wf_top_pred, wf_top_true).item()
                total_wf_top_mape += mean_absolute_percentage_error(wf_top_pred, wf_top_true).item()
                
                total_wf_bot_mae += l1_loss(wf_bot_pred, wf_bot_true).item()
                total_wf_bot_mse += mse_loss(wf_bot_pred, wf_bot_true).item()
                total_wf_bot_mape += mean_absolute_percentage_error(wf_bot_pred, wf_bot_true).item()
                
                total_ce_mae += l1_loss(ce_pred, ce_true).item()
                total_ce_mse += mse_loss(ce_pred, ce_true).item()
                total_ce_mape += mean_absolute_percentage_error(ce_pred, ce_true).item()
            
        print({
            f"test/WF Top MAE": total_wf_top_mae / num_batches,
            f"test/WF Top RMSE": (total_wf_top_mse / num_batches)**(1/2),
            f"test/WF Top MAPE": total_wf_top_mape / num_batches,
            f"test/WF Bottom MAE": total_wf_top_mae / num_batches,
            f"test/WF Bottom RMSE": (total_wf_top_mse / num_batches)**(1/2),
            f"test/WF Bottom MAPE": total_wf_top_mape / num_batches,
            f"test/CE MAE": total_ce_mae / num_batches,
            f"test/CE RMSE": (total_ce_mse / num_batches)**(1/2),
            f"test/CE MAPE": total_ce_mape / num_batches
        })

    @property
    def output_irreps_lst(self) -> List[Irreps]:
        return [Irreps("3x0e")]

if __name__ == "__main__":
    eval = WFEvaluator()
    eval.main()