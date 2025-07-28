# Training Experiments for Predetermined Splits + Z-Symmetry Breaking

## Overview

This document outlines all the training experiments for testing:
1. **Predetermined splits** vs random splits
2. **Z-symmetry breaking** for surface properties
3. **Model comparison**: iComformer vs eComformer
4. **Multiple data splits**: elemental, space_group, structureid

## Experiment Matrix

### iComformer Experiments (6 total)
| Property | Data Split | Z-Symmetry | Script |
|----------|------------|------------|--------|
| WF | elemental | False | `train_iComformer_WF_elemental.py` |
| WF | space_group | False | `train_iComformer_WF_space_group.py` |
| WF | structureid | False | `train_iComformer_WF_structureid.py` |
| cleavage_energy | elemental | False | `train_iComformer_cleavage_energy_elemental.py` |
| cleavage_energy | space_group | False | `train_iComformer_cleavage_energy_space_group.py` |
| cleavage_energy | structureid | False | `train_iComformer_cleavage_energy_structureid.py` |

### eComformer Experiments (12 total)
| Property | Data Split | Z-Symmetry | Script |
|----------|------------|------------|--------|
| **WF - No Z-Symmetry Breaking (Baseline)** |
| WF | elemental | False | `train_eComformer_WF_elemental_no_zsym.py` |
| WF | space_group | False | `train_eComformer_WF_space_group_no_zsym.py` |
| WF | structureid | False | `train_eComformer_WF_structureid_no_zsym.py` |
| **WF - With Z-Symmetry Breaking** |
| WF | elemental | True | `train_eComformer_WF_elemental_zsym.py` |
| WF | space_group | True | `train_eComformer_WF_space_group_zsym.py` |
| WF | structureid | True | `train_eComformer_WF_structureid_zsym.py` |
| **Cleavage Energy - No Z-Symmetry Breaking** |
| cleavage_energy | elemental | False | `train_eComformer_cleavage_energy_elemental_no_zsym.py` |
| cleavage_energy | space_group | False | `train_eComformer_cleavage_energy_space_group_no_zsym.py` |
| cleavage_energy | structureid | False | `train_eComformer_cleavage_energy_structureid_no_zsym.py` |
| **Cleavage Energy - With Z-Symmetry Breaking** |
| cleavage_energy | elemental | True | `train_eComformer_cleavage_energy_elemental_zsym.py` |
| cleavage_energy | space_group | True | `train_eComformer_cleavage_energy_space_group_zsym.py` |
| cleavage_energy | structureid | True | `train_eComformer_cleavage_energy_structureid_zsym.py` |

## Data Sources

| Split Type | File Path |
|------------|-----------|
| Elemental | `/home/mudaliar.k/data/combined_elemental.csv` |
| Space Group | `/home/mudaliar.k/data/combined_space_group.csv` |
| Structure ID | `/home/mudaliar.k/data/combined_structureid.csv` |

## Training Configuration

All experiments use:
- **Epochs**: 350
- **Batch Size**: 64
- **Max Neighbors**: 25
- **Cutoff**: 4.0
- **Additional Features**: `use_lattice=True`, `use_angle=True`
- **Data Loader**: `save_dataloader=True` for efficiency

### Output Features
- **WF**: `output_features=2` (WF_bottom + WF_top)
- **Cleavage Energy**: `output_features=1` (single value)

## Expected Results Analysis

### 1. Predetermined Splits Impact
- **Hypothesis**: Better generalization than random splits
- **Measure**: Compare test MAE across different split types
- **Expected**: structure_id > space_group > elemental (increasing difficulty)

### 2. Z-Symmetry Breaking Impact
- **iComformer**: Minor improvement (provides directional guidance)
- **eComformer (No Z-Symmetry)**: Poor performance (SE(3) conflicts with surfaces)
- **eComformer (With Z-Symmetry)**: Significant improvement

### 3. Model Comparison
- **Work Function**: eComformer+zsym > iComformer > eComformer (no zsym)
- **Cleavage Energy**: Similar pattern expected

## Running Experiments

### Individual Experiment
```bash
cd /home/mudaliar.k/github/Comformer_uv
sbatch bash_training_scripts/training_split_scripts/run_iComformer_WF_elemental.sh
```

### Batch Submission (Example)
```bash
# Submit all iComformer experiments
for script in bash_training_scripts/training_split_scripts/run_iComformer*.sh; do
    sbatch "$script"
done

# Submit eComformer experiments with z-symmetry breaking
for script in bash_training_scripts/training_split_scripts/run_eComformer*zsym.sh; do
    sbatch "$script"
done
```

## Output Organization

Each experiment creates its own output directory:
```
output/
├── iComformer_WF_elemental_splits/
├── iComformer_WF_space_group_splits/
├── iComformer_WF_structureid_splits/
├── eComformer_WF_elemental_splits_no_zsym/
├── eComformer_WF_elemental_splits_zsym/
└── ... (all other experiments)
```

## Key Metrics to Compare

1. **Test MAE**: Primary performance metric
2. **Validation curves**: Check for overfitting
3. **Training time**: Efficiency comparison
4. **Split effectiveness**: Compare performance across split types
5. **Z-symmetry impact**: Compare with/without symmetry breaking

## Next Steps

1. **Submit initial experiments**: Start with one per category to verify setup
2. **Monitor performance**: Check first few experiments for issues
3. **Full batch submission**: Once validated, submit all experiments
4. **Results analysis**: Compare across all dimensions

This comprehensive experiment design will provide definitive evidence for both the predetermined splits feature and the z-symmetry breaking approach.