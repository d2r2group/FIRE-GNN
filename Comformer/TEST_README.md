# Z-Symmetry Breaking Tests

This directory contains test scripts to verify that the z-symmetry breaking feature is correctly implemented and has the expected effect on model predictions.

## Test Scripts

### Basic Test: `test_zsymbreak.py`

This script performs the following tests:

1. Creates a synthetic test structure with atoms at different z-coordinates
2. Verifies that z-coordinates are correctly extracted during graph creation
3. Creates a model with z-symmetry breaking enabled and another without
4. Traces through the forward pass to verify z-coordinates are accessed and used
5. Confirms that model outputs differ between the two configurations

**Usage:**
```bash
python test_zsymbreak.py
```

**Expected output:** The script will output a series of checks, with ✅ indicating success and ❌ indicating failure. All checks should pass if the z-symmetry breaking feature is correctly implemented.

### Comparative Test: `compare_zsymbreak.py`

This script performs a more detailed comparison using actual data:

1. Loads a small sample of real data (5 points by default)
2. Creates two identical models, one with z-symmetry breaking and one without
3. Compares model predictions to identify differences
4. Analyzes whether z-symmetry breaking improves predictions
5. Visualizes z-coordinate embeddings to see what the model learns

**Usage:**
```bash
python compare_zsymbreak.py
```

**Expected output:** The script will output a comparison of model predictions and squared errors, showing whether z-symmetry breaking improves predictions. It also creates a visualization of the z-coordinate embeddings saved as `z_embedding_visualization.png`.

## How Z-Symmetry Breaking Should Work

When correctly implemented, the z-symmetry breaking feature should:

1. Extract z-coordinates from atomic positions in `graphs.py`
2. Add this z-coordinate data to the graph representation
3. Process z-coordinates through a dedicated embedding network in the model
4. Combine z-coordinate features with regular node features after message passing
5. Alter the model's output to account for the z-dimension information

The test scripts verify each of these steps.

## Interpreting Test Results

### If tests pass:
- Z-symmetry breaking is correctly implemented at the code level
- The feature is actively affecting model predictions
- You should see different outputs when the feature is enabled vs. disabled

### If tests show prediction improvements:
- Z-symmetry breaking is helping the model make better predictions
- The effect should be stronger for structures with greater z-coordinate variation
- This indicates the feature is providing valuable information to the model

### If tests fail:
- Check error messages to identify the specific issue
- Verify the implementation of z-coordinate extraction in `graphs.py`
- Check the model's handling of z-coordinates in the forward pass

## Running on the Cluster

These test scripts are designed to run locally on a small sample of data. If you want to run them on the cluster:

1. Copy the scripts to your cluster environment
2. Adjust any data paths in the scripts to point to your cluster data locations
3. Run the scripts with the appropriate Python environment:

```bash
# On cluster
module load anaconda3/2024.06
module load cuda/12.1
conda activate comformer_uv
python test_zsymbreak.py
```

## Full Training Comparison

To fully validate the z-symmetry breaking feature, you should compare full training runs:

1. Train a model with z-symmetry breaking enabled
2. Train an identical model without z-symmetry breaking
3. Compare final validation metrics and test performance

The existing training scripts `train_D2R2_WF.py` and `train_D2R2_WF_zsymbreak.py` can be used for this comparison.