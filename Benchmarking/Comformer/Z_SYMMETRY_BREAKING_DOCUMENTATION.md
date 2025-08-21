# Z-Symmetry Breaking for Surface Property Prediction

## Overview

This document describes the implementation of z-symmetry breaking in Comformer models for predicting surface properties like work function and cleavage energy. The approach addresses the fundamental challenge that surfaces break the 3D translational symmetry inherent in bulk crystals.

## Physics Motivation

### Why Break Z-Symmetry?

1. **Surface Properties are Directional**: Work function values differ between top and bottom surfaces of a slab
2. **Broken 3D Symmetry**: Surfaces inherently break the periodic symmetry along the z-axis
3. **SE(3) → SE(2) Transition**: Surface systems should preserve in-plane (x,y) symmetries while breaking z-reflection symmetry

### Target Properties

- **Work Function (WF)**: Energy required to remove an electron from the surface
  - WF_top ≠ WF_bottom for asymmetric surfaces
  - Directionally dependent
- **Cleavage Energy**: Energy required to cleave a crystal along a plane
  - Related to surface stability and formation

## Implementation Details

### Configuration Parameters

Both `iComformerConfig` and `eComformerConfig` now include:
```python
break_z_symmetry: bool = False  # Default: disabled for backward compatibility
```

### Architecture Changes

#### 1. Z-Coordinate Embedding Network
When `break_z_symmetry=True`, both models include a small neural network:
```python
self.z_embedding = nn.Sequential(
    nn.Linear(1, 16),
    nn.SiLU(),
    nn.Linear(16, config.node_features),  # Match node_features for direct addition
    nn.SiLU(),
)
```

#### 2. Early Z-Feature Injection
Z-coordinates are processed **before message passing** in the forward method:

```python
# Early z-symmetry breaking: inject surface information before message passing
if self.break_z_symmetry and hasattr(data, 'z_coords') and data.z_coords is not None:
    # Convert raw z-coordinates to surface-relative coordinates (frame-invariant)
    raw_z = data.z_coords.squeeze(1)  # Remove extra dimension
    slab_center = torch.mean(raw_z)  # Center of slab in z-direction
    relative_z = raw_z - slab_center  # Relative to slab center
    z_spread = torch.std(raw_z)  # Characteristic slab thickness
    
    # Avoid division by zero for perfectly flat structures
    if z_spread > 1e-6:
        normalized_z = relative_z / z_spread  # Normalized by thickness
    else:
        normalized_z = relative_z  # Keep relative coordinates if spread is tiny
    
    z_features = self.z_embedding(normalized_z.unsqueeze(1))
    node_features = node_features + z_features  # Direct addition to node features
```

### Surface-Relative Coordinate System

#### Problem with Raw Z-Coordinates
Raw z-coordinates are **frame-dependent**:
- Translating or rotating the slab changes absolute z-values
- Same physical structure → different coordinate values
- No intrinsic meaning for surface comparison

#### Solution: Surface-Relative Coordinates
1. **Center Calculation**: `slab_center = mean(z_coordinates)`
2. **Relative Positioning**: `relative_z = z - slab_center`
3. **Thickness Normalization**: `normalized_z = relative_z / std(z_coordinates)`

#### Physical Meaning
- **negative values**: atoms on "bottom" side of slab
- **positive values**: atoms on "top" side of slab  
- **zero**: atoms near middle of slab
- **magnitude**: relative distance from slab center (normalized by thickness)

### Data Flow

#### 1. Graph Construction (`graphs.py`)
```python
# Extract raw z-coordinates (unchanged - always available)
z_coords = torch.tensor(cart_coords[:, 2], dtype=torch.get_default_dtype()).unsqueeze(1)
g = Data(..., z_coords=z_coords)
```

#### 2. Model Processing
- **If `break_z_symmetry=False`**: z_coords ignored, model works normally
- **If `break_z_symmetry=True`**: 
  1. Convert to surface-relative coordinates
  2. Embed through z_embedding network
  3. Add to initial node features
  4. Proceed with message passing

#### 3. Message Passing Impact
- **Early injection**: All subsequent layers operate with z-directional information
- **Preserved in-plane symmetry**: Only z-direction is treated specially
- **Learned representations**: Model learns z-directional preferences from start

## Architectural Philosophy

### Why Early Injection (Before Message Passing)?

**Advantages:**
1. **Physical Consistency**: Surfaces break z-symmetry from the beginning
2. **Information Propagation**: Z-directionality informs all message passing layers
3. **Architectural Coherence**: All layers operate with same symmetry assumptions

**Previous Approach Issues:**
- Late injection (after message passing) created architectural inconsistency
- Model learned equivariant representations, then suddenly broke symmetry
- Led to overfitting and poor generalization

### SE(3) Equivariance Considerations

#### For iComformer (Invariant Model)
- Naturally handles asymmetry through data-driven learning
- Z-symmetry breaking provides directional guidance
- Less critical but still beneficial for surface properties

#### For eComformer (Equivariant Model)
- **Essential for surface properties**: SE(3) equivariance enforces symmetries incompatible with surfaces
- **Controlled breaking**: Only z-direction treated specially, preserves in-plane symmetries
- **Mathematical consistency**: Early injection maintains architectural coherence

## Usage Guidelines

### When to Enable Z-Symmetry Breaking

**Enable (`break_z_symmetry=True`) for:**
- Surface properties: work function, cleavage energy, surface energy
- Directional properties where top ≠ bottom
- eComformer models predicting surface properties

**Disable (`break_z_symmetry=False`) for:**
- Bulk properties: formation energy, bandgap, elastic moduli
- Properties with full 3D symmetry
- When backward compatibility is required

### Model Combinations

| Model | Property | break_z_symmetry | Rationale |
|-------|----------|------------------|-----------|
| iComformer | WF, cleavage_energy | True | Provides directional guidance |
| iComformer | WF, cleavage_energy | False | Baseline comparison |
| eComformer | WF, cleavage_energy | True | **Essential** - breaks incompatible SE(3) symmetry |
| eComformer | WF, cleavage_energy | False | Baseline (likely poor performance) |
| Both | Bulk properties | False | Maintains proper 3D symmetry |

## Implementation Notes

### Backward Compatibility
- Default `break_z_symmetry=False` ensures existing code works unchanged
- No performance impact when disabled
- z_embedding layers only created when needed

### Memory and Computation
- **Minimal overhead**: Small 16→node_features network
- **Early computation**: Surface-relative coordinates computed once per forward pass
- **Efficient**: Direct addition to node features (no concatenation)

### Error Handling
- **Missing z_coords**: Gracefully ignored, model works normally
- **Flat structures**: Division by zero protection for uniform z-coordinates
- **Device compatibility**: All operations preserve tensor device/dtype

### Debugging and Validation
- **Logging**: Models log when z-symmetry breaking is active
- **Inspection**: z_coords available in data objects for debugging
- **Gradual rollout**: Can test on subset of properties first

## Expected Results

### Performance Improvements
- **eComformer**: Significant improvement for surface properties with symmetry breaking
- **iComformer**: Moderate improvement, provides directional inductive bias
- **Reduced overfitting**: More physically consistent representations

### Failure Modes to Watch For
- **Frame dependence**: If surface-relative coordinates don't work, may need different normalization
- **Scale sensitivity**: Very thick/thin slabs might need adjusted normalization
- **Convergence issues**: Early z-injection might affect training dynamics

## Future Improvements

### Potential Enhancements
1. **Adaptive normalization**: Slab-specific thickness scaling
2. **Learned surface detection**: Automatic surface region identification
3. **Multi-scale features**: Different z-resolutions for different layers
4. **SE(2) equivariant layers**: True 2D+discrete symmetry implementation

### Research Directions
1. **Symmetry group analysis**: Formal mathematical treatment of broken symmetries
2. **Transfer learning**: Pretrain on bulk, fine-tune with z-breaking for surfaces
3. **Multi-property consistency**: Ensure WF_top + WF_bottom relationships are preserved

## Conclusion

This z-symmetry breaking implementation provides a minimally invasive, physically motivated approach to handle surface properties in Comformer models. The early injection strategy maintains architectural consistency while providing the directional information needed for accurate surface property prediction.

The approach is particularly critical for equivariant models where the inherent SE(3) symmetry constraints are incompatible with the directional nature of surface properties.