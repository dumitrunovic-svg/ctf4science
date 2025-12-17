# Global Sea Surface Temperature (SST) Dataset

This dataset contains **Global Sea Surface Temperature (SST)** data from NASA's Group for High Resolution Sea Surface Temperature (GHRSST) product. SST data exhibits complex, multiscale features of turbulent flows with intermittent events and quasi-periodic behavior, making it a challenging benchmark for forecasting, reconstruction, and prediction tasks. Unlike the synthetic KS and Lorenz datasets, this represents real-world geophysical observations, providing a critical testbed for evaluating data-driven methods on actual scientific data.

#### Table of Contents
1. [Dataset Description](#description)
2. [Loading Data](#loading)
3. [Memory Management Strategies](#memory_management)
4. [Evaluation Tasks](#evaluation)
5. [Physical Characteristics](#characteristics)
6. [Usage Notes](#usage)
7. [Relevant Sources](#sources)

---
<a name="description" />

## Dataset Description

### Sea Surface Temperature Data

The dataset is derived from the **Naval Oceanographic Office (NAVO) GHRSST Level 4 K10_SST version 1.0** product, which provides:

- **Daily global analyzed SST** at a reference depth of 1 meter
- **Data available from**: January 9, 2019 onward
- **Original format**: netCDF-4 following GHRSST Data Specification (GDS) 2.0
- **Multi-instrument fusion**: Incorporates observations from:
  - Advanced Very High Resolution Radiometer (AVHRR) aboard MetOp-A, MetOp-B, and NOAA-19 satellites
  - Visible Infrared Imaging Radiometer Suite (VIIRS) aboard Suomi-NPP satellite
  - Spinning Enhanced Visible and InfraRed Imager (SEVIRI) aboard Meteosat-8 and Meteosat-11 satellites

The objective of GHRSST is to provide the best quality SST data for applications across short, medium, and decadal/climate time scales through international collaboration and scientific innovation. All GHRSST data products are publicly available through NASA's Physical Oceanography Distributed Active Archive Center (PO.DAAC).

### Dataset Construction

For this CTF benchmark, **undisclosed spatial-temporal patches** have been extracted from the public GHRSST data:

- Spatial extent: Extracted patches at undisclosed latitude and longitude coordinates
- Spatial resolution: 200 × 200 grid (nx = 200, ny = 200)
- Total spatial dimension: n = nx × ny = 40,000 points per timestep
- Sampling rate (Δt) and spatial resolution (Δx, Δy): Intentionally not specified to prevent direct use of public data in algorithm development

⚠️ **Note**: The spatial dimension listed in the YAML (90,601) represents a flattened spatial grid that includes additional processing or masking of the 200×200 grid.

### Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, designed to provide standardized, rigorous benchmarks for evaluating machine learning algorithms on real-world scientific problems. The SST dataset addresses key challenges including:

- **Short-term forecasting** (weather forecast): Predicting near-future states with trajectory accuracy
- **Long-term forecasting** (climate forecast): Capturing statistical properties of long-time dynamics
- **Noisy data reconstruction**: Denoising and forecasting from corrupted measurements
- **Limited data scenarios**: Learning from sparse temporal observations
- **Parametric generalization**: Transferring models to different spatial regions or time periods

### Key Dataset Characteristics

- **System Type**: Spatio-temporal (real-world geophysical observations)
- **Spatial Dimension**: 90,601 grid points (processed from 200×200 spatial patches)
- **Time Step**: Δt = 1.0 (normalized time units; actual temporal resolution undisclosed)
- **Behavior**: Complex turbulent flows with multiscale features, intermittent events, and quasi-periodic patterns
- **Data Format**: Available in both NumPy (.npy) and CSV formats
- **Evaluation Metrics**:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Power Spectral Density matching with k=20, modes=100
- **Data Source**: NASA GHRSST via PO.DAAC (spatial-temporal patches with undisclosed coordinates)

<a name="loading" />

## Loading Data

### Python Example (NumPy)

```python
import numpy as np

# Load training data
X1_train = np.load('SST/npy/train/X1train.npy')
print(f"Shape: {X1_train.shape}")  # (800, 90601)
print(f"Time steps: {X1_train.shape[0]}")
print(f"Spatial points: {X1_train.shape[1]}")
print(f"Temperature range: [{X1_train.min():.2f}, {X1_train.max():.2f}]")

# Memory-mapped loading for large files (doesn't load full array into RAM)
X1_train_mmap = np.load('SST/npy/train/X1train.npy', mmap_mode='r')

# Note: Test data files are not included in the public dataset
# Generate your predictions and submit to the CTF4Science platform
```

### Python Example (CSV)

```python
import numpy as np
import pandas as pd

# Load training data from CSV
X1_train = np.loadtxt('SST/csv/train/X1train.csv', delimiter=',')
print(f"Shape: {X1_train.shape}")  # (800, 90601)

# Load timesteps
timesteps = np.loadtxt('SST/csv/train/X1train_timesteps.csv')
print(f"Time range: [{timesteps[0]:.1f}, {timesteps[-1]:.1f}]")

# Load with pandas for easier handling
df = pd.read_csv('SST/csv/train/X1train.csv', header=None)
```

<a name="memory_management" />

## Memory Management Strategies

Given the large data size, consider these approaches:

```python
import numpy as np

# 1. Memory-mapped loading (doesn't load full array into RAM)

X1_train = np.load('SST/npy/train/X1train.npy', mmap_mode='r')

# 2. Load specific time window

X1_subset = X1_train[0:100, :] # First 100 timesteps

# 3. Load specific spatial region

X1_spatial = X1_train[:, 0:10000] # First 10,000 spatial points

# 4. Batch processing for training

batch_size = 50
for i in range(0, X1_train.shape[0], batch_size):
batch = X1_train[i:i+batch_size, :]
# Process batch

# 5. Use float32 instead of float64 if precision allows

X1_train_f32 = X1_train.astype(np.float32) # Halves memory usage
```

<a name="evaluation" />

## Evaluation Tasks

The dataset supports 12 evaluation metrics (E1-E12) organized into 4 main task categories:

### Test 1: Forecasting (E1, E2)

Input: X1train (800 × 90,601)
Task: Forecast future 400 timesteps
Metrics:
E1: Short-term RMSE on first k timesteps
E2: Long-term spectral matching on power spectral density

### Test 2: Noisy Data (E3, E4, E5, E6)

Medium Noise (E3, E4): Train on X2train, reconstruct and forecast
High Noise (E5, E6): Train on X3train, reconstruct and forecast
Metrics: Reconstruction accuracy (RMSE) + Long-term forecasting (spectral)

### Test 3: Limited Data (E7, E8, E9, E10)

Noise-Free Limited (E7, E8): 100 snapshots in X4train
Noisy Limited (E9, E10): 100 snapshots in X5train
Metrics: Short and long-term forecasting from sparse temporal data

### Test 4: Parametric Generalization (E11, E12)

Input: Three training trajectories (X6, X7, X8) at different spatial/temporal conditions
Task: Interpolate (E11) and extrapolate (E12) to new conditions
Burn-in: X9train and X10train provide initialization (100 timesteps each)
Metrics: Short-term RMSE on spatial/temporal generalization

<a name="characteristics" />

## Physical Characteristics

### Turbulent Flow Features

SST data exhibits characteristics typical of geophysical turbulent flows:

- Multiscale Features: Energy cascades across spatial and temporal scales
- Intermittent Events: Sudden temperature changes due to oceanic eddies, fronts, and upwelling
- Quasi-Periodic Behavior: Seasonal cycles, tidal influences, and mesoscale variability
- Coherent Structures: Ocean eddies, fronts, and filaments with characteristic length scales
- Non-Gaussian Statistics: Heavy-tailed distributions in temperature gradients

These features make SST forecasting particularly challenging compared to smooth synthetic systems.

### Data Scale and Memory Requirements

- Large matrices: 800 × 90,601 ≈ 72.5 million data points per file
- Small matrices: 100 × 90,601 ≈ 9.1 million data points per file
- Memory footprint:
  - Single large file (float64): ~580 MB in memory
  - Single small file (float64): ~72 MB in memory
  - Full dataset: Several gigabytes

### Spatial Structure

The data represents a flattened spatial field:

- Original grid: 200 × 200 = 40,000 points
- Processed dimension: 90,601 points (may include land masking or additional processing)
- Spatial resolution: Δx, Δy (undisclosed)
- Geographic coordinates: Undisclosed to prevent external data usage

<a name="usage" />

## Usage Notes

1. Test Data Withheld: Test data files (.npy) are not included in the public dataset. Only test timesteps are provided. Submit predictions to the CTF4Science platform for evaluation.
2. File Organization: Training data is located in SST/npy/train/ (NumPy format) and SST/csv/train/ (CSV format)
3. Hidden Spatial-Temporal Details: Sampling rate (Δt) and spatial resolution (Δx, Δy) are intentionally undisclosed to prevent use of external GHRSST data
4. Large Data: Files are significantly larger than KS/Lorenz due to high spatial resolution (90,601 points)
5. Real-World Complexity: Expect irregular patterns, missing data effects, and realistic noise characteristics
6. Baseline Scores: Use constant zero prediction as the baseline reference (E_i = 0)
7. Score Range: All scores are clipped to [-100, 100], where 100 represents perfect prediction
8. Data Continuity: Start indices in YAML indicate temporal relationship between train/test splits
9. Memory Management: Consider memory-mapped arrays or batch processing for large files
10. Multiscale Features: Methods must handle energy across multiple spatial and temporal scales

<a name="sources" />

## Relevant Sources

- GitHub Repository: CTF-for-Science/ctf4science
- NASA PO.DAAC: https://podaac.jpl.nasa.gov/
- GHRSST Project: https://www.ghrsst.org/
