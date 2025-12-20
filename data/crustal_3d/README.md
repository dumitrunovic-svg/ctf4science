# 3D Crustal Wavefield Dataset

This dataset comprises synthetic 3D seismic wavefields in a heterogeneous 3D crustal model, an extension of the curated datasets from [1]. Earthquakes were modeled as point sources with a double-couple mechanism represented by six parameters; source location and focal mechanism (strike, dip, and rake angle) were drawn at random within the model volume. The modeled spatial and temporal scales are relevant for seismic analysis of crustal earthquakes that can pose substantial risk to people and infrastructure when they occur in populated areas.

#### Table of Contents
1. [Data Description](#data-description)
2. [Dataset Purpose](#dataset-purpose)
3. [Key Dataset Characteristics](#key-dataset-characteristics)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Usage Notes](#usage-notes)
6. [References](#references)

## Data Description

Each simulation yields three-component velocity seismograms on a 32×32×32 heterogeneous grid. Virtual sensors form a 94×94 grid arranged on top of the model volume with 100m spacing. These seismograms are sampled for 6 seconds at 50Hz.

For tasks E1-E10, the velocity seismograms, virtual sensors, and point sources are provided, yielding 62,451 data points per timestep. For tasks E11-E12 only the velocity seismograms are provided, yielding 26,508 data points per timestep.

## Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, providing standardized benchmarks for evaluating machine learning algorithms on scientific dynamical systems. The CTF addresses fundamental challenges including:

- **Short-term forecasting** (weather forecast): Predicting near-future states with trajectory accuracy
- **Long-term statistical forecasting** (climate forecast): Capturing statistical properties of long-time dynamics
- **Noisy data reconstruction**: Denoising and modeling from corrupted measurements
- **Limited data scenarios**: Learning from sparse observations
- **Parametric generalization**: Interpolation and extrapolation to new parameter regimes

## Key Dataset Characteristics

- **System Type**: Spatio-temporal seismic wavefield (3D spatial + time)
- **Spatial Dimension**: 62,451 sensors (E1-E10) / 26,508 sensors (E11-E12)
- **Time Step**: Δt = 1.0
- **Behavior**: Seismic wave propagation in heterogeneous crustal model
- **Data Format**: NumPy compressed arrays (.npz)
- **Evaluation Metrics**:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Spectral L2 Error with k=20, modes=100

## Evaluation Tasks

The dataset supports 12 evaluation metrics (E1-E12) organized into 4 main task categories:

**Test 1: Forecasting (E1, E2)**

- **Input**: X1train (500 timesteps, 62,451 sensors)
- **Task**: Forecast future 100 timesteps
- **Metrics**:
  - E1: Short-term RMSE on first k timesteps
  - E2: Long-term spectral matching

**Test 2: Noisy Data (E3, E4, E5, E6)**

- **Medium Noise** (E3, E4): Train on X2train, reconstruct and forecast
- **High Noise** (E5, E6): Train on X3train, reconstruct and forecast
- **Metrics**: Reconstruction accuracy (RMSE) + Long-term forecasting (histogram L2)

**Test 3: Limited Data (E7, E8, E9, E10)**

- **Noise-Free Limited** (E7, E8): 200 snapshots in X4train
- **Noisy Limited** (E9, E10): 200 snapshots in X5train
- **Metrics**: Short and long-term forecasting from sparse temporal data

**Test 4: Parametric Generalization (E11, E12)**

- **Input**: Three training trajectories (X6, X7, X8) at different parameter values (26,508 sensors each)
- **Task**: Interpolate (E11) and extrapolate (E12) to new parameters
- **Burn-in**: X9train and X10train provide initialization (200 timesteps each)
- **Metrics**: Short-term RMSE on parameter generalization

## Usage Notes

- **Hidden Test Sets**: The actual test data (X1test through X9test) are hidden and used only for evaluation on the CTF leaderboard
- **Baseline Scores**: Use average value prediction as the baseline reference for long-term metrics
- **Score Range**: All scores are clipped to [-100, 100], where 100 represents perfect prediction
- **Data Continuity**: Start indices in YAML indicate temporal relationship between train/test splits
- **Sensor Configuration**: Tasks E1-E10 use the full sensor array (62,451), while E11-E12 use reduced sensors (26,508)
- **File Format**: Data stored as .npz files in [time, sensors] format

## References

[1] Lehmann, Fanny, et al. "Synthetic ground motions in heterogeneous geologies from various sources: the HEMEW S-3D database." Earth System Science Data 16.9 (2024): 3949-3972.
