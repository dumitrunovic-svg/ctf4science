# Ocean DAS Dataset

This dataset uses a novel geophysical sensing technology called DAS (Distributed Acoustic Sensing) [1]. DAS uses fiber optic cables to measure strain changes along their length, effectively turning the fiber into an array of thousands of sensors. This dataset comprises data from a shallow offshore DAS installation about 30m below sea level where surface gravity waves are particularly dispersive. The data is sampled at 5Hz but low-pass filtered to 1Hz.

#### Table of Contents
1. [DAS Technology](#das-technology)
2. [Dataset Purpose](#dataset-purpose)
3. [Key Dataset Characteristics](#key-dataset-characteristics)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Usage Notes](#usage-notes)
6. [References](#references)

## DAS Technology

Distributed Acoustic Sensing (DAS) converts fiber optic cables into dense arrays of seismic sensors by measuring the backscattered light from laser pulses. Each point along the fiber acts as an independent strain sensor, enabling continuous monitoring of ground motion with unprecedented spatial resolution. This technology has applications in earthquake early warning, infrastructure monitoring, and ocean observation.

## Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, providing standardized benchmarks for evaluating machine learning algorithms on scientific dynamical systems. The CTF addresses fundamental challenges including:

- **Short-term forecasting** (weather forecast): Predicting near-future states with trajectory accuracy
- **Long-term statistical forecasting** (climate forecast): Capturing statistical properties of long-time dynamics
- **Noisy data reconstruction**: Denoising and modeling from corrupted measurements
- **Limited data scenarios**: Learning from sparse observations
- **Parametric generalization**: Interpolation and extrapolation to new parameter regimes

## Key Dataset Characteristics

- **System Type**: Spatio-temporal acoustic/strain measurements (1D spatial + time)
- **Spatial Dimension**: 3,000 sensors (virtual sensors along fiber optic cable)
- **Time Step**: Δt = 1.0
- **Data Format**: NumPy compressed arrays (.npz)
- **Evaluation Metrics**:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Spectral L2 Error with k=20, modes=100

## Evaluation Tasks

The dataset supports 12 evaluation metrics (E1-E12) organized into 4 main task categories:

**Test 1: Forecasting (E1, E2)**

- **Input**: X1train (2000 timesteps, 3000 sensors)
- **Task**: Forecast future 1000 timesteps
- **Metrics**:
  - E1: Short-term RMSE on first k timesteps
  - E2: Long-term spectral matching

**Test 2: Noisy Data (E3, E4, E5, E6)**

- **Medium Noise** (E3, E4): Train on X2train, reconstruct and forecast
- **High Noise** (E5, E6): Train on X3train, reconstruct and forecast
- **Metrics**: Reconstruction accuracy (RMSE) + Long-term forecasting (histogram L2)

**Test 3: Limited Data (E7, E8, E9, E10)**

- **Noise-Free Limited** (E7, E8): 500 snapshots in X4train
- **Noisy Limited** (E9, E10): 500 snapshots in X5train
- **Metrics**: Short and long-term forecasting from sparse temporal data

**Test 4: Parametric Generalization (E11, E12)**

- **Input**: Three training trajectories (X6, X7, X8) at different conditions
- **Task**: Interpolate (E11) and extrapolate (E12) to new conditions
- **Burn-in**: X9train and X10train provide initialization (500 timesteps each)
- **Metrics**: Short-term RMSE on parameter generalization

## Usage Notes

- **Hidden Test Sets**: The actual test data (X1test through X9test) are hidden and used only for evaluation on the CTF leaderboard
- **Baseline Scores**: Use average value prediction as the baseline reference for long-term metrics
- **Score Range**: All scores are clipped to [-100, 100], where 100 represents perfect prediction
- **Data Continuity**: Start indices in YAML indicate temporal relationship between train/test splits
- **File Format**: Data stored as .npz files in [time, sensors] format

## References

[1] Yin, Jiuxun, et al. "Real‐data testing of distributed acoustic sensing for offshore earthquake early warning." The Seismic Record 3.4 (2023): 269-277.
