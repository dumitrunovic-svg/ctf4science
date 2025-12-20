# Kuramoto-Sivashinsky Dataset

This dataset contains numerical simulations of the **Kuramoto-Sivashinsky (KS) equation**, a fourth-order nonlinear partial differential equation (PDE) that exhibits spatio-temporal chaos. The KS equation is a canonical example used in scientific machine learning to benchmark data-driven algorithms for dynamical systems modeling, forecasting, and reconstruction.

#### Table of Contents
1. [The Kuramoto-Sivashinsky Equation](#the-kuramoto-sivashinsky-equation)
2. [Dataset Purpose](#dataset-purpose)
3. [Key Dataset Characteristics](#key-dataset-characteristics)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Usage Notes](#usage-notes)

## The Kuramoto-Sivashinsky Equation

The KS equation is defined as:

```
u_t + uu_x + u_xx + μ u_xxxx = 0
```

where:

- `u(x,t)` is the solution on a spatial domain `x ∈ [0, 32π]` with periodic boundary conditions
- `μ` is a parameter controlling the fourth-order diffusion term
- The equation exhibits spatio-temporal chaotic behavior, making it particularly challenging for forecasting algorithms


## Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, providing standardized benchmarks for evaluating machine learning algorithms on scientific dynamical systems. The CTF addresses fundamental challenges including:

- **Short-term forecasting** (weather forecast): Predicting near-future states with trajectory accuracy
- **Long-term statistical forecasting** (climate forecast): Capturing statistical properties of long-time dynamics
- **Noisy data reconstructi**on**: Denoising and modeling from corrupted measurements
- **Limited data scenarios**: Learning from sparse observations
- **Parametric generalization**: Interpolation and extrapolation to new parameter regimes

## Key Dataset Characteristics

- **System Type**: Spatio-temporal PDE (1D spatial + time)
- **Spatial Dimension**: 1024 grid points across domain [0, 32π]
- **Time Step**: Δt = 0.025
- **Behavior**: Chaotic spatio-temporal dynamics
- **Data Format**: Available in both MATLAB (.mat) and CSV formats
- **Evaluation Metrics**:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Power Spectral Density matching with k=20, modes=100

## Evaluation Tasks

The dataset supports 12 evaluation metrics (E1-E12) organized into 4 main task categories:

**Test 1: Forecasting (E1, E2)**

-  **Input**: X1train
-  **Task**: Forecast future 1000 timesteps
-  **Metrics**:
  - E1: Short-term RMSE on first k timesteps
  - E2: Long-term histogram matching of state distributions (x, y, z separately)

**Test 2: Noisy Data (E3, E4, E5, E6)**

- **Medium Noise** (E3, E4): Train on X2train, reconstruct and forecast
- **High Noise** (E5, E6): Train on X3train, reconstruct and forecast
- **Metrics**: Reconstruction accuracy (RMSE) + Long-term forecasting (histogram L2)

**Test 3: Limited Data (E7, E8, E9, E10)**

- **Noise-Free Limited** (E7, E8): 100 snapshots in X4train
- **Noisy Limited** (E9, E10): 100 snapshots in X5train
- **Metrics**: Short and long-term forecasting from sparse temporal data

**Test 4: Parametric Generalization (E11, E12)**

- **Input**: Three training trajectories (X6, X7, X8) at different parameter values
- **Task**: Interpolate (E11) and extrapolate (E12) to new parameters
- **Burn-in**: X9train and X10train provide initialization (100 timesteps each)
- **Metrics**: Short-term RMSE on parameter generalization

## Usage Notes

- **Hidden Test Sets**: The actual test data (X1test through X9test) are hidden and used only for evaluation on the CTF leaderboard
- **Baseline Scores**: Use average value prediction as the baseline reference for long-term metrics
- **Score Range**: All scores are clipped to [-100, 100], where 100 represents perfect prediction
- **Data Continuity**: Start indices in YAML indicate temporal relationship between train/test splits
- **Chaotic Divergence**: Trajectory-level matching is only meaningful for short-term forecasting (E1, E3, E5, E7, E9, E11, E12)
- **Statistical Matching**: Long-term metrics (E2, E4, E6, E8, E10) evaluate attractor statistics, not trajectories
- **File Formats**: Choose .mat for MATLAB/Python (scipy) workflows or .csv for language-agnostic access
