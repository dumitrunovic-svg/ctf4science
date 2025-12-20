# Lorenz System Dataset

This dataset contains numerical simulations of the **Lorenz system**, one of the most influential and widely-studied dynamical systems in history. The Lorenz equations exhibit chaotic behavior and are a canonical benchmark for evaluating data-driven algorithms in dynamical systems modeling, forecasting, and control.

_Note: This version of the dataset **does not** contain the test matrices. Please refer to `ODE_Lorenz` for an equivalent set of data that **does** contain test matrices._

#### Table of Contents
1. [The Lorenz Equations](#the-lorenz-equations)
2. [Dataset Purpose](#dataset-purpose)
3. [Key Dataset Characteristics](#key-dataset-characteristics)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Long-Term Evaluation Metric](#long-term-evaluation-metric-histogram-comparison)
6. [Chaos and Lyapunov Time](#chaos-and-lyapunov-time)
7. [Usage Notes](#usage-notes)

## The Lorenz Equations

The Lorenz system is defined by three coupled ordinary differential equations (ODEs):

```
dx/dt = σ(y - x)
dy/dt = rx - xz - y
dz/dt = xy - bz
```

where:

- `x, y, z` are the three state variables
- `σ = 10` (Prandtl number)
- `b = 8/3` (geometric factor)
- `r = 28` (Rayleigh number, at which the system exhibits chaotic behavior)

The system produces the famous "butterfly attractor" in 3D phase space, characterized by sensitive dependence on initial conditions and bounded aperiodic trajectories.

## Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, providing standardized benchmarks for evaluating machine learning algorithms on scientific dynamical systems. The CTF addresses fundamental challenges including:

- **Short-term forecasting**: Predicting near-future trajectories within Lyapunov time
- **Long-term statistical forecasting**: Capturing the probability distribution of states over long horizons
- **Noisy data reconstructi**on**: Denoising and modeling from corrupted measurements
- **Limited data scenarios**: Learning dynamics from sparse temporal observations
- **Parametric generalization**: Transferring learned models to different parameter regimes

## Key Dataset Characteristics

- System Type: Low-dimensional chaotic ODE (3 state variables)
- Spatial Dimension: 3 (x, y, z coordinates)
- Time Step: Δt = 0.05
- Behavior: Chaotic trajectories with sensitivity to initial conditions
- Data Format: Available in both MATLAB (.mat) and CSV formats
- Evaluation Metrics:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Histogram L2 error comparing state distributions (bins=41)

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

## Long-Term Evaluation Metric (Histogram Comparison)

Unlike our <ins>KS dataset</ins> which uses power spectral density, the Lorenz system uses **histogram-based distribution matching** for long-term forecasting evaluation:

- **Bins**: 41 bins for each state variable
- **Method**: Compute histograms of x, y, z over the last k timesteps
- **Error**: L1 norm difference between predicted and true histograms, averaged over x, y, z
- **Rationale**: Beyond the Lyapunov time (~3 time units), exact trajectory matching is impossible due to chaos. Instead, we evaluate whether the predicted trajectory explores the same regions of phase space with the correct statistical distribution.

## Chaos and Lyapunov Time

The Lorenz system is **chaotic**, meaning:

- Small differences in initial conditions grow exponentially
- Long-term exact trajectory prediction is fundamentally impossible
- The **Lyapunov time** is approximately 1.1 time units (or ~22 integration steps)
- After ~3 Lyapunov times, trajectory divergence is complete
- Therefore, long-term evaluation focuses on statistical properties, not trajectory matching

## Usage Notes

- **Hidden Test Sets**: The actual test data (X1test through X9test) are hidden and used only for evaluation on the CTF leaderboard
- **Baseline Scores**: Use average value prediction as the baseline reference for long-term metrics
- **Score Range**: All scores are clipped to [-100, 100], where 100 represents perfect prediction
- **Data Continuity**: Start indices in YAML indicate temporal relationship between train/test splits
- **Chaotic Divergence**: Trajectory-level matching is only meaningful for short-term forecasting (E1, E3, E5, E7, E9, E11, E12)
- **Statistical Matching**: Long-term metrics (E2, E4, E6, E8, E10) evaluate attractor statistics, not trajectories
- **File Formats**: Choose .mat for MATLAB/Python (scipy) workflows or .csv for language-agnostic access
