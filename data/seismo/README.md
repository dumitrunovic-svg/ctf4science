# Seismic Dataset

This dataset is generated using the Python library Instaseis, which produces synthetic seismograms from Green's function databases. These seismograms record waveforms of ground motion in response to an earthquake at a given source location. Instaseis can generate data in terms of displacement, velocity, or acceleration. In this dataset, we use velocity, which has units of m/s.

#### Table of Contents
1. [Stations](#stations-sensors)
2. [Sources](#sources)
3. [The Data](#the-data)
4. [Data Normalization](#data-normalization)
5. [Dataset Purpose](#dataset-purpose)
6. [Key Dataset Characteristics](#key-dataset-characteristics)
7. [Evaluation Tasks](#evaluation-tasks)
8. [Usage Notes](#usage-notes)

## Stations (sensors)

Stations (the virtual seismometers) are assumed to be on the Earth's surface. Their locations are recorded as x-y-z or latitude-longitude fields on a generic sphere with a 6371km radius. For each earthquake, each station records a 3-component waveform: Z (vertical), N (north-south horizontal), and E (east-west horizontal). We only take the Z component for simplicity.

## Sources

Earthquake events are generated with random values of location, magnitude, and other characteristics such as type of earthquake (normal-thrust vs strike-slip).

## The Data

For each earthquake, each station (sensor) records 1 time series of velocities in the Z (vertical) direction. A typical training set has the shape of (2000, 2048), corresponding to 2000 time-steps and 2048 sensors.

## Data Normalization

Waveform time series could differ by orders of magnitude due to 1. the log scale of earthquake magnitudes and 2. the reduction of wave amplitudes due to the distance between the station and the source. Within the time series, there are also large variations in the amplitude.

To mitigate this, we normalise waveform traces to unit variance and zero mean. The benefit is that neural network models can focus on waveform shape. The drawback of normalisation is that we lose the physical meaning of amplitude (e.g. difference between nearby and far stations, or between a magnitude 4 and magnitude 7 earthquake), but this is acceptable because the waveform shapes are the key characteristic of the time series that we want the models to learn and predict.

The normalisation is done for each individual time series, since, for instance, X1train and X1test need to be normalised by a common mean and variance to ensure continuity and predictability. We normalise time series per group, using the mean and variance of the whole group. The grouping is as follows:

- X1train, X1test
- X2train, X2test, X3test
- X3train, X4test, X5test
- X4train, X6test
- X5train, X7test
- X6train
- X7train
- X8train
- X9train, X8test
- X10train, X9test

## Dataset Purpose

This dataset is part of the **Common Task Framework (CTF) for Science**, providing standardized benchmarks for evaluating machine learning algorithms on scientific dynamical systems. The CTF addresses fundamental challenges including:

- **Short-term forecasting** (weather forecast): Predicting near-future states with trajectory accuracy
- **Long-term statistical forecasting** (climate forecast): Capturing statistical properties of long-time dynamics
- **Noisy data reconstruction**: Denoising and modeling from corrupted measurements
- **Limited data scenarios**: Learning from sparse observations
- **Parametric generalization**: Interpolation and extrapolation to new parameter regimes

## Key Dataset Characteristics

- **System Type**: Spatio-temporal seismic waveforms (surface stations + time)
- **Spatial Dimension**: 2,048 sensors (stations on Earth's surface)
- **Time Step**: Δt = 1.0
- **Behavior**: Seismic wave propagation from earthquake sources
- **Data Format**: NumPy compressed arrays (.npz)
- **Normalization**: Zero mean, unit variance per time series group
- **Evaluation Metrics**:
  - Short-term: Root Mean Square Error (RMSE)
  - Long-term: Spectral L2 Error with k=20, modes=100

## Evaluation Tasks

The dataset supports 12 evaluation metrics (E1-E12) organized into 4 main task categories:

**Test 1: Forecasting (E1, E2)**

- **Input**: X1train (2000 timesteps, 2048 sensors)
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

- **Input**: Three training trajectories (X6, X7, X8) at different earthquake configurations
- **Task**: Interpolate (E11) and extrapolate (E12) to new configurations
- **Burn-in**: X9train and X10train provide initialization (500 timesteps each)
- **Metrics**: Short-term RMSE on parameter generalization

## Usage Notes

- **Hidden Test Sets**: The actual test data (X1test through X9test) are hidden and used only for evaluation on the CTF leaderboard
- **Baseline Scores**: Use average value prediction as the baseline reference for long-term metrics
- **Score Range**: All scores are clipped to [-100, 100], where 100 represents perfect prediction
- **Data Continuity**: Start indices in YAML indicate temporal relationship between train/test splits
- **Normalization Groups**: Pay attention to the normalization groupings when processing data—train/test pairs share normalization statistics
- **File Format**: Data stored as .npz files in [time, sensors]
- **Sensor Locations**: Station coordinates are provided in `sensor_locations.csv`
