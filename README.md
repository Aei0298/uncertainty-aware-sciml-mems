# Monte Carlo Analysis for Information Engineering

## Uncertainty-Aware Scientific Machine Learning for System Identification and Equation Discovery in MEMS Gyroscope Dynamics

## Overview

This project investigates how the dynamics and physical parameters of a simplified MEMS gyroscope can be identified from uncertain and noisy sensor data.

The work is developed within the broader topic of **Monte Carlo Analysis for Information Engineering** and combines methods from:

- dynamical systems
- signal processing
- classical system identification
- equation discovery
- statistical and machine learning
- probabilistic modelling
- physics-informed machine learning
- causal analysis
- uncertainty quantification

The central scientific theme is:

> **Learning uncertain physical dynamics from sensor data.**

A controlled synthetic MEMS dynamical system is used so that the true governing equations and physical parameters are known in advance. This makes it possible to compare different modelling and identification approaches against a known ground truth and to systematically study their behaviour under uncertainty and measurement noise.

---

## Research Objectives

The project aims to:

1. Develop a simplified dynamical model representative of MEMS gyroscope behaviour.
2. Simulate sensor responses under varying physical and environmental conditions.
3. Introduce measurement noise, sensor bias, gain variation, and parameter drift.
4. Propagate uncertainty through the dynamical system using Monte Carlo simulation.
5. Extract relevant information from simulated sensor signals using signal-processing methods.
6. Estimate physical parameters using classical system-identification techniques.
7. Investigate equation discovery using **Sparse Identification of Nonlinear Dynamics (SINDy)**.
8. Investigate **symbolic regression** as an additional equation-discovery approach.
9. Compare classical, statistical, black-box, probabilistic, and physics-informed modelling methods.
10. Quantify predictive and parameter uncertainty.
11. Investigate selected causal relationships between environmental variables, parameter drift, and sensor response.
12. Evaluate the robustness of the different approaches through Monte Carlo analysis.

---

## Dynamical Model

The initial MEMS representation is based on a forced mass-spring-damper system:

\[
m\ddot{x}(t) + c\dot{x}(t) + kx(t) = F(t)
\]

where:

- \(m\) — effective mass
- \(c\) — damping coefficient
- \(k\) — stiffness
- \(F(t)\) — applied excitation
- \(x(t)\) — displacement response

The corresponding acceleration is obtained from:

\[
\ddot{x}(t)
=
\frac{F(t)-c\dot{x}(t)-kx(t)}{m}
\]

A nonlinear extension can additionally be considered:

\[
m\ddot{x}(t)
+
c\dot{x}(t)
+
kx(t)
+
\alpha x^3(t)
=
F(t)
\]

where:

- \(\alpha\) — nonlinear stiffness coefficient

The model is intentionally simplified. It is used as a **representative dynamical sensor system** rather than as a complete industrial MEMS gyroscope model.

The advantage of this controlled formulation is that the true physical parameters and governing dynamics remain known, allowing identification and learning methods to be evaluated quantitatively.

---

## Scientific Workflow

```text
Physical MEMS model
        │
        ▼
Excitation
        │
        ▼
Dynamic response
        │
        ▼
Sensor noise, bias and parameter drift
        │
        ▼
Signal processing
        │
        ▼
Classical system identification
        │
        ▼
Equation discovery
        │
        ├── SINDy
        └── Symbolic regression
        │
        ▼
Statistical and black-box machine learning
        │
        ▼
Probabilistic machine learning
        │
        ▼
Physics-informed / grey-box modelling
        │
        ▼
Causal analysis
        │
        ▼
Monte Carlo uncertainty quantification
```

---

## Methodological Components

### 1. Dynamical-System Simulation

The physical system is simulated numerically to generate displacement, velocity, and acceleration responses.

The model provides a controlled environment in which parameters such as

\[
m,\qquad c,\qquad k
\]

can be varied systematically.

---

### 2. Sensor Modelling

Synthetic sensor measurements are generated from the simulated mechanical response.

The measurement process can include:

- additive measurement noise
- sensor bias
- gain variation
- parameter drift
- environmental variability

This creates more realistic identification conditions than using ideal deterministic signals alone.

---

### 3. Signal Processing

Signal-processing techniques are used to extract dynamical information from the simulated sensor signals.

The analysis includes methods such as:

- time-domain signal analysis
- Fast Fourier Transform (FFT)
- power spectral analysis
- frequency estimation
- filtering
- feature extraction

These methods provide information about characteristic frequencies and dynamic behaviour before model identification is performed.

---

### 4. Classical System Identification

Classical identification methods are used to estimate physical system parameters from measured signals.

The estimated quantities can include:

\[
m,\qquad c,\qquad k
\]

and derived quantities such as the natural frequency:

\[
\omega_n = \sqrt{\frac{k}{m}}
\]

and damping characteristics.

The estimated parameters can then be compared directly with the known ground-truth values of the synthetic system.

---

### 5. Equation Discovery

Two data-driven equation-discovery approaches are investigated.

#### Sparse Identification of Nonlinear Dynamics — SINDy

SINDy searches for sparse governing equations from a library of candidate mathematical terms.

The objective is to recover equations resembling the underlying physical dynamics directly from simulated measurement data.

#### Symbolic Regression

Symbolic regression searches for analytical mathematical expressions that describe relationships in the data.

This provides an alternative approach for discovering interpretable equations without explicitly prescribing their final functional form.

---

### 6. Statistical and Black-Box Machine Learning

Data-driven regression models are investigated as alternative approaches for predicting system behaviour or estimating physical quantities.

These models provide a comparison between:

- physics-based modelling
- classical parameter estimation
- interpretable equation discovery
- purely data-driven prediction

---

### 7. Probabilistic Machine Learning

Probabilistic models are used to represent predictive uncertainty explicitly.

Instead of producing only a point estimate,

\[
\hat{y}
\]

a probabilistic model can estimate a predictive distribution:

\[
p(y \mid x)
\]

This allows uncertainty in predictions to be quantified directly.

---

### 8. Physics-Informed Machine Learning

Physics-informed approaches incorporate knowledge of the governing dynamical system into the learning process.

A generic physics-informed objective can be written as

\[
\mathcal{L}
=
\mathcal{L}_{\text{data}}
+
\lambda
\mathcal{L}_{\text{physics}}
\]

where:

- \(\mathcal{L}_{\text{data}}\) measures agreement with observations
- \(\mathcal{L}_{\text{physics}}\) penalizes violation of the governing physical equations
- \(\lambda\) controls the relative contribution of the physics constraint

The goal is to investigate whether physical knowledge improves robustness when measurements are noisy or uncertain.

---

### 9. Causal Analysis

Selected causal relationships are investigated between variables such as:

- environmental conditions
- parameter drift
- stiffness variation
- sensor measurements
- predicted system response

The aim is to distinguish predictive relationships from physically meaningful intervention effects where appropriate.

---

### 10. Monte Carlo Uncertainty Quantification

Monte Carlo simulation is used as the main uncertainty-analysis framework.

For uncertain
