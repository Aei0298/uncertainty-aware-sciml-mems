# Monte Carlo Analysis for Information Engineering

## Uncertainty-Aware Scientific Machine Learning for System Identification and Equation Discovery in MEMS Gyroscope Dynamics

## Overview

This project investigates how the dynamics and physical parameters of a simplified MEMS gyroscope can be identified from uncertain and noisy sensor data.

The work is developed within the broader topic of **Monte Carlo Analysis for Information Engineering** and combines methods from:

- Dynamical systems
- Signal processing
- Classical system identification
- Equation discovery
- Statistical and machine learning
- Probabilistic modelling
- Physics-informed machine learning
- Causal analysis
- Uncertainty quantification

The central scientific theme is:

> **Learning uncertain physical dynamics from sensor data.**

A controlled synthetic MEMS dynamical system is used so that the true governing equations and physical parameters are known in advance.

This makes it possible to compare different modelling and identification approaches against a known ground truth and to systematically investigate their behaviour under uncertainty, parameter variability, and measurement noise.

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

# Dynamical Model

The initial MEMS representation is based on a forced mass-spring-damper system:

$$
m\ddot{x}(t) + c\dot{x}(t) + kx(t) = F(t)
$$

where:

- $m$ — effective mass
- $c$ — damping coefficient
- $k$ — stiffness
- $F(t)$ — applied excitation
- $x(t)$ — displacement response

The corresponding acceleration is obtained from:

$$
\ddot{x}(t)
=
\frac{F(t)-c\dot{x}(t)-kx(t)}{m}
$$

A nonlinear extension can additionally be considered:

$$
m\ddot{x}(t)
+
c\dot{x}(t)
+
kx(t)
+
\alpha x^3(t)
=
F(t)
$$

where:

- $\alpha$ — nonlinear stiffness coefficient

The model is intentionally simplified.

It is used as a **representative dynamical sensor system** rather than as a complete industrial MEMS gyroscope model.

The advantage of this controlled formulation is that the true physical parameters and governing equations remain known, allowing identification and learning methods to be evaluated quantitatively.

---

# Scientific Workflow

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
