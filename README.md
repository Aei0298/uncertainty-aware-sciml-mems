# Monte Carlo Analysis for Information Engineering

## Uncertainty-Aware Scientific Machine Learning for System Identification and Equation Discovery in MEMS Gyroscope Dynamics

### Project Overview

This project investigates how the dynamics and physical parameters of a simplified MEMS gyroscope can be identified from uncertain and noisy sensor data.

The study is developed within the topic **Monte Carlo Analysis for Information Engineering** and combines dynamical systems, signal processing, classical system identification, equation discovery, statistical and machine learning methods, probabilistic modelling, physics-informed machine learning, causal analysis, and uncertainty quantification.

The central scientific theme is:

> **Learning uncertain physical dynamics from sensor data.**

The project uses a controlled synthetic MEMS dynamical system so that the true governing dynamics and physical parameters are known. This allows the performance of different modelling and identification approaches to be evaluated against a known ground truth.

---

## Main Objectives

The project aims to:

1. Develop a simplified dynamical model representing MEMS gyroscope behaviour.
2. Simulate sensor responses under uncertain physical and environmental conditions.
3. Introduce measurement noise, bias, gain variation, and parameter drift.
4. Use Monte Carlo simulation to propagate uncertainty through the system.
5. Apply signal-processing methods to extract information from the simulated sensor measurements.
6. Perform classical system identification of physical parameters.
7. Investigate equation discovery using Sparse Identification of Nonlinear Dynamics (SINDy).
8. Investigate symbolic regression as an additional equation-discovery method.
9. Compare classical, statistical, black-box, probabilistic, and physics-informed learning approaches.
10. Quantify predictive and parameter uncertainty.
11. Investigate selected causal relationships between environmental variables, parameter drift, and measured sensor responses.
12. Evaluate the robustness of the different approaches using Monte Carlo analysis.

---

## Core Dynamical Model

The initial simplified MEMS model is based on a forced mass-spring-damper system:

\[
m\ddot{x}(t) + c\dot{x}(t) + kx(t) = F(t)
\]

A nonlinear extension will also be investigated:

\[
m\ddot{x}(t)
+ c\dot{x}(t)
+ kx(t)
+ \alpha x^3(t)
= F(t)
\]

where:

- \(m\) is the effective mass,
- \(c\) is the damping coefficient,
- \(k\) is the stiffness,
- \(\alpha\) represents nonlinear stiffness,
- \(F(t)\) is the applied excitation,
- \(x(t)\) is the displacement response.

The model is intentionally simplified and is used as a representative dynamical sensor system rather than as an exact industrial MEMS gyroscope model.

---

## Scientific Workflow

```text
MEMS physical model
        ↓
excitation
        ↓
dynamic response
        ↓
sensor noise and drift
        ↓
signal processing
        ↓
classical system identification
        ↓
equation discovery
        ↓
statistical / black-box ML
        ↓
probabilistic ML
        ↓
physics-informed / grey-box ML
        ↓
causal analysis
        ↓
Monte Carlo uncertainty quantification