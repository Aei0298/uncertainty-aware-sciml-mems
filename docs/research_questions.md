# Research Questions

## Central Research Problem

The project investigates how reliably physical dynamics can be learned from uncertain and noisy MEMS sensor data.

The study combines physical modelling, system identification, equation discovery, machine learning, probabilistic modelling, physics-informed learning, causal analysis, and Monte Carlo uncertainty quantification.

---

## RQ1 — Dynamical Response under Uncertainty

**How do uncertain physical parameters, environmental conditions, and measurement noise influence the dynamic response of a MEMS gyroscope?**

Main variables may include:

- mass,
- damping,
- stiffness,
- nonlinear stiffness,
- temperature,
- mechanical stress,
- ageing,
- sensor noise,
- sensor gain,
- sensor bias.

---

## RQ2 — System Identification

**How accurately can classical and data-driven system-identification approaches recover the underlying physical parameters from noisy sensor measurements?**

Primary evaluation quantities:

- parameter bias,
- MAE,
- RMSE,
- variance,
- robustness to measurement noise.

---

## RQ3 — Equation Discovery

**How reliably can SINDy and symbolic regression recover the governing dynamics and reveal unmodelled nonlinear behaviour?**

The known simulation equation provides ground truth for evaluating:

- correct term recovery,
- coefficient accuracy,
- false-positive terms,
- false-negative terms,
- robustness to noise.

---

## RQ4 — Physics-Informed Learning

**Does incorporating physical knowledge through grey-box or physics-informed machine learning improve prediction, parameter identification, generalization, or extrapolation relative to purely data-driven models?**

---

## RQ5 — Probabilistic and Causal Learning

**How can probabilistic and causal machine-learning methods support uncertainty-aware interpretation and help distinguish environmental effects from true physical parameter drift?**

This question will remain limited in scope.

Probabilistic modelling will focus on predictive uncertainty.

Causal analysis will focus on selected known relationships from the synthetic data-generating process.

---

## RQ6 — Monte Carlo Robustness and UQ

**How can Monte Carlo analysis quantify the robustness and uncertainty of system identification, equation discovery, and learned dynamical models?**

This is the overarching research question connecting the full project to the Information Engineering topic.

---

## Research Hierarchy

The research questions are not independent.

They form the following progression:

```text
RQ1
Understand uncertain MEMS dynamics
        ↓
RQ2
Estimate physical parameters
        ↓
RQ3
Discover governing equations
        ↓
RQ4
Evaluate physics-informed learning
        ↓
RQ5
Quantify and interpret uncertainty
        ↓
RQ6
Evaluate robustness through Monte Carlo analysis