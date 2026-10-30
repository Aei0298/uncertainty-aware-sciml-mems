# Experiment Log

This file records important experimental decisions, parameter changes, failed approaches, and validated results.

The purpose is to preserve a transparent history of the computational study.

---

## Experiment 001 — Project Initialization

**Date:** 2026-08-12

### Objective

Create the project structure and define the initial scientific scope.

### Status

Completed.

### Decisions

- Application domain: simplified MEMS gyroscope dynamics.
- Core course topic: Monte Carlo Analysis for Information Engineering.
- Main scientific theme: learning uncertain physical dynamics from sensor data.
- Initial physical model: nonlinear mass-spring-damper system.
- Planned equation-discovery methods: SINDy and symbolic regression.
- Planned advanced modelling:
  - probabilistic ML,
  - physics-informed / grey-box ML,
  - focused causal analysis.
- Monte Carlo uncertainty quantification will remain the primary cross-cutting methodology.

### Next Experiment

Experiment 002:

Define and validate the deterministic MEMS dynamical-system model.



### Configuration Note

The initial parameter values and uncertainty distributions are development settings intended for code validation.

They are not yet claimed to represent realistic industrial MEMS gyroscope specifications.

Parameter ranges will be reviewed against scientific literature before the final Monte Carlo study.