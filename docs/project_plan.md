# Project Plan

## 1. Project Goal

The goal of this project is to investigate the reliability of learning physical dynamics from uncertain MEMS sensor data using Monte Carlo analysis and Scientific Machine Learning methods.

The project will progress from a known dynamical model toward increasingly data-driven approaches.

The main comparison will involve:

- classical system identification,
- equation discovery,
- black-box machine learning,
- probabilistic machine learning,
- physics-informed / grey-box learning,
- causal analysis.

Monte Carlo analysis will be used to quantify robustness and uncertainty throughout the study.

---

## 2. Development Strategy

The project will be developed incrementally.

Each stage must be validated before the next stage is started.

The progression is:

1. deterministic physical simulation,
2. uncertain simulation,
3. signal processing,
4. classical identification,
5. equation discovery,
6. machine learning,
7. probabilistic learning,
8. physics-informed learning,
9. causal analysis,
10. integrated Monte Carlo uncertainty evaluation.

---

## 3. Project Phases

### Phase 1 — MEMS Dynamical System

Notebook:

`01_mems_dynamical_system.ipynb`

Tasks:

- define the governing ODE,
- define state variables,
- define physical parameters,
- implement excitation signals,
- solve the ODE numerically,
- calculate displacement, velocity, and acceleration,
- visualize the response,
- compare linear and nonlinear behaviour,
- verify physical plausibility.

Completion criterion:

The deterministic simulator produces stable and interpretable sensor responses for known parameter values.

---

### Phase 2 — Monte Carlo Simulation

Notebook:

`02_monte_carlo_simulation.ipynb`

Tasks:

- define uncertain parameters,
- assign probability distributions,
- generate reproducible Monte Carlo samples,
- simulate ensembles of sensor responses,
- store simulation metadata and outputs,
- evaluate convergence of selected quantities.

Completion criterion:

The simulator can generate reproducible ensembles of uncertain MEMS responses.

---

### Phase 3 — Signal Processing

Notebook:

`03_signal_processing.ipynb`

Tasks:

- inspect time-domain signals,
- apply filtering if required,
- calculate FFT,
- estimate PSD,
- extract resonance information,
- compute selected time-domain and frequency-domain features.

Completion criterion:

A reproducible feature dataset is generated from simulated sensor responses.

---

### Phase 4 — Classical System Identification

Notebook:

`04_classical_system_identification.ipynb`

Tasks:

- define white-box parameter identification,
- estimate physical parameters,
- compare estimates with ground truth,
- quantify bias and estimation error,
- evaluate sensitivity to noise.

Completion criterion:

The baseline system-identification method is validated under controlled conditions.

---

### Phase 5 — SINDy Equation Discovery

Notebook:

`05_sindy_equation_discovery.ipynb`

Tasks:

- prepare state and derivative data,
- define candidate-function libraries,
- apply sparse equation discovery,
- compare discovered equations with the true governing equation,
- investigate sensitivity to noise and library choice.

Completion criterion:

The method can recover the known governing structure under suitable conditions.

---

### Phase 6 — Symbolic Regression

Notebook:

`06_symbolic_regression.ipynb`

Tasks:

- define symbolic-search variables and operators,
- discover candidate equations,
- compare discovered expressions with physical ground truth,
- compare symbolic regression with SINDy.

Completion criterion:

At least one interpretable symbolic model can be quantitatively evaluated.

---

### Phase 7 — Statistical and Black-Box Machine Learning

Notebook:

`07_statistical_and_blackbox_ml.ipynb`

Tasks:

- implement statistical baseline,
- implement nonlinear ML benchmark,
- predict selected parameters or dynamic quantities,
- evaluate interpolation and extrapolation performance.

Completion criterion:

A clean data-driven benchmark exists for comparison with physics-based methods.

---

### Phase 8 — Probabilistic Machine Learning

Notebook:

`08_probabilistic_ml.ipynb`

Tasks:

- implement probabilistic prediction,
- calculate predictive means and uncertainties,
- assess uncertainty calibration,
- evaluate uncertainty under extrapolation.

Completion criterion:

The model provides scientifically interpretable predictive uncertainty.

---

### Phase 9 — Physics-Informed / Grey-Box ML

Notebook:

`09_physics_informed_ml.ipynb`

Tasks:

- define known physics component,
- define learnable residual or physics loss,
- train hybrid model,
- compare with black-box model,
- analyze generalization and extrapolation.

Completion criterion:

The value of incorporating physical knowledge can be quantitatively evaluated.

---

### Phase 10 — Causal Analysis

Notebook:

`10_causal_ml.ipynb`

Tasks:

- define causal graph,
- define interventions,
- investigate selected environmental effects,
- compare estimated effects with known simulation ground truth.

Completion criterion:

At least one causal question is addressed using a clearly defined data-generating process.

---

### Phase 11 — Integrated Monte Carlo UQ

Notebook:

`11_monte_carlo_uq_evaluation.ipynb`

Tasks:

- compare all major methods under uncertainty,
- evaluate parameter recovery,
- evaluate equation-recovery reliability,
- quantify predictive uncertainty,
- quantify robustness to noise,
- perform sensitivity analysis,
- summarize scientific conclusions.

Completion criterion:

The project can answer the main research questions using quantitative evidence.

---

## 4. Reporting Strategy

The LaTeX report will be written progressively.

The report will not be used as the primary location for exploratory analysis.

Only validated methods, results, figures, and interpretations will be transferred into the final report.

---

## 5. Scope Control

The project will prioritize depth over the number of algorithms.

Advanced methods will only be included if they contribute directly to the research questions.

In particular:

- Monte Carlo analysis remains the central methodology.
- System identification and equation discovery are core.
- Physics-informed ML is a major extension.
- Probabilistic ML supports uncertainty quantification.
- Causal ML remains a focused supporting analysis rather than an independent project.