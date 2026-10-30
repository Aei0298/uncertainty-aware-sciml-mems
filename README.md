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

```math
m\ddot{x}(t) + c\dot{x}(t) + kx(t) = F(t)
```

where:

- $m$ — effective mass
- $c$ — damping coefficient
- $k$ — stiffness
- $F(t)$ — applied excitation
- $x(t)$ — displacement response

The corresponding acceleration is obtained from:

```math
\ddot{x}(t)
=
\frac{F(t)-c\dot{x}(t)-kx(t)}{m}
```

A nonlinear extension can additionally be considered:

```math
m\ddot{x}(t)
+
c\dot{x}(t)
+
kx(t)
+
\alpha x^3(t)
=
F(t)
```

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
```

---

# Methodological Components

## 1. Dynamical-System Simulation

The physical system is simulated numerically to generate displacement, velocity, and acceleration responses.

The system state can be written as:

```math
\mathbf{x}(t)
=
\begin{bmatrix}
x(t) \\
\dot{x}(t)
\end{bmatrix}
```

The governing second-order differential equation is transformed into a first-order system suitable for numerical integration.

The model provides a controlled environment in which parameters such as:

```math
m,\qquad c,\qquad k
```

can be varied systematically.

The resulting dynamic responses form the physical foundation of the complete project.

---

## 2. Sensor Modelling

Synthetic sensor measurements are generated from the simulated mechanical response.

The measurement process can include:

- Additive measurement noise
- Sensor bias
- Gain variation
- Parameter drift
- Environmental variability

A simplified measurement representation can be written as:

```math
y(t)
=
g\,x(t)
+
b
+
\varepsilon(t)
```

where:

- $y(t)$ — measured signal
- $x(t)$ — true system response
- $g$ — sensor gain
- $b$ — sensor bias
- $\varepsilon(t)$ — measurement noise

This creates more realistic identification conditions than using ideal deterministic signals alone.

---

## 3. Signal Processing

Signal-processing techniques are used to extract dynamical information from the simulated sensor signals.

The analysis includes methods such as:

- Time-domain signal analysis
- Fast Fourier Transform (FFT)
- Power spectral analysis
- Welch power spectral density estimation
- Frequency estimation
- Filtering
- Feature extraction

The discrete Fourier transform can be expressed as:

```math
X_k
=
\sum_{n=0}^{N-1}
x_n
e^{-i2\pi kn/N}
```

Frequency-domain methods provide information about characteristic frequencies and dynamic behaviour before model identification is performed.

For a linear mass-spring system, the undamped natural angular frequency is:

```math
\omega_n
=
\sqrt{\frac{k}{m}}
```

and the corresponding natural frequency is:

```math
f_n
=
\frac{1}{2\pi}
\sqrt{\frac{k}{m}}
```

---

## 4. Classical System Identification

Classical identification methods are used to estimate physical system parameters from measured signals.

The estimated quantities can include:

```math
m,\qquad c,\qquad k
```

as well as derived quantities such as natural frequency and damping characteristics.

For example, the damping ratio of the linear system is:

```math
\zeta
=
\frac{c}{2\sqrt{km}}
```

The estimated parameters can then be compared directly with the known ground-truth values of the synthetic system.

This provides a reference against which more advanced data-driven methods can be evaluated.

---

## 5. Equation Discovery

Two main data-driven equation-discovery approaches are investigated.

### Sparse Identification of Nonlinear Dynamics — SINDy

SINDy searches for sparse governing equations from a library of candidate mathematical terms.

The general formulation is:

```math
\dot{\mathbf{X}}
=
\Theta(\mathbf{X})\Xi
```

where:

- $\mathbf{X}$ contains measured system states
- $\dot{\mathbf{X}}$ contains their time derivatives
- $\Theta(\mathbf{X})$ is a library of candidate nonlinear functions
- $\Xi$ contains the identified sparse coefficients

The objective is to recover equations resembling the underlying physical dynamics directly from simulated measurement data.

For the mechanical system, the expected structure is related to:

```math
\ddot{x}
=
\frac{1}{m}F
-
\frac{c}{m}\dot{x}
-
\frac{k}{m}x
```

A successful equation-discovery method should therefore identify the physically relevant terms while suppressing unnecessary ones.

---

### Symbolic Regression

Symbolic regression searches for analytical mathematical expressions that describe relationships in the data.

Unlike conventional regression models, the mathematical structure itself is part of the optimization process.

A symbolic model may therefore take a general form such as:

```math
y
=
f(x_1,x_2,\ldots,x_n)
```

where the function $f$ is discovered from the data rather than predefined.

This provides an alternative approach for discovering interpretable equations without explicitly prescribing their final functional form.

---

## 6. Statistical and Black-Box Machine Learning

Data-driven regression models are investigated as alternative approaches for predicting system behaviour or estimating physical quantities.

These models provide a comparison between:

- Physics-based modelling
- Classical parameter estimation
- Interpretable equation discovery
- Purely data-driven prediction

The purpose is not only to maximize predictive performance, but also to investigate the trade-off between:

- Accuracy
- Interpretability
- Physical consistency
- Robustness to noise
- Generalization under uncertainty

---

## 7. Probabilistic Machine Learning

Probabilistic models are used to represent predictive uncertainty explicitly.

Instead of producing only a point estimate:

```math
\hat{y}
```

a probabilistic model estimates a predictive distribution:

```math
p(y\mid x)
```

This allows uncertainty in predictions to be quantified directly.

Gaussian Process regression is one of the probabilistic approaches investigated.

A Gaussian Process is defined as:

```math
f(x)
\sim
\mathcal{GP}
\left(
m(x),
k(x,x')
\right)
```

where:

- $m(x)$ — mean function
- $k(x,x')$ — covariance kernel

The resulting predictions include both a predicted mean and an uncertainty estimate.

---

## 8. Physics-Informed Machine Learning

Physics-informed approaches incorporate knowledge of the governing dynamical system into the learning process.

A generic physics-informed objective can be written as:

```math
\mathcal{L}
=
\mathcal{L}_{\text{data}}
+
\lambda
\mathcal{L}_{\text{physics}}
```

where:

- $\mathcal{L}_{\text{data}}$ measures agreement with observations
- $\mathcal{L}_{\text{physics}}$ penalizes violation of the governing physical equations
- $\lambda$ controls the relative contribution of the physics constraint

For the linear MEMS model, a physics residual can be defined as:

```math
r(t)
=
m\ddot{x}(t)
+
c\dot{x}(t)
+
kx(t)
-
F(t)
```

A physics-informed model should ideally satisfy:

```math
r(t)
\approx
0
```

while also reproducing the available measurement data.

The goal is to investigate whether incorporating physical knowledge improves robustness when measurements are noisy or uncertain.

---

## 9. Causal Analysis

Selected causal relationships are investigated between variables such as:

- Environmental conditions
- Parameter drift
- Stiffness variation
- Sensor measurements
- Predicted system response

The objective is to distinguish predictive relationships from physically meaningful intervention effects where appropriate.

For example, an intervention on stiffness may be represented conceptually as:

```math
do(k = k^\ast)
```

and its influence on system response can then be evaluated.

This extends the analysis beyond correlation-based prediction toward questions about how changes in physical parameters affect measurable behaviour.

---

## 10. Monte Carlo Uncertainty Quantification

Monte Carlo simulation is used as the main uncertainty-analysis framework.

Let the uncertain parameter vector be:

```math
\boldsymbol{\theta}
=
\begin{bmatrix}
m &
c &
k &
\ldots
\end{bmatrix}^{T}
```

Samples are drawn from prescribed probability distributions:

```math
\boldsymbol{\theta}^{(i)}
\sim
p(\boldsymbol{\theta})
```

For every realization, the dynamical system is simulated:

```math
\boldsymbol{\theta}^{(i)}
\rightarrow
\text{simulation}
\rightarrow
y^{(i)}
```

Repeating this process for many realizations produces an ensemble:

```math
\left\{
y^{(1)},
y^{(2)},
\ldots,
y^{(N)}
\right\}
```

The sample mean can then be estimated as:

```math
\bar{y}
=
\frac{1}{N}
\sum_{i=1}^{N}
y^{(i)}
```

and the sample standard deviation as:

```math
s_y
=
\sqrt{
\frac{1}{N-1}
\sum_{i=1}^{N}
\left(
y^{(i)}-\bar{y}
\right)^2
}
```

The coefficient of variation is calculated using:

```math
CV
=
\frac{s_y}{\bar{y}}
```

The Monte Carlo ensemble is used to investigate:

- Mean response
- Standard deviation
- Output distributions
- Coefficient of variation
- Convergence behaviour
- Sensitivity measures
- Parameter uncertainty
- Predictive uncertainty
- Robustness of identification methods

---

# Model Comparison Philosophy

The project is structured around a comparison of several modelling paradigms.

| Approach | Main Purpose |
|---|---|
| Physics-based simulation | Generate known dynamical ground truth |
| Signal processing | Extract information from sensor measurements |
| Classical system identification | Estimate interpretable physical parameters |
| SINDy | Discover sparse governing equations |
| Symbolic regression | Discover analytical relationships |
| Statistical / black-box ML | Learn predictive mappings from data |
| Probabilistic ML | Quantify predictive uncertainty |
| Physics-informed ML | Combine physical constraints with learning |
| Causal analysis | Investigate intervention-based relationships |
| Monte Carlo simulation | Quantify uncertainty and robustness |

The objective is not simply to identify a single best-performing model.

Instead, the project investigates the strengths and limitations of each methodology under controlled levels of uncertainty, noise, and parameter variation.

---

# Key Scientific Questions

The project addresses questions such as:

- How accurately can physical parameters be recovered from noisy sensor data?
- How sensitive are identification results to measurement noise?
- How strongly does physical parameter uncertainty influence the resulting system response?
- Can equation-discovery methods recover the known governing dynamics?
- How robust is SINDy when measurement noise increases?
- How does symbolic regression compare with sparse equation discovery?
- How do black-box models compare with physics-based and interpretable approaches?
- Can probabilistic models provide meaningful predictive uncertainty?
- Does incorporating physical knowledge improve model robustness?
- Which uncertain physical parameters have the strongest influence on the measured response?
- How stable are model conclusions across repeated Monte Carlo realizations?
- Can causal analysis distinguish intervention effects from purely statistical associations?

---

# Repository Structure

```text
uncertainty-aware-sciml-mems/
│
├── configs/
│   ├── base_config.yaml
│   ├── model_config.yaml
│   └── monte_carlo_config.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── monte_carlo/
│
├── docs/
│   ├── experiment_log.md
│   ├── methodology.md
│   ├── project_plan.md
│   └── research_questions.md
│
├── figures/
│   ├── causal_ml/
│   ├── dynamics/
│   ├── equation_discovery/
│   ├── machine_learning/
│   ├── physics_informed_ml/
│   ├── probabilistic_ml/
│   ├── signal_processing/
│   ├── symbolic_regression/
│   ├── system_identification/
│   └── uncertainty/
│
├── notebooks/
│   ├── 01_mems_dynamical_system.ipynb
│   ├── 02_monte_carlo_simulation.ipynb
│   ├── 03_signal_processing.ipynb
│   ├── 04_classical_system_identification.ipynb
│   ├── 05_sindy_equation_discovery.ipynb
│   ├── 06_symbolic_regression.ipynb
│   ├── 07_statistical_and_blackbox_ml.ipynb
│   ├── 08_probabilistic_ml.ipynb
│   ├── 09_physics_informed_ml.ipynb
│   ├── 10_causal_ml.ipynb
│   └── 11_monte_carlo_uq_evaluation.ipynb
│
├── results/
│   ├── causal_ml/
│   ├── equation_discovery/
│   ├── machine_learning/
│   ├── monte_carlo_uq/
│   ├── physics_informed_ml/
│   ├── probabilistic_ml/
│   ├── signal_processing/
│   ├── symbolic_regression/
│   └── system_identification/
│
├── src/
│   ├── causal_analysis.py
│   ├── dynamics.py
│   ├── equation_discovery.py
│   ├── excitation.py
│   ├── ml_models.py
│   ├── physics_informed.py
│   ├── probabilistic_models.py
│   ├── sensors.py
│   ├── signal_processing.py
│   ├── simulation.py
│   ├── system_identification.py
│   ├── uncertainty.py
│   └── utils.py
│
├── tests/
│   ├── test_dynamics.py
│   ├── test_signal_processing.py
│   ├── test_system_identification.py
│   └── test_uncertainty.py
│
├── environment.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Notebook Workflow

The notebooks follow the scientific progression of the project.

### `01_mems_dynamical_system.ipynb`

Defines and investigates the baseline MEMS dynamical system.

### `02_monte_carlo_simulation.ipynb`

Introduces uncertainty in model parameters and performs Monte Carlo simulations.

### `03_signal_processing.ipynb`

Analyses simulated sensor signals in the time and frequency domains.

### `04_classical_system_identification.ipynb`

Estimates physical parameters using classical identification approaches.

### `05_sindy_equation_discovery.ipynb`

Uses SINDy to recover governing dynamical equations.

### `06_symbolic_regression.ipynb`

Investigates symbolic regression for analytical equation discovery.

### `07_statistical_and_blackbox_ml.ipynb`

Evaluates statistical and conventional machine-learning approaches.

### `08_probabilistic_ml.ipynb`

Investigates probabilistic prediction and uncertainty using methods such as Gaussian Processes.

### `09_physics_informed_ml.ipynb`

Introduces physical constraints into the machine-learning formulation.

### `10_causal_ml.ipynb`

Investigates selected causal relationships between system parameters and dynamic responses.

### `11_monte_carlo_uq_evaluation.ipynb`

Evaluates uncertainty propagation, convergence, sensitivity, and robustness across the complete workflow.

---

# Software and Tools

The project is implemented primarily in Python.

Main tools and libraries include:

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- PySINDy
- Gaussian Process modelling
- Symbolic regression tools
- pytest
- YAML configuration files

Exact package dependencies are listed in:

```text
requirements.txt
```

and:

```text
environment.yml
```

---

# Testing

Core functionality is supported by automated tests covering areas such as:

- Dynamical-system behaviour
- Signal-processing functions
- System-identification calculations
- Uncertainty-analysis utilities

The tests are located in:

```text
tests/
```

and can be executed using:

```bash
pytest
```

---

# Reproducibility

The repository is structured to separate:

- Physical models
- Numerical simulation
- Signal processing
- Machine-learning methods
- Configuration files
- Generated data
- Results
- Figures
- Validation tests

This modular structure is intended to improve reproducibility and make individual parts of the workflow easier to inspect and extend.

---

# Project Focus

This project combines three main areas:

```text
Mechanical and dynamical modelling
                +
Scientific machine learning
                +
Uncertainty quantification
```

with the overall objective of developing and evaluating methods for:

> **Data-driven identification of uncertain physical systems from sensor measurements.**

The broader motivation is to investigate how physical modelling, uncertainty quantification, and modern data-driven methods can complement one another when analysing engineering systems under imperfect measurement conditions.
