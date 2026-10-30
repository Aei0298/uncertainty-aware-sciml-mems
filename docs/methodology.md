# Methodology

## 1. Study Design

This project uses a controlled simulation study.

A simplified nonlinear MEMS dynamical system will be treated as the ground-truth data-generating process.

Because the governing equation and true parameters are known, identification and learning methods can be evaluated quantitatively.

The study will combine deterministic simulation with Monte Carlo uncertainty analysis.

---

## 2. Ground-Truth Dynamical Model

The initial model is:

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

- \(m\) is effective mass,
- \(c\) is damping,
- \(k\) is linear stiffness,
- \(\alpha\) is nonlinear stiffness,
- \(F(t)\) is the excitation,
- \(x(t)\) is displacement.

The linear case is obtained by setting:

\[
\alpha = 0.
\]

---

## 3. State-Space Representation

Define:

\[
x_1 = x
\]

and

\[
x_2 = \dot{x}.
\]

Then:

\[
\dot{x}_1 = x_2
\]

and

\[
\dot{x}_2 =
\frac{
F(t)
-
cx_2
-
kx_1
-
\alpha x_1^3
}{m}.
\]

This form will be used for numerical integration.

---

## 4. Excitation

Multiple excitation strategies may be evaluated.

The initial deterministic study will use controlled excitation suitable for identifying system dynamics.

Candidate signals include:

- sinusoidal excitation,
- chirp excitation,
- multi-sine excitation.

The excitation strategy will be finalized before Monte Carlo simulation.

---

## 5. Measurement Model

The simulated physical response will be separated from the measurement process.

A generic measurement model is:

\[
y_{\text{meas}}(t)
=
g\,y_{\text{true}}(t)
+
b
+
\epsilon(t)
\]

where:

- \(g\) is sensor gain,
- \(b\) is sensor bias,
- \(\epsilon(t)\) is measurement noise.

This separation is required to distinguish physical-system uncertainty from measurement uncertainty.

---

## 6. Monte Carlo Analysis

Uncertain model and measurement parameters will be represented using probability distributions.

For each realization:

\[
\theta^{(i)} \sim p(\theta)
\]

and the dynamical system will be simulated.

The resulting ensemble will be used for:

- uncertainty propagation,
- system-identification robustness,
- equation-discovery robustness,
- predictive uncertainty,
- sensitivity analysis.

---

## 7. Signal Processing

The sensor responses will be analyzed in both the time and frequency domains.

Potential features include:

- RMS,
- variance,
- peak amplitude,
- kurtosis,
- FFT amplitudes,
- PSD,
- resonance frequency,
- spectral energy.

The final feature set will be selected based on physical relevance and redundancy.

---

## 8. System Identification

A classical white-box method will estimate physical parameters assuming the model structure is known.

Estimated parameters will be compared directly with simulation ground truth.

---

## 9. Equation Discovery

Two equation-discovery approaches are planned.

### 9.1 SINDy

SINDy will identify sparse governing equations from a predefined candidate-function library.

### 9.2 Symbolic Regression

Symbolic regression will search for interpretable mathematical expressions using a broader set of possible functional forms.

Both approaches will be compared against the known ground-truth equation.

---

## 10. Machine Learning

A statistical baseline and a nonlinear black-box model will be used as data-driven comparison methods.

The exact prediction target will be finalized after the deterministic and signal-processing stages are validated.

---

## 11. Probabilistic Machine Learning

Probabilistic models will be used to estimate predictive uncertainty in addition to point predictions.

A Gaussian Process is currently the preferred initial approach.

---

## 12. Physics-Informed / Grey-Box Learning

The physics-informed model will combine known physical structure with a learnable component.

A likely architecture is:

\[
f_{\text{hybrid}}
=
f_{\text{physics}}
+
f_{\text{ML,residual}}.
\]

The objective is to investigate whether physical structure improves generalization, robustness, or data efficiency.

---

## 13. Causal Analysis

The synthetic simulator provides a known causal data-generating structure.

A small causal analysis will investigate selected relationships such as:

\[
T \rightarrow k \rightarrow y
\]

or:

\[
\text{ageing}
\rightarrow
k,c
\rightarrow
y.
\]

Estimated intervention effects can therefore be compared against known simulation effects.

---

## 14. Evaluation

Evaluation may include:

- RMSE,
- MAE,
- bias,
- coefficient error,
- equation term recovery,
- uncertainty interval coverage,
- predictive calibration,
- Monte Carlo convergence,
- robustness to noise,
- extrapolation performance.

---

## 15. Reproducibility

All simulations and analyses will:

- use documented random seeds,
- read major parameters from configuration files,
- save numerical outputs systematically,
- generate figures reproducibly,
- separate reusable code from notebook analysis.