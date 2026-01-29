# Policy Evaluation Uncertainty: Confidence Interval Coverage under Dependence

This repository contains a small, reproducible study on **policy evaluation** uncertainty in reinforcement learning (RL).  
We investigate when a standard **Wald (normal-approximation) confidence interval** for the mean return provides correct coverage, and how **temporal dependence** (trajectory-like correlation) can break classical assumptions.

## 1. Problem Statement

In policy evaluation, a fixed policy \(\pi\) is assessed by its (episodic) value
\[
V(\pi) = \mathbb{E}_\pi[R],
\]
where \(R\) denotes a random episode return (or a return computed from a trajectory segment).

A common estimator is the sample mean
\[
\hat V = \bar R = \frac{1}{n}\sum_{i=1}^n R_i,
\]
and a widely used uncertainty quantification method is the **naive Wald confidence interval**
\[
\bar R \pm z_{1-\alpha/2}\frac{s}{\sqrt{n}},
\]
which is typically justified by iid sampling and asymptotic normality (CLT).

However, in RL, data are often collected along **trajectories** and can be strongly correlated. This raises a practical statistical question:

> **Does a nominal 95% confidence interval still achieve ~95% empirical coverage when returns are correlated?  
> If not, can a dependence-aware method (e.g., block bootstrap) improve coverage?**

The key quantity we estimate empirically is the **coverage probability**:
\[
\widehat{\text{Cov}} = \frac{1}{M}\sum_{m=1}^M \mathbf{1}\{V(\pi)\in CI^{(m)}\},
\]
computed over \(M\) repeated simulations.

## 2. Methodology

### 2.1 Experimental Design (Monte Carlo Study)

We run a Monte Carlo experiment with two sampling regimes:

1. **Approximately iid returns** (independent episodes):  
   \(R_1, \dots, R_n\) are generated independently from a distribution with mean \(V(\pi)\).

2. **Correlated returns** (trajectory-like dependence):  
   \(R_1, \dots, R_n\) are generated from a dependent process (e.g., an AR(1) structure) to mimic temporal correlation encountered in trajectory data.

For each regime and each sample size \(n\), we repeat the following procedure \(M\) times:

- Generate a return sample \(R_1,\dots,R_n\).
- Construct a confidence interval for \(V(\pi)\).
- Record whether the true value \(V(\pi)\) lies inside the interval.

### 2.2 Confidence Intervals

We compare:

- **Naive Wald CI (iid-based)**:
  \[
  CI_{\text{Wald}} = \bar R \pm z_{1-\alpha/2}\frac{s}{\sqrt{n}}.
  \]
  This interval ignores dependence and treats the standard error as \(s/\sqrt{n}\).

- **Moving Block Bootstrap CI (dependence-aware)**:
  We resample contiguous blocks of length \(L\) from the observed sequence, stitch blocks to form bootstrap samples, and compute the bootstrap distribution of \(\bar R\).  
  The confidence interval is taken as the empirical \(\alpha/2\) and \(1-\alpha/2\) quantiles of bootstrap means.

### 2.3 Evaluation Metric

For a nominal \(1-\alpha\) interval (typically \(0.95\)), we report:

- **Empirical coverage** as a function of \(n\), under iid and correlated regimes.
- Qualitative comparison of under-coverage (too narrow intervals) and improvements from bootstrap methods.

## Repository Structure

- `src/` : reproducible simulation code (script-based runs)
- `notebooks/` : exploratory notebooks (optional)
- `results/figures/` : saved plots (coverage curves, etc.)

## Getting Started

### Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
