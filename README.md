# Policy Evaluation Uncertainty: Confidence Interval Coverage under Dependence

This repository contains a small, reproducible study on **policy evaluation uncertainty** in reinforcement learning (RL).
We investigate when a standard Wald (normal-approximation) confidence interval for the mean return provides correct coverage,
and how **temporal dependence** (trajectory-like correlation) can break classical statistical assumptions.

---

## 1. Problem Statement

In policy evaluation, a fixed policy π is assessed by its (episodic) value

    V(π) = E[R]

where R denotes a random episode return (or a return computed from a trajectory segment).

A common estimator of V(π) is the sample mean

    V_hat = (1/n) * sum_{i=1}^n R_i

and a widely used uncertainty quantification method is the **naive Wald confidence interval**

    V_hat ± z_(1−α/2) * (s / sqrt(n)),

which is typically justified by independent sampling and asymptotic normality (CLT).

However, in reinforcement learning, data are often collected along **trajectories** and are therefore correlated.
This raises a fundamental statistical question:

> **Does a nominal 95% confidence interval still achieve ~95% empirical coverage when returns are correlated?  
> If not, can a dependence-aware method (e.g. block bootstrap) improve coverage?**

The key quantity we estimate empirically is the **coverage probability**

    Coverage = (1/M) * sum_{m=1}^M I{ V(π) ∈ CI^(m) }

computed over M repeated Monte Carlo simulations.

---

## 2. Methodology

### 2.1 Experimental Design (Monte Carlo Study)

We run a Monte Carlo experiment under two sampling regimes:

#### 1. Approximately iid returns (independent episodes)

Returns R₁, …, Rₙ are generated independently from a distribution with mean V(π).
This corresponds to the classical statistical setting where standard confidence intervals are valid.

#### 2. Correlated returns (trajectory-like dependence)

Returns R₁, …, Rₙ are generated from a dependent process (e.g. an AR(1) structure) to mimic
temporal correlation encountered in RL trajectories.

---

### 2.2 Confidence Intervals

We compare two inference procedures:

#### Naive Wald confidence interval (iid assumption)

The standard normal-approximation interval:

    CI_Wald = V_hat ± z_(1−α/2) * (s / sqrt(n))

This interval ignores temporal dependence and therefore underestimates uncertainty when samples are correlated.

#### Moving block bootstrap confidence interval

To account for dependence, we apply a **moving block bootstrap**:

- contiguous blocks of length L are resampled with replacement,
- blocks are stitched together to form bootstrap samples,
- the bootstrap distribution of the mean is used to construct the confidence interval.

---

### 2.3 Evaluation Metric

For a nominal confidence level (e.g. 95%), we report:

- empirical coverage as a function of sample size n,
- comparison between iid and correlated regimes,
- improvement obtained by dependence-aware methods.

---

## Getting Started

### Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

