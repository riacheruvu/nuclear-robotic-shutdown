# 🛡️ Risk-Informed Qualification Framework for Robotic Remote Shutdown

**AA228V / CS238V Final Project | Stanford University**

This framework provides a rigorous validation pipeline for a mobile robot tasked with navigating a high-radiation environment to perform a manual valve shutdown. It evaluates system safety by combining **Linearized Interval-Arithmetic Reachability** with **GPU-Accelerated Defensive Mixture Importance Sampling**.

## 📖 Background & Publication

This repository is the implementation of the framework discussed in:  
**[Modeling Safety-Critical Robots for Nuclear Engineering](https://riacheruvu.medium.com/risk-informed-robotic-shutdown-for-nuclear-engineering-1c154c72fbaa)**  
*Published on Medium | April 2026*

---

## 🏗️ 7‑D Augmented Dynamics

To capture realistic actuation delays common in remote-handling robotics, the system state \(s\) is augmented with a two-step lag buffer:

\[
s = [x,\; y,\; \theta,\; v_{t-1},\; \omega_{t-1},\; v_{t-2},\; \omega_{t-2}]^T
\]

In this model, the kinematics at time \(t\) are driven by the control command issued at \(t-2\), forcing the safety framework to account for the physical drift between command and execution.

---

## 🚀 Key Features

- **7‑D Augmented Dynamics:** Models kinematics with a two-step temporal lag buffer (\(u_{t-2}\) drives motion) to simulate realistic actuation delays.  
- **Formal Reachability:** Propagates axis-aligned bounding boxes (`IntervalBoxes`) through linearized dynamics with Minkowski-sum noise inflation to provide formal safety bounds.  
- **Defensive Mixture Importance Sampling (IS):** GPU-accelerated estimator that biases both Gaussian slip noise and Bernoulli “salt-and-pepper” sensor scrambles to accurately estimate rare failure probabilities (\(P_{\text{fail}}\)).  
- **Radiation Physics:** Includes a \(^{60}\text{Co}\) gamma-ray model with air attenuation and background dose rates for realistic risk assessment.  
- **Animated Validation Dashboard:** Generates real-time visualizations of stochastic rollouts, formal bounds, and radiation heatmaps.

---

## 📂 Project Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point for the validation pipeline and metrics reporting. |
| `dynamics.py` | Defines the 7‑D POMDP dynamics and proportional heading controller. |
| `reachability.py` | Implements interval-arithmetic reachability with \(\sqrt{\Delta t}\) noise scaling. |
| `importance_sampling.py` | Core GPU logic for parallel stochastic rollouts and IS weighting. |
| `ablation.py` | Compares Naive Monte Carlo vs. Defensive and Aggressive IS strategies. |
| `config.py` | Constants for environment (\(^{60}\text{Co}\) activity), robot gains, and safety thresholds. |
| `visualization.py` | Gamma-dose heatmap and `FuncAnimation` dashboard logic. |

---

## 🛠️ Technical Methodology

### 1. Reachability Analysis

The framework propagates uncertainty through:

\[
s_{t+1} = f(s_t) + w_t
\]

where \(w_t\) includes both slip noise and observation-induced control noise. A linearized Jacobian \(A = \partial f / \partial s\) bounds the state evolution within a safe corridor \(R_{\text{safe}}\).

---

### 2. Importance Sampling

To capture rare failures that Naive Monte Carlo misses, we use a **Defensive Mixture**:

- **Nominal Distribution (\(\alpha = 0.7\))**  
  Baseline noise \((\sigma_{\text{slip}},\; P_{\text{scramble}})\)

- **Biased Distribution (\(1 - \alpha = 0.3\))**  
  Noise amplified by  
  \[
  \sigma_{\text{slip}} \times \text{bias\_factor}, \quad
  P_{\text{scramble}} \times \text{bias\_factor}
  \]

The **Joint Likelihood Fix** tracks both continuous Gaussian slips and discrete Bernoulli scrambles to prevent weight collapse under high-variance conditions.

---

## 📊 Performance Thresholds

1. **Task Liveness:** Mission success rate (reaching the valve within **0.15 m**) must exceed **90%** at a 95% CI.  
2. **Safety Integrity:** Total cumulative dose must remain below  
   \[
   D_{\max} = 50.0\ \text{mSv}
   \]

---

## 🏃 Getting Started

### Prerequisites

- Python 3.8+  
- `torch` (with CUDA support)  
- `numpy`, `matplotlib`

### Execution

Run the full validation pipeline:

```bash
python main.py
```

This produces:

- `nuclear_shutdown_final.png` — static snapshot  
- `shutdown_animation.mp4` — full dashboard animation
