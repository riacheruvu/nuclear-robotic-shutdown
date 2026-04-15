# 🛡️ Risk-Informed Qualification Framework for Robotic Remote Shutdown

**AA228V / CS238V Final Project | Stanford University**

This framework provides a rigorous validation pipeline for a mobile robot tasked with navigating a high-radiation environment to perform a manual valve shutdown. It evaluates system safety by combining **Linearized Interval-Arithmetic Reachability** with **GPU-Accelerated Defensive Mixture Importance Sampling**.

## 📖 Background & Publication

This repository is the implementation of the framework discussed in:
**[Modeling Safety-Critical Robots for Nuclear Engineering](https://riacheruvu.medium.com/risk-informed-robotic-shutdown-for-nuclear-engineering-1c154c72fbaa)**
*Published on Medium | April 2026*

-----

## 🏗️ 7-D Augmented Dynamics

To capture realistic actuation delays common in remote-handling robotics, the system state $s$ is augmented with a two-step lag buffer:
$$s = [x, y, \theta, v_{t-1}, \omega_{t-1}, v_{t-2}, \omega_{t-2}]^T$$
In this model, the kinematics at time $t$ are driven by the control command issued at $t-2$. This forces the safety framework to account for the physical "drift" that occurs between command and execution.

-----

## 🚀 Key Features

  * **7-D Augmented Dynamics:** Models kinematics with a two-step temporal lag buffer ($u_{t-2}$ drives motion) to simulate realistic actuation delays.
  * **Formal Reachability:** Propagates axis-aligned bounding boxes (`IntervalBoxes`) through linearized dynamics with Minkowski-sum noise inflation to provide formal safety bounds.
  * **Defensive Mixture Importance Sampling (IS):** A GPU-accelerated estimator that biases both Gaussian slip noise and Bernoulli "salt-and-pepper" sensor scrambles to accurately estimate rare failure probabilities ($P_{fail}$).
  * **Radiation Physics:** Includes a $^{60}\text{Co}$ specific gamma-ray model with air attenuation and background dose rates for realistic risk assessment.
  * **Animated Validation Dashboard:** Generates real-time visualizations of stochastic rollouts, formal bounds, and radiation heatmaps.

-----

## 🛠️ Technical Methodology

### 1\. Reachability Analysis

The framework propagates uncertainty through the system using:
$$s_{t+1} = f(s_t) + w_t$$
where $w_t$ includes both slip noise and observation-induced control noise. It uses a linearized Jacobian $A = \partial f / \partial s$ to bound the state evolution within a corridor $R_{safe}$.

### 2\. Importance Sampling

To capture rare failures that Naive Monte Carlo might miss, we use a Defensive Mixture:

  * **Nominal Distribution:** $\alpha = 0.7$
  * **Biased Distribution:** $\sigma_{slip} \times \text{bias\_factor}$

The **Joint Likelihood Fix** tracks both the continuous Gaussian slips and the discrete Bernoulli sensor scrambles ($P_{scramble}$) to prevent weight collapse during high-variance scenarios.

-----

## 📊 Performance Thresholds

The framework qualifies the robot based on two primary metrics defined in `main.py`:

1.  **Task Liveness:** Mission success rate (reaching the valve within **0.15m**) must exceed **90%** at a 95% Confidence Interval.
2.  **Safety Integrity:** Total cumulative dose must remain below $D_{max} =$ **50.0 mSv**.

-----

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | Central entry point for the validation pipeline and metrics reporting. |
| `dynamics.py` | 7-D POMDP dynamics and proportional heading controller. |
| `reachability.py` | Interval-arithmetic reachability with $\sqrt{DT}$ noise scaling. |
| `importance_sampling.py` | Core GPU logic for parallel stochastic rollouts and IS weighting. |
| `ablation.py` | Comparative study of Naive MC vs. Defensive/Aggressive IS strategies. |
| `config.py` | Environment constants, robot gains, and safety thresholds. |
| `visualization.py` | Logic for gamma-dose heatmaps and `FuncAnimation` dashboards. |

-----

## 🏃 Getting Started

### Prerequisites

  * Python 3.8+
  * `torch` (with CUDA support for acceleration)
  * `numpy`, `matplotlib`

### Execution

Run the full validation pipeline:

```bash
python main.py
```

This will generate `nuclear_shutdown_final.png` (static snapshot) and `shutdown_animation.mp4` (full dashboard animation).