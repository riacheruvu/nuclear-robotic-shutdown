# 🛡️ Risk-Informed Qualification Framework for Robotic Remote Shutdown

**AA228V / CS238V Final Project | Stanford University**

This framework provides a basic validation pipeline for a mobile robot tasked with navigating a high-radiation environment to perform a manual valve shutdown. It evaluates system safety by combining **Linearized Interval-Arithmetic Reachability** with **GPU-Accelerated Defensive Mixture Importance Sampling**.

## 🚀 Key Features

* **7-D Augmented Dynamics:** Models kinematics with a two-step temporal lag buffer ($u_{t-2}$ drives motion) to simulate realistic actuation delays.
* **Formal Reachability:** Propagates axis-aligned bounding boxes (IntervalBoxes) through linearized dynamics with Minkowski-sum noise inflation to provide formal safety bounds.
* **Defensive Mixture Importance Sampling (IS):** A GPU-accelerated estimator that biases both Gaussian slip noise and Bernoulli "salt-and-pepper" sensor scrambles to accurately estimate rare failure probabilities ($P_{fail}$).
* **Radiation Physics:** Includes a Co-60 specific gamma-ray model with air attenuation and background dose rates for realistic risk assessment.
* **Animated Validation Dashboard:** Generates real-time visualizations of stochastic rollouts, formal bounds, and radiation heatmaps.

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `main.py` | The central entry point for the validation pipeline and metrics reporting. |
| `dynamics.py` | Defines the 7-D POMDP dynamics and the proportional heading controller. |
| `reachability.py` | Implements interval-arithmetic reachability with $\sqrt{DT}$ noise scaling. |
| `importance_sampling.py` | Contains the core GPU logic for parallel stochastic rollouts and IS weighting. |
| `ablation.py` | Compares Naive Monte Carlo against Defensive and Aggressive IS strategies. |
| `config.py` | Constants for the environment ($^{60}\text{Co}$ activity), robot gains, and safety thresholds. |
| `visualization.py` | Logic for the gamma-dose heatmap and `FuncAnimation` dashboard. |

---

## 🛠️ Technical Methodology

### 1. Reachability Analysis
The framework propagates uncertainty through the system using:
$$s_{t+1} = f(s_t) + w_t$$
where $w_t$ includes both slip noise and observation-induced control noise. It uses a linearized Jacobian $A = \partial f / \partial s$ to bound the state evolution within a corridor $R_{safe}$.

### 2. Importance Sampling
To capture rare failures that Naive Monte Carlo might miss, we use a Defensive Mixture:
* **Nominal Distribution:** $\alpha = 0.7$
* **Biased Distribution:** $\sigma_{slip} \times \text{bias\_factor}$
The joint likelihood tracks both the continuous Gaussian slips and the discrete Bernoulli sensor scrambles ($P_{scramble}$) to prevent weight collapse.

---

## 📊 Performance Thresholds

The framework qualifies the robot based on two primary metrics:
1.  **Task Liveness:** Mission success rate (reaching the valve within $0.15\text{m}$) must exceed **90%** at a 95% Confidence Interval.
2.  **Safety Integrity:** Total cumulative dose must remain below $D_{max} = 50.0\text{ mSv}$.

---

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