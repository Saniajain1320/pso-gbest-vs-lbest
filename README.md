# 🌸 PSO Lab 9 – NMIMS MPSTME
**Evolutionary Computing | Department of AI**

## Overview
Interactive Streamlit dashboard implementing **Particle Swarm Optimization** for Lab Experiment 9.

### Tasks
| Task | Description |
|------|-------------|
| **Task 1** | Minimize `f(x₁,x₂) = 100·(x₁ − x₂²)² + (1 − x₁)²` with `−5 ≤ x₁,x₂ ≤ 5` using PSO |
| **Task 2** | Compare gbest-PSO vs lbest-PSO in terms of convergence, runtime, and solution quality |

---

## Project Structure
```
pso_lab/
├── pso.py           # Core PSO algorithms (gbest, lbest, comparison utility)
├── app.py           # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

---

## Algorithm Details

### Global Best PSO (gbest / Star Topology)
Velocity update rule:
```
v_i(t+1) = w·v_i(t) + c₁·r₁·(y_i − x_i(t)) + c₂·r₂·(ŷ − x_i(t))
```
- **ŷ** = global best position (shared across entire swarm)
- Faster convergence, higher risk of premature convergence

### Local Best PSO (lbest / Ring Topology)
Same formula but **ŷ** = best position among ring neighbours only.
- Slower propagation, better exploration, lower premature convergence risk

### Parameters
| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Inertia weight | w | 0.7 | Controls exploration vs exploitation |
| Cognitive coeff | c₁ | 1.5 | Attraction to personal best |
| Social coeff | c₂ | 1.5 | Attraction to (local/global) best |
| Neighbourhood size | k | 3 | Ring neighbours per particle (lbest only) |

---

## Dashboard Features
- **Theory tab** – PSO concepts + interactive 3D fitness landscape
- **Task 1 tab** – Run gbest/lbest/both; see convergence curve + animated swarm explorer
- **Task 2 tab** – Multi-run statistical comparison, box plots, bar charts, written analysis

---

## References
- Eiben & Smith, *Introduction to Evolutionary Computing*, Springer 2015
- Kennedy & Eberhart, *Particle Swarm Optimization*, ICNN 1995
- https://www.maths.uq.edu.au/MASCOS/Multi-Agent04/Fleetwood.pdf
