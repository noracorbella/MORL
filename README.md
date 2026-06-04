# Multi-Objective Reinforcement Learning — TFM

This repository contains the code developed for a Master's Thesis on **Multi-Objective Reinforcement Learning (MORL)**. It implements and compares four planning algorithms across several environments, including a custom autonomous driving scenario and standard MORL benchmarks.

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **VI** | Standard Value Iteration |
| **CHVI** | Convex Hull Value Iteration — computes the full Pareto-optimal set of policies |
| **LGVI-LexMax** | Lexicographic Goal Value Iteration with lexicographic maximisation |
| **LGVI-LexHull** | Lexicographic Goal Value Iteration with convex hull over lexicographic sets |

---

## Repository Structure

```
TFM/
├── AutomatedDriving/       # Custom autonomous driving environment
├── OtherEnvs/              # Standard MORL benchmark environments
└── Plots/                  # Visualisation scripts and generated figures
```

### AutomatedDriving

A custom stochastic environment modelling an autonomous vehicle facing an ethical dilemma: three competing reward signals (car safety, pedestrian 1, pedestrian 2). Implements all four algorithms in both deterministic and stochastic variants.

```
AutomatedDriving/
├── Main_def.py             # Entry point — configure and run here
├── environment/            # Environment definition (grid, agent, items)
├── algorithms/             # VI, CHVI, LGVI-LexMax, LGVI-LexHull
├── utils/                  # Policy evaluation, auxiliary functions, plots
├── Comparison/             # Benchmarking scripts (runtime and solution quality)
├── benchmark_results/      # Saved benchmark outputs (.txt)
└── Other/                  # Intermediate development versions (kept for reference)
```

**Run:**
```bash
python AutomatedDriving/Main_def.py
```
Edit the `CONFIGURATION` block at the top of `Main_def.py` to select the algorithm and settings.

---

### OtherEnvs

Five standard MORL benchmark environments. Each follows an identical folder structure.

| Environment | Rewards | Description |
|-------------|---------|-------------|
| **DeepSeaTreasure** | 2 | Classic MORL benchmark — time vs treasure |
| **DeepSeaTreasureConcave** | 2 | Variant with a concave Pareto front |
| **DeepSeaTreasureMirrored** | 2 | Variant with a mirrored Pareto front |
| **ResourceGathering** | 3 | Agent collects resources while avoiding enemies |
| **BreakableBottles** | 3 | Safety-constrained navigation |
| **FruitTree** | 6 | Tree-structured environment with six objectives |

Each environment shares the same layout:
```
<EnvName>/
├── <Env>_main.py           # Entry point — configure and run here
├── <Env>_Benchmark.py      # Runtime benchmarking
├── environment/            # Environment class
├── algorithms/             # VI, CHVI, LGVI-LexMax, LGVI-LexHull
└── utils/                  # Policy evaluation, hull utilities, auxiliary functions
```

**Run** (example for ResourceGathering):
```bash
python OtherEnvs/ResourceGathering/RG_main.py
```
Edit the `CONFIGURATION` block at the top of each `*_main.py` to choose the algorithm, weights, priority order, and whether to train or load a saved policy.

---

### Plots

Visualisation tools and pre-generated figures.

```
Plots/
├── HullPlotter2D.py        # 2D Pareto front plotter
├── HullPlotter3D.py        # 3D Pareto front plotter
└── *.png                   # Generated hull and benchmark figures
```

---

## Dependencies

```bash
pip install numpy tqdm mo-gymnasium
```

> `mo-gymnasium` is only required for the environments under `OtherEnvs/`.  
> `AutomatedDriving` has no external environment dependencies.

---

## Configuration

Every entry point (`Main_def.py` / `*_main.py`) has a `CONFIGURATION` block at the top. Key parameters:

- **`algorithm_used`** — select VI, CHVI, LGVI-LexMax or LGVI-LexHull
- **`weights`** — scalarisation weights (used by VI and to extract a policy from CHVI)
- **`priority`** — lexicographic priority order over objectives (used by LGVI)
- **`Training`** — `True` to train and save, `False` to load a previously saved policy
- **`Calculate_hulls`** — `True` to recompute, `False` to load saved hulls (CHVI/LGVI-LexHull)
- **`Evaluate_policy`** — run Monte Carlo evaluation over `n_eval_episodes` episodes
- **`Test`** — visualise the policy in the environment's graphical interface
