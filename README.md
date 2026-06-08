# Multi-Objective Reinforcement Learning for Computing All Optimal Policies with Many Objectives

### Master's Thesis

**University Master's Degree in Modelling for Science and Engineering**

**Author:** Nora Corbella Alcántara  
**Supervisor:** Dr. Manel Rodríguez Soto

June 2026

---

This repository contains the code developed for the Master's Thesis on Multi-Objective Reinforcement Learning (MORL). It implements and compares four planning algorithms across several environments, including a custom autonomous driving scenario and standard MORL benchmarks.

## Algorithms

* Value Iteration (VI)
* Convex Hull Value Iteration (CHVI)
* Lexicographic Value Iteration (LexVI/LGVI-Lexmax)
* Lexicographic Hull Value Iteration (LHVI/LGVI-Lexhull)

## Repository Structure

```
TFM/
├── AutomatedDriving/       # Custom autonomous driving environment
├── OtherEnvs/              # Standard MORL benchmark environments
└── Plots/                  # Visualisation scripts and generated figures
```

### AutomatedDriving

A custom stochastic environment modelling an autonomous vehicle facing an ethical dilemma: three competing reward signals (car safety, pedestrian 1, pedestrian 2). Implements all four algorithms in both deterministic and stochastic variants.

The ADS environment was originally introduced in \
url{[Rodriguez Soto et al.](https://doi.org/10.1016/j.artint.2025.104460)}
```
AutomatedDriving/
├── Main_def.py             # Entry point: configure and run here
├── environment/            # Environment definition (grid, agent, items)
├── algorithms/             # VI, CHVI, LGVI-LexMax, LGVI-LexHull
├── utils/                  # Policy evaluation, auxiliary functions, plots
├── Comparison/             # Benchmarking scripts (runtime and solution quality)
├── benchmark_results/      # Saved benchmark outputs (.txt)
```

**Run:**
```bash
python AutomatedDriving/Main.py
```
Edit the `CONFIGURATION` block at the top of `Main.py` to select the algorithm and settings.

---

### OtherEnvs

Five standard MORL benchmark environments. Each follows an identical folder structure.

From \url{https://github.com/Farama-Foundation/MO-Gymnasium}

Each environment shares the same layout:
```
<EnvName>/
├── <Env>_main.py           # Entry point - configure and run here
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

---

## Configuration

Every entry point (`Main.py` / `*_main.py`) has a `CONFIGURATION` block at the top. Key parameters:

- **`algorithm_used`** - select VI, CHVI, LGVI-LexMax or LGVI-LexHull
- **`weights`** - scalarisation weights (used by VI)
- **`priority`** - lexicographic priority order over objectives (used by LGVI)
- **`Training`** - `True` to train and save, `False` to load a previously saved policy
- **`Calculate_hulls`** - `True` to recompute, `False` to load saved hulls (CHVI/LGVI-LexHull)
- **`Evaluate_policy`** - run Monte Carlo evaluation over `n_eval_episodes` episodes
- **`Test`** - visualise the policy in the environment's graphical interface
