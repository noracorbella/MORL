# Multi-Objective Reinforcement Learning for Automated Driving Systems

This repository contains the research code for a Master's Thesis on applying **multi-objective reinforcement learning** to ethical decision-making in autonomous vehicles — specifically how an AV should navigate when it must balance competing objectives: reaching its goal, avoiding internal collisions, and protecting pedestrians.

## Research Overview

A controlled car must navigate a grid while two pedestrians move through the environment. Three objectives compete:

- **Objective 0 — Vehicle goal**: reaching the destination
- **Objective 1 — Internal safety**: avoiding collisions with obstacles
- **Objective 2 — External safety**: avoiding collisions with pedestrians

Classical single-objective RL collapses these into a scalar reward using manually specified weights, which is brittle and opaque. This project implements and compares two weight-free multi-objective approaches:

1. **Convex Hull Value Iteration (CHVI)** — computes the full Pareto front; any policy can be extracted post-hoc by specifying weights
2. **Lexicographic Value Iteration (LG-VI)** — eliminates weight specification entirely by defining a strict priority ordering over objectives

The progression across three versions shows how the algorithms evolve from a deterministic baseline to handling stochastic pedestrian behavior to removing the need for weights altogether.

## Repository Structure

```
AutomatedDriving/
├── ADS_code/        Version 1 — Deterministic baseline (Q-Learning, VI, Convex Hull VI)
├── Stochastic/      Version 2 — Stochastic pedestrian behavior (all algorithms adapted)
├── Lexicographic/   Version 3 — Weight-free lexicographic MORL (all 6 priority orderings)
└── Comparison/      Algorithm benchmarking: runtime and policy comparison
```

Each version is self-contained with its own `README.md`.

## Environment

| Property | Value |
|---|---|
| Grid | 9×7 cells (63 total) |
| Cell types | Accessible (car), Pedestrian-only, Crosswalk (everyone), Inaccessible |
| Agents | 1 controlled car + 2 pedestrians |
| State space | ~250,000 states (`car_pos × ped1_pos × ped2_pos`) |
| Actions | 6 (RIGHT, UP, LEFT + fast variants) |
| Reward | Vectorial `[r_car, r_ped1, r_ped2]` |

## Algorithms

| Algorithm | Module | Weight-free? | Stochastic? |
|---|---|---|---|
| Q-Learning | All | No | Via sampling |
| Value Iteration | All | No | Yes (v2+) |
| Convex Hull VI (CHVI) | ADS_code, Stochastic | No (post-hoc) | Yes (Stochastic) |
| Lexicographic VI (LG-VI) | Lexicographic | **Yes** | Yes |
| Lexicographic Convex Hull VI | Lexicographic | **Yes** | Yes |

## Quick Start

```bash
# Run the algorithm benchmark (trains from scratch — see timing table below)
cd Comparison
python Benchmark1_CHVI.py          # CHVI: ~222 s total
python Benchmark2_LGVI.py          # 6x LG-VI: ~131 s total
python Benchmark3_LGVI_lexhull.py  # LG Hull VI: ~139 s total

# Train and visualise a lexicographic policy
cd Lexicographic
python Main_lexmax_simple.py
```

## Requirements

```
Python 3.7+
numpy
scipy
pygame
tqdm
matplotlib
```

Install with:
```bash
pip install numpy scipy pygame tqdm matplotlib
```

## Generated Files

The `policies/`, `allpolicies/`, and `videos/` directories are excluded from this repository (see `.gitignore`). All algorithms train from scratch. Typical training times:

| Module | Algorithm | Approx. time |
|---|---|---|
| ADS_code | Value Iteration | < 1 min |
| Stochastic | VI Stochastic | ~2 min |
| Stochastic | Convex Hull VI | ~5 min |
| Lexicographic | LG-VI (one priority) | ~2 min |
| Lexicographic | LG Convex Hull VI | ~2.5 min |

See each module's `README.md` for full training and testing instructions.

## Citation

Corbella Alcántara, Nora. *Multi-Objective Reinforcement Learning for Ethical Decision-Making in Automated Driving Systems*. Master's Thesis, Universitat Politècnica de Catalunya, 2026.

## License

Apache License 2.0 — see [LICENSE](../LICENSE).