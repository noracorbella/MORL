# Multi-Objective Reinforcement Learning: Computing All Optimal Policies with Many Objectives

### Master's Thesis - University Master's Degree in Modelling for Science and Engineering

**Author:** Nora Corbella Alcántara
**Supervisor:** Dr. Manel Rodríguez Soto
June 2026

---

This repository implements four multi-objective value-iteration algorithms on a MOMDP grid environment:

* **VI** - Value Iteration.
* **CHVI** - Convex Hull Value Iteration (computes the whole
  convex hull, from which a policy for any weight or lexicographic order can be extracted afterwards).
* **LexVI** - Lexicographic Value Iteration (solves one lexicographic
  order of the objectives).
* **LHVI** - Lexicographic Hull Value Iteration (a single run yields
  an optimal policy for every lexicographic order (`d!` of them for `d`
  objectives)).

The algorithms depend only on the [`MOEnv`](morl/core/env_interface.py) class, and not on any concrete environment.

---

## Repository structure

```
morl/
├── core/
│   ├── env_interface.py      #   MOEnv abstract environment interface
│   ├── hull_ops.py           #   convex hull operations (for CHVI and LHVI)
│   └── validate_env.py       #   checker for a new environment
└── algorithms/
    ├── vi.py                 #   value_iteration
    ├── chvi.py               #   convexhull_vi
    ├── lexvi.py              #   lexicographic_vi
    └── lhvi.py               #   lexicographic_hull_vi (+ extract_lex_policy)

environments/                 # MOEnv wrappers for the environments
├── deep_sea_treasure/        #   wrapper.py for DST (for standard, concave and mirrored)
│   ├── wrapper.py
├── resource_gathering/
│   ├── wrapper.py
└── automated_driving/        #   wrapper + simulator
    ├── wrapper.py
    └── simulator/            #  ADS simulator

experiments/                  # ways to use the above
├── run.py                    #   runner
└── benchmark.py              #   timing benchmark tool
```

Deep Sea Treasure and Resource Gathering wrap
[MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium) environments.

Automated Driving wraps the simulator on `environments/automated_driving/simulator/`.

---

## Setup

Requires **Python 3.12**. Install the dependencies (see
[`requirements.txt`](requirements.txt) - `numpy`, `scipy`, `tqdm`,
`mo-gymnasium`) into a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`mo-gymnasium` is used for the Deep Sea Treasure and Resource Gathering wrappers.

---

## Running the built-in environments

Everything is driven from the `CONFIGURATION` block at the top of
[`experiments/run.py`](experiments/run.py):

```python
ENVIRONMENT = "dst"      # dst | dstc | dstm | rg | ads
ALGORITHM   = "vi"       # vi | chvi | lexvi | lhvi

WEIGHTS  = [1, 100]      # VI only: scalarisation weights (len == n_objectives)
PRIORITY = [0, 1]        # LexVI: the lex order; CHVI/LHVI: order for policy extraction
THETA    = 0.01          # convergence threshold
GAMMA    = None          # None -> use the wrapper's built-in default for that env

MODE      = "train"      # train (run the algorithm, then save) | load (read a saved policy)
EVALUATE  = True         # roll the policy out and report its return
VISUALISE = False        # render the policy in a pygame window
```

Then:

```bash
python experiments/run.py
```

This constructs the wrapper, runs the algorithm, prints the analytic value 
at the start state and an empirical rollout return, saves the policy (`train`)
or loads it (`load`), and if `VISUALISE = True` it animates the policy in the environment's window.

The five environment are `dst`, `dstc`, `dstm` (Deep Sea Treasure and its concave 
/ mirrored variants), `rg` (Resource Gathering) and `ads` (Automated Driving). 
`WEIGHTS` is used only by `vi`, `PRIORITY` is the lexicographic order for
`lexvi` and the order used to extract a policy from the hull for `chvi`/`lhvi`.

---

## The four algorithms

Each entry point takes an `MOEnv` and a convergence threshold `theta`.
Policies are returned as dictionaries keyed by state.

| Algorithm | Entry point | Returns |
|---|---|---|
| VI | `value_iteration(env, weights, theta)` | `(policy, Q)` - `Q[(s, a)]` scalar |
| LexVI | `lexicographic_vi(env, priority, theta)` | `(policy, Q)` - `Q[(s, a)]` vector |
| CHVI | `convexhull_vi(env, theta)` | `(None, Q_hulls)` |
| LHVI | `lexicographic_hull_vi(env, theta)` | `(None, Q_hulls)` |

For the hull algorithms, `Q_hulls[(s, a)]` is a numpy array of the optimal
value vectors of that state-action's hull. They return `policy = None` because a
hull represents a *set* of optimal policies. You can extract a concrete one for a given
priority order with:

```python
policy = extract_lex_policy(Q_hulls, env, priority)   # from morl.algorithms.lhvi
```

---

## How we apply the algorithms to a specific environment

Implement one `MOEnv` subclass and all four algorithms apply automatically.

### 1. The `MOEnv` subclass

[`morl.core.env_interface.MOEnv`](morl/core/env_interface.py) 

For an environment two classes and four methods need provinding.

| member | meaning |
|---|---|
| `n_objectives: int` | length of every reward vector (constant). |
| `gamma: float` | discount factor, `0 <= gamma <= 1` (constant). |
| `states()` | all states, terminal ones included. |
| `actions(state)` | the actions available in a non-terminal state (may be empty on terminal states). |
| `transitions(state, action)` | the distribution as a list of `(prob, next_state, reward_vector)`. |
| `is_terminal(state)` | whether the state is terminal. |

Three points of `transitions` are:

* **Probabilities** are non-negative and sum to `1.0`; every `next_state` is one of
  `states()`; every `reward_vector` has length `n_objectives`.
* **Determinism is a single outcome:** a deterministic transition is simply
  `[(1.0, next_state, reward_vector)]`.
* **Terminal reward rides the incoming transition.** A terminal state's value is
  the zero vector; the reward for *reaching* it is carried on the transition that
  leads into it, and the algorithms never back up terminal states.

---

## Reproducing the results

All results in the thesis were produced with these settings:

| Environment | `gamma` | `theta` | Priority orders |
|---|---|---|---|
| `dst`, `dstc`, `dstm` | 0.99 | 0.01 | `[0, 1]`, `[1, 0]` (d! = 2) |
| `rg` | 0.7 | 0.01 | `[2, 1, 0]` (d! = 6) |
| `ads` | 0.7 | 0.01 | `[2, 1, 0]` and `[0, 2, 1]` (d! = 6) |

`GAMMA = None` in the configuration blocks already selects these gammas (they are
the wrappers' defaults), and `THETA = 0.01` is the default everywhere. For ADS,
the wrapper's default `degree_of_stochasticity = 1` (a pedestrian on cell `[3, 3]`
moves stochastically) is the setting used for all thesis results. Note that the
first ADS run builds its transition model (a few minutes) and caches it to disk;
later runs load the cache in under a second.

### Value hulls at the start state

Set in `experiments/run.py`:

```python
ENVIRONMENT = "dst"    # or dstc | dstm | rg | ads
ALGORITHM   = "chvi"   # or "lhvi"
MODE        = "train"
```

and run `python experiments/run.py`. Training prints the
`VALUE HULL AT INITIAL STATE` block: the achievable value vectors at the start
state (the Pareto front for CHVI, the lexicographically optimal subset for LHVI),
plus the vertex selected for the configured `PRIORITY`. These vertex lists are
the hull results reported in the thesis.

### Policy evaluation returns

Set `EVALUATE = True` (and `MODE = "load"` to reuse a trained policy). The runner
rolls the extracted policy out from the start state and reports the mean and
standard deviation of the discounted vector return. On the deterministic
environments (`dst`, `dstc`, `dstm`) every episode is identical, so one episode
suffices and the empirical return equals the analytic `V(start)`. On the
stochastic environments (`rg`, `ads`) increase `N_EVAL_EPISODES`; the sampling is
seeded by `EVAL_SEED`, so a given seed and episode count is exactly reproducible.

### Timing benchmarks

Set the environment and algorithm in `experiments/benchmark.py`:

```python
ENVIRONMENT = "dst"    # dst | dstc | dstm | rg | ads
ALGORITHM   = "chvi"   # chvi | lexvi | lhvi
```

and run `python experiments/benchmark.py`. Each run reports:

* `model_build` - building the transition model from scratch (ADS deliberately
  bypasses its disk cache here so this is the real cost);
* `convergence` and per-order `extraction`;
* `algo total` (convergence + extraction) - the number used to compare
  CHVI / LexVI / LHVI against each other;
* `combined` (model build + convergence + extraction) - the full from-scratch cost.

Timings are wall-clock and machine-dependent: absolute values will differ from
the thesis tables on other hardware, but the relative comparisons between
algorithms are what the thesis results rest on.

### Hull figures

The 2D/3D hull figures were produced with
[`Plots/HullPlotter2D.py`](Plots/HullPlotter2D.py) and
[`Plots/HullPlotter3D.py`](Plots/HullPlotter3D.py). These scripts do not read
files: the vertex arrays (`chvi_points`, `lexhull_points`) are pasted in at the
top of each script. To regenerate a figure, run CHVI/LHVI as above, copy the
printed start-state hull vertices into those arrays, and run the script
(requires `matplotlib`, not in `requirements.txt`).


---

## Attribution and license

The Automated Driving scenario originates from Rodríguez-Soto et al.,
["Multi-objective reinforcement learning for designing ethical multi-agent
environments"](https://doi.org/10.1016/j.artint.2025.104460) (*Artificial
Intelligence*, 2025). Deep Sea Treasure and Resource Gathering are standard
benchmarks from [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

This repository is released under the GNU General Public License v3 - see
[`LICENSE`](LICENSE).
