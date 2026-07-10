# Multi-Objective Reinforcement Learning: Computing All Optimal Policies with Many Objectives

### Master's Thesis - University Master's Degree in Modelling for Science and Engineering

**Author:** Nora Corbella Alcántara
**Supervisor:** Dr. Manel Rodríguez Soto
June 2026

---

This repository implements four multi-objective value-iteration algorithms on a MOMDP grid environment:

* **VI** - Value Iteration (weighted).
* **CHVI** - Convex Hull Value Iteration (weight-free, computes the whole
  convex hull, from which a policy for any weight or lexicographic order can be
  extracted afterwards).
* **LexVI** - Lexicographic Value Iteration (weight-free, solves one lexicographic
  order of the objectives).
* **LHVI** - Lexicographic Hull Value Iteration (weight-free, a single run yields
  an optimal policy for **every** lexicographic order (all `d!` of them for `d`
  objectives)).

The algorithms depend only on an abstract interface, [`MOEnv`](morl/core/env_interface.py), 
and never on any concrete environment. To run the algorithms on a new problem you implement 
one `MOEnv`subclass.

---

## Repository structure

```
morl/                         # the algorithms and the interface they depend on
├── core/
│   ├── env_interface.py      #   MOEnv abstract environment interface
│   ├── hull_ops.py           #   convex hull operations shared by CHVI and LHVI
│   └── validate_env.py       #   check_moenv_contract - a reusable contract checker
└── algorithms/
    ├── vi.py                 #   value_iteration
    ├── chvi.py               #   convexhull_vi
    ├── lexvi.py              #   lexicographic_vi
    └── lhvi.py               #   lexicographic_hull_vi (+ extract_lex_policy)

environments/                 # MOEnv wrappers for the environments
├── deep_sea_treasure/        #   wrapper.py for DST (for standard, concave and mirrored)
│   ├── wrapper.py
│   └── check_wrapper.py      #   runs check_moenv_contract on the wrapper
├── resource_gathering/
│   ├── wrapper.py
│   └── check_wrapper.py
└── automated_driving/        #   wrapper + simulator
    ├── wrapper.py
    ├── check_wrapper.py
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
python3.12 -m venv .venv
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
or loads it (`load`), and - if `VISUALISE = True` and a display is available -
animates the policy in the environment's window.

The five environment are `dst`, `dstc`, `dstm` (Deep Sea Treasure and its concave 
/ mirrored variants), `rg` (Resource Gathering) and `ads` (Automated Driving). 
`WEIGHTS` is used only by `vi`, `PRIORITY` is the lexicographic order for
`lexvi` and the order used to extract a policy from the hull for `chvi`/`lhvi`.

---

## The four algorithms

Each entry point takes an `MOEnv` and a convergence threshold `theta` (max-norm).
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

Three points of `transitions` are worth stating explicitly (the full contract is in the 
`MOEnv` docstrings):

* **Probabilities** are non-negative and sum to `1.0`; every `next_state` is one of
  `states()`; every `reward_vector` has length `n_objectives`.
* **Determinism is a single outcome:** a deterministic transition is simply
  `[(1.0, next_state, reward_vector)]`.
* **Terminal reward rides the incoming transition.** A terminal state's value is
  the zero vector; the reward for *reaching* it is carried on the transition that
  leads into it, and the algorithms never back up terminal states.

---

## Notes

* Each built-in wrapper ships a `check_wrapper.py` that runs
  `check_moenv_contract` on it.
* [`experiments/benchmark.py`](experiments/benchmark.py) times the weight-free
  algorithms (CHVI, LexVI, LHVI) across the environments, separating model-build,
  convergence and policy-extraction.

---

## Attribution and license

The Automated Driving scenario originates from Rodríguez-Soto et al.,
["Multi-objective reinforcement learning for designing ethical multi-agent
environments"](https://doi.org/10.1016/j.artint.2025.104460) (*Artificial
Intelligence*, 2025). Deep Sea Treasure and Resource Gathering are standard
benchmarks from [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

This repository is released under the GNU General Public License v3 - see
[`LICENSE`](LICENSE).
