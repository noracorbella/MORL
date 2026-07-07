# Multi-Objective Reinforcement Learning: Computing All Optimal Policies with Many Objectives

### Master's Thesis — University Master's Degree in Modelling for Science and Engineering

**Author:** Nora Corbella Alcántara
**Supervisor:** Dr. Manel Rodríguez Soto
June 2026

---

This repository implements four multi-objective value-iteration algorithms behind a
single, small environment interface, so that **any** of the algorithms can be
applied to **any** multi-objective MDP you can express through that interface:

* **VI** — scalarised Value Iteration (weighted; solves one linear trade-off).
* **CHVI** — Convex Hull Value Iteration (weight-free; computes the whole
  convex-hull Pareto front, from which a policy for any weight or priority can be
  extracted afterwards).
* **LexVI** — Lexicographic Value Iteration (weight-free; solves one strict
  priority ordering of the objectives).
* **LHVI** — Lexicographic Hull Value Iteration (weight-free; a single solve yields
  an optimal policy for **every** priority ordering — all `d!` of them for `d`
  objectives).

The design goal is reusability: the algorithms depend **only** on an abstract
interface, [`MOEnv`](morl/core/env_interface.py), and never on any concrete
environment. To run the algorithms on a new problem you implement one `MOEnv`
subclass — see the [worked example](#applying-the-algorithms-to-your-own-environment).

---

## Repository structure

```
morl/                         # the algorithms and the interface they depend on
├── core/
│   ├── env_interface.py      #   MOEnv — the abstract environment interface
│   ├── hull_ops.py           #   convex-hull algebra shared by CHVI and LHVI
│   └── validate_env.py       #   check_moenv_contract — a reusable contract checker
└── algorithms/
    ├── vi.py                 #   value_iteration
    ├── chvi.py               #   convexhull_vi
    ├── lexvi.py              #   lexicographic_vi
    └── lhvi.py               #   lexicographic_hull_vi (+ extract_lex_policy)

environments/                 # MOEnv wrappers for the built-in environments
├── deep_sea_treasure/        #   wrapper.py serves DST, concave and mirrored (via env_id)
│   ├── wrapper.py
│   └── check_wrapper.py      #   runs check_moenv_contract on the wrapper
├── resource_gathering/
│   ├── wrapper.py
│   └── check_wrapper.py
└── automated_driving/        #   self-contained: wrapper + its bundled simulator
    ├── wrapper.py
    ├── check_wrapper.py
    └── simulator/            #   the custom (non-gym) ADS simulator + assets

experiments/                  # ways to use the above
├── run.py                    #   unified runner (single CONFIGURATION block)
└── benchmark.py              #   timing benchmark tool
```

Deep Sea Treasure and Resource Gathering wrap
[MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium) environments;
Automated Driving wraps a bespoke simulator bundled under
`environments/automated_driving/simulator/`.

---

## Setup

Requires **Python 3.12**. Install the dependencies (see
[`requirements.txt`](requirements.txt) — `numpy`, `scipy`, `tqdm`,
`mo-gymnasium`) into a virtual environment:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`mo-gymnasium` is used by the Deep Sea Treasure and Resource Gathering wrappers.
Implementing your own `MOEnv` needs only `numpy` (and `scipy` for the hull
algorithms).

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
VISUALISE = False        # local only: render the policy in a pygame window
```

Then:

```bash
python experiments/run.py
```

This constructs the chosen wrapper, runs the chosen algorithm, prints the
**analytic value at the start state** and an **empirical rollout return**, saves
the policy (`train`) or loads it (`load`), and — if `VISUALISE = True` and a
display is available — animates the policy in the environment's window.

The five environment keys are `dst`, `dstc`, `dstm` (Deep Sea Treasure and its
concave / mirrored variants), `rg` (Resource Gathering) and `ads` (Automated
Driving). `WEIGHTS` is used only by `vi`; `PRIORITY` is the lexicographic order for
`lexvi` and the order used to extract a policy from the hull for `chvi`/`lhvi`.

---

## The four algorithms

Each entry point takes an `MOEnv` and a convergence threshold `theta` (max-norm).
Policies are returned as dictionaries keyed by state.

| Algorithm | Entry point | Returns |
|---|---|---|
| VI | `value_iteration(env, weights, theta)` | `(policy, Q)` — `Q[(s, a)]` scalar |
| LexVI | `lexicographic_vi(env, priority, theta)` | `(policy, Q)` — `Q[(s, a)]` vector |
| CHVI | `convexhull_vi(env, theta)` | `(None, Q_hulls)` |
| LHVI | `lexicographic_hull_vi(env, theta)` | `(None, Q_hulls)` |

For the hull algorithms, `Q_hulls[(s, a)]` is a numpy array of the Pareto-optimal
value vectors of that state-action's hull. They return `policy = None` because a
hull represents a *set* of optimal policies; extract a concrete one for a given
priority order with:

```python
policy = extract_lex_policy(Q_hulls, env, priority)   # from morl.algorithms.lhvi
```

CHVI and LHVI compute the full family in a single solve; LexVI solves one priority
order per run.

---

## Applying the algorithms to your own environment

This is the point of the interface: implement one `MOEnv` subclass and all four
algorithms apply to your problem unchanged.

### 1. The `MOEnv` contract

Subclass [`morl.core.env_interface.MOEnv`](morl/core/env_interface.py) and provide
two attributes and four methods:

| member | meaning |
|---|---|
| `n_objectives: int` | length of every reward vector (constant). |
| `gamma: float` | discount factor, `0 <= gamma <= 1` (constant). |
| `states()` | all states, terminal ones included; each **hashable and canonical** (equal situations must be the identical object). |
| `actions(state)` | the actions available in a non-terminal state (may be empty on terminal states). |
| `transitions(state, action)` | the distribution as a list of `(prob, next_state, reward_vector)` triples. |
| `is_terminal(state)` | whether the state is terminal (absorbing). |

Three points of the `transitions` contract are worth stating explicitly (the full
contract is in the `MOEnv` docstrings):

* **Probabilities** are non-negative and sum to `1.0`; every `next_state` is one of
  `states()`; every `reward_vector` has length `n_objectives`.
* **Determinism is a single outcome:** a deterministic transition is simply
  `[(1.0, next_state, reward_vector)]`.
* **Terminal reward rides the incoming transition.** A terminal state's value is
  the zero vector; the reward for *reaching* it is carried on the transition that
  leads into it, and the algorithms never back up terminal states.

### 2. A minimal, copy-runnable example

A four-state, two-objective toy MOMDP. From `start` the agent chooses path `A`
(favouring objective 0) or path `B` (favouring objective 1); each ends in the
terminal `end`. There are two Pareto-optimal policies.

```python
from morl.core.env_interface import MOEnv


class TinyMOEnv(MOEnv):
    n_objectives = 2
    gamma = 0.9

    def states(self):
        return ["start", "A", "B", "end"]

    def is_terminal(self, state):
        return state == "end"

    def actions(self, state):
        return {"start": [0, 1], "A": [0], "B": [0], "end": []}[state]

    def transitions(self, state, action):
        # each outcome is (prob, next_state, reward_vector); deterministic here,
        # so every distribution is a single (prob=1.0, ...) triple
        table = {
            ("start", 0): [(1.0, "A", [1.0, 0.0])],
            ("start", 1): [(1.0, "B", [0.0, 1.0])],
            ("A", 0):     [(1.0, "end", [2.0, 0.0])],
            ("B", 0):     [(1.0, "end", [0.0, 2.0])],
        }
        return table[(state, action)]
```

(For a *stochastic* transition you would return several triples whose
probabilities sum to 1, e.g.
`[(0.1, "hit", [-1.0, 0.0]), (0.9, "safe", [0.0, 0.0])]`.)

### 3. Validate it against the contract

Before solving, check the environment satisfies the contract — this catches the
common mistakes (probabilities that don't sum to 1, an unreachable/unlisted
`next_state`, a wrong-length reward vector):

```python
from morl.core.validate_env import check_moenv_contract

check_moenv_contract(TinyMOEnv())   # prints a PASS/FAIL report; returns True if all pass
```

### 4. Solve it

```python
from morl.algorithms.chvi import convexhull_vi
from morl.algorithms.lexvi import lexicographic_vi
from morl.algorithms.lhvi import lexicographic_hull_vi, extract_lex_policy
import numpy as np

env = TinyMOEnv()

# CHVI: the Pareto front of achievable value vectors at the start state
_, q_hulls = convexhull_vi(env, theta=1e-9)
front = np.concatenate([q_hulls[("start", a)] for a in env.actions("start")], axis=0)
print(front)                     # -> the two vertices [2.8, 0.0] and [0.0, 2.8]

# LexVI: one policy per priority order
policy, _ = lexicographic_vi(env, priority=[0, 1], theta=1e-9)
print(policy["start"])           # -> 0  (path A maximises objective 0 first)

# LHVI: all priority orders from a single solve
_, q_hulls = lexicographic_hull_vi(env, theta=1e-9)
print(extract_lex_policy(q_hulls, env, [1, 0])["start"])   # -> 1 (path B)
```

### 5. A real reference

For a production-quality `MOEnv` — wrapping an external environment, caching
transitions, handling terminal cells — read
[`environments/deep_sea_treasure/wrapper.py`](environments/deep_sea_treasure/wrapper.py)
and its contract check
[`environments/deep_sea_treasure/check_wrapper.py`](environments/deep_sea_treasure/check_wrapper.py).
The Resource Gathering (stochastic) and Automated Driving (custom simulator)
wrappers show the same pattern on harder environments.

---

## Notes

* Each built-in wrapper ships a `check_wrapper.py` that runs
  `check_moenv_contract` on it — a quick way to see the contract exercised on a
  real environment.
* [`experiments/benchmark.py`](experiments/benchmark.py) times the weight-free
  algorithms (CHVI, LexVI, LHVI) across the environments, separating model-build,
  convergence and policy-extraction. It is provided as a tool; reported timings and
  their analysis are in the thesis.

---

## Attribution and license

The Automated Driving scenario originates from Rodríguez-Soto et al.,
["Multi-objective reinforcement learning for designing ethical multi-agent
environments"](https://doi.org/10.1016/j.artint.2025.104460) (*Artificial
Intelligence*, 2025). Deep Sea Treasure and Resource Gathering are standard
benchmarks from [MO-Gymnasium](https://github.com/Farama-Foundation/MO-Gymnasium).

This repository is released under the GNU General Public License v3 — see
[`LICENSE`](LICENSE).
