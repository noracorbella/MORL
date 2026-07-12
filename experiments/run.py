"""
Runner for the MORL algorithms on any wrapped environment.

Edit the CONFIGURATION block below to choose an environment, an algorithm, and
its settings, then run this file.
This does the following:
* constructs the chosen MOEnv wrapper
* runs the unified algorithm
* extracts a policy
* optionally saves/loads the policy
* and evaluates it with a single generic rollout.
"""

import os
import pickle
import sys
from itertools import permutations

import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environments.deep_sea_treasure.wrapper import DeepSeaTreasureEnv
from environments.resource_gathering.wrapper import ResourceGatheringEnv
from environments.automated_driving.wrapper import AutomatedDrivingEnv

from morl.algorithms.vi import value_iteration
from morl.algorithms.lexvi import lexicographic_vi, lex_max
from morl.algorithms.chvi import convexhull_vi
from morl.algorithms.lhvi import lexicographic_hull_vi, extract_lex_policy, lex_hull
from morl.core.hull_ops import get_hull, canonical_order


# ===================== CONFIGURATION =====================
ENVIRONMENT = "ads"      # dst | dstc | dstm | rg | ads
ALGORITHM   = "lhvi"       # vi | chvi | lexvi | lhvi

# --- algorithm settings ---
WEIGHTS  = [1, 100, 10000]      # VI only: scalarisation weights (len must == n_objectives)
PRIORITY = [1, 2, 0]        # LexVI: the lex order; CHVI/LHVI: order used for policy extraction
THETA    = 0.01          # convergence threshold
GAMMA    = None          # None -> use the wrapper's built-in default for that env

# --- run control ---
MODE      = "load"      # train (run the algorithm, then save) | load (read a saved policy)
EVALUATE  = False         # roll out the resulting policy and report vector return
N_EVAL_EPISODES = 1000
MAX_STEPS       = 200
EVAL_SEED       = 0      # seeds the rollout sampling so stochastic-env eval is reproducible
POLICY_PATH     = None   # None -> experiments/policies/ path
VISUALISE       = True  # local only: render the policy in the env's pygame window
                         # (needs a real display; skipped if the env has no renderer)
# ========================================================


# env-name -> wrapper (takes a kwargs dict so GAMMA can be injected/omitted).
ENVIRONMENTS = {
    "dst":  lambda g: DeepSeaTreasureEnv(env_id="deep-sea-treasure-v0", **g),
    "dstc": lambda g: DeepSeaTreasureEnv(env_id="deep-sea-treasure-concave-v0", **g),
    "dstm": lambda g: DeepSeaTreasureEnv(env_id="deep-sea-treasure-mirrored-v0", **g),
    "rg":   lambda g: ResourceGatheringEnv(**g),
    "ads":  lambda g: AutomatedDrivingEnv(**g),
}


def make_env(name, gamma):
    """
    Construct the wrapper for ``name`` at ``gamma``.
    """
    g = {} if gamma is None else {"gamma": gamma}
    return ENVIRONMENTS[name](g)


def start_state_hull(q_hulls, env, kind):
    """The value hull at the initial state: the achievable value-vector frontier
    from ``env.start_state``, taken over the union of that state's action hulls.

    Reproduces the old ``get_initial_state_hull``: a convex hull for CHVI
    (``kind="chvi"``), the lexicographic hull for LHVI (``kind="lhvi"``). This is
    the per-state V-ring at the start (not every state's hull). Read-only -- built
    from the already-computed Q-ring.
    """
    s0 = env.start_state
    vertices = np.concatenate([q_hulls[(s0, a)] for a in env.actions(s0)], axis=0)
    if len(vertices) > 1:
        vertices = get_hull(vertices) if kind == "chvi" else lex_hull(vertices, env.n_objectives)
    return canonical_order(vertices)


def run_algorithm(env):
    """
    Run the algorithm and return ``(policy, all_policies, v_start, start_hull, q_hulls)``.

    ``policy`` is the dict (state -> action) used for evaluation.
    ``all_policies`` is ``None`` except for LHVI, where it is
    ``{priority_tuple: policy_dict}`` for every one of the ``n_objectives!``
    priority orders (LHVI computes them all in one pass).
    ``v_start`` is the analytic value at ``env.start_state`` for the scalar/vector
    algorithms (VI scalar, LexVI vector), else ``None``. ``start_hull`` is the
    start-state value hull for the hull algorithms (CHVI/LHVI), else ``None``.
    ``q_hulls`` is the full Q-ring for CHVI/LHVI, else ``None`` -- returned so
    ``main`` can persist it, letting ``load`` extract a policy for ANY priority
    later without re-solving (VI/LexVI have no such reusable artifact: their Q
    table is computed under a fixed weight/priority baked into convergence
    itself, so there is nothing cheap to re-extract).
    """
    s0 = env.start_state

    if ALGORITHM == "vi":
        policy, Q = value_iteration(env, weights=WEIGHTS, theta=THETA)
        v_start = max(Q[(s0, a)] for a in env.actions(s0))  # V(s0) = max_a Q(s0,a)
        return policy, None, v_start, None, None

    if ALGORITHM == "lexvi":
        policy, Q = lexicographic_vi(env, priority=PRIORITY, theta=THETA)
        v_start = Q[(s0, policy[s0])]  # vector value of the chosen action at s0
        return policy, None, v_start, None, None

    if ALGORITHM == "chvi":
        # Hull-valued: extract a concrete policy from the Q-hull via the
        # lex-max operation for PRIORITY.
        _none, q_hulls = convexhull_vi(env, theta=THETA)
        policy = extract_lex_policy(q_hulls, env, PRIORITY)
        return policy, None, None, start_state_hull(q_hulls, env, "chvi"), q_hulls

    if ALGORITHM == "lhvi":
        _none, q_hulls = lexicographic_hull_vi(env, theta=THETA)
        all_policies = {
            tuple(order): extract_lex_policy(q_hulls, env, list(order))
            for order in permutations(range(env.n_objectives))
        }
        policy = all_policies[tuple(PRIORITY)]
        return policy, all_policies, None, start_state_hull(q_hulls, env, "lhvi"), q_hulls

    raise ValueError(f"Unknown ALGORITHM {ALGORITHM!r}")


def evaluate(env, policy, gamma, n_episodes, max_steps, seed):
    """
    Roll the policy out and report the mean discounted vector return.

    A single MOEnv rollout:
    from ``env.start_state`` repeatedly take ``policy[state]`` and sample the next
    state from ``env.transitions`` according to its probabilities, accumulating
    ``gamma**step * reward_vector`` (step 0-indexed from the start).
    An episode ends on a terminal state or after ``max_steps``. 
    Sampling is seeded so stochastic-environment returns are reproducible,
    deterministic environments should give the same rollout regardless.
    """
    rng = np.random.default_rng(seed)
    returns = []
    lengths = []
    for _ in tqdm(range(n_episodes), desc="Evaluating"):
        state = env.start_state
        total = np.zeros(env.n_objectives)
        step = 0
        while step < max_steps and not env.is_terminal(state):
            action = policy[state]
            outcomes = env.transitions(state, action)
            probs = [p for (p, _ns, _r) in outcomes]
            idx = rng.choice(len(outcomes), p=probs)
            _p, next_state, reward = outcomes[idx]
            total += (gamma ** step) * np.asarray(reward, dtype=float)
            state = next_state
            step += 1
        returns.append(total)
        lengths.append(step)
    returns = np.array(returns)
    return {
        "mean_return": returns.mean(axis=0),
        "std_return": returns.std(axis=0),
        "mean_length": float(np.mean(lengths)),
        "episode_returns": returns,
    }


def print_start_hull(start_hull, start_state):
    """Feature 1: print the start-state value hull (CHVI/LHVI), matching the old
    'VALUE HULL AT INITIAL STATE' report -- the set of Pareto-optimal value
    vectors achievable from the start."""
    print(f"\n{'=' * 60}")
    print("VALUE HULL AT INITIAL STATE")
    print('=' * 60)
    print(f"Initial state: {start_state}")
    print(f"Number of vertices: {len(start_hull)}")
    print("Vertices (Pareto-optimal value vectors):")
    for v in start_hull:
        print(f"  {np.array2string(np.asarray(v), precision=4, floatmode='fixed', suppress_small=True)}")
    print('=' * 60)


def print_analytic_v_start(algorithm, priority, v_start, start_hull, start_state):
    """Feature 2: print the analytic value at the start state -- the value the
    algorithm computed, distinct from the empirical rollout return. For the hull
    algorithms this is the single hull vertex the PRIORITY-extracted policy
    selects (one vertex read out of the start hull via lex-max)."""
    print(f"\nAnalytic V(start)  [start_state = {start_state}]")
    if algorithm == "vi":
        print(f"  scalar V(start) = {v_start:.4f}")
    elif algorithm == "lexvi":
        print(f"  vector V(start) = {np.array2string(np.asarray(v_start), precision=4, floatmode='fixed', suppress_small=True)}")
    else:  # chvi / lhvi: the vertex the extracted policy for PRIORITY lands on
        vertex = start_hull[lex_max(start_hull, priority)]
        print(f"  vertex for priority {priority} = "
              f"{np.array2string(np.asarray(vertex), precision=4, floatmode='fixed', suppress_small=True)}")


def default_policy_path():
    """
    Auto path for the policy file, from env/algorithm and, for algorithms whose
    solve is tied to one priority/weight vector, that priority or weight.

    CHVI and LHVI save the priority-INDEPENDENT ``q_hulls`` (one solve serves
    every priority), so their filename does not encode PRIORITY -- the same file
    is found again by ``load`` no matter what PRIORITY is currently configured.
    VI (tied to WEIGHTS) and LexVI (tied to PRIORITY, baked into its convergence)
    have no such reusable artifact, so their filename keeps encoding it.
    """
    if ALGORITHM == "vi":
        tag = "-".join(map(str, WEIGHTS))
    elif ALGORITHM == "lexvi":
        tag = "-".join(map(str, PRIORITY))
    else:  # chvi, lhvi
        tag = "hulls"
    return os.path.join(
        REPO_ROOT, "experiments", "policies", f"{ENVIRONMENT}_{ALGORITHM}_{tag}.pkl"
    )


def save_policy(path, policy, all_policies, gamma, v_start, q_hulls):
    """
    Pickle the policy plus the metadata needed to reload it correctly (notably
    ``gamma``, so ``load`` rebuilds the env identically).

    For CHVI/LHVI, the full ``q_hulls`` Q-ring is pickled instead of a frozen
    start-state hull: ``load`` derives the start-state hull and extracts a policy
    FRESH from ``q_hulls`` for whatever PRIORITY is currently configured, so
    switching priorities never re-solves and never risks replaying a value that
    was actually computed under a different priority. For LHVI, ``all_policies``
    (every priority order's policy, already computed in the one solve) is kept
    too, for convenience. VI/LexVI have no such reusable artifact -- their Q
    table is computed under a fixed weight/priority baked into convergence
    itself -- so ``v_start`` and ``policy`` are simply saved and replayed as-is.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {
        "environment": ENVIRONMENT,
        "algorithm": ALGORITHM,
        "gamma": gamma,
        "theta": THETA,
        "weights": WEIGHTS if ALGORITHM == "vi" else None,
        "priority": None if ALGORITHM == "vi" else PRIORITY,
        "policy": policy,
        "all_policies": all_policies,  # {priority_tuple: policy_dict} for LHVI, else None
        "v_start": v_start,            # analytic V(start) for VI/LexVI, else None
        "q_hulls": q_hulls,            # full Q-ring for CHVI/LHVI, else None
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)
    print(f"Saved policy to {path}")


def load_policy(path):
    """
    Load a saved policy.
    """
    with open(path, "rb") as f:
        blob = pickle.load(f)
    print(f"Loaded policy from {path}")
    return blob


def main():
    path = POLICY_PATH if POLICY_PATH is not None else default_policy_path()

    print("="*60)
    print(f"Environment: {ENVIRONMENT} | Algorithm: {ALGORITHM} | Mode: {MODE}")    
    print("="*60)

    if MODE == "train":
        env = make_env(ENVIRONMENT, GAMMA)
        print(f"Discount gamma = {env.gamma}, theta = {THETA}")
        if ALGORITHM == "vi":
            print(f"Weights = {WEIGHTS}")
        else:
            print(f"Priority = {PRIORITY}")
        policy, all_policies, v_start, start_hull, q_hulls = run_algorithm(env)
        save_policy(path, policy, all_policies, env.gamma, v_start, q_hulls)

    elif MODE == "load":
        blob = load_policy(path)
        # Rebuild the env with the SAME gamma the policy was trained under.
        env = make_env(ENVIRONMENT, blob["gamma"])
        print(f"Discount gamma = {env.gamma} (from saved policy)")

        if ALGORITHM in ("chvi", "lhvi"):
            q_hulls = blob.get("q_hulls")
            if q_hulls is None:
                print("(this saved policy has no q_hulls -- it predates hull "
                      "persistence; falling back to its frozen policy, which was "
                      "extracted for whatever PRIORITY it was trained with)")
                policy = blob["policy"]
                start_hull = None
            else:
                # Cheap: extract fresh for the CURRENT PRIORITY, no re-solve.
                print(f"Extracting policy for priority {PRIORITY} from the saved "
                      f"q_hulls (no re-solve)...")
                policy = extract_lex_policy(q_hulls, env, PRIORITY)
                start_hull = start_state_hull(q_hulls, env, ALGORITHM)
            v_start = None
        else:
            policy = blob["policy"]
            v_start = blob.get("v_start")
            start_hull = None

    else:
        raise ValueError(f"Unknown MODE {MODE!r}")

    # Feature 1: print the start-state value hull (CHVI/LHVI).
    if ALGORITHM in ("chvi", "lhvi"):
        if start_hull is not None:
            print_start_hull(start_hull, env.start_state)
        else:
            print("\n(start-state hull not in this saved policy; re-run in train "
                  "mode to populate it)")

    # Feature 2: print the analytic V(start), distinct from the rollout return.
    if v_start is not None or start_hull is not None:
        print_analytic_v_start(ALGORITHM, PRIORITY, v_start, start_hull, env.start_state)
    else:
        print("\n(analytic V(start) not in this saved policy; re-run in train mode)")

    if EVALUATE:
        print(f"\nEvaluating over {N_EVAL_EPISODES} episode(s), "
              f"max_steps={MAX_STEPS}, seed={EVAL_SEED}...")
        results = evaluate(env, policy, env.gamma, N_EVAL_EPISODES, MAX_STEPS, EVAL_SEED)
        mean_vec = results["mean_return"]
        std_vec = results["std_return"]
        print(f"Mean episode length : {results['mean_length']:.2f}")
        print(f"Mean discounted vector return : {np.array2string(mean_vec, precision=4, floatmode='fixed', suppress_small=True)}")
        print(f"Std  discounted vector return : {np.array2string(std_vec, precision=4, floatmode='fixed', suppress_small=True)}")
        if ALGORITHM == "vi":
            w = np.array(WEIGHTS, dtype=float)
            scal = np.dot(results["episode_returns"], w)
            print(f"Weights = {WEIGHTS}")
            print(f"Mean scalarised return : {scal.mean():.4f}  (std {scal.std():.4f})")

    # Optional local visualisation: only if requested AND the env provides the
    # optional render_policy hook (envs without a renderer skip it gracefully).
    # Off the automated/validation flow -- needs a real display.
    if VISUALISE:
        if hasattr(env, "render_policy"):
            print("\nVisualising policy (close the window or Ctrl-C to stop)...")
            env.render_policy(policy)
        else:
            print(f"\n(no renderer for environment '{ENVIRONMENT}'; skipping visualisation)")

    env.close()


if __name__ == "__main__":
    main()
