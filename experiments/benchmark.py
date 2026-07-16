"""
Timing benchmark.

Edit the CONFIGURATION block below to choose an environment and an algorithm,
then run this file to time that one case.

Three algorithms can be timed (VI is weight-based and not part of this H2 timing):

  * CHVI   : one convex-hull solve, then extract d! lexicographic policies.
  * LexVI  : d! independent lexicographic solves (one per priority order).
  * LHVI   : one lexicographic-hull solve producing all d! policies in one pass.

Measurement
  * wall-clock ``time.time()``
  * a single run that builds the model from scratch
  * per-environment theta / gamma / d!:
        DST, DSTc, DSTm : theta=0.01, gamma=0.99, d! = 2! = 2
        RG              : theta=0.01, gamma=0.7,  d! = 3! = 6
        ADS             : theta=0.01, gamma=0.7,  d! = 3! = 6
"""

import os
import sys
import time
from itertools import permutations

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from environments.deep_sea_treasure.wrapper import DeepSeaTreasureEnv
from environments.resource_gathering.wrapper import ResourceGatheringEnv
from environments.automated_driving.wrapper import AutomatedDrivingEnv

from morl.algorithms.lexvi import lexicographic_vi
from morl.algorithms.chvi import convexhull_vi
from morl.algorithms.lhvi import lexicographic_hull_vi, extract_lex_policy


# ===================== CONFIGURATION =====================
ENVIRONMENT = "rg"      # dst | dstc | dstm | rg | ads
ALGORITHM   = "lexvi"     # chvi | lexvi | lhvi 
THETA       = 0.01       # convergence threshold
# ========================================================

ENV_FACTORIES = {
    "dst":  lambda: DeepSeaTreasureEnv(env_id="deep-sea-treasure-v0"),
    "dstc": lambda: DeepSeaTreasureEnv(env_id="deep-sea-treasure-concave-v0"),
    "dstm": lambda: DeepSeaTreasureEnv(env_id="deep-sea-treasure-mirrored-v0"),
    "rg":   lambda: ResourceGatheringEnv(),
    "ads":  lambda: AutomatedDrivingEnv(use_cache=False),
}


def build_model_timed(name):
    """Construct the wrapper and force the full transition model to be built from
    scratch, returning ``(env, non_terminal_states, orders, model_build_seconds)``.
    """
    t0 = time.time()
    env = ENV_FACTORIES[name]()
    non_terminal = [s for s in env.states() if not env.is_terminal(s)]
    for s in non_terminal:
        for a in env.actions(s):
            env.transitions(s, a)
    model_build = time.time() - t0
    orders = [list(o) for o in permutations(range(env.n_objectives))]
    return env, non_terminal, orders, model_build


def bench_chvi(env, orders):
    """One CHVI solve, then time each of the d! lex-policy extractions."""
    t = time.time()
    _none, q_hulls = convexhull_vi(env, theta=THETA)
    convergence = time.time() - t
    extraction = {}
    for order in orders:
        t = time.time()
        extract_lex_policy(q_hulls, env, order)
        extraction[tuple(order)] = time.time() - t
    return {"convergence": convergence, "extraction": extraction,
            "extraction_total": sum(extraction.values())}


def bench_lexvi(env, orders):
    """d! independent LexVI solves (one per priority order)."""
    runs = {}
    for order in orders:
        t = time.time()
        lexicographic_vi(env, priority=order, theta=THETA)
        runs[tuple(order)] = time.time() - t
    total = sum(runs.values())
    return {"runs": runs, "total": total, "average": total / len(orders)}


def bench_lhvi(env, orders):
    """One LHVI solve (all d! policies), then time each of the d! extractions."""
    t = time.time()
    _none, q_hulls = lexicographic_hull_vi(env, theta=THETA)
    convergence = time.time() - t
    extraction = {}
    for order in orders:
        t = time.time()
        extract_lex_policy(q_hulls, env, order)
        extraction[tuple(order)] = time.time() - t
    return {"convergence": convergence, "extraction": extraction,
            "extraction_total": sum(extraction.values())}


def _fmt_orders(d):
    return "  ".join(f"{list(k)}: {v:.3f}" for k, v in d.items())


def main():
    print(f"Environment: {ENVIRONMENT} | Algorithm: {ALGORITHM}")

    env, non_terminal, orders, model_build = build_model_timed(ENVIRONMENT)
    d = len(orders)
    print(f"theta={THETA}, gamma={env.gamma}, d!={d}, non-terminal states={len(non_terminal)}")
    print(f"model_build (from scratch): {model_build:.3f} s")

    if ALGORITHM == "chvi":
        r = bench_chvi(env, orders)
        algo_total = r["convergence"] + r["extraction_total"]
        print("\nCHVI  (one solve -> extract d! policies):")
        print(f"  convergence                : {r['convergence']:.3f} s")
        print(f"  extraction total           : {r['extraction_total']:.3f} s   "
              f"(per order: {_fmt_orders(r['extraction'])})")
        print(f"  algo total (conv+extr)     : {algo_total:.3f} s")
        print(f"  combined (model+conv+extr) : {model_build + algo_total:.3f} s   "
              f"[full: build model from scratch + solve + extract all policies; "
              f"matches old benchmark's TOTAL TIME]")

    elif ALGORITHM == "lexvi":
        r = bench_lexvi(env, orders)
        print("\nLexVI  (d! independent solves):")
        print(f"  per order              : {_fmt_orders(r['runs'])}")
        print(f"  total                  : {r['total']:.3f} s")
        print(f"  average per run (/{d})   : {r['average']:.3f} s")
        print(f"  combined (model+total) : {model_build + r['total']:.3f} s   ")

    elif ALGORITHM == "lhvi":
        r = bench_lhvi(env, orders)
        algo_total = r["convergence"] + r["extraction_total"]
        print("\nLHVI  (one solve -> all d! policies):")
        print(f"  convergence                : {r['convergence']:.3f} s")
        print(f"  extraction total           : {r['extraction_total']:.3f} s   "
              f"(per order: {_fmt_orders(r['extraction'])})")
        print(f"  algo total (conv+extr)     : {algo_total:.3f} s")
        print(f"  combined (model+conv+extr) : {model_build + algo_total:.3f} s   "
              f"[full: build model from scratch + solve + extract all policies; "
              f"matches old benchmark's TOTAL TIME]")

    else:
        raise ValueError(f"Unknown ALGORITHM {ALGORITHM!r}")

    env.close()


if __name__ == "__main__":
    main()
