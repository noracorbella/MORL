"""H2 timing benchmark for the unified MORL algorithms.

Reproduces the OLD per-environment benchmark measurement (the ``*_Benchmark.py``
files and AutomatedDriving's top-level ``Benchmark.py``) on the unified,
MOEnv-only algorithms, across all five environments. It is a faithful port of the
old measurement, not a new methodology; see the notes below for the few places
the unified structure forces a (documented) difference.

Three algorithms are timed -- the weight-free MORL trio, exactly as the old code:

  * CHVI   : one convex-hull solve, then extract d! lexicographic policies.
  * LexVI  : d! independent lexicographic solves (one per priority order).
  * LHVI   : one lexicographic-hull solve producing all d! policies in one pass.

(Scalar VI is weight-based and is not part of this comparison -- the old
benchmarks do not time it either.)

Measurement, matched to the old code:
  * wall-clock ``time.time()``, reported in seconds to 3 decimals;
  * a single run, no warm-up, no averaging over repeats;
  * per-environment theta / gamma / d!:
        DST, DSTc, DSTm : theta=0.01, gamma=0.99, d! = 2! = 2
        RG              : theta=0.01, gamma=0.7,  d! = 3! = 6
        ADS             : theta=0.01, gamma=0.7,  d! = 3! = 6
    (theta is 0.01 everywhere, matching the top-level AutomatedDriving/Benchmark.py
    -- NOT the Comparison/ folder, which is theta-inconsistent 0.01 vs 1.0.)

model_build handling (the one documented structural difference):
  The old code built the transition model *inside* the timed region (lazily, in
  iteration 1). The unified wrappers own the model -- DST/RG cache it lazily in
  transitions(), ADS precomputes it in the constructor -- so here model_build is
  timed as its OWN line: construct the wrapper and force a full transitions sweep
  from scratch (ADS built with use_cache=False for a from-scratch, comparable
  number). The H2 head-to-head is then run on convergence + extraction with
  model_build EXCLUDED (fairer across all three, since the old code baked
  model_build into CHVI's number but broke it out for LHVI). A ``combined`` figure
  (model_build + convergence) is also reported so the unified numbers cross-check
  against the old model-build-inclusive ones.
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
ENVS = ["dst", "dstc", "dstm", "rg", "ads"]   # subset to run (edit to skip slow ADS)
THETA = 0.01                                   # same for every env (top-level ADS Benchmark)
# ========================================================

# Wrapper factories. ADS is built from scratch (use_cache=False) so its model_build
# is the real ~90 s precompute, comparable to the old MNS build; the algorithms then
# run on the resulting in-memory transition cache.
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

    The sweep over every (state, action) forces the lazy cache for DST/RG (and is a
    cheap pass over the already-precomputed ADS cache), so model_build is the
    from-scratch model-build cost, uniform across environments and comparable to the
    old benchmark's model-build phase.
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


def run_env(name):
    """Run the three benchmarks on one environment and return its results dict."""
    env, non_terminal, orders, model_build = build_model_timed(name)
    d = len(orders)
    res = {
        "gamma": env.gamma, "n_objectives": env.n_objectives, "d_factorial": d,
        "n_non_terminal": len(non_terminal), "model_build": model_build,
        "chvi": bench_chvi(env, orders),
        "lexvi": bench_lexvi(env, orders),
        "lhvi": bench_lhvi(env, orders),
    }
    env.close()
    return res


def print_env_table(name, r):
    mb = r["model_build"]
    chvi, lexvi, lhvi = r["chvi"], r["lexvi"], r["lhvi"]
    chvi_algo = chvi["convergence"] + chvi["extraction_total"]
    lhvi_algo = lhvi["convergence"] + lhvi["extraction_total"]

    print(f"\n{'=' * 72}")
    print(f"ENV: {name}   (theta={THETA}, gamma={r['gamma']}, d!={r['d_factorial']}, "
          f"non-terminal states={r['n_non_terminal']})")
    print('=' * 72)
    print(f"model_build (from scratch): {mb:.3f} s   [EXCLUDED from H2 comparison]")

    print("\nB1 CHVI  (one solve -> extract d! policies):")
    print(f"  convergence            : {chvi['convergence']:.3f} s")
    print(f"  extraction total       : {chvi['extraction_total']:.3f} s   "
          f"(per order: {_fmt_orders(chvi['extraction'])})")
    print(f"  algo total (conv+extr) : {chvi_algo:.3f} s")
    print(f"  combined (model+conv)  : {mb + chvi['convergence']:.3f} s   "
          f"[~ old CHVI model-build-inclusive number]")

    print("\nB2 LexVI  (d! independent solves):")
    print(f"  per order              : {_fmt_orders(lexvi['runs'])}")
    print(f"  total                  : {lexvi['total']:.3f} s")
    print(f"  average per run (/{r['d_factorial']})   : {lexvi['average']:.3f} s")
    if name in ("rg", "ads"):
        print(f"    note: old {name.upper()} per-env benchmark divided by 2 over "
              f"{r['d_factorial']} runs (copy-paste bug); this /{r['d_factorial']} "
              f"average differs for that reason, not a timing change.")

    print("\nB3 LHVI  (one solve -> all d! policies):")
    print(f"  convergence            : {lhvi['convergence']:.3f} s")
    print(f"  extraction total       : {lhvi['extraction_total']:.3f} s   "
          f"(per order: {_fmt_orders(lhvi['extraction'])})")
    print(f"  algo total (conv+extr) : {lhvi_algo:.3f} s")
    print(f"  combined (model+conv)  : {mb + lhvi['convergence']:.3f} s")

    lex_total = lexvi["total"]
    lhvi_vs_lex = lex_total / lhvi_algo if lhvi_algo > 0 else float("nan")
    winner = "LHVI faster" if lhvi_vs_lex > 1 else "LexVI faster"
    print("\nH2  (algorithm time, model_build excluded):")
    print(f"  LHVI  one-solve-all-orders : {lhvi_algo:.3f} s")
    print(f"  LexVI d! independent runs  : {lex_total:.3f} s")
    print(f"  ratio LexVI_total / LHVI_algo = {lhvi_vs_lex:.2f}  "
          f"(>1 means LHVI wins)  -> {winner}")
    print(f"  CHVI  one-solve-all-orders : {chvi_algo:.3f} s   "
          f"(LHVI/CHVI algo ratio = {lhvi_algo / chvi_algo:.2f})")


def main():
    print("Unified H2 timing benchmark  (single run, wall-clock seconds)")
    results = {}
    for name in ENVS:
        print(f"\n>>> running {name} ...", flush=True)
        results[name] = run_env(name)
        print_env_table(name, results[name])

    print(f"\n{'=' * 72}")
    print("H2 SUMMARY  (algorithm time; LHVI one solve for all d! vs d! LexVI runs)")
    print('=' * 72)
    print(f"{'env':6} {'LHVI algo':>12} {'LexVI total':>12} {'CHVI algo':>12} "
          f"{'LexVI/LHVI':>12}  (>1: LHVI wins)")
    for name in ENVS:
        r = results[name]
        lhvi_algo = r["lhvi"]["convergence"] + r["lhvi"]["extraction_total"]
        chvi_algo = r["chvi"]["convergence"] + r["chvi"]["extraction_total"]
        lex_total = r["lexvi"]["total"]
        ratio = lex_total / lhvi_algo if lhvi_algo > 0 else float("nan")
        print(f"{name:6} {lhvi_algo:12.3f} {lex_total:12.3f} {chvi_algo:12.3f} "
              f"{ratio:12.2f}")


if __name__ == "__main__":
    main()
