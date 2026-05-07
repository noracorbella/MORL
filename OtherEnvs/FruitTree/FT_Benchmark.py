"""
FT_Benchmark.py
---------------
Unified benchmark runner for the Fruit Tree algorithms.

BENCHMARKS
----------
  1. CHVI   : Convex Hull Value Iteration + extract N lexicographic policies via lex_max.
  2. LGVI   : Nx Lexicographic Value Iteration (one run per priority order).
  3. LGVI-H : Lexicographic Hull Value Iteration + extract N policies in one pass.

Note: with 6 objectives there are 6! = 720 priority orders. Running all 720 for
benchmarks 2 and 3 is feasible (tree is small) but produces many policy files.
BENCHMARK_N_ORDERS controls how many to run (set to 720 for full benchmark,
or a smaller number like 6 for a quick test with one order per objective first).
"""

import os
import sys
import time
from itertools import permutations
import numpy as np


_HERE        = os.path.dirname(os.path.abspath(__file__))
POLICIES_DIR = os.path.join(_HERE, "ft_benchmark_policies")
RESULTS_DIR  = os.path.join(_HERE, "ft_benchmark_results")

sys.path.insert(0, os.path.join(_HERE, "environment"))
sys.path.insert(0, os.path.join(_HERE, "algorithms"))
sys.path.insert(0, os.path.join(_HERE, "utils"))

from FT_Environment import FTEnvironment
from FT_CHVI import convexhull_VI
from FT_LG_utils import extract_lex_policy_from_qhulls
from FT_LGVI_lexmax import LG_VI_lexmax
from FT_LGVI_lexhull_timing_v2 import LG_VI_lexhull

BENCHMARK1_CHVI         = 1
BENCHMARK2_LGVI_LEXMAX  = 2
BENCHMARK3_LGVI_LEXHULL = 3

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_TO_RUN   = BENCHMARK1_CHVI   # BENCHMARK1_CHVI | BENCHMARK2_LGVI_LEXMAX | BENCHMARK3_LGVI_LEXHULL
TREE_DEPTH         = 7                 # 5, 6 or 7
BENCHMARK_N_ORDERS = 270                 # number of priority orders to benchmark (max 720)

# Parameters
theta           = 1e-4
discount_factor = 1.0

# Files
CHVI_MNS_FILE         = os.path.join(POLICIES_DIR, f"FT_CHVI_d{TREE_DEPTH}_MNS.pkl")
CHVI_QHULLS_FILE      = os.path.join(POLICIES_DIR, f"FT_CHVI_d{TREE_DEPTH}_qhulls.pkl")
LGVI_LEXMAX_MNS_FILE  = os.path.join(POLICIES_DIR, f"FT_LGVI_lexmax_d{TREE_DEPTH}_MNS.pkl")
LGVI_LEXHULL_MNS_FILE = os.path.join(POLICIES_DIR, f"FT_LGVI_lexhull_d{TREE_DEPTH}_MNS.pkl")
LGVI_LEXHULL_V_FILE   = os.path.join(POLICIES_DIR, f"FT_LGVI_lexhull_d{TREE_DEPTH}_V_hulls.pkl")
LGVI_LEXHULL_Q_FILE   = os.path.join(POLICIES_DIR, f"FT_LGVI_lexhull_d{TREE_DEPTH}_Q_hulls.pkl")

# Generate priority orders (all 720, then take first BENCHMARK_N_ORDERS)
_all_orders = [list(p) for p in permutations(range(6))]
LEX_ORDERS  = [
    ("-".join(str(x) for x in p), p)
    for p in _all_orders[:BENCHMARK_N_ORDERS]
]
LEX_ORDER_LABELS = {tuple(priority): label for label, priority in LEX_ORDERS}


def _delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted cached file: {path}")
    else:
        print(f"No cached file found: {path}")


def _write_results(filename, report):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# BENCHMARK 1: CHVI + Extract N Lexicographic Policies
# ---------------------------------------------------------------------------

def run_benchmark1(theta, discount_factor):
    n = len(LEX_ORDERS)
    print(f"\nBENCHMARK 1: CHVI + {n} Lexicographic Policy Extractions  (depth={TREE_DEPTH})")

    print("\nClearing cached files...")
    _delete_if_exists(CHVI_MNS_FILE)
    _delete_if_exists(CHVI_QHULLS_FILE)

    timing = {}

    # --- run CHVI ---
    print("\nRun Convex Hull Value Iteration")
    env = FTEnvironment(weights=None, tree_depth=TREE_DEPTH)
    t_chvi_start = time.time()
    Q_hulls = convexhull_VI(
        env,
        theta=theta,
        discount_factor=discount_factor,
        MNS_filename=CHVI_MNS_FILE,
        q_hulls_file=CHVI_QHULLS_FILE,
    )
    timing["CHVI"] = time.time() - t_chvi_start
    print(f"CHVI elapsed time: {timing['CHVI']:.3f} s")

    print(f"\nExtracting {n} lexicographic policies via lex_max")
    extraction_times = {}

    for priority_label, priority in LEX_ORDERS:
        env = FTEnvironment(weights=None, tree_depth=TREE_DEPTH)
        t_start = time.time()
        policy = extract_lex_policy_from_qhulls(Q_hulls, priority, env, env.n_actions)
        extraction_times[priority_label] = time.time() - t_start
        print(f"[{priority_label}]:  {extraction_times[priority_label]:.4f} s")
        np.save(os.path.join(POLICIES_DIR, f"FT_CHVI_lexmax_d{TREE_DEPTH}_{priority_label}_policy.npy"), policy)

    timing["Policy extraction"] = sum(extraction_times.values())
    # total time
    timing["TOTAL"] = timing["CHVI"] + timing["Policy extraction"]

    lines = []
    lines.append(f"BENCHMARK 1: CHVI + {n} Lexicographic Policy Extractions (depth={TREE_DEPTH})")
    lines.append("-" * 60)
    lines.append(f"CHVI: {timing['CHVI']:.3f} s")
    lines.append("Policy extraction per order:")
    for priority_label, dur in extraction_times.items():
        lines.append(f"  [{priority_label}]: {dur:.4f} s")
    lines.append(f"Total extraction: {timing['Policy extraction']:.3f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {timing['TOTAL']:.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results(f"benchmark1_CHVI_d{TREE_DEPTH}.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 2: Nx Lexicographic Value Iteration
# ---------------------------------------------------------------------------

def run_benchmark2(theta, discount_factor):
    n = len(LEX_ORDERS)
    print(f"\nBENCHMARK 2: {n}x Lexicographic Value Iteration  (depth={TREE_DEPTH})")

    print("\nClear cached files")
    _delete_if_exists(LGVI_LEXMAX_MNS_FILE)

    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"FT_LGVI_lexmax_d{TREE_DEPTH}_{label}_policy.npy"))

    run_times     = {}
    t_total_start = time.time()

    for label, priority in LEX_ORDERS:
        policy_file = os.path.join(POLICIES_DIR, f"FT_LGVI_lexmax_d{TREE_DEPTH}_{label}_policy.npy")

        print(f"\nPriority order: {priority}")
        env = FTEnvironment(weights=None, tree_depth=TREE_DEPTH)

        t_run_start = time.time()
        policy, Q = LG_VI_lexmax(
            env,
            theta=theta,
            discount_factor=discount_factor,
            priority=priority,
            MNS_filename=LGVI_LEXMAX_MNS_FILE,
        )
        run_times[label] = time.time() - t_run_start
        print(f"Elapsed time for [{label}]: {run_times[label]:.3f} s")

        np.save(policy_file, policy)

    total_time = time.time() - t_total_start

    lines = []
    lines.append(f"BENCHMARK 2: {n}x Lexicographic Value Iteration (depth={TREE_DEPTH})")
    lines.append("-" * 60)
    for label, priority in LEX_ORDERS:
        lines.append(f"Priority [{label}]: {run_times[label]:.3f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f} s")
    lines.append(f"Average per run: {(total_time / n):.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results(f"benchmark2_LGVI_d{TREE_DEPTH}.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 3: Lexicographic Hull VI + Extract N Policies
# ---------------------------------------------------------------------------

def run_benchmark3(theta, discount_factor):
    n = len(LEX_ORDERS)
    print(f"\nBENCHMARK 3: Lexicographic Hull VI + {n} Policy Extractions  (depth={TREE_DEPTH})")
    print("\nClear cached files")
    _delete_if_exists(LGVI_LEXHULL_MNS_FILE)
    _delete_if_exists(LGVI_LEXHULL_V_FILE)
    _delete_if_exists(LGVI_LEXHULL_Q_FILE)

    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"FT_LGVI_lexhull_d{TREE_DEPTH}_{label}_policy.npy"))

    print("\nRun Lexicographic Hull Value Iteration.")
    env = FTEnvironment(weights=None, tree_depth=TREE_DEPTH)

    t_start = time.time()
    policies, Q_hulls, timing = LG_VI_lexhull(
        env,
        theta=theta,
        discount_factor=discount_factor,
        MNS_filename=LGVI_LEXHULL_MNS_FILE,
    )
    total_time = time.time() - t_start
    del Q_hulls

    print(f"\nElapsed time for LG_VI_lexhull: {total_time:.3f} s")

    print(f"\nSaving {n} extracted policies...")
    for label, priority in LEX_ORDERS:
        priority_tuple = tuple(priority)
        if priority_tuple in policies:
            policy_file = os.path.join(POLICIES_DIR, f"FT_LGVI_lexhull_d{TREE_DEPTH}_{label}_policy.npy")
            np.save(policy_file, policies[priority_tuple])
            print(f"  Saved [{label}] to {policy_file}")

    lines = []
    lines.append(f"BENCHMARK 3: Lexicographic Hull VI + {n} Policy Extractions (depth={TREE_DEPTH})")
    lines.append("-" * 60)
    lines.append(f"Iteration 1 (model build + first VI sweep): {timing['model_build']:.3f} s")
    lines.append(f"Remaining iterations: {timing['remaining_vi']:.3f} s")
    lines.append(f"Policy extraction (all 720 orders): {timing['policy_extraction']:.3f} s")
    for priority_tuple, dur in list(timing["extraction_times"].items())[:n]:
        label = LEX_ORDER_LABELS.get(priority_tuple, str(list(priority_tuple)))
        lines.append(f"  [{label}]: {dur:.4f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results(f"benchmark3_LGVI_hull_d{TREE_DEPTH}.txt", report)


if __name__ == "__main__":
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    if BENCHMARK_TO_RUN == BENCHMARK1_CHVI:
        run_benchmark1(theta, discount_factor)

    if BENCHMARK_TO_RUN == BENCHMARK2_LGVI_LEXMAX:
        run_benchmark2(theta, discount_factor)

    if BENCHMARK_TO_RUN == BENCHMARK3_LGVI_LEXHULL:
        run_benchmark3(theta, discount_factor)