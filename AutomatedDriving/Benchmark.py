"""
Benchmark.py
------------
Unified benchmark runner for the AutomatedDriving algorithms.

BENCHMARKS
----------
  1. CHVI   : Convex Hull Value Iteration + extract 6 lexicographic policies
              via lex_max.
  2. LGVI   : 6x Lexicographic Value Iteration (one run per priority order).
  3. LGVI-H : Lexicographic Hull Value Iteration + extract 6 policies in one
              pass.
"""

import os
import sys
import time
import numpy as np


_HERE            = os.path.dirname(os.path.abspath(__file__))
POLICIES_DIR     = os.path.join(_HERE, "benchmark_policies")
RESULTS_DIR      = os.path.join(_HERE, "benchmark_results")

sys.path.insert(0, os.path.join(_HERE, "environment"))
sys.path.insert(0, os.path.join(_HERE, "algorithms"))
sys.path.insert(0, os.path.join(_HERE, "utils"))

from ADS_Environment import Environment
from CHVI_stochastic import convexhull_VI
from LG_utils import extract_lex_policy_from_qhulls
from LGVI_stochastic_lexmax import LG_VI_lexmax
from LGVI_stochastic_lexhull_timing import LG_VI_lexhull

BENCHMARK1_CHVI        = 1
BENCHMARK2_LGVI_LEXMAX = 2
BENCHMARK3_LGVI_LEXHULL = 3

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_TO_RUN = BENCHMARK1_CHVI # BENCHMARK1_CHVI | BENCHMARK2_LGVI_LEXMAX | BENCHMARK3_LGVI_LEXHULL

# Parameters
theta = 0.01
discount_factor = 0.7

# files
CHVI_MNS_FILE    = os.path.join(POLICIES_DIR, "CHVI_stochastic_MNS.pkl")
CHVI_QHULLS_FILE = os.path.join(POLICIES_DIR, "CHVI_stochastic_qhulls.pkl")
LGVI_LEXMAX_MNS_FILE    = os.path.join(POLICIES_DIR, "LGVI_lexmax_MNS.pkl")
LGVI_LEXHULL_MNS_FILE   = os.path.join(POLICIES_DIR, "LGVI_lexhull_MNS.pkl")
LGVI_LEXHULL_V_FILE     = os.path.join(POLICIES_DIR, "LGVI_lexhull_V_hulls.pkl")
LGVI_LEXHULL_Q_FILE     = os.path.join(POLICIES_DIR, "LGVI_lexhull_Q_hulls.pkl")

LEX_ORDERS = [
    ("[0, 1, 2]", [0, 1, 2]),
    ("[0, 2, 1]", [0, 2, 1]),
    ("[1, 0, 2]", [1, 0, 2]),
    ("[1, 2, 0]", [1, 2, 0]),
    ("[2, 0, 1]", [2, 0, 1]),
    ("[2, 1, 0]", [2, 1, 0]),
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
# BENCHMARK 1: Convex Hull Value Iteration + Extract 6 Lexicographic Policies
# Measures the time to:
#   1. Build the model
#   2. Run CHVI until convergence
#   3. Extract one policy per lexicographic priority order using lex_max.
# ---------------------------------------------------------------------------

def run_benchmark1(theta, discount_factor):
    print("\nBENCHMARK 1: CHVI + 6 Lexicographic Policy Extractions")
    print("\nClear cached files")

    print("\nClearing cached files…")
    _delete_if_exists(CHVI_MNS_FILE)
    _delete_if_exists(CHVI_QHULLS_FILE)

    timing = {}

    # --- run CHVI ---
    print("\nRunning Convex Hull Value Iteration …")
    env = Environment(weights=None)
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

    # extract one policy per lexicographic order using lex_max
    print("\nExtracting 6 lexicographic policies via lex_max")
    extraction_times = {}

    for priority_label, priority in LEX_ORDERS:
        env = Environment(weights=None)
        t_start = time.time()
        policy = extract_lex_policy_from_qhulls(Q_hulls, priority, env, env.n_actions)
        extraction_times[priority_label] = time.time() - t_start
        print(f"[{priority_label}]:  {extraction_times[priority_label]:.3f}")
        np.save(
            os.path.join(POLICIES_DIR,
                         f"CHVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"),
            policy,
        )

    timing["Policy extraction"] = sum(extraction_times.values())
    # total time
    timing["TOTAL"] = timing["CHVI"] + timing["Policy extraction"]

    lines = []
    lines.append("BENCHMARK 1: Convex Hull Value Iteration + Extract 6 Lexicographic Policies")
    lines.append("-" * 60)
    lines.append(f"CHVI: {timing['CHVI']:.3f}")

    lines.append("Policy extraction per order:")
    for priority_label, dur in extraction_times.items():
        lines.append(f"{priority_label}: {dur:.3f}")
    lines.append(f"Total extraction:{timing['Policy extraction']:.3f}")
    lines.append("")
    lines.append(f" TOTAL TIME: {timing['TOTAL']:.3f}")
    
    report = "\n".join(lines)
    print("\n" + report)
    _write_results("benchmark1_CHVI.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 2: 6x Lexicographic Value Iteration
# Measures the time to run LG_VI_lexmax separately for each of the 6
# lexicographic priority orders and extract the optimal policy for each.

#   1. Delete MNS cache before first LG order
#   2. For orders 2-6 reuse the cached MNS
# ---------------------------------------------------------------------------

def run_benchmark2(theta, discount_factor):

    print("\nBENCHMARK 2: 6x Lexicographic Value Iteration")
    print("\nClear cached files")

    _delete_if_exists(LGVI_LEXMAX_MNS_FILE)

    # delete any previously saved policies for these orders
    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"))

    run_times     = {}   # per-order total time (VI + policy extraction)

    t_total_start = time.time()

    # full LG_VI_lexmax run per priority order
    for label, priority in LEX_ORDERS:
        policy_file = os.path.join(POLICIES_DIR, f"LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy")

        print(f"\nPriority order: {priority}")

        env = Environment(weights=None)

        t_run_start = time.time()

        policy, Q = LG_VI_lexmax(
            env,
            theta=theta,
            discount_factor=discount_factor,
            priority=priority,
            MNS_filename=LGVI_LEXMAX_MNS_FILE,
        )

        run_times[label] = time.time() - t_run_start
        print(f"Elapsed time for {label}: {run_times[label]:.3f}")

        np.save(policy_file, policy)

    total_time = time.time() - t_total_start

    lines = []
    lines.append("BENCHMARK 2: 6x Lexicographic Value Iteration")
    lines.append("-" * 60)
    for label, priority in LEX_ORDERS:
        lines.append(f"Priority {label}: {run_times[label]:.3f}")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f}")
    lines.append(f"Average per run: {(total_time / 6):.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results("benchmark2_LGVI.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 3: Lexicographic Hull Value Iteration + Extract 6 Policies
# Measures the time to:
#   1. Run LG_VI_lexhull once until convergence (building the model)
#   2. Extract one policy per lexicographic priority order
# ---------------------------------------------------------------------------

def run_benchmark3(theta, discount_factor):
    print("\nBENCHMARK 3: Lexicographic Hull VI + 6 Policy Extractions")
    print("\nClear cached files")

    _delete_if_exists(LGVI_LEXHULL_MNS_FILE)
    _delete_if_exists(LGVI_LEXHULL_V_FILE)
    _delete_if_exists(LGVI_LEXHULL_Q_FILE)

    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"LG_VI_lexhull_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy"))

    # run LG_VI_lexhull 
    print("\nRun Lexicographic Hull Value Iteration.")
    env = Environment(weights=None)

    t_start = time.time()
    policies, Q_hulls, timing = LG_VI_lexhull(
        env,
        theta=theta,
        discount_factor=discount_factor,
        MNS_filename=LGVI_LEXHULL_MNS_FILE,
    )

    total_time = time.time() - t_start
    del Q_hulls 

    print(f"\nElapsed time for LG_VI_lexhull: {total_time:.3f}s")

    # save extracted policies
    print("\nSave extracted policies")
    for priority_tuple, policy in policies.items():
        policy_file = os.path.join(POLICIES_DIR, f"LG_VI_lexhull_{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}_policy.npy",)
        np.save(policy_file, policy)
        label = LEX_ORDER_LABELS.get(priority_tuple, str(list(priority_tuple)))
        print(f"Saved {label} in {policy_file}")

    lines = []
    lines.append("BENCHMARK 3: Lexicographic Hull VI + Extract 6 Policies")
    lines.append("-" * 60)
    lines.append(f"Iteration 1 (model build + first VI sweep): {timing['model_build']:.3f}")
    lines.append(f"Remaining Iterations: {timing['remaining_vi']:.3f}")
    lines.append(f"Policy extraction: {timing['policy_extraction']:.3f}")
    for priority_tuple, dur in timing["extraction_times"].items():
        label = LEX_ORDER_LABELS.get(priority_tuple, str(list(priority_tuple)))
        lines.append(f"{label}: {dur:.3f}")
    lines.append(f"Total extraction: {timing['policy_extraction']:.3f}")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results("benchmark3_LGVI_hull.txt", report)


if __name__ == "__main__":
    os.makedirs(POLICIES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    if BENCHMARK_TO_RUN == BENCHMARK1_CHVI:
        run_benchmark1(theta, discount_factor)

    if BENCHMARK_TO_RUN == BENCHMARK2_LGVI_LEXMAX:
        run_benchmark2(theta, discount_factor)

    if BENCHMARK_TO_RUN == BENCHMARK3_LGVI_LEXHULL:
        run_benchmark3(theta, discount_factor)
