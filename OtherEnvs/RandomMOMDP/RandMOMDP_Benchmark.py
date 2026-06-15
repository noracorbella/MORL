"""
RandMOMDP_Benchmark.py
----------------------
Unified benchmark runner for the random MOMDP algorithms.

BENCHMARKS
----------
  1. CHVI   : Convex Hull Value Iteration + extract K! lexicographic policies via lex_max.
  2. LGVI   : K! x Lexicographic Value Iteration (one run per priority order).
  3. LGVI-H : Lexicographic Hull Value Iteration + extract K! policies in one pass.
"""

import os
import sys
import time
import numpy as np


_HERE        = os.path.dirname(os.path.abspath(__file__))
POLICIES_DIR = os.path.join(_HERE, "randmomdp_benchmark_policies")
RESULTS_DIR  = os.path.join(_HERE, "randmomdp_benchmark_results")

sys.path.insert(0, os.path.join(_HERE, "environment"))
sys.path.insert(0, os.path.join(_HERE, "algorithms"))
sys.path.insert(0, os.path.join(_HERE, "utils"))

from RandMOMDP_Environment import RandMOMDPEnvironment
from RandMOMDP_CHVI import convexhull_VI
from RandMOMDP_LG_utils import extract_lex_policy_from_qhulls, generate_all_priority_orders
from RandMOMDP_LGVI_lexmax import LG_VI_lexmax
from RandMOMDP_LGVI_lexhull_timing import LG_VI_lexhull

BENCHMARK1_CHVI         = 1
BENCHMARK2_LGVI_LEXMAX  = 2
BENCHMARK3_LGVI_LEXHULL = 3

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BENCHMARK_TO_RUN = BENCHMARK1_CHVI  # BENCHMARK1_CHVI | BENCHMARK2_LGVI_LEXMAX | BENCHMARK3_LGVI_LEXHULL

theta = 0.01
discount_factor = 0.7

env_kwargs = dict(seed=45, states=100, actions=5, actions_per_state=5,
                  successors=1, objectives=3, terminal_states=1)

# files
CHVI_MNS_FILE         = os.path.join(POLICIES_DIR, "RandMOMDP_CHVI_MNS.pkl")
CHVI_QHULLS_FILE      = os.path.join(POLICIES_DIR, "RandMOMDP_CHVI_qhulls.pkl")
LGVI_LEXMAX_MNS_FILE  = os.path.join(POLICIES_DIR, "RandMOMDP_LGVI_lexmax_MNS.pkl")
LGVI_LEXHULL_MNS_FILE = os.path.join(POLICIES_DIR, "RandMOMDP_LGVI_lexhull_MNS.pkl")
LGVI_LEXHULL_V_FILE   = os.path.join(POLICIES_DIR, "RandMOMDP_LGVI_lexhull_V_hulls.pkl")
LGVI_LEXHULL_Q_FILE   = os.path.join(POLICIES_DIR, "RandMOMDP_LGVI_lexhull_Q_hulls.pkl")

# All K! lexicographic orders derived from the number of objectives.
_N_OBJ = env_kwargs["objectives"]
LEX_ORDERS = [(str(order), order) for order in generate_all_priority_orders(_N_OBJ)]
LEX_ORDER_LABELS = {tuple(priority): label for label, priority in LEX_ORDERS}


def _pstr(priority):
    return "-".join(str(p) for p in priority)


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
# BENCHMARK 1: Convex Hull Value Iteration + Extract K! Lexicographic Policies
# ---------------------------------------------------------------------------

def run_benchmark1(theta, discount_factor):
    print("\nBENCHMARK 1: CHVI + Lexicographic Policy Extractions")
    print("\nClearing cached files…")
    _delete_if_exists(CHVI_MNS_FILE)
    _delete_if_exists(CHVI_QHULLS_FILE)

    timing = {}

    print("\nRun Convex Hull Value Iteration")
    env = RandMOMDPEnvironment(weights=None, **env_kwargs)
    t_chvi_start = time.time()
    Q_hulls = convexhull_VI(
        env, theta=theta, discount_factor=discount_factor,
        MNS_filename=CHVI_MNS_FILE, q_hulls_file=CHVI_QHULLS_FILE,
    )
    timing["CHVI"] = time.time() - t_chvi_start
    print(f"CHVI elapsed time: {timing['CHVI']:.3f} s")

    print("\nExtracting lexicographic policies via lex_max")
    extraction_times = {}

    for priority_label, priority in LEX_ORDERS:
        env = RandMOMDPEnvironment(weights=None, **env_kwargs)
        t_start = time.time()
        policy = extract_lex_policy_from_qhulls(Q_hulls, priority, env, env.n_actions)
        extraction_times[priority_label] = time.time() - t_start
        print(f"[{priority_label}]:  {extraction_times[priority_label]:.3f} s")
        np.save(os.path.join(POLICIES_DIR, f"RandMOMDP_CHVI_lexmax_{_pstr(priority)}_policy.npy"), policy)

    timing["Policy extraction"] = sum(extraction_times.values())
    timing["TOTAL"] = timing["CHVI"] + timing["Policy extraction"]

    lines = []
    lines.append("BENCHMARK 1: Convex Hull Value Iteration + Extract Lexicographic Policies")
    lines.append("-" * 60)
    lines.append(f"CHVI: {timing['CHVI']:.3f} s")
    lines.append("Policy extraction per order:")
    for priority_label, dur in extraction_times.items():
        lines.append(f"  {priority_label}: {dur:.3f} s")
    lines.append(f"Total extraction: {timing['Policy extraction']:.3f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {timing['TOTAL']:.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results("benchmark1_CHVI.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 2: K! x Lexicographic Value Iteration
# ---------------------------------------------------------------------------

def run_benchmark2(theta, discount_factor):
    print("\nBENCHMARK 2: Lexicographic Value Iteration (one run per order)")
    print("\nClearing cached files…")
    _delete_if_exists(LGVI_LEXMAX_MNS_FILE)

    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"RandMOMDP_LGVI_lexmax_{_pstr(priority)}_policy.npy"))

    run_times     = {}
    t_total_start = time.time()

    for label, priority in LEX_ORDERS:
        policy_file = os.path.join(POLICIES_DIR, f"RandMOMDP_LGVI_lexmax_{_pstr(priority)}_policy.npy")

        print(f"\nPriority order: {priority}")
        env = RandMOMDPEnvironment(weights=None, **env_kwargs)

        t_run_start = time.time()
        policy, Q = LG_VI_lexmax(
            env, theta=theta, discount_factor=discount_factor, priority=priority,
            MNS_filename=LGVI_LEXMAX_MNS_FILE,
        )
        run_times[label] = time.time() - t_run_start
        print(f"Elapsed time for {label}: {run_times[label]:.3f} s")

        np.save(policy_file, policy)

    total_time = time.time() - t_total_start

    lines = []
    lines.append("BENCHMARK 2: Lexicographic Value Iteration (one run per order)")
    lines.append("-" * 60)
    for label, priority in LEX_ORDERS:
        lines.append(f"Priority {label}: {run_times[label]:.3f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f} s")
    lines.append(f"Average per run: {(total_time / len(LEX_ORDERS)):.3f} s")

    report = "\n".join(lines)
    print("\n" + report)
    _write_results("benchmark2_LGVI.txt", report)


# ---------------------------------------------------------------------------
# BENCHMARK 3: Lexicographic Hull Value Iteration + Extract K! Policies
# ---------------------------------------------------------------------------

def run_benchmark3(theta, discount_factor):
    print("\nBENCHMARK 3: Lexicographic Hull VI + Policy Extractions")
    print("\nClearing cached files…")
    _delete_if_exists(LGVI_LEXHULL_MNS_FILE)
    _delete_if_exists(LGVI_LEXHULL_V_FILE)
    _delete_if_exists(LGVI_LEXHULL_Q_FILE)

    for label, priority in LEX_ORDERS:
        _delete_if_exists(os.path.join(POLICIES_DIR, f"RandMOMDP_LGVI_lexhull_{_pstr(priority)}_policy.npy"))

    print("\nRun Lexicographic Hull Value Iteration.")
    env = RandMOMDPEnvironment(weights=None, **env_kwargs)

    t_start = time.time()
    policies, Q_hulls, timing = LG_VI_lexhull(
        env, theta=theta, discount_factor=discount_factor,
        MNS_filename=LGVI_LEXHULL_MNS_FILE,
    )
    total_time = time.time() - t_start
    del Q_hulls

    print(f"\nElapsed time for LG_VI_lexhull: {total_time:.3f} s")

    print("\nSave extracted policies")
    for priority_tuple, policy in policies.items():
        policy_file = os.path.join(POLICIES_DIR, f"RandMOMDP_LGVI_lexhull_{_pstr(priority_tuple)}_policy.npy")
        np.save(policy_file, policy)
        label = LEX_ORDER_LABELS.get(priority_tuple, str(list(priority_tuple)))
        print(f"Saved {label} to {policy_file}")

    lines = []
    lines.append("BENCHMARK 3: Lexicographic Hull VI + Extract Policies")
    lines.append("-" * 60)
    lines.append(f"Iteration 1 (model build + first VI sweep): {timing['model_build']:.3f} s")
    lines.append(f"Remaining iterations: {timing['remaining_vi']:.3f} s")
    lines.append(f"Policy extraction: {timing['policy_extraction']:.3f} s")
    for priority_tuple, dur in timing["extraction_times"].items():
        label = LEX_ORDER_LABELS.get(priority_tuple, str(list(priority_tuple)))
        lines.append(f"  {label}: {dur:.3f} s")
    lines.append("")
    lines.append(f"TOTAL TIME: {total_time:.3f} s")

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
