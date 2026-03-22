"""
BENCHMARK 2: 6x Lexicographic Value Iteration
--------------------------------------------------------------------------------------
Measures the time to run LG_VI_lexmax separately for each of the 6
lexicographic priority orders and extract the optimal policy for each.

  1. Delete MNS cache before first LG order
  2. For orders 2-6 reuse the cached MNS
"""

import numpy as np
import time
import os

def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted cached file: {path}")
    else:
        print(f"No file to delete: {path}")

LEX_ORDERS = [
    ("[0, 1, 2]", [0, 1, 2]),
    ("[0, 2, 1]", [0, 2, 1]),
    ("[1, 0, 2]", [1, 0, 2]),
    ("[1, 2, 0]", [1, 2, 0]),
    ("[2, 0, 1]", [2, 0, 1]),
    ("[2, 1, 0]", [2, 1, 0]),
]


MNS_FILE     = "policies/LG_VI_lexmax_MNS.pkl"
RESULTS_FILE = "benchmark_results/benchmark2_LGVI_lexhull.txt"


if __name__ == "__main__":

    from ADS_Environment import Environment
    from LG_VI_stoc_lexmax import LG_VI_lexmax

    os.makedirs("policies", exist_ok=True)

    # delete any cached files so we time from scratch
    print("\nBENCHMARK 2: 6x Lexicographic Value Iteration")
    print("\nClear cached files")
    
    delete_if_exists(MNS_FILE)
    
    # delete any previously saved policies for these orders
    for label, priority in LEX_ORDERS:
        delete_if_exists(f"policies/LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy")

    run_times     = {}   # per-order total time (VI + policy extraction)
    vi_times      = {}   # VI-only time per order
    extract_times = {}   # policy extraction time per order (already inside LG_VI_lexmax)

    t_total_start = time.time()

    # full LG_VI_lexmax run per priority order
    for label, priority in LEX_ORDERS:
        policy_file = f"policies/LG_VI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}.npy"

        print(f"\nPriority order: {priority}")

        env = Environment(weights=None)

        t_run_start = time.time()

        policy, Q = LG_VI_lexmax(
            env,
            theta=1.0,
            discount_factor=0.7,
            priority=priority,
            MNS_filename=MNS_FILE,
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

    os.makedirs("benchmark_results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(report + "\n")

    print(f"\nResults saved to: {RESULTS_FILE}")