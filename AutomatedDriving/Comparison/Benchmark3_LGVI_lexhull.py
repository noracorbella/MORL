"""
BENCHMARK 3: Lexicographic Hull Value Iteration + Extract 6 Policies
------------------------------------------------------------------------------
Measures the time to:
  1. Run LG_VI_lexhull once until convergence (building the model)
  2. Extract one policy per lexicographic priority order
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

LEX_ORDER_LABELS = {
    (0, 1, 2): "[0, 1, 2]",
    (0, 2, 1): "[0, 2, 1]",
    (1, 0, 2): "[1, 0, 2]",
    (1, 2, 0): "[1, 2, 0]",
    (2, 0, 1): "[2, 0, 1]",
    (2, 1, 0): "[2, 1, 0]",
}

MNS_FILE      = "policies/LG_VI_lexhull_MNS.pkl"
V_HULLS_FILE  = "policies/LG_VI_lexhull_V_hulls_benchmark.pkl"
Q_HULLS_FILE  = "policies/LG_VI_lexhull_Q_hulls_benchmark.pkl"
RESULTS_FILE = "benchmark_results/benchmark3_LGVI_lexhull.txt"

if __name__ == "__main__":

    from ADS_Environment import Environment
    from LG_VI_stoc_lexhull_v3 import LG_VI_lexhull

    os.makedirs("policies", exist_ok=True)

    # delete any cached files so we time from scratch
    print("\nBENCHMARK 3: Lexicographic Hull VI + 6 Policy Extractions")
    print("\nClear cached files")
    delete_if_exists(MNS_FILE)
    delete_if_exists(V_HULLS_FILE)
    delete_if_exists(Q_HULLS_FILE)
    for priority in LEX_ORDER_LABELS:
        delete_if_exists(f"policies/LG_VI_lexhull_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy")

    # run LG_VI_lexhull 
    print("\nRun Lexicographic Hull Value Iteration.")
    env = Environment(weights=None)

    t_start = time.time()

    policies, Q_hulls, timing = LG_VI_lexhull(
        env,
        theta=1.0,
        discount_factor=0.7,
        MNS_filename=MNS_FILE,
    )

    total_time = time.time() - t_start
    del Q_hulls  

    print(f"\nElapsed time for LG_VI_lexhull: {total_time:.3f}")

    # save extracted policies
    print("\nSave extracted policies")
    for priority_tuple, policy in policies.items():
        policy_file = f"policies/LG_VI_lexhull_{priority_tuple[0]}-{priority_tuple[1]}-{priority_tuple[2]}_policy.npy"
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

    with open(RESULTS_FILE, "w") as f:
        f.write(report + "\n")

    print(f"\nResults saved to: {RESULTS_FILE}")