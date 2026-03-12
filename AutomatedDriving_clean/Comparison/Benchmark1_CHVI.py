"""
BENCHMARK 1: Convex Hull Value Iteration + Extract 6 Lexicographic Policies
--------------------------------------------------------------------------------------
Measures the time to:
  1. Build the model
  2. Run CHVI until convergence
  3. Extract one policy per lexicographic priority order using lex_max.
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


def extract_lex_policy_from_qhulls(Q_hulls, priority, env, n_actions):
    """
    Extract the optimal lexicographic policy for a given priority order
    directly from the CHVI Q-hulls using lex_max.
    Args:
        Q_hulls : dict (Q-hulls from convexhull_VI)
        priority : list (lexicographic priority order)
        env : Environment
        n_actions: int
    Returns:
        policy : np.ndarray ([n_cells, n_cells, n_cells])
    """
    from LG_utils import lex_max

    n_cells = env.map_num_cells
    policy  = np.zeros([n_cells, n_cells, n_cells], dtype=int)

    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                state_tuple = (c, p1, p2)

                # For each action pick the best hull vector with lex max
                action_best_vectors = []
                for action in range(n_actions):
                    hull = Q_hulls[state_tuple + (action,)]
                    if not isinstance(hull, np.ndarray):
                        hull = np.array(hull)
                    best_idx = lex_max(hull, priority=priority)
                    action_best_vectors.append(hull[best_idx])

                # Among best vectors, pick the best action
                action_best_vectors = np.array(action_best_vectors)
                best_action = lex_max(action_best_vectors, priority=priority)
                policy[c, p1, p2] = best_action

    return policy


LEX_ORDERS = [
    ("[0, 1, 2]", [0, 1, 2]),
    ("[0, 2, 1]", [0, 2, 1]),
    ("[1, 0, 2]", [1, 0, 2]),
    ("[1, 2, 0]", [1, 2, 0]),
    ("[2, 0, 1]", [2, 0, 1]),
    ("[2, 1, 0]", [2, 1, 0]),
]

MNS_FILE     = "policies/CH_VI_stoc_MNS.pkl"
QHULLS_FILE  = "policies/CH_VI_qhulls_benchmark.pkl"
RESULTS_FILE = "benchmark_results/benchmark1_CHVI.txt"

if __name__ == "__main__":

    from ADS_Environment import Environment
    from CH_VI_stochastic_v2 import convexhull_VI

    os.makedirs("policies", exist_ok=True)

    # delete any cached files so we time from scratch
    print("\nBENCHMARK 1: CHVI + 6 Lexicographic Policy Extractions")
    print("\nClear cached files")
    
    delete_if_exists(MNS_FILE)
    delete_if_exists(QHULLS_FILE)

    timing = {}

    # run CHVI 
    print("\nRun Convex Hull Value Iteration")
    env = Environment(weights=None)

    t_chvi_start = time.time()
    Q_hulls = convexhull_VI(
        env,
        theta=0.01,
        discount_factor=0.7,
        MNS_filename=MNS_FILE,
        q_hulls_file=QHULLS_FILE,
    )

    timing["CHVI"] = time.time() - t_chvi_start
    print(f"\nCHVI elapsed time: {timing['CHVI']}")

    # extract one policy per lexicographic order using lex_max
    print("\nExtract 6 lexicographic policies using lex_max")
    extraction_times = {}

    for priority_label, priority in LEX_ORDERS:
        env = Environment(weights=None)

        t_start = time.time()
        policy = extract_lex_policy_from_qhulls(Q_hulls, priority, env, env.n_actions)

        duration = time.time() - t_start
        extraction_times[priority_label] = duration
        print(f"[{priority_label}]:  {duration:.3f}")

        # save policy
        np.save(f"policies/CHVI_lexmax_{priority[0]}-{priority[1]}-{priority[2]}_policy.npy", policy)

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

    os.makedirs("benchmark_results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        f.write(report + "\n")

    print(f"\nResults saved to: {RESULTS_FILE}")