import numpy as np
import os
from itertools import combinations

# def main():
#     names = list(policies.keys())
#     all_same = True
#     for name_a, name_b in combinations(names, 2):
#         identical, _ = policies_are_identical(policies[name_a], policies[name_b])
#         if not identical:
#             all_same = False
#             break


if __name__ == "__main__":
    weights = [10000, 100, 1]  
    priority = [0, 1, 2] 
    priority_str = f"{priority[0]}-{priority[1]}-{priority[2]}"
    discount_factor = 0.7

    policy_files = {
        "VI": f"allpolicies/VI_stochastic_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy",
        "CHVI": f"allpolicies/CHVI_{weights[0]}-{weights[1]}-{weights[2]}-policy.npy",
        "LG_VI_lexmax": f"allpolicies/LG_VI_lexmax_{priority_str}_policy.npy",
        "LG_VI_lexhull": f"allpolicies/LG_VI_lexhull_{priority_str}_policy.npy"
    }

    print("\nLoading policies...")

    policies = {}
    for i, path in policy_files.items():
        if os.path.exists(path):
            policies[i] = np.load(path, allow_pickle=True)
            print(f"Loaded {path}  (shape: {policies[i].shape})")
        else:
            print(f"Not found: {path}")

    names = list(policies.keys())
    n_states = next(iter(policies.values())).size

    print(f"Pairwise comparison for loaded policies  ({n_states} states)")
    print(f"{'-'*60}")
    print(f"  {'Pair':<20} {'Identical?':>10}  {'Diff states':>12}  {'% agree':>8}")
    print(f"  {'-'*56}")

    results = {}
    for name_a, name_b in combinations(names, 2):
        p_a = policies[name_a]
        p_b = policies[name_b]

        if p_a.shape != p_b.shape:
            raise ValueError(f"Shape mismatch: {p_a} vs {p_b.shape}")
        diff_mask = p_a != p_b
        diff_states = np.where(diff_mask)[0]
        identical = len(diff_states) == 0

        percentage = 100.0 * (n_states - len(diff_states)) / n_states
        label = f"{name_a} vs {name_b}"
        results[(name_a, name_b)] = (identical, diff_states)
        tag = "YES" if identical else "NO"
        print(f"  {label:<20} {tag:>10}  {len(diff_states):>12}  {percentage:>7.2f}%")


    for (name_a, name_b), (identical, diff_states) in results.items():
        if not identical:
            print(f"  States where {name_a} diff {name_b}  ({len(diff_states)} states):")
            p_a = policies[name_a]
            p_b = policies[name_b]

            for s in diff_states[:20]:
                print(f"    state {s:5d}:  {name_a}-->action {p_a[s]}  |  {name_b}-->action {p_b[s]}")
            if len(diff_states) > 20:
                print(f"    ... and {len(diff_states) - 20} more states.")
            print()