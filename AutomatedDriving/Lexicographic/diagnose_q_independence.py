"""
Diagnostic script to verify that Q-vectors are independent of reference priority.

This script tests the hypothesis that:
1. Q-vectors converge to the same values regardless of reference priority used during training
2. The reference priority only affects the convergence path, not the final Q-values

If this hypothesis is true, then our LG_VI_lexhull implementation is correct.
If Q-values differ, we need to track all priorities separately during training.
"""

import numpy as np
import time
import gc
from LG_VI_stoc_lexhull import LG_VI_lexhull
from ADS_Environment import ADS_Environment


def compare_all_policies(policies_1, policies_2, name1, name2):
    """Compare all 6 policies between two experiments."""
    priority_orderings = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    all_match = True

    for priority_ordering in priority_orderings:
        pol_1 = policies_1[priority_ordering]
        pol_2 = policies_2[priority_ordering]

        match = np.array_equal(pol_1, pol_2)

        if match:
            print(f"    ✓ Priority {priority_ordering}: MATCH")
        else:
            diff_count = np.sum(pol_1 != pol_2)
            total = pol_1.size
            pct = (diff_count / total) * 100
            print(f"    ✗ Priority {priority_ordering}: DIFFER ({diff_count}/{total} = {pct:.2f}%)")
            all_match = False

    return all_match


def compare_q_tables(Q1, Q2, name1, name2, tolerance=1e-6):
    """Compare two Q-tables and report differences."""
    print(f"\nComparing Q-tables: {name1} vs {name2}")
    print("-" * 80)

    if Q1.shape != Q2.shape:
        print(f"  ✗ SHAPE MISMATCH: {Q1.shape} vs {Q2.shape}")
        return False

    abs_diff = np.abs(Q1 - Q2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    median_diff = np.median(abs_diff)

    # Count how many values differ significantly
    significant_diffs = np.sum(abs_diff > tolerance)
    total_values = Q1.size
    pct_different = (significant_diffs / total_values) * 100

    print(f"  Max difference:    {max_diff:.2e}")
    print(f"  Mean difference:   {mean_diff:.2e}")
    print(f"  Median difference: {median_diff:.2e}")
    print(f"  Values > {tolerance:.0e}:  {significant_diffs:,}/{total_values:,} ({pct_different:.2f}%)")

    match = np.allclose(Q1, Q2, atol=tolerance)

    if match:
        print(f"  ✓ Q-TABLES MATCH (within tolerance {tolerance:.0e})")
    else:
        print(f"  ✗ Q-TABLES DIFFER (above tolerance {tolerance:.0e})")

        # Find location of max difference
        max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"\n  Location of max difference: {max_idx}")
        print(f"    {name1}: {Q1[max_idx]}")
        print(f"    {name2}: {Q2[max_idx]}")
        print(f"    Difference: {abs_diff[max_idx]:.2e}")

    return match


def main():
    """Run the diagnostic."""

    print("=" * 80)
    print("DIAGNOSTIC: Testing Q-vector Independence from Reference Priority")
    print("=" * 80)

    print("\nHypothesis:")
    print("  Q-vectors should converge to the same values regardless of which")
    print("  reference priority is used during training.")
    print("\nTest:")
    print("  Run LG_VI_lexhull with 3 different reference priorities and compare")
    print("  the resulting Q-tables pairwise to avoid memory issues.")

    # Test with three different reference priorities
    test_priorities = [
        [0, 1, 2],  # Car first
        [1, 2, 0],  # Pedestrian 1 first
        [2, 1, 0],  # Pedestrian 2 first
    ]

    print(f"\nTesting with reference priorities: {test_priorities}")

    # Parameters
    theta = 1.0
    discount_factor = 0.7

    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS (with pairwise comparison)")
    print("=" * 80)

    # Run first experiment and store results
    print(f"\n[Experiment 1/3] Running with reference priority {test_priorities[0]}")
    print("-" * 80)

    env = ADS_Environment()
    start_time = time.time()
    policies_1, Q_1 = LG_VI_lexhull(
        env,
        theta=theta,
        discount_factor=discount_factor,
        priority=test_priorities[0]
    )
    time_1 = time.time() - start_time

    print(f"\n✓ Completed in {time_1:.2f}s")
    print(f"  Q-table shape: {Q_1.shape}")
    print(f"  Number of policies extracted: {len(policies_1)}")

    # Clean up environment
    del env
    gc.collect()

    # Run second experiment, compare, then keep only Q_2 for next comparison
    print(f"\n[Experiment 2/3] Running with reference priority {test_priorities[1]}")
    print("-" * 80)

    env = ADS_Environment()
    start_time = time.time()
    policies_2, Q_2 = LG_VI_lexhull(
        env,
        theta=theta,
        discount_factor=discount_factor,
        priority=test_priorities[1]
    )
    time_2 = time.time() - start_time

    print(f"\n✓ Completed in {time_2:.2f}s")
    print(f"  Q-table shape: {Q_2.shape}")
    print(f"  Number of policies extracted: {len(policies_2)}")

    # Compare Q_1 vs Q_2
    print("\n" + "=" * 80)
    print("COMPARING: Experiment 1 vs Experiment 2")
    print("=" * 80)
    match_12 = compare_q_tables(Q_1, Q_2, "Priority (0,1,2)", "Priority (1,2,0)")

    # Compare policies
    print("\n  Comparing extracted policies:")
    policies_match_12 = compare_all_policies(policies_1, policies_2, "Exp 1", "Exp 2")

    # Clean up Q_1 and policies_1 - we don't need them anymore
    del Q_1, policies_1, env
    gc.collect()

    # Run third experiment and compare with Q_2
    print(f"\n[Experiment 3/3] Running with reference priority {test_priorities[2]}")
    print("-" * 80)

    env = ADS_Environment()
    start_time = time.time()
    policies_3, Q_3 = LG_VI_lexhull(
        env,
        theta=theta,
        discount_factor=discount_factor,
        priority=test_priorities[2]
    )
    time_3 = time.time() - start_time

    print(f"\n✓ Completed in {time_3:.2f}s")
    print(f"  Q-table shape: {Q_3.shape}")
    print(f"  Number of policies extracted: {len(policies_3)}")

    # Compare Q_2 vs Q_3
    print("\n" + "=" * 80)
    print("COMPARING: Experiment 2 vs Experiment 3")
    print("=" * 80)
    match_23 = compare_q_tables(Q_2, Q_3, "Priority (1,2,0)", "Priority (2,1,0)")

    # Compare policies
    print("\n  Comparing extracted policies:")
    policies_match_23 = compare_all_policies(policies_2, policies_3, "Exp 2", "Exp 3")

    # Clean up
    del Q_2, Q_3, policies_2, policies_3, env
    gc.collect()

    # Overall results
    all_match = match_12 and match_23
    all_policies_match = policies_match_12 and policies_match_23

    times = {
        "Priority (0,1,2)": time_1,
        "Priority (1,2,0)": time_2,
        "Priority (2,1,0)": time_3
    }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nQ-tables match: {'✓ YES' if all_match else '✗ NO'}")
    print(f"Policies match: {'✓ YES' if all_policies_match else '✗ NO'}")

    print("\nTiming:")
    for name, t in times.items():
        print(f"  {name}: {t:.2f}s")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if all_match and all_policies_match:
        print("\n✓ HYPOTHESIS CONFIRMED!")
        print("\nQ-vectors are INDEPENDENT of reference priority.")
        print("The reference priority only affects the convergence path,")
        print("not the final Q-values.")
        print("\nThis means our LG_VI_lexhull implementation is CORRECT:")
        print("  1. Train once with any reference priority")
        print("  2. Q-vectors converge to priority-independent values")
        print("  3. Extract all 6 policies from the same Q-table")
        print("\nNo changes needed to the implementation!")
        return 0
    else:
        print("\n✗ HYPOTHESIS REJECTED!")
        print("\nQ-vectors DEPEND on reference priority.")
        print("This means we need to modify our implementation to track")
        print("value functions separately for each priority during training.")
        print("\nRECOMMENDED FIX:")
        print("  Implement Option C: Track V_dict for all 6 priorities")
        print("  and check convergence for all of them.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
