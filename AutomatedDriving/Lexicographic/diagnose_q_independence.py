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
from LG_VI_stoc_lexhull import LG_VI_lexhull
from ADS_Environment import ADS_Environment


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
    print("  the resulting Q-tables.")

    # Test with three different reference priorities
    test_priorities = [
        [0, 1, 2],  # Car first
        [1, 2, 0],  # Pedestrian 1 first
        [2, 1, 0],  # Pedestrian 2 first
    ]

    print(f"\nTesting with reference priorities: {test_priorities}")

    Q_tables = {}
    policies_dicts = {}
    times = {}

    # Parameters
    theta = 1.0
    discount_factor = 0.7

    print("\n" + "=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    for priority in test_priorities:
        priority_name = f"Priority {tuple(priority)}"
        print(f"\n[Experiment] Running with reference priority {priority}")
        print("-" * 80)

        # Create fresh environment
        env = ADS_Environment()

        # Run LG_VI_lexhull with this reference priority
        start_time = time.time()
        policies, Q = LG_VI_lexhull(
            env,
            theta=theta,
            discount_factor=discount_factor,
            priority=priority
        )
        elapsed = time.time() - start_time

        # Store results
        Q_tables[priority_name] = Q
        policies_dicts[priority_name] = policies
        times[priority_name] = elapsed

        print(f"\n✓ Completed in {elapsed:.2f}s")
        print(f"  Q-table shape: {Q.shape}")
        print(f"  Number of policies extracted: {len(policies)}")

    # Compare Q-tables
    print("\n" + "=" * 80)
    print("COMPARING Q-TABLES")
    print("=" * 80)

    names = list(Q_tables.keys())
    all_match = True

    # Compare Q1 vs Q2
    match_12 = compare_q_tables(
        Q_tables[names[0]],
        Q_tables[names[1]],
        names[0],
        names[1]
    )
    all_match = all_match and match_12

    # Compare Q1 vs Q3
    match_13 = compare_q_tables(
        Q_tables[names[0]],
        Q_tables[names[2]],
        names[0],
        names[2]
    )
    all_match = all_match and match_13

    # Compare Q2 vs Q3
    match_23 = compare_q_tables(
        Q_tables[names[1]],
        Q_tables[names[2]],
        names[1],
        names[2]
    )
    all_match = all_match and match_23

    # Compare policies (they should all be the same)
    print("\n" + "=" * 80)
    print("COMPARING EXTRACTED POLICIES")
    print("=" * 80)

    print("\nNote: All experiments should extract the same 6 policies,")
    print("even though they used different reference priorities during training.")

    # For each priority ordering, compare the extracted policies
    priority_orderings = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
    policies_match = True

    for priority_ordering in priority_orderings:
        print(f"\n  Comparing policies for priority {priority_ordering}:")

        # Get policies from all three experiments
        pol_1 = policies_dicts[names[0]][priority_ordering]
        pol_2 = policies_dicts[names[1]][priority_ordering]
        pol_3 = policies_dicts[names[2]][priority_ordering]

        # Compare
        match_12_pol = np.array_equal(pol_1, pol_2)
        match_13_pol = np.array_equal(pol_1, pol_3)
        match_23_pol = np.array_equal(pol_2, pol_3)

        all_policies_match = match_12_pol and match_13_pol and match_23_pol

        if all_policies_match:
            print(f"    ✓ All policies MATCH")
        else:
            print(f"    ✗ Policies DIFFER")
            if not match_12_pol:
                diff_count = np.sum(pol_1 != pol_2)
                print(f"      {names[0]} vs {names[1]}: {diff_count} states differ")
            if not match_13_pol:
                diff_count = np.sum(pol_1 != pol_3)
                print(f"      {names[0]} vs {names[2]}: {diff_count} states differ")
            if not match_23_pol:
                diff_count = np.sum(pol_2 != pol_3)
                print(f"      {names[1]} vs {names[2]}: {diff_count} states differ")
            policies_match = False

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nQ-tables match: {'✓ YES' if all_match else '✗ NO'}")
    print(f"Policies match: {'✓ YES' if policies_match else '✗ NO'}")

    print("\nTiming:")
    for name, t in times.items():
        print(f"  {name}: {t:.2f}s")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if all_match and policies_match:
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
