"""
Verification script to compare LG_VI_lexmax and LG_VI_lexhull implementations.

This script:
1. Runs LG_VI_lexmax for all 6 lexicographic priority orders
2. Runs LG_VI_lexhull once (computes all 6 policies simultaneously)
3. Verifies that both methods produce identical policies and Q-values
4. Compares execution time between the two approaches
"""

import numpy as np
import time
from LG_utils import generate_all_priority_orders
from LG_VI_stoc_lexmax import LG_VI_lexmax
from LG_VI_stoc_lexhull import LG_VI_lexhull
from ADS_Environment import ADS_Environment


def compare_policies(policy1, policy2, priority_name):
    """Compare two policies and return statistics."""
    if policy1.shape != policy2.shape:
        return {
            'match': False,
            'error': f"Shape mismatch: {policy1.shape} vs {policy2.shape}"
        }

    matches = np.sum(policy1 == policy2)
    total = policy1.size
    match_percentage = (matches / total) * 100

    return {
        'match': matches == total,
        'matches': matches,
        'total': total,
        'percentage': match_percentage,
        'priority': priority_name
    }


def compare_q_values(Q1, Q2, priority_name, tolerance=1e-9):
    """Compare two Q-value tables and return statistics."""
    if Q1.shape != Q2.shape:
        return {
            'match': False,
            'error': f"Shape mismatch: {Q1.shape} vs {Q2.shape}"
        }

    abs_diff = np.abs(Q1 - Q2)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)

    within_tolerance = np.allclose(Q1, Q2, atol=tolerance)

    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'tolerance': tolerance,
        'priority': priority_name
    }


def main():
    """Run the verification."""

    print("="*80)
    print("VERIFICATION: LG_VI_lexmax vs LG_VI_lexhull")
    print("="*80)

    # Initialize environment
    print("\n[1/4] Initializing environment...")
    env = ADS_Environment()

    # Parameters
    theta = 1.0
    discount_factor = 0.7
    n_objectives = 3

    # Get all priority orders
    all_priorities = generate_all_priority_orders(n_objectives)
    print(f"      Will test {len(all_priorities)} priority orders: {all_priorities}")

    # ========================================================================
    # APPROACH 1: Run LG_VI_lexmax for each priority order
    # ========================================================================
    print("\n[2/4] Running LG_VI_lexmax for each priority order...")
    print("-"*80)

    lexmax_policies = {}
    lexmax_Q_tables = {}
    lexmax_times = {}

    total_lexmax_time_start = time.time()

    for priority in all_priorities:
        priority_tuple = tuple(priority)
        print(f"\n  Running for priority {priority_tuple}...")

        # Reinitialize environment for fair comparison
        env = ADS_Environment()

        start_time = time.time()
        policy, Q = LG_VI_lexmax(env, theta=theta, discount_factor=discount_factor, priority=priority)
        end_time = time.time()

        elapsed = end_time - start_time
        lexmax_times[priority_tuple] = elapsed
        lexmax_policies[priority_tuple] = policy
        lexmax_Q_tables[priority_tuple] = Q

        print(f"  ✓ Completed in {elapsed:.2f}s")

    total_lexmax_time = time.time() - total_lexmax_time_start

    print(f"\n  TOTAL TIME (lexmax): {total_lexmax_time:.2f}s")
    print(f"  Average per priority: {total_lexmax_time/len(all_priorities):.2f}s")

    # ========================================================================
    # APPROACH 2: Run LG_VI_lexhull once
    # ========================================================================
    print("\n[3/4] Running LG_VI_lexhull (all priorities at once)...")
    print("-"*80)

    # Reinitialize environment for fair comparison
    env = ADS_Environment()

    lexhull_time_start = time.time()
    lexhull_policies, lexhull_Q = LG_VI_lexhull(env, theta=theta, discount_factor=discount_factor, priority=[0,1,2])
    lexhull_time = time.time() - lexhull_time_start

    print(f"\n  TOTAL TIME (lexhull): {lexhull_time:.2f}s")

    # ========================================================================
    # VERIFICATION: Compare results
    # ========================================================================
    print("\n[4/4] Verifying results...")
    print("-"*80)

    all_policies_match = True
    all_q_match = True

    print("\n  Policy Comparison:")
    print("  " + "-"*76)

    for priority in all_priorities:
        priority_tuple = tuple(priority)

        # Compare policies
        policy_result = compare_policies(
            lexmax_policies[priority_tuple],
            lexhull_policies[priority_tuple],
            priority_tuple
        )

        if policy_result['match']:
            print(f"  ✓ Priority {priority_tuple}: Policies MATCH (100%)")
        else:
            print(f"  ✗ Priority {priority_tuple}: Policies DIFFER "
                  f"({policy_result['percentage']:.2f}% match)")
            all_policies_match = False

    print("\n  Q-Value Comparison:")
    print("  " + "-"*76)

    # Note: All lexmax runs produce the same Q-table (vectorial),
    # so we just need to compare one of them with lexhull
    reference_priority = (0, 1, 2)
    q_result = compare_q_values(
        lexmax_Q_tables[reference_priority],
        lexhull_Q,
        "All priorities"
    )

    if q_result['match']:
        print(f"  ✓ Q-tables MATCH (max diff: {q_result['max_diff']:.2e}, "
              f"tolerance: {q_result['tolerance']:.2e})")
    else:
        print(f"  ✗ Q-tables DIFFER (max diff: {q_result['max_diff']:.2e}, "
              f"mean diff: {q_result['mean_diff']:.2e})")
        all_q_match = False

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nCorrectness:")
    print(f"  Policies match: {'✓ YES' if all_policies_match else '✗ NO'}")
    print(f"  Q-values match: {'✓ YES' if all_q_match else '✗ NO'}")

    print(f"\nPerformance:")
    print(f"  LG_VI_lexmax (6 runs):  {total_lexmax_time:.2f}s")
    print(f"  LG_VI_lexhull (1 run):  {lexhull_time:.2f}s")

    speedup = total_lexmax_time / lexhull_time
    print(f"\n  Speedup: {speedup:.2f}x")

    if speedup > 1:
        print(f"  → LG_VI_lexhull is {speedup:.2f}x FASTER")
    else:
        print(f"  → LG_VI_lexmax is {1/speedup:.2f}x FASTER")

    print("\n" + "="*80)

    # ========================================================================
    # DETAILED TIMING BREAKDOWN
    # ========================================================================
    print("\nDetailed Timing Breakdown:")
    print("-"*80)
    print(f"{'Priority':<20} {'Time (s)':<15} {'% of Total':<15}")
    print("-"*80)

    for priority in all_priorities:
        priority_tuple = tuple(priority)
        t = lexmax_times[priority_tuple]
        pct = (t / total_lexmax_time) * 100
        print(f"{str(priority_tuple):<20} {t:>10.2f}     {pct:>10.1f}%")

    print("-"*80)
    print(f"{'TOTAL (lexmax)':<20} {total_lexmax_time:>10.2f}s")
    print(f"{'TOTAL (lexhull)':<20} {lexhull_time:>10.2f}s")
    print("="*80)

    # Final verdict
    if all_policies_match and all_q_match:
        print("\n✓ VERIFICATION PASSED: Both methods produce identical results!")
        return 0
    else:
        print("\n✗ VERIFICATION FAILED: Methods produce different results!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
