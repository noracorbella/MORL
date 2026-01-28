# ADAPTED FOR Q TABLES NOT BEING STORED IN LEXMAX

import numpy as np
import time
import gc
from LG_utils import generate_all_priority_orders
from LG_VI_stoc_lexmax import LG_VI_lexmax
from LG_VI_stoc_lexhull import LG_VI_lexhull
from ADS_Environment import Environment

"""
Verification script to compare LG_VI_lexmax and LG_VI_lexhull implementations.
 
This script:
1. Runs LG_VI_lexmax for all 6 lexicographic priority orders
2. Runs LG_VI_lexhull once (computes all 6 policies simultaneously)
3. Verifies that both methods produce identical policies and Q-values
4. Compares execution time between the two approaches
"""
 
 
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
 
 
def main(quick_test=False):
    """Run the verification.
 
    Args:
        quick_test: If True, only test 1 priority orders for quick verification
    """
 
    print("="*80)
    print("VERIFICATION: LG_VI_lexmax vs LG_VI_lexhull")
    print("="*80)
 
    # Parameters
    theta = 1.0
    discount_factor = 0.7
    n_objectives = 3
 
    # Get priority orders to test
    all_priorities = generate_all_priority_orders(n_objectives)
 
    if quick_test:
        # For quick testing: only test 2 priorities
        priorities_to_test = [all_priorities[0]]
        print(f"\n[QUICK TEST MODE]")
        print(f"Testing {len(priorities_to_test)} priority orders: {priorities_to_test}")
    else:
        priorities_to_test = all_priorities
        print(f"\n[FULL TEST MODE]")
        print(f"Testing all {len(priorities_to_test)} priority orders: {priorities_to_test}")
 
    # ========================================================================
    # APPROACH 1: Run LG_VI_lexmax for each priority order
    # ========================================================================
    print("\n[2/4] Running LG_VI_lexmax for each priority order...")
    print("-"*80)
 
    lexmax_policies = {}
    lexmax_Q_reference = None  # Only store one Q-table (they're all the same)
    lexmax_times = {}
 
    total_lexmax_time_start = time.time()
 
    for priority in priorities_to_test:
        priority_tuple = tuple(priority)
        print(f"\n  Running for priority {priority_tuple}...")
 
        # Reinitialize environment for fair comparison
        env = Environment()
 
        start_time = time.time()
        return_Q = (lexmax_Q_reference is None)
        policy, Q = LG_VI_lexmax(env, theta=theta, discount_factor=discount_factor, 
                             priority=priority, return_Q=return_Q)
        # policy, Q = LG_VI_lexmax(env, theta=theta, discount_factor=discount_factor, priority=priority)
        end_time = time.time()
 
        elapsed = end_time - start_time
        lexmax_times[priority_tuple] = elapsed
        lexmax_policies[priority_tuple] = policy
 
        # Only store Q-table from first run (all runs produce same Q)
        if lexmax_Q_reference is None and Q is not None:
            lexmax_Q_reference = Q.copy()
        
        else:
            q_consistency = compare_q_values(Q, lexmax_Q_reference, f"Priority {priority_tuple}")
            if not q_consistency['match']:
                print(f" WARNING: Q-table differs from first run")
                print(f" Max diff: {q_consistency['max_diff']:.2e}")
 
        # Explicitly clean up large objects
        del policy, env
        if Q is not None:
            del Q
        gc.collect()
 
        print(f"Completed in {elapsed:.2f}s (memory cleaned)")
 
    total_lexmax_time = time.time() - total_lexmax_time_start
 
    print(f"\n  TOTAL TIME (lexmax): {total_lexmax_time:.2f}s")
    print(f"  Average per priority: {total_lexmax_time/len(priorities_to_test):.2f}s")
 
    # ========================================================================
    # APPROACH 2: Run LG_VI_lexhull once
    # ========================================================================
    print("\n[3/4] Running LG_VI_lexhull (all priorities at once)...")
    print("-"*80)
 
    # Reinitialize environment for fair comparison
    env = Environment()
 
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
 
    for priority in priorities_to_test:
        priority_tuple = tuple(priority)
 
        # Compare policies
        policy_result = compare_policies(
            lexmax_policies[priority_tuple],
            lexhull_policies[priority_tuple],
            priority_tuple
        )
 
        if policy_result['match']:
            print(f"Priority {priority_tuple}: Policies MATCH (100%)")
        else:
            print(f"Priority {priority_tuple}: Policies DIFFER "
                  f"({policy_result['percentage']:.2f}% match)")
            all_policies_match = False
 
    print("\n  Q-Value Comparison:")
    print("  " + "-"*76)
 
    # Note: All lexmax runs produce the same Q-table (vectorial),
    # so we just need to compare the reference Q-table with lexhull
    q_result = compare_q_values(
        lexmax_Q_reference,
        lexhull_Q,
        "All priorities"
    )
 
    if q_result['match']:
        print(f"Q-tables MATCH (max diff: {q_result['max_diff']:.2e}, "
              f"tolerance: {q_result['tolerance']:.2e})")
    else:
        print(f"Q-tables DIFFER (max diff: {q_result['max_diff']:.2e}, "
              f"mean diff: {q_result['mean_diff']:.2e})")
        all_q_match = False
 
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
 
    print(f"\nCorrectness:")
    print(f"  Policies match: {' YES' if all_policies_match else ' NO'}")
    print(f"  Q-values match: {' YES' if all_q_match else ' NO'}")
 
    print(f"\nPerformance:")
    print(f"  LG_VI_lexmax ({len(priorities_to_test)} runs):  {total_lexmax_time:.2f}s")
    print(f"  LG_VI_lexhull (1 run):   {lexhull_time:.2f}s")
 
 
    # ========================================================================
    # DETAILED TIMING BREAKDOWN
    # ========================================================================
    print("\nDetailed Timing Breakdown:")
    print("-"*80)
    print(f"{'Priority':<20} {'Time (s)':<15} {'% of Total':<15}")
    print("-"*80)
 
    for priority in priorities_to_test:
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
        print("\n SUCCESS: Both methods produce identical results!")
        return 0
    else:
        print("\n FAIL: Methods produce different results!")
        return 1
 
 
if __name__ == "__main__":
    import sys
 
    # Check for --quick flag
    quick_test = "--quick" in sys.argv
 
    if quick_test:
        print("\n[Starting in QUICK TEST mode - only testing 2 priorities]")
        print("[Use no arguments for full test of all 6 priorities]\n")
    else:
        print("\n[Starting in FULL TEST mode - testing all 6 priorities]")
        print("[Use --quick flag for quick test with only 2 priorities]\n")
 
    exit_code = main(quick_test=quick_test)
    exit(exit_code)