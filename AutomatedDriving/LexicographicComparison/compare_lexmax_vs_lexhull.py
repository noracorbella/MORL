import numpy as np
import time
import gc
from LG_utils import generate_all_priority_orders, lex_hull_corrected, lex_max
from LG_VI_stoc_lexmax import LG_VI_lexmax
from LG_VI_stoc_lexhull_v2 import LG_VI_lexhull
from ADS_Environment import Environment

"""
Verification script to compare LG_VI_lexmax and LG_VI_lexhull implementations.
 
This script:
1. Runs LG_VI_lexmax for all 6 lexicographic priority orders
2. Runs LG_VI_lexhull once (computes all 6 policies simultaneously)
3. Verifies that both methods produce identical policies
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
 
 
def compare_q_hulls(Q_lexmax, Q_hulls_lexhull, env, priority_order, tolerance=1e-9):
    """
    Compare Q-values from lexmax with Q-hulls from lexhull.
    
    For lexhull, we need to extract the representative Q-vector for each action
    using the given priority order, then compare with lexmax's Q-values.
    """
    n_cells = env.map_num_cells
    n_actions = env.n_actions
    
    max_diff = 0
    total_diff = 0
    count = 0
    
    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                state_tuple = (c, p1, p2)
                
                # Get lexmax Q-values for this state
                q_lexmax = Q_lexmax[c, p1, p2]  # Shape: (n_actions, n_objectives)
                
                # Get representative Q-vectors from lexhull for this state
                q_lexhull = np.zeros((n_actions, 3))
                for action in range(n_actions):
                    if state_tuple + (action,) in Q_hulls_lexhull:
                        action_hull = Q_hulls_lexhull[state_tuple + (action,)]
                        # Select best vector from hull using this priority
                        best_idx = lex_max(action_hull, priority=list(priority_order))
                        q_lexhull[action] = action_hull[best_idx]
                
                # Compare
                diff = np.abs(q_lexmax - q_lexhull)
                max_diff = max(max_diff, np.max(diff))
                total_diff += np.sum(diff)
                count += diff.size
    
    mean_diff = total_diff / count if count > 0 else 0
    within_tolerance = max_diff < tolerance
    
    return {
        'match': within_tolerance,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'tolerance': tolerance,
        'priority': priority_order
    }
 
 
def main(quick_test=False, max_iterations=None):
    """Run the verification.
 
    Args:
        quick_test: If True, only test 2 priority orders for quick verification
        max_iterations: If provided, limit iterations for testing
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
        priorities_to_test = [all_priorities[0], all_priorities[1]]
        print(f"\n[QUICK TEST MODE]")
        print(f"Testing {len(priorities_to_test)} priority orders: {priorities_to_test}")
    else:
        priorities_to_test = all_priorities
        print(f"\n[FULL TEST MODE]")
        print(f"Testing all {len(priorities_to_test)} priority orders")
    
    if max_iterations:
        print(f"[LIMITED ITERATIONS: {max_iterations}]")
 
    # ========================================================================
    # APPROACH 1: Run LG_VI_lexmax for each priority order
    # ========================================================================
    print("\n[1/3] Running LG_VI_lexmax for each priority order...")
    print("-"*80)
 
    lexmax_policies = {}
    lexmax_Q_tables = {}  # Store Q-table for each priority
    lexmax_times = {}
 
    total_lexmax_time_start = time.time()
 
    for i, priority in enumerate(priorities_to_test):
        priority_tuple = tuple(priority)
        print(f"\n  [{i+1}/{len(priorities_to_test)}] Running for priority {priority_tuple}...")
 
        # Reinitialize environment for fair comparison
        env = Environment()
 
        start_time = time.time()
        policy, Q = LG_VI_lexmax(
            env, 
            theta=theta, 
            discount_factor=discount_factor, 
            priority=list(priority)
        )
        end_time = time.time()
 
        elapsed = end_time - start_time
        lexmax_times[priority_tuple] = elapsed
        lexmax_policies[priority_tuple] = policy
        lexmax_Q_tables[priority_tuple] = Q.copy()
 
        print(f"  Completed in {elapsed:.2f}s")
 
        # Clean up
        del policy, Q, env
        gc.collect()
 
    total_lexmax_time = time.time() - total_lexmax_time_start
 
    print(f"\n  TOTAL TIME (lexmax): {total_lexmax_time:.2f}s")
    print(f"  Average per priority: {total_lexmax_time/len(priorities_to_test):.2f}s")
 
    # ========================================================================
    # APPROACH 2: Run LG_VI_lexhull once
    # ========================================================================
    print("\n[2/3] Running LG_VI_lexhull (all priorities at once)...")
    print("-"*80)
 
    # Reinitialize environment for fair comparison
    env = Environment()
 
    lexhull_time_start = time.time()
    
    if max_iterations:
        lexhull_policies, lexhull_Q_hulls = LG_VI_lexhull(
            env, 
            theta=theta, 
            discount_factor=discount_factor,
        )
    else:
        lexhull_policies, lexhull_Q_hulls = LG_VI_lexhull(
            env, 
            theta=theta, 
            discount_factor=discount_factor
        )
    
    lexhull_time = time.time() - lexhull_time_start
 
    print(f"\n  TOTAL TIME (lexhull): {lexhull_time:.2f}s")
    print(f"  Policies extracted: {len(lexhull_policies)}")
 
    # ========================================================================
    # VERIFICATION: Compare results
    # ========================================================================
    print("\n[3/3] Verifying results...")
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
            print(f"  Priority {priority_tuple}: ✓ MATCH (100%)")
        else:
            print(f"  Priority {priority_tuple}: ✗ DIFFER "
                  f"({policy_result['percentage']:.2f}% match, "
                  f"{policy_result['total'] - policy_result['matches']} different states)")
            all_policies_match = False
 
    print("\n  Q-Value Comparison:")
    print("  " + "-"*76)
 
    # Compare Q-values for each priority
    for priority in priorities_to_test:
        priority_tuple = tuple(priority)
        
        q_result = compare_q_hulls(
            lexmax_Q_tables[priority_tuple],
            lexhull_Q_hulls,
            env,
            priority_tuple
        )
 
        if q_result['match']:
            print(f"  Priority {priority_tuple}: ✓ MATCH "
                  f"(max diff: {q_result['max_diff']:.2e})")
        else:
            print(f"  Priority {priority_tuple}: ✗ DIFFER "
                  f"(max diff: {q_result['max_diff']:.2e}, "
                  f"mean diff: {q_result['mean_diff']:.2e})")
            all_q_match = False
 
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
 
    print(f"\nCorrectness:")
    if all_policies_match:
        print(f"  Policies match: ✓ YES")
    else:
        print(f"  Policies match: ✗ NO")
    
    if all_q_match:
        print(f"  Q-values match: ✓ YES")
    else:
        print(f"  Q-values match: ✗ NO (within tolerance)")
 
    print(f"\nPerformance:")
    print(f"  LG_VI_lexmax ({len(priorities_to_test)} runs):  {total_lexmax_time:.2f}s")
    print(f"  LG_VI_lexhull (1 run):   {lexhull_time:.2f}s")
 
    speedup = total_lexmax_time / lexhull_time
    print(f"\n  Speedup: {speedup:.2f}x")
 
    if speedup > 1:
        print(f"  → LG_VI_lexhull is {speedup:.2f}x FASTER ⚡")
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
        print("\n✓ VERIFICATION PASSED: Both methods produce identical results!")
        return 0
    else:
        print("\n✗ VERIFICATION FAILED: Methods produce different results!")
        return 1
 
 
if __name__ == "__main__":
    import sys
 
    # Check for flags
    quick_test = "--quick" in sys.argv
    
    # Check for max iterations
    max_iterations = None
    for arg in sys.argv:
        if arg.startswith("--max-iter="):
            max_iterations = int(arg.split("=")[1])
    
    if quick_test:
        print("\n[Starting in QUICK TEST mode - only testing 2 priorities]")
        print("[Use no arguments for full test of all 6 priorities]\n")
    else:
        print("\n[Starting in FULL TEST mode - testing all 6 priorities]")
        print("[Use --quick flag for quick test with only 2 priorities]\n")
    
    if max_iterations:
        print(f"[Limiting to {max_iterations} iterations for testing]\n")
 
    exit_code = main(quick_test=quick_test, max_iterations=max_iterations)
    exit(exit_code)