# Lexicographic Multi-Objective Reinforcement Learning for Automated Driving

This directory contains implementations of various reinforcement learning algorithms for solving a multi-objective automated driving scenario with lexicographic approaches that don't require weight specification.

## Core Files

### Main Entry Points

**`Main.py`** (Original Interface)
- Main execution script using string-based priority specification
- Supports 4 algorithms: Q-Learning, Value Iteration, Lexicographic VI, Convex Hull VI
- Priority specified as `'car'` or `'pedestrian'` (line 50)
- Compatible with original LG_VI_stochastic.py

**`Main_v2.py`** (Enhanced Interface)
- Updated main script with flexible priority specification
- Priority specified as list: `[0,1,2]` for car first, `[1,2,0]` for ped1 first, `[2,1,0]` for ped2 first
- Line 49: Set `LG_priority` as a list of objective indices
- Example: `LG_priority = [2,1,0]` prioritizes pedestrian2 > pedestrian1 > car
- Automatically generates policy filenames based on priority order
- Uses LG_VI_stochastic_v2.py for enhanced flexibility

### Environment & Agents

**`ADS_Environment.py`**
- Defines the `Environment` class implementing the MDP
- Grid-based map: 9x7 with 63 cells (accessible, pedestrian, everyone, inaccessible)
- Manages three agents: 1 car + 2 pedestrians
- State transition function `step()` (line 511)
- Vectorial rewards for three objectives (line 434)
- Collision detection and safety calculation (line 356-408)
- State space: ~250,000 states (car_pos × ped1_pos × ped2_pos)

**`ItemAndAgent.py`**
- `Item` class: Basic objects with position
- `Agent` class: Extends Item with movement capabilities
- Actions: RIGHT=0, UP=1, LEFT=2, plus fast versions (3,4,5)
- `Move` class: Represents movement requests with collision checking
- Pedestrian move map with configurable stochasticity (line 135-152)

**`constants.py`**
- Reward constants for safety violations
- `SAFETY_INTERNAL = -10`: Car hitting obstacles
- `SAFETY_EXTERNAL = -10`: Base penalty for hitting pedestrians
- `SAFETY_EXTERNAL_INJURY_MULTIPLIER = 0.3`: Non-lethal collisions
- `SAFETY_EXTERNAL_LETHAL_MULTIPLIER = 1.0`: Lethal collisions
- `degree_of_stochasticity = 1`: Controls pedestrian randomness (0-3)

## Algorithm Implementations

### Traditional Single-Objective Approaches (Weighted)

**`q_learning_algorithm.py`**
- Standard Q-Learning (Sutton & Barto, Section 6.5)
- Epsilon-greedy exploration with exponential decay
- Requires weight vector for scalarization
- Q-table: `[n_cells, n_cells, n_cells, n_actions]`
- Trains through trial-and-error experience

**`VI_Stochastic.py`**
- Value Iteration for stochastic environment
- Handles pedestrian stochasticity via expected value computation
- Builds transition model in first iteration for efficiency (line 74-112)
- Requires weight vector for scalarization
- Convergence when delta < theta

### Lexicographic Approaches (Weight-Free)

**`LG_VI_stochastic.py`** (Original)
- Lexicographic Value Iteration (Vamplew et al., 2021)
- Priority specification: `'car'` or `'pedestrian'` (strings)
- `lex_max()` function (line 4-45): Lexicographic action selection
- Maintains vectorial Q-values: `Q[state, action] = [r0, r1, r2]`
- Two modes:
  - `priority='car'`: Order [r_car, r_ped1, r_ped2]
  - `priority='pedestrian'`: Order [r_ped1, r_ped2, r_car]

**`LG_VI_stochastic_v2.py`**
- Updated Lexicographic VI with flexible priority specification
- Priority as list: `priority=[0,1,2]` instead of strings
- Line 6: `def LG_VI(env, theta=1.0, discount_factor=0.7, priority=[0,1,2])`
- Can express all 6 possible orderings:
  - `[0,1,2]`: Car > Internal Safety > External Safety
  - `[0,2,1]`: Car > External Safety > Internal Safety
  - `[1,0,2]`: Internal Safety > Car > External Safety
  - `[1,2,0]`: Internal Safety > External Safety > Car
  - `[2,0,1]`: External Safety > Car > Internal Safety
  - `[2,1,0]`: External Safety > Internal Safety > Car
- Imports `lex_max()` from LG_utils for cleaner code organization
- Line 134: Uses `lex_max(q_vectors, priority=priority)` for action selection

**`LG_utils.py`**
- Centralized utilities for lexicographic operations
- **`lex_max(q_vectors, priority=[0,1,2], tol=1e-9)`** (line 8-54):
  - Performs lexicographic maximization over Q-vectors
  - Takes priority as flexible list of objective indices
  - Returns index of lexicographically best action
  - Can handle numpy arrays or lists as priority
- **`generate_all_priority_orders(n_objectives)`** (line 57-63):
  - Generates all possible permutations of objectives
  - For n=3, returns 6 orderings: `[[0,1,2], [0,2,1], [1,0,2], ...]`
  - Useful for exhaustive policy comparison
- **`lex_hull(q_vectors, n_objectives=3, tol=1e-9)`** (line 67-90):
  - Computes optimal action for every possible lexicographic ordering
  - Returns dict mapping priority tuples to optimal actions
  - Example: `{(0,1,2): 3, (1,2,0): 1, ...}`
  - Enables analysis of how priority order affects decisions

**`LG_CH_VI_stochastic.py`**
- Lexicographic Convex Hull Value Iteration
- Based on Barrett & Narayanswamy (2008)
- `LG_CH_VI()` (line 5-191): Computes Q-hulls instead of single Q-values
- Each Q_hull contains multiple non-dominated Q-vectors
- Bellman hull backup: Q(s,a) = CH(∪ [prob × (r + γ × hull(s'))]) (line 104-142)
- Three policy extraction methods:
  1. **Weighted**: Given weights, find max w·q (line 194-230)
  2. **Lexicographic**: Find lexicographically best Q-vector (line 292-330)
  3. **Priority shortcuts**: lexP (car) / lexA (pedestrian) (line 332-339)

### Convex Hull Operations

**`CH_operations.py`**
- Mathematical utilities for convex hull manipulation
- `get_hull(points)` (line 18-37): Compute convex hull using scipy
- `translate_hull(point, gamma, hull)` (line 40-56): Bellman operation r + γH
- `sum_hulls(hull1, hull2)` (line 61-84): Minkowski sum
- `max_q_value(weight, hull)` (line 87-105): Extract Q-value for weight
- `non_dominated(solutions)` (line 5-15): Pareto filtering

## Analysis & Testing Tools

**`auxiliary_functions.py`**
- Testing and evaluation utilities
- `test_policy_vectorial(env, policy, num_episodes, verbose)` (line 106-173):
  - Evaluates policy over multiple episodes
  - Returns vectorial rewards separately (not scalarized)
  - Computes mean, std, collision rate
- `example_execution(env, policy, render)` (line 40-93):
  - Simulates policy with optional visualization
  - Prints state, action, reward at each step
- `tester` / `tester_vect` classes (line 95-242):
  - Multithreaded policy testing with Pygame visualization
  - `tester_vect` uses vectorial rewards

**`policy_comparison.py`**
- Tools for comparing policies from different priority orderings
- **`extract_V_from_Q(policy, Q)`** (line 4-25):
  - Extracts value function V from Q-table given policy
  - V[s] = Q[s, π(s)] for all states
- **`compare_policies(env, policy1, policy2, Q1, Q2, initial_state)`** (line 27-64):
  - Compares two policies at initial state
  - Checks if they yield same value
  - Returns: are_identical, V1_init, V2_init, V1, V2
- **`compare_all_states(V1, V2, tol)`** (line 67-105):
  - Compares value functions across all states
  - Returns statistics: total states, identical states, differences
- **Main script** (line 108-164):
  - Loads policies for different priority orderings
  - Example: Compare `[0,1,2]` vs `[2,1,0]`
  - Shows how many states have different actions
  - Prints example states where policies differ

**`diagnostic_compare_policies.py`**
- Compares Value Iteration (weighted) vs Lexicographic VI
- Loads policies with weights [1, 100, 100] vs lex priorities
- Counts state-by-state differences
- Shows action distribution
- Tests both policies and compares mean rewards

**`window.py`**
- Pygame-based visualization
- `Window` class (line 62-234): Graphical rendering
- Displays grid, car (C), pedestrians (P), obstacles, goals (X)
- Color coding: gray (accessible), black (inaccessible), blue (crosswalk)
- Controls: SPACE (pause), ESCAPE (exit)

**`graphics.py`**
- Matplotlib plotting for training curves
- Loads reward data from 'graphics.npy'
- Plots moving average (window=500) of objectives
- Shows convergence over training episodes

## Pre-Trained Policies

The `policies/` directory contains pre-trained models:

### Lexicographic VI (Original)
- `LGVI_car_priority-policy.npy` & `-Q.npy`: Priority 'car'
- `LGVI_pedestrian_priority-policy.npy` & `-Q.npy`: Priority 'pedestrian'

### Lexicographic VI v2 (Flexible Priorities)
- `LGVI_v2_0-1-2_priority-policy.npy` & `-Q.npy`: Car > Ped1 > Ped2
- `LGVI_v2_1-2-0_priority-policy.npy` & `-Q.npy`: Ped1 > Ped2 > Car
- `LGVI_v2_2-1-0_priority-policy.npy` & `-Q.npy`: Ped2 > Ped1 > Car

### Convex Hull VI
- `CHVI_qhulls.npy`: Complete Q-hull set (7.5 MB)
- `LGCH_qhulls.npy`: Alternative hull set
- `CHVI_lex_pedestrian_priority-policy.npy`: Extracted with pedestrian priority

### Traditional Methods
- `q_learning_policy.npy`: Q-Learning trained policy
- `value_iteration_policy.npy`: Value Iteration trained policy


### Training Lexicographic VI v2 with Custom Priority

1. Open `Main_v2.py`
2. Set `algorithm_used = LEXICOGRAPHIC_VALUE_ITERATION` (line 22)
3. Set `LG_Training = True` (line 48)
4. Choose priority order (line 49):
   ```python
   LG_priority = [2,1,0]  # Ped2 > Ped1 > Car
   # OR
   LG_priority = [1,2,0]  # Ped1 > Ped2 > Car
   # OR
   LG_priority = [0,2,1]  # Car > Ped2 > Ped1
   # ... etc

5. Run: python Main_v2.py
6. Policy saved to policies/LGVI_v2_{priority}_priority-policy.npy

### Comparing Two Priority Orderings
1. Train policies for two different priorities (e.g., `[0,1,2]` and `[2,1,0]`)
2. Open policy_comparison.py
3. Set priority1 and priority2 (line 114-115):
```python
priority1 = [0,1,2]  # Car priority
priority2 = [2,1,0]  # Ped2 priority
```
4. Run: python policy_comparison.py
5. Output shows:
    - Percentage of states with identical values
    - States where values differ
    - Percentage of states with different actions

### Training Convex Hull VI (Train Once, Extract Many)
1. Compute hulls (expensive, one-time):
    - Set algorithm_used = CONVEX_HULL_VALUE_ITERATION in Main.py
    - Set CHVI_Calculate_hulls = True
    - Run: python Main.py
    - Hulls saved to policies/CHVI_qhulls.npy
2. Extract policy (fast, repeatable):
    - Set CHVI_Calculate_hulls = False
    - Choose extraction method:
        - Lexicographic: Set CHVI_extraction_method = `'lexicographic'`
            - Set CHVI_priority = `'car'` or `'pedestrian'`
        - Weighted: Set CHVI_extraction_method = `'weighted'`
            - Set `weights = [w0, w1, w2]`
    - Run: python Main.py
    - Extract as many policies as you want

### Testing a Pre-Trained Policy
1. Choose main script: Main.py or Main_v2.py
2. Set Training = False or LG_Training = False
3. Set policy filename to load
4. Configure testing:
```python
Test_with_vectorial = True  # See separate objectives
Test_Episodes = 100
Show_visualisation = True  # Pygame 
```
5. Run main script

### File Dependencies
Main.py / Main_v2.py
├── ADS_Environment.py
│   ├── ItemAndAgent.py
│   │   └── constants.py
│   └── window.py
├── q_learning_algorithm.py
├── VI_Stochastic.py
├── LG_VI_stochastic.py [v1]
├── LG_VI_stochastic_v2.py [v2]
│   └── LG_utils.py 
├── LG_CH_VI_stochastic.py
│   └── CH_operations.py
└── auxiliary_functions.py

policy_comparison.py 
└── Loads policies from policies/

debug_LGCHVI.py
└── Loads Q-hulls from policies/