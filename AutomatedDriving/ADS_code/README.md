# Version 1: Deterministic Multi-Objective ADS
 
This is the initial implementation of the multi-objective automated driving system. It establishes the baseline environment and implements classical reinforcement learning algorithms with weighted scalarisation.
 
## Overview
 
This version models a deterministic automated driving scenario where:
- A car must navigate to a goal location
- Two pedestrians move through the environment with fixed, deterministic behavior
- Three competing objectives must be balanced through manual weight specification
 
## Key Characteristics
 
### What This Version Provides
 
- **Deterministic Environment**: All agent behaviors are fully predictable
- **Weighted Scalarisation**: Multiple objectives combined using fixed weights
- **Classical RL Algorithms**: Q-Learning and Value Iteration
- **Convex Hull VI**: Multi-objective optimization without pre-specified weights
 
## Files
 
### Main Entry Point
 
**`Main.py`**
- Central execution script for training and testing
- Line 8: Choose algorithm (Q_LEARNING, VALUE_ITERATION, CONVEX_HULL_VI)
- Line 13: Toggle training mode (True/False)
- Line 14: Set weight vector `[w_car, w_ped1, w_ped2]`
- Line 15: Select which algorithm to use
- Line 47: Load and test trained policies
 
### Environment
 
**`ADS_Environment.py`**
- Implements the MDP (Markov Decision Process)
- Grid-based map: 9x7 cells (63 total)
- Cell types: Accessible (car), Pedestrian-only, Everyone (crosswalk), Inaccessible
- Three agents: 1 car (controlled), 2 pedestrians (deterministic heuristic)
- State: `[car_cell, ped1_cell, ped2_cell]`
- Actions: RIGHT=0, UP=1, LEFT=2, plus fast versions (3,4,5)
- Vectorial reward: `[r_car, r_ped1, r_ped2]` (scalarized using weights)
 
**`ItemAndAgent.py`**
- `Item` class: Base class for objects with position
- `Agent` class: Movable entities with collision detection
- `Move` class: Represents movement with direction and speed
- Defines action constants and movement logic
 
**`constants.py`**
- Penalty values for safety violations
- `SAFETY_INTERNAL = -10`: Hitting obstacles
- `SAFETY_EXTERNAL = -10`: Hitting pedestrians
- Injury multipliers for different collision severities
 
### Algorithms
 
**`q_learning_algorithm.py`**
- Standard tabular Q-Learning (Sutton & Barto, 1998)
- Epsilon-greedy exploration with exponential decay
- Learning rate (alpha) and discount factor (gamma) configurable
- Scalarises rewards using environment weights
- Returns: policy (action per state), Q-table
 
**`value_iteration_algorithm_original.py`**
- Dynamic programming approach for optimal policy
- Iterates until convergence (delta < theta)
- Deterministic environment allows exact value computation
- Faster convergence than Q-Learning
- Returns: policy, Q-table
 
**`convexhull_VI.py`**
- Multi-objective Value Iteration using convex hulls
- Computes Pareto-optimal Q-vector sets
- `convexhull_VI()`: Builds hulls for all state-action pairs
- `extract_policy_for_weights()`: Derives policy for given weights
- Train once, extract many policies for different weights
 
**`CH_operations.py`**
- Utility functions for convex hull operations
- `get_hull(points)`: Compute convex hull from point set
- `translate_hull(point, gamma, hull)`: Bellman backup operation
- `sum_hulls(hull1, hull2)`: Minkowski sum of hulls
- `non_dominated(solutions)`: Filter dominated points
 
### Utilities
 
**`auxiliary_functions.py`**
- `tester`: Visualized policy execution with Pygame
- `example_execution`: Run policy and print trajectory
- Evaluation utilities for testing trained policies
 
**`window.py`**
- Pygame-based visualization system
 
**`graphics.py`**
- Matplotlib plotting for training curves
- Visualizes reward evolution over episodes
- Moving average smoothing
 
## Usage
 
### Training a Policy
 
1. **Edit Main.py:**
   ```python
   Training = True
   weights = [1.0, 100.0, 100.0]  # Car, Ped1, Ped2
   algorithm_used = VALUE_ITERATION
   ```
 
2. **Run:**
   ```bash
   python Main.py
   ```
 
### Testing a Policy
 
1. **Edit Main.py:**
   ```python
   Training = False
   test_policy_name = "policies/policy_democratic.npy"
   ```
 
2. **Run:**
   ```bash
   python Main.py
   ```
 
3. **Pygame window shows policy execution**
 
### Using Convex Hull VI
 
**Step 1: Compute Hulls** (expensive, one-time)
```python
algorithm_used = CONVEX_HULL_VI
Training = True
# Computes Q-hulls for all state-action pairs
```
 
**Step 2: Extract Policies** (fast, repeatable)
```python
# Load hulls and extract policy for specific weights
weights = [1.0, 50.0, 100.0]
policy = extract_policy_for_weights(q_hulls, weights, env, env.n_actions)
```
 
## File Overview
 
```
ADS_code/
├── README.md                          # This file
├── Main.py                            # Main execution script
├── ADS_Environment.py                 # Environment/MDP definition
├── ItemAndAgent.py                    # Agent classes
├── constants.py                       # Reward constants
├── q_learning_algorithm.py            # Q-Learning implementation
├── value_iteration_algorithm_original.py  # Value Iteration
├── convexhull_VI.py                   # Convex Hull Value Iteration
├── convexhull.py                      # Early convex hull experiments
├── CH_operations.py                   # Convex hull utilities
├── auxiliary_functions.py             # Testing utilities
├── window.py                          # Pygame visualization
├── graphics.py                        # Matplotlib plotting
├── wastebasket.png                    # Icon for visualization
└── policies/                          # Trained policies (generated, not in repo)
```