# Version 2: Stochastic Multi-Objective ADS
 
This extends Version 1 by adding stochastic pedestrian behavior while maintaining the weighted scalarization approach.
 
### Stochastic Transitions
 
The key innovation is handling non-deterministic pedestrian movement:
 
- At certain map locations, pedestrians randomly choose from multiple actions
- Controlled by `degree_of_stochasticity` parameter (0-3)
 
### Algorithm Adaptations
 
All algorithms updated to handle stochasticity:
 
- **Value Iteration**: Computes expected values over stochastic outcomes
- **Convex Hull VI**: Sums probability-weighted hulls
- **Q-Learning**: Naturally handles stochasticity through sampling
 

## Core Files
 
### Main Entry Point
 
**`Main.py`**
- Central execution script adapted for stochasticity
- Line 13: `algorithm_used` - Choose algorithm
- Line 17: `Calculate_hulls` - For Convex Hull VI (compute vs. extract)
- Line 20: `Training` - Train new policy or load existing
- Line 22: `weights` - Objective weights `[w_car, w_ped1, w_ped2]`
- Two-phase workflow for Convex Hull VI:
  - Phase 1: Compute Q-hulls (set `Calculate_hulls=True`)
  - Phase 2: Extract policies for different weights (set `Calculate_hulls=False`)
 
### Environment
 
**`ADS_Environment.py`**
- Extended from Version 1 with stochastic transitions
- Same grid structure: 9x7 cells
- **Key difference**: `step()` method handles stochastic pedestrian actions
- At stochastic locations, pedestrians randomly select from available actions
- State transitions now return probability distributions
 
**`ItemAndAgent.py`**
- Same as Version 1
- Agent movement logic unchanged
- `move_map` defines stochastic action sets at certain positions
 
**`constants.py`**
- Same penalty values as Version 1
- **New**: `degree_of_stochasticity` parameter (0-3)
  - 0: Deterministic (like Version 1)
  - 1-3: Increasing levels of randomness in pedestrian behavior
 
### Algorithms
 
**`VI_Stochastic.py`** (Updated from Version 1)
- Stochastic Value Iteration
- **Key innovation**: Expected value computation (line 74-112)
- For each action, computes: `Q(s,a)`
- Builds transition model on first iteration for efficiency
- Stores all possible next states with probabilities
- Convergence when `delta < theta`
 
**`CH_VI_stochastic.py`** (Updated from Version 1)
- Stochastic Convex Hull Value Iteration
- Handles probabilistic transitions in hull operations
- `convexhull_VI()`: Main algorithm computing Q-hulls
- For stochastic states, sums probability-weighted hulls.
- `extract_policy_for_weights()`: Derives policy for given weights
- Two extraction modes:
  - Weighted: `policy = argmax_a w·q` for any weight vector
  - Lexicographic: Not yet supported (see Version 3)
 
**`CH_operations.py`**
- Same utility functions as Version 1
- `get_hull()`, `translate_hull()`, `sum_hulls()`, etc.
- No changes needed - hull operations are weight-agnostic
 
**`q_learning_algorithm.py`**
- Q-Learning naturally handles stochasticity through sampling
- Epsilon-greedy exploration
- Learns from experience without explicit model
- Slower to converge than VI but model-free
 
### Utilities
 
**`auxiliary_functions.py`**
- `tester`: Visualized policy testing
- Updated to handle stochastic outcomes
 
**`window.py`**, **`graphics.py`**
- Same as Version 1
 
## Usage
 
### Training a Stochastic Policy
 
**Using Value Iteration:**
 
```python
# Edit Main.py
algorithm_used = VALUE_ITERATION
Training = True
weights = [1.0, 100.0, 100.0]
 
# Run
python Main.py
 
# Policy saved to: policies/VI_stochastic_1.0-100.0-100.0-policy.npy
```
 
**Using Convex Hull VI:**
 
**Phase 1: Compute Q-hulls** (one-time, expensive)
```python
algorithm_used = CONVEX_HULL_VI
Calculate_hulls = True
weights = [1.0, 100.0, 100.0]  # Only needed for environment init
 
# Run - takes significant time
python Main.py
 
# Hulls saved to: policies/CH_qhulls.npy (~7-8 MB)
```
 
**Phase 2: Extract policies** (fast, repeatable)
```python
Calculate_hulls = False
weights = [1.0, 50.0, 100.0]  # Try different weights!
 
# Run
python Main.py
 
# Policy saved to: policies/CHVI_1.0-50.0-100.0-policy.npy
```
 
### Testing Policies
 
```python
Training = False
test_policy_name = "policies/VI_stochastic_1.0-100.0-100.0-policy.npy"
 
python Main.py
```
 
## Stochasticity Details
 
### How It Works
 
1. **Pedestrian Move Map** (in `ItemAndAgent.py`):
   - Defines available actions at each position
   - Example at position [3,3]: `[UP, LEFT, RIGHT, DOWN]`
   - Pedestrian randomly chooses one with equal probability
 
2. **Algorithm Adaptation**:
   - **Deterministic states**: Single outcome (prob=1.0)
   - **Stochastic states**: Multiple outcomes with probabilities
   - Example: If ped1 is at [3,3] and ped2 is not stochastic:
     - 4 possible next states (one per ped1 action)
     - Each has probability 0.25
 
3. **Expected Value Computation**:
   ```
   Q(s,a) = sum_outcomes prob × (reward + gamma × V(next_state))
   ```
 
### Degree of Stochasticity
 
Set in `constants.py`:
- `degree_of_stochasticity = 0`: Deterministic (Version 1 behavior)
- `degree_of_stochasticity = 1`: Low randomness
- `degree_of_stochasticity = 2`: Medium randomness
- `degree_of_stochasticity = 3`: High randomness
 

## File Overview
 
```
Stochastic/
├── README.md                      # This file
├── Main.py                        # Main execution script
├── ADS_Environment.py             # Stochastic environment
├── ItemAndAgent.py                # Agent classes (unchanged)
├── constants.py                   # Constants + stochasticity degree
├── VI_Stochastic.py              # Stochastic Value Iteration
├── CH_VI_stochastic.py           # Stochastic Convex Hull VI
├── CH_operations.py              # Convex hull utilities
├── q_learning_algorithm.py       # Q-Learning (unchanged)
├── auxiliary_functions.py        # Testing utilities
├── window.py                     # Pygame visualization
├── graphics.py                   # Plotting
├── wastebasket.png               # Visualization icon
└── policies/                     # Trained policies (generated, not in repo)
    ├── VI_stochastic_*.npy       # VI policies
    ├── CHVI_*.npy                # Extracted policies
    └── CH_qhulls.npy             # Q-hulls (large!)
```
 
 