# Multi-Objective Reinforcement Learning for Automated Driving Systems
 
This repository contains the implementation and evolution of multi-objective reinforcement learning (MORL) algorithms applied to an ethical automated driving scenario. The project explores different approaches to handling competing objectives in autonomous vehicle decision-making.
 
## Project Overview
 
The core problem models an autonomous vehicle navigating an urban environment while balancing three competing objectives:
 
1. **Individual Achievement**: Reaching the goal efficiently
2. **Internal Safety**: Avoiding damage to the vehicle from obstacles
3. **External Safety**: Preventing collisions with pedestrians
 
This repository documents the progressive development of MORL solutions, from deterministic weighted approaches to lexicographic priority-based methods.
 
## Repository Structure
 
The project is organized chronologically to show the evolution of the implementation:
 
### **AutomatedDriving/ADS_Code/** - Version 1: Deterministic Baseline
Initial implementation with weighted scalarisation in a deterministic environment.
- Basic Q-Learning and Value Iteration
- Convex Hull Value Iteration for multi-objective optimisation
- Deterministic agent behavior
 
### **AutomatedDriving/Stochastic/** - Version 2: Adding Uncertainty
Extended the baseline to handle stochastic pedestrian behavior.
- Adapted algorithms for stochastic transitions
- Pedestrians have probabilistic movement at certain locations
- Requires weight specification
 
### **AutomatedDriving/Lexicographic/** - Version 3: Weight-Free MORL
Implementation using lexicographic ordering to eliminate weight specification.
- **Lexicographic Value Iteration**: Priority-based decision making without weights
- **Flexible priority specification**: All 6 possible objective orderings
- **Policy comparison tools**: Analyse effects of different priorities
- **Convex Hull methods**: Train once, extract policies
  
### **Bibliografia/** - References and Learning Materials
Collection of papers, tutorials, and presentations on MORL and reinforcement learning.
 
## Quick Start
 
### Running the Current Implementation (Lexicographic)
 
```bash
cd AutomatedDriving/Lexicographic
 
# Train a policy prioritizing pedestrian safety
python Main_v2.py
# Edit the file to set: LG_priority = [1,2,0]  # Ped1 > Ped2 > Car
 ```
 
See `AutomatedDriving/Lexicographic/README.md` for detailed usage instructions.
 
## Key Concepts
 
### Multi-Objective Reinforcement Learning (MORL)
 
Traditional RL optimizes a single scalar reward. MORL handles multiple, often conflicting objectives:
 
- **Scalarisation Approach** (Versions 1 & 2): Combine objectives using weights
  - `r_total = w_1·r_car + w_2·r_ped1 + w_3·r_ped2`
  - **Challenge**: Choosing appropriate weights is difficult and domain-specific
 
- **Lexicographic Approach** (Version 3): Strict priority ordering
  - Compare objectives sequentially by importance
  - No weights needed - only ordinal preferences
  - Example: `[2,1,0]` means prioritize r_ped2 first, r_ped1 second, r_car last
 
### Lexicographic Ordering
 
A lexicographic ordering evaluates objectives in strict priority order:
 
1. Maximise the highest priority objective
2. If tied, maximise the second priority objective
3. Continue until tie is broken or all objectives considered
 
### Convex Hull Value Iteration
 
A technique that computes the set of all Pareto-optimal Q-vectors:
 
- **Train once**: Compute convex hulls of Q-values for all state-action pairs
- **Extract many**: Derive optimal policies for any weight vector or lexicographic ordering
- **Flexibility**: Explore the full spectrum of possible trade-offs post-training
 
 
## 🛠️ Technical Requirements
 
- Python 3.7+
- NumPy
- Pygame
- tqdm 
- scipy
- matplotlib 