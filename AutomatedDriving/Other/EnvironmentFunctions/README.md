### Environment
 
**`ADS_Environment.py`**
- Implements the MDP
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