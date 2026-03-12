# Convex Hull Value Iteration: A Learning Guide

This guide provides a comprehensive walkthrough of Convex Hull Value Iteration (CHVI) algorithms for Multi-Objective Reinforcement Learning (MORL). We combine mathematical theory with detailed code explanations to help you understand how these algorithms work.

---

## Table of Contents

1. [Introduction: Why Convex Hulls?](#1-introduction-why-convex-hulls)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Core Operations (CH_operations.py)](#3-core-operations-ch_operationspy)
4. [Deterministic CHVI (convexhull_VI.py)](#4-deterministic-chvi-convexhull_vipy)
5. [Stochastic CHVI (CH_VI_stochastic.py)](#5-stochastic-chvi-ch_vi_stochasticpy)
6. [Lexicographic Hull VI (LG_VI_stoc_lexhull.py)](#6-lexicographic-hull-vi-lg_vi_stoc_lexhullpy)
7. [Comparison Summary](#7-comparison-summary)

---

## 1. Introduction: Why Convex Hulls?

### The Problem with Weighted Scalarisation

In traditional MORL, we combine multiple objectives into a single scalar:

```
r_total = w_1 * r_obj1 + w_2 * r_obj2 + w_3 * r_obj3
```

**Problems:**
- You must choose weights **before** training
- Training is expensive - you need to retrain for each weight combination
- Hard to know good weights without experimentation

### The Convex Hull Solution

**Key Insight from Barrett & Narayanan (2008):** Instead of committing to weights upfront, compute the **set of all Pareto-optimal value vectors** for each state-action pair. These sets form convex hulls in the objective space.

**Benefits:**
1. **Train once**: Compute Q-hulls covering all possible trade-offs
2. **Extract many**: Derive optimal policy for *any* weight vector instantly
3. **No commitment**: Explore different priorities without retraining

### Visual Intuition

Consider 2 objectives (e.g., car efficiency vs. pedestrian safety):

```
Objective 2 (Safety)
    ^
    |     * Pareto-optimal points
    |   *   \
    |  *     \  <- Convex hull boundary
    | *       \
    |*---------*
    +-----------------> Objective 1 (Efficiency)
```

Each point on the Pareto frontier represents a different optimal trade-off. The convex hull captures all these trade-offs efficiently.

---

## 2. Mathematical Foundations

### 2.1 Multi-Objective MDPs (MOMDPs)

A Multi-Objective MDP extends standard MDPs with vector-valued rewards:

- **States** S: Set of environment states
- **Actions** A: Set of possible actions
- **Transition** T(s'|s,a): Probability of reaching s' from s via action a
- **Reward** R(s,a): A **vector** [r_1, r_2, ..., r_n] instead of scalar

For our driving scenario:
```
R(s,a) = [r_car, r_ped1, r_ped2]
```

### 2.2 Pareto Dominance

**Definition:** Vector **v** dominates vector **u** (written v ≻ u) if:
- v_i >= u_i for all objectives i
- v_j > u_j for at least one objective j

**Non-dominated set:** Points where no point dominates another.

```python
# From CH_operations.py:5-15
def non_dominated(solutions):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points
            dominated = (np.asarray(solutions[is_efficient]) <= c).all(axis=1)
            is_efficient[is_efficient] = np.invert(dominated)
            is_efficient[i] = 1  # Keep the point itself
    return solutions[is_efficient]
```

**How it works:**
1. Start assuming all points are efficient
2. For each point `c`, find points dominated by `c`
3. Mark dominated points as inefficient
4. Return only non-dominated points

### 2.3 Convex Hull in Multi-Objective Space

The convex hull of a point set is the smallest convex polytope containing all points.

**Why convex hulls matter for MORL:**
- For linear scalarisation (weighted sum), the optimal solution is always on the convex hull
- Points inside the hull are always dominated by some convex combination of hull vertices
- We only need to store hull vertices, not all Pareto points

### 2.4 Barrett & Narayanan's Key Definitions (2008)

The paper "Learning All Optimal Policies with Multiple Criteria" defines three fundamental operations:

#### Definition 1: Translation and Scaling

For immediate reward vector **r** and discount factor γ:

```
T(r, γ, H) = {r + γ * h : h ∈ H}
```

This is the Bellman backup for hulls - adds reward and discounts future values.

#### Definition 2: Minkowski Sum

For two hulls H₁ and H₂:

```
H₁ ⊕ H₂ = CH({h₁ + h₂ : h₁ ∈ H₁, h₂ ∈ H₂})
```

Used for combining probabilistic outcomes in stochastic settings.

#### Definition 3: Q-value Extraction

For weight vector **w**:

```
Q(s,a,w) = max_{q ∈ Hull(s,a)} w · q
```

Extract the best Q-value for specific weights via dot product maximization.

---

## 3. Core Operations (CH_operations.py)

This file implements the fundamental convex hull operations. Let's examine each function.

### 3.1 Computing the Convex Hull

```python
# CH_operations.py:18-37
def get_hull(points):
    """
    From a set of points, computes its associated convex hull.
    """
    # Step 1: Filter to non-dominated points first
    points = non_dominated(np.array(points))

    # Step 2: Use scipy's ConvexHull algorithm
    try:
        hull = ConvexHull(points)
        hull_points = [points[vertex] for vertex in hull.vertices]
        # Step 3: Filter hull vertices to non-dominated ones
        vertexs = non_dominated(np.array(hull_points))
    except:
        # Not enough points for convex hull (collinear, etc.)
        vertexs = points

    return np.array(vertexs)
```

**Algorithm flow:**

```
Input points → Non-dominated filter → ConvexHull → Non-dominated filter → Output
     [8]              [4]                              [3]
```

**Why double filtering?**
- First filter: Reduces computational cost
- Second filter: ConvexHull returns all vertices; we only want Pareto-optimal ones

**Example:**
```python
points = [[1, 5], [2, 4], [3, 3], [4, 2], [5, 1], [2, 2]]
#                                                  ↑ dominated
get_hull(points)  # Returns [[1,5], [5,1]] (or similar Pareto front)
```

### 3.2 Translation Operation (Bellman Backup)

```python
# CH_operations.py:40-56
def translate_hull(point, gamma, hull):
    """
    Translation and scaling operation (Definition 1 from Barrett & Narayanan).

    Computes: point + gamma * hull
    """
    if len(hull) == 0:
        hull = np.array([point])
    else:
        # Scale by discount factor
        hull = np.multiply(hull, gamma, casting="unsafe")
        if len(point) > 0:
            # Add immediate reward
            hull = np.add(hull, point, casting="unsafe")
    return hull
```

**Mathematical meaning:**

```
Q(s,a) = r(s,a) + γ * V(s')
```

In hull form:
```
Hull_Q(s,a) = r(s,a) + γ * Hull_V(s')
```

**Visual example (2D):**

```
Original hull V(s'):          After translate_hull([1,2], 0.9, hull):

    [4,6]                         [0.9*4+1, 0.9*6+2] = [4.6, 7.4]
      *                                   *
     / \                                 / \
    *   *                               *   *
  [2,4] [6,2]                      [2.8,5.6] [6.4,3.8]
```

### 3.3 Minkowski Sum of Hulls

```python
# CH_operations.py:61-84
def sum_hulls(hull_1, hull_2):
    """
    Sum operation of convex hulls (Definition 2 from Barrett & Narayanan).

    Minkowski sum: combines all possible pairwise additions.
    """
    if len(hull_1) == 0:
        return hull_2
    elif len(hull_2) == 0:
        return hull_1

    new_points = None

    # Generate all pairwise sums
    for i in range(len(hull_1)):
        if new_points is None:
            new_points = translate_hull(hull_1[i].copy(), 1, hull_2.copy())
        else:
            new_points = np.concatenate(
                (new_points, translate_hull(hull_1[i].copy(), 1, hull_2.copy())),
                axis=0
            )

    # Compute convex hull of all sums
    return get_hull(new_points)
```

**When is this used?**

In stochastic environments with multiple possible outcomes:

```
Q(s,a) = Σ_i p_i * [r_i + γ * V(s'_i)]
```

We can't just average hulls. Instead, we use the Minkowski sum of probability-weighted hulls.

**Visual intuition:**

```
Hull_1:          Hull_2:          Minkowski Sum:
   *                 *                   *
  / \               / \                 /   \
 *   *             *   *               *     *
                                      / \   / \
                                     *   * *   *
```

### 3.4 Q-value Extraction for Weights

```python
# CH_operations.py:87-105
def max_q_value(weight, hull):
    """
    Extraction of the Q-value (Definition 3 from Barrett & Narayanan).

    Returns: max_{q ∈ hull} w · q
    """
    scalarised = []

    for i in range(len(hull)):
        f = np.dot(weight, hull[i])
        scalarised.append(f)

    scalarised = np.array(scalarised)
    return np.max(scalarised)
```

**Geometric interpretation:**

The weight vector defines a direction. We're finding the hull vertex that's furthest in that direction.

```
w = [0.7, 0.3]  (prefer objective 1)

         Obj2
          ^
          |   *A        w · A = 0.7*2 + 0.3*5 = 2.9
          |  / \        w · B = 0.7*5 + 0.3*2 = 4.1  <- MAX
          | /   \       w · C = 0.7*3 + 0.3*1 = 2.4
          |*     *
          |B      C
          +-----------> Obj1
```

**Usage:** Once hulls are computed, extract optimal policy for *any* weights in O(|hull|) time.

---

## 4. Deterministic CHVI (convexhull_VI.py)

This implements Convex Hull Value Iteration for deterministic environments.

### 4.1 Algorithm Overview

```
CONVEX HULL VALUE ITERATION (Deterministic)
============================================
Input: MDP with vector rewards
Output: Q_hulls for all (state, action) pairs

1. Initialize Q_hulls[(s,a)] = {0} for all s, a
2. Repeat until convergence:
   For each state s:
     For each action a:
       next_s = T(s, a)  # Deterministic transition

       # Union of hulls for all actions in next state
       V_hull(next_s) = CH(∪_{a'} Q_hull(next_s, a'))

       # Bellman backup
       Q_hull(s,a) = r(s,a) + γ * V_hull(next_s)

3. Return Q_hulls
```

### 4.2 Initialization

```python
# convexhull_VI.py:14-24
Q_hulls = {}
for c in env.states_agent_left:
    for p1 in env.states_agent_right:
        for p2 in env.states_agent_right:
            for a in range(n_actions):
                Q_hulls[(c, p1, p2, a)] = [np.zeros(n_rewards)]
```

**Key point:** Each hull starts as a single point at the origin [0, 0, 0].

### 4.3 Model Building (First Iteration)

```python
# convexhull_VI.py:54-65
if iteration == 1:
    env.reset(state_translated[0], state_translated[1], state_translated[2])
    next_state, reward, done_array = env.step([action])
    done = done_array[0]

    # Cache for future iterations
    model_next_state[c, p1, p2, action] = next_state
    model_next_reward[c, p1, p2, action] = reward
    model_next_done[c, p1, p2, action] = done
else:
    # Use cached model
    next_state = model_next_state[c, p1, p2, action]
    reward = model_next_reward[c, p1, p2, action]
    done = model_next_done[c, p1, p2, action]
```

**Optimization:** The environment is deterministic, so transitions are cached after the first iteration. This dramatically speeds up subsequent iterations.

### 4.4 The Bellman Hull Backup

```python
# convexhull_VI.py:67-108
if done:
    # Terminal state: Q(s,a) = reward vector only
    new_hull = [reward.copy()]
else:
    # Step 1: Collect all Q-vectors from next state
    next_c, next_p1, next_p2 = next_state
    all_next_q_vectors = []

    for next_action in range(n_actions):
        next_hull = Q_hulls[(next_c, next_p1, next_p2, next_action)]
        for q_vec in next_hull:
            all_next_q_vectors.append(np.array(q_vec))

    # Step 2: Compute convex hull of union
    all_next_q_vectors = np.array(all_next_q_vectors)
    next_state_hull = get_hull(all_next_q_vectors)

    # Step 3: Bellman backup: r + γ * hull
    new_hull = translate_hull(reward, discount_factor, next_state_hull)

    # Step 4: Compute hull of result
    if len(new_hull) > 1:
        new_hull = get_hull(new_hull)
```

**Visual trace:**

```
State s, Action a
      |
      v
    [reward = [2, -1, 0]]
      |
      | + γ *
      v
   next_state s'
      |
      v
   Union of Q_hulls(s', a') for all a'
      |
      v
   ConvexHull of union = V_hull(s')
      |
      v
   translate_hull(reward, γ, V_hull(s'))
      |
      v
   New Q_hull(s, a)
```

### 4.5 Convergence Check

```python
# convexhull_VI.py:119-125
if old_hull_array.shape == new_hull_array.shape:
    max_diff = np.max(np.abs(new_hull_array - old_hull_array))
else:
    # Different number of vertices - mark as changed
    max_diff = float('inf')

delta = max(delta, max_diff)
```

**Challenge:** Hulls can have different numbers of vertices between iterations. The algorithm handles this by treating shape changes as infinite difference.

### 4.6 Policy Extraction

```python
# convexhull_VI.py:152-179
def extract_policy_for_weights(Q_hulls, weights, env, n_actions):
    """Extract optimal policy for a specific weight vector."""
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)

    for c in env.states_agent_left:
        for p1 in env.states_agent_right:
            for p2 in env.states_agent_right:
                best_value = -np.inf
                best_action = 0

                for action in range(n_actions):
                    hull = np.array(Q_hulls[(c, p1, p2, action)])

                    # Find best Q-value for this weight
                    q_value = max_q_value(weights, hull)

                    if q_value > best_value:
                        best_value = q_value
                        best_action = action

                policy[c, p1, p2] = best_action

    return policy
```

**Key insight:** After training, extracting a policy is just:
1. For each state, for each action: compute weighted max over hull
2. Select action with highest weighted max
3. O(|S| * |A| * |hull|) - very fast!

---

## 5. Stochastic CHVI (CH_VI_stochastic.py)

This extends CHVI to handle stochastic transitions (probabilistic pedestrian movement).

### 5.1 The Stochastic Challenge

In deterministic CHVI:
```
Q(s,a) = r(s,a) + γ * V(s')
```

In stochastic CHVI:
```
Q(s,a) = Σᵢ pᵢ * [rᵢ + γ * V(s'ᵢ)]
```

We must combine multiple possible outcomes weighted by probability.

### 5.2 Identifying Stochastic States

```python
# CH_VI_stochastic.py:27-28
pedestrian_stochastic_actions = env.agents[1].move_map[3][3]
stochastic_state = [3, 3]  # Cell where pedestrians have multiple options
```

Pedestrians at position [3,3] (the crosswalk decision point) have probabilistic movement.

### 5.3 Enumerating Outcomes

```python
# CH_VI_stochastic.py:59-98
if not p1_is_stochastic and not p2_is_stochastic:
    # Deterministic: single outcome
    env.reset(...)
    next_state, reward, done_array = env.step([action])
    prob = 1.0
    outcomes.append((next_state, reward, done, prob))

elif p1_is_stochastic and not p2_is_stochastic:
    # Ped1 stochastic: iterate over ped1's possible actions
    for p1_action in pedestrian_stochastic_actions:
        env.reset(...)
        next_state, reward_vect, done_array = env.step([action, p1_action, 8000])
        prob = 1.0 / len(pedestrian_stochastic_actions)
        outcomes.append((next_state, reward_vect, done, prob))

elif not p1_is_stochastic and p2_is_stochastic:
    # Ped2 stochastic: similar
    for p2_action in pedestrian_stochastic_actions:
        ...

else:
    # Both stochastic: enumerate all combinations
    for p1_action in pedestrian_stochastic_actions:
        for p2_action in pedestrian_stochastic_actions:
            env.reset(...)
            next_state, reward_vect, done_array = env.step([action, p1_action, p2_action])
            prob = 1.0 / (len(pedestrian_stochastic_actions) ** 2)
            outcomes.append(...)
```

**Outcome structure:**
```python
outcomes = [
    (next_state_1, reward_1, done_1, prob_1),
    (next_state_2, reward_2, done_2, prob_2),
    ...
]
```

### 5.4 Combining Stochastic Outcomes

```python
# CH_VI_stochastic.py:103-152
outcome_hulls = []

for next_state, reward_vect, done, prob in outcomes:
    if done:
        # Terminal: just scaled reward
        outcome_hull = prob * np.array([reward_vect])
    else:
        # Compute V_hull for next state
        next_c, next_p1, next_p2 = next_state
        all_next_q_vectors = []

        for next_action in range(n_actions):
            next_hull = Q_hulls[(next_c, next_p1, next_p2, next_action)]
            all_next_q_vectors.extend(next_hull)

        next_state_hull = get_hull(np.array(all_next_q_vectors))

        # Bellman backup for this outcome
        outcome_hull = translate_hull(reward_vect, discount_factor, next_state_hull)

        # Scale by probability
        outcome_hull = prob * outcome_hull

    outcome_hulls.append(outcome_hull)

# Combine outcomes using Minkowski sum
if len(outcome_hulls) == 1:
    new_hull = outcome_hulls[0]
else:
    combined_hull = outcome_hulls[0]
    for outcome_hull in outcome_hulls[1:]:
        combined_hull = sum_hulls(combined_hull, outcome_hull)
    new_hull = combined_hull
```

### 5.5 Why Minkowski Sum?

**Mathematical justification:**

For two independent random outcomes with hulls H₁ (prob p₁) and H₂ (prob p₂):

```
E[H] = p₁ * H₁ ⊕ p₂ * H₂
```

Where ⊕ is the Minkowski sum.

**Intuition:** The expected value of a random variable over hulls is the "average" of all possible hull combinations. Minkowski sum captures all these combinations.

**Example with 2 outcomes:**

```
Outcome 1 (prob=0.5):     Outcome 2 (prob=0.5):     Combined:
  Hull_1 = [[2,4]]         Hull_2 = [[6,2]]

  0.5 * Hull_1 = [[1,2]]   0.5 * Hull_2 = [[3,1]]

  Minkowski sum: [[1,2]] ⊕ [[3,1]] = [[1+3, 2+1]] = [[4,3]]
```

---

## 6. Lexicographic Hull VI (LG_VI_stoc_lexhull.py)

This combines lexicographic ordering with hull-based computation.

### 6.1 Lexicographic vs. Weighted Scalarisation

**Weighted approach:** `Score = w₁*r₁ + w₂*r₂ + w₃*r₃`

**Lexicographic approach:** Compare objectives in strict priority order.

For priority [0, 1, 2] (car > ped1 > ped2):
1. Choose action maximizing r_car
2. If tied, choose action maximizing r_ped1
3. If still tied, choose action maximizing r_ped2

### 6.2 The Lexicographic Max Operation

```python
# LG_utils.py:8-54
def lex_max(q_vectors, priority=[0,1,2], tol=1e-9):
    """
    Lexicographic maximization implementing Eq. (5) from Vamplew et al. (2021).
    """
    n_actions = q_vectors.shape[0]
    best_actions = list(range(n_actions))  # All actions initially candidates

    for obj_idx in priority:  # Process objectives in priority order
        if len(best_actions) == 1:
            break

        # Get objective values for remaining candidates
        obj_values = [q_vectors[action, obj_idx] for action in best_actions]
        max_val = np.max(obj_values)

        # Keep only actions that achieve the max
        new_best_actions = []
        for i, action in enumerate(best_actions):
            if abs(obj_values[i] - max_val) < tol:
                new_best_actions.append(action)

        best_actions = new_best_actions

    return best_actions[0]
```

**Example trace:**

```
q_vectors:        priority = [2, 1, 0]  (ped2 > ped1 > car)
Action 0: [10, 5, 3]
Action 1: [8, 7, 3]
Action 2: [6, 7, 2]

Step 1 (obj 2 - ped2): max=3, candidates={0,1}
Step 2 (obj 1 - ped1): max=7, candidates={1}
Result: Action 1
```

### 6.3 Computing Lexicographic Hull

```python
# LG_utils.py:67-90
def lex_hull(q_vectors, n_objectives=3, tol=1e-9):
    """
    Calculate lexicographic max for ALL possible priority orderings.
    """
    priority_orders = generate_all_priority_orders(n_objectives)
    lex_optimal_actions = {}

    for order in priority_orders:
        best_action = lex_max(q_vectors, priority=order, tol=tol)
        order_tuple = tuple(order)
        lex_optimal_actions[order_tuple] = best_action

    return lex_optimal_actions
```

**Output structure:**
```python
{
    (0, 1, 2): 3,  # Car priority: action 3 is optimal
    (0, 2, 1): 3,  # Car > Ped2 > Ped1: action 3
    (1, 0, 2): 1,  # Ped1 priority: action 1
    (1, 2, 0): 1,
    (2, 0, 1): 5,  # Ped2 priority: action 5
    (2, 1, 0): 5,
}
```

### 6.4 Algorithm Flow

```python
# LG_VI_stoc_lexhull.py:46-164
while True:
    for each state (c, p1, p2):
        q_vectors = np.zeros((n_actions, n_objectives))

        for action in range(n_actions):
            # Handle stochastic outcomes (same as CH_VI_stochastic)
            outcomes = compute_outcomes(state, action)

            q_vector = np.zeros(n_objectives)
            for next_state, reward_vect, done, prob in outcomes:
                if done:
                    q_vector += prob * reward_vect
                else:
                    next_value = V[next_state]
                    q_vector += prob * (reward_vect + discount_factor * next_value)

            q_vectors[action] = q_vector

        # Store Q-vectors (the "hull" of Q-values)
        Q[c, p1, p2] = q_vectors

        # Compute lex_max for ALL priority orderings
        lex_optimal_actions = lex_hull(q_vectors, n_objectives)

        # Use reference priority for convergence
        best_action = lex_optimal_actions[(0, 1, 2)]
        v_new = q_vectors[best_action]

        # Update value function
        V[c, p1, p2] = v_new
```

### 6.5 Extracting All Policies

```python
# LG_VI_stoc_lexhull.py:166-184
all_priority_orders = generate_all_priority_orders(n_objectives)
policies = {}

for priority_order in all_priority_orders:
    priority_tuple = tuple(priority_order)
    policy = np.zeros([n_cells, n_cells, n_cells], dtype=int)

    for c, p1, p2 in all_states:
        lex_optimal_actions = lex_hull(Q[c, p1, p2], n_objectives)
        policy[c, p1, p2] = lex_optimal_actions[priority_tuple]

    policies[priority_tuple] = policy
```

**Result:** 6 policies (3! = 6 permutations of 3 objectives), each optimal for its priority ordering.

### 6.6 Key Difference from CHVI

| Aspect | Convex Hull VI | Lexicographic Hull VI |
|--------|----------------|----------------------|
| **Q-value storage** | Convex hull (multiple vertices) | Single Q-vector per action |
| **Policy extraction** | For any weight vector | For any priority ordering |
| **Flexibility** | Continuous weight space | Discrete orderings (n!) |
| **Computation** | Hull operations | Vector comparisons |

---

## 7. Comparison Summary

### Algorithm Progression

```
┌─────────────────────────────────────────────────────────────────┐
│                    VALUE ITERATION FAMILY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard VI (scalar)                                           │
│       ↓                                                         │
│  CHVI Deterministic ─────────────────→ Lex Hull VI              │
│       ↓                                      ↓                  │
│  CHVI Stochastic ←─────────────────── Lex Hull VI Stochastic    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Comparison

| Feature | convexhull_VI | CH_VI_stochastic | LG_VI_stoc_lexhull |
|---------|---------------|------------------|---------------------|
| **Transitions** | Deterministic | Stochastic | Stochastic |
| **Stores** | Q-hulls | Q-hulls | Q-vectors |
| **Extract for** | Any weights | Any weights | Any priority order |
| **Math operation** | Hull union + translate | Hull sum (Minkowski) | Lex comparison |
| **Memory** | O(|hull vertices|) | O(|hull vertices|) | O(n_objectives) |
| **Policies** | Infinite (continuous w) | Infinite | n_objectives! |

### When to Use What

1. **CHVI Deterministic:** When environment is deterministic and you want to explore weight trade-offs

2. **CHVI Stochastic:** When environment has probabilistic transitions and you need weight flexibility

3. **Lexicographic Hull VI:** When:
   - You prefer strict priority orderings over weighted trade-offs
   - You want interpretable policies ("safety first")
   - You want all priority-ordered policies from one training run

### Computational Complexity

Per iteration:
- **CHVI:** O(|S| * |A| * hull_operations)
- **Lex Hull VI:** O(|S| * |A| * |outcomes| * n_objectives!)

The hull operations in CHVI can be expensive for high-dimensional objective spaces, while Lex Hull VI scales with factorial of objectives.

---

## References

1. **Barrett, L., & Narayanan, S. (2008).** "Learning All Optimal Policies with Multiple Criteria." *ICML 2008*. - Foundation for convex hull operations in MORL.

2. **Vamplew, P., et al. (2021).** "Lexicographic Multi-Objective Reinforcement Learning." - Lexicographic ordering approach (Equation 5).

3. **Roijers, D. M., et al. (2013).** "A Survey of Multi-Objective Sequential Decision-Making." - Comprehensive MORL overview.

---

## Code File Summary

| File | Purpose | Key Functions |
|------|---------|---------------|
| `CH_operations.py` | Core hull operations | `get_hull`, `translate_hull`, `sum_hulls`, `max_q_value` |
| `convexhull_VI.py` | Deterministic CHVI | `convexhull_VI`, `extract_policy_for_weights` |
| `CH_VI_stochastic.py` | Stochastic CHVI | `convexhull_VI` (handles probabilistic outcomes) |
| `LG_VI_stoc_lexhull.py` | Lexicographic approach | `LG_VI_lexhull` (computes all priority orderings) |
| `LG_utils.py` | Lex utilities | `lex_max`, `lex_hull`, `generate_all_priority_orders` |
