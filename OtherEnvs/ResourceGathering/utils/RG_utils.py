import numpy as np


def reset_to(env, row, col, has_gold, has_gem):
       """
       Reset the resource-gathering environment to a specific state
       (modified reset methond in mo_gymnasium/envs/resource_gathering.py)
       """
       obs, _ = env.reset(row=row, col=col, has_gold=has_gold, has_gem=has_gem)
       return obs



def step_from(env, row, col, has_gold, has_gem, action):
       """
       Execute action from state (row, col, has_gold, has_gem)
       """ 
       reset_to(env, row, col, has_gold, has_gem)
       obs, reward_vect, terminated, truncated, info = env.step(action)
       next_state = tuple(int(x) for x in obs)
       return next_state, np.array(reward_vect, dtype=float), bool(terminated or truncated)

def get_outcomes(env, state, action):
    """
    Return all (next_state, reward_vect, done, prob) outcomes for (state, action)

    param: env    : Environment
    param: state  : tuple (row, col, has_gold, has_gem)
    param: action : int

    return: outcomes : list of (next_state, reward_vect, done, prob)
    """
    row, col, has_gold, has_gem = state

    # Compute next position
    dr, dc   = env.dir[action]
    next_row = row + dr
    next_col = col + dc

    # Boundary: if invalid, agent stays
    if not (0 <= next_row < env.n_rows and 0 <= next_col < env.n_cols):
        next_row, next_col = row, col

    cell = env.map[next_row][next_col]

    # Resource collection happens on entry
    new_has_gold = 1 if (cell == "R1" or has_gold == 1) else 0
    new_has_gem  = 1 if (cell == "R2" or has_gem  == 1) else 0
 
    next_state = (next_row, next_col, new_has_gold, new_has_gem)

    if cell in ("E1", "E2"):
        p = env.enemy_kill_prob 
        return [
            (next_state, np.array([-1., 0., 0.]), True,  p),
            (next_state, np.array([ 0., 0., 0.]), False, 1.0 - p),
        ]

    elif cell == "H":
        reward = np.array([0., float(new_has_gold), float(new_has_gem)])
        return [(next_state, reward, True, 1.0)]

    else:
        # Deterministic non-terminal (empty cell, R1, R2)
        return [(next_state, np.zeros(3), False, 1.0)]
