import numpy as np


def navigate_to(env, target_row, target_col):
    """
    Run original env.reset() from the DSTc environment (which resets to (0, 0))
    and navigate to (target_row, target_col) by going RIGHT until the correct column and 
    DOWN unitl the correct row.
    """
    env.reset()

    for col in range(target_col):
        obs, reward, terminated, truncated, info = env.step(3) # action 3 corresponds to going right
        if terminated or truncated:
            raise RuntimeError(f"Episode ended unexpectedly while navigating to ({target_row}, {target_col})")
        
    for row in range(target_row):
        obs, reward, terminated, truncated, info = env.step(1) # action 1 corresponds to going down
        if terminated or truncated:
            raise RuntimeError(f"Episode ended unexpectedly while navigating to ({target_row}, {target_col})")
        

def step_from(env, row, col, action):
       """
       Navigate to (row, col) and execute action from there.

       :return: next_state (tuble), reward_vect (np.array), done (bool)
       """ 
       navigate_to(env, row, col)
       obs, reward_vect, terminated, truncated, info = env.step(action)
       return (int(obs[0]), int(obs[1])), np.array(reward_vect, dtype=float), bool(terminated or truncated)
    
