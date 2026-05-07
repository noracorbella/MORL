import numpy as np


def navigate_to_dstm(env, target_row, target_col):
    """
    Run original env.reset() from the DST environment (which resets to (0, 0))
    and navigate to (target_row, target_col) by going RIGHT until the correct column and 
    DOWN unitl the correct row.
    """
    obs, _ = env.reset()
    start_row, start_col = int(obs[0]), int(obs[1])

    # move horizontally from start_col to target_col
    if target_col > start_col:
         for col in range(target_col - start_col):
            obs, reward, terminated, truncated, info = env.step(3) # action 3 corresponds to going right
            if terminated or truncated:
                raise RuntimeError(f"Episode ended unexpectedly while navigating to ({target_row}, {target_col})")
    elif target_col < start_col:
        for col in range(start_col - target_col):  
            obs, reward, terminated, truncated, info = env.step(2)
            if terminated or truncated:
                raise RuntimeError(f"Episode ended unexpectedly while navigating to ({target_row}, {target_col})")

    # move down from start_row to target_row
    for row in range(target_row - start_row):      
        obs, reward, terminated, truncated, info = env.step(1)
        if terminated or truncated:
            raise RuntimeError(f"Episode ended unexpectedly while navigating to ({target_row}, {target_col})")

def step_from(env, row, col, action):
       """
       Navigate to (row, col) and execute action from there.

       :return: next_state (tuble), reward_vect (np.array), done (bool)
       """ 
       navigate_to_dstm(env, row, col)
       obs, reward_vect, terminated, truncated, info = env.step(action)
       return (int(obs[0]), int(obs[1])), np.array(reward_vect, dtype=float), bool(terminated or truncated)
    
