import numpy as np


def reset_to(env, depth, node):
    """
    Reset the fruit-tree environment to a specific state
    via de modified reset() method.
    """
    obs, _ = env.reset(depth=depth, node=node)
    return obs

def step_from(env, depth, node, action):
       """
       Execute action from state (depth, node)
       """ 
       reset_to(env, depth, node)
       obs, reward_vect, terminated, truncated, info = env.step(action)
       return (int(obs[0]), int(obs[1])), np.array(reward_vect, dtype=float), bool(terminated or truncated)
    
def get_outcomes(env, state, action):
    """
    Return all (next_state, reward_vect, done, prob) outcomes.
    Parameters
    env    : FTEnvironment
    state  : tuple (depth, node)
    action : int (0=left, 1=right)
 
    Returns
    outcomes : list of (next_state, reward_vect, done, prob)
    """

    depth, node = state
    
    next_depth = depth + 1
    next_node = 2 * node + action # left child: 2n, right child: 2n+1

    next_state = (next_depth, next_node)
    done = (next_depth == env.tree_depth)

    # reward is the fruit vector at the next node (zero except at leaves)
    reward_vect = env.get_tree_value(next_depth, next_node)

    return [(next_state, reward_vect, done, 1.0)]