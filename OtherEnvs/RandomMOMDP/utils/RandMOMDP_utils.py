import numpy as np


def get_outcomes(env, state, action):
    """
    Return all (next_state, reward_vect, done, prob) outcomes for (state, action)

    param: env    : Environment
    param: state  : int, the current state
    param: action : int

    return: outcomes : list of (next_state, reward_vect, done, prob)
    """
    # Unavailable action -> no outcomes. The action mask must be respected
    if not env.action_mask[state, action]:
        return []

    probs = env.T[state, action]                      # (S,)
    successors = np.flatnonzero(probs > 0.0)

    outcomes = []
    for next_state in successors:
        next_state = int(next_state)
        prob = float(probs[next_state])
        reward_vect = env.R[:, state, action, next_state].astype(float)  # (K,)
        done = next_state in env.terminal_states
        outcomes.append((next_state, reward_vect, done, prob))

    return outcomes
