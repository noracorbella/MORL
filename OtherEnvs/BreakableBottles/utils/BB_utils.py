import numpy as np

def obs_to_state(obs):
     """
     Convert dict observation into tuple
     """
     d = obs['bottles_dropped']
     return (int(obs['location']), int(obs['bottles_carrying']), int(obs['bottles_delivered']),
        int(d[0] > 0), int(d[1] > 0), int(d[2] > 0),)


def reset_to(env, loc, carrying, delivered, d0, d1, d2):
       """
       Reset the breakable-bottles environment to a specific state
       (modified reset methond in mo_gymnasium/envs/breakable_bottles.py)
       """
       obs, _ = env.reset(location=loc, bottles_carrying=carrying, bottles_delivered=delivered, bottles_dropped=[d0, d1, d2])
       return obs_to_state(obs)



def step_from(env, state, action):
    """
    Execute action from state (row, col, has_gold, has_gem)
    """ 
    loc, carrying, delivered, d0, d1, d2 = state
    reset_to(env, loc, carrying, delivered, d0, d1, d2)
    obs, reward_vect, terminated, truncated, _ = env.step(action)
    return obs_to_state(obs), np.array(reward_vect, dtype=float), bool(terminated or truncated)

def _potential(dropped):
    """potential = -1 if any bottle on ground, else 0"""
    return -1 if any(d > 0 for d in dropped) else 0

def get_outcomes(env, state, action):
    """
    Return all (next_state, reward_vect, done, prob) outcomes for (state, action).
 
    env    : BBEnvironment
    state  : tuple (loc, carrying, delivered, d0, d1, d2)
    action : int (0=left, 1=right, 2=pickup)
 
    outcomes : list of (next_state, reward_vect, done, prob)
    """
    loc, carrying, delivered, d0, d1, d2 = state
    dropped     = [d0, d1, d2]
    size        = env.size
    old_pot     = _potential(dropped)
    is_stochastic = (
        action in (0, 1)
        and 1 <= loc <= size - 2   # middle locations
        and carrying > 1
    )
 
    if is_stochastic:
        return [
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=True,  prob=env.prob_drop, old_pot=old_pot),
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=False, prob=1.0 - env.prob_drop, old_pot=old_pot),
        ]
    else:
        return [
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=False, prob=1.0, old_pot=old_pot)
        ]

def _single_outcome(env, loc, carrying, delivered, dropped, action, drop, prob, old_pot):
    """Compute one deterministic branch of the transition."""
    size         = env.size
    new_dropped  = list(dropped)
    new_carrying = carrying
    new_delivered= delivered
    new_loc      = loc
    bottle_rew   = 0
 
    if action == 0: # LEFT
        if loc > 0:
            if drop:
                new_carrying -= 1
                new_dropped[loc - 1] = min(new_dropped[loc - 1] + 1, 1)
            new_loc = loc - 1
 
    elif action == 1: # RIGHT
        if loc < size - 1:
            if drop:
                new_carrying -= 1
                new_dropped[loc - 1] = min(new_dropped[loc - 1] + 1, 1)
            new_loc = loc + 1
            # deliver if arriving at destination
            if new_loc == size - 1 and new_carrying > 0:
                num_del    = min(new_delivered + new_carrying, 2) - new_delivered
                new_delivered = min(new_delivered + new_carrying, 2)
                new_carrying  = 0
                bottle_rew    = env.bottle_reward * num_del
 
    elif action == 2: # PICKUP
        if loc == 0 and carrying < 2:
            new_carrying += 1
        elif (1 <= loc <= size - 2
              and new_dropped[loc - 1] > 0
              and carrying < 2
              and env.unbreakable_bottles):
            new_dropped[loc - 1] -= 1
            new_carrying += 1
 
    new_pot   = _potential(new_dropped)
    done      = (new_delivered == 2)
    next_state = (new_loc, new_carrying, new_delivered,
                  int(new_dropped[0] > 0),
                  int(new_dropped[1] > 0),
                  int(new_dropped[2] > 0))
 
    reward = np.array([
        env.time_penalty,
        float(bottle_rew),
        float(new_pot - old_pot),
    ])
 
    return (next_state, reward, done, prob)