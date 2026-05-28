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
    loc, carrying, delivered, d0, d1, d2 = state  # unpack state tuple (6 components)
    dropped = [d0, d1, d2] # drop flags (segment i lies between locations i and i+1)
    size = env.size 
    old_pot = _potential(dropped) # potential before the step: 0 if ground is clear, -1 if any bottle is dropped
    
    # a drop can only happen when: 
    # 1. the agent is moving (not picking up)
    # 2. through a middle location (not the source or destination endpoints)
    # 3. while holding more than one bottle
    is_stochastic = (action in (0, 1) and 1 <= loc <= size - 2 and carrying > 1)

    if is_stochastic: # two possible outcomes: one bottle drops or it doesn't                              
        return [
            # one bottle falls (prob = prob_drop)
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=True,  prob=env.prob_drop, old_pot=old_pot),  
            # no drop (prob = 1 - prob_drop)
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=False, prob=1.0 - env.prob_drop, old_pot=old_pot),  
        ]
    else:  # deterministic: single outcome with probability 1                                         
        return [
            _single_outcome(env, loc, carrying, delivered, dropped,
                            action, drop=False, prob=1.0, old_pot=old_pot)
        ]

def _single_outcome(env, loc, carrying, delivered, dropped, action, drop, prob, old_pot):
    """Compute one deterministic branch of the transition."""
    size          = env.size 
    new_dropped   = list(dropped)    
    new_carrying  = carrying        
    new_delivered = delivered       
    new_loc       = loc        
    bottle_rew    = 0 # delivery reward; non-zero only when bottles are handed in at the destination

    if action == 0: # LEFT
        if loc > 0: # can only move left if not already at the leftmost location                                                      
            if drop:
                new_carrying -= 1 # agent loses one bottle                                   
                new_dropped[loc - 1] = min(new_dropped[loc - 1] + 1, 1) # mark segment loc-1 as occupied
            new_loc = loc - 1 # move to the adjacent location on the left

    elif action == 1: # RIGHT
        if loc < size - 1: # can only move right if not already at the rightmost location
            if drop:
                new_carrying -= 1 # agent loses one bottle
                new_dropped[loc - 1] = min(new_dropped[loc - 1] + 1, 1) # mark the segment just left as occupied
            new_loc = loc + 1 # move to the adjacent location on the right
            # deliver if arriving at destination
            if new_loc == size - 1 and new_carrying > 0: # stepped into the destination while holding bottles
                num_del       = min(new_delivered + new_carrying, 2) - new_delivered  # how many bottles count as newly delivered (max 2)
                new_delivered = min(new_delivered + new_carrying, 2) # update cumulative delivered count (max 2)
                new_carrying  = 0 # agent drops all bottles at the destination
                bottle_rew    = env.bottle_reward * num_del # reward scaled by how many bottles were just handed in

    elif action == 2: # PICKUP
        if loc == 0 and carrying < 2: # at the source and not at carrying capacity: collect a fresh bottle
            new_carrying += 1
        
        # middle location where
        # 1. bottle is lying on the ground
        # 2. agent has room to carry it
        # 3. mode allows to recover dropped bottles
        elif (1 <= loc <= size - 2 and new_dropped[loc - 1] > 0 and carrying < 2 and env.unbreakable_bottles):                              
            new_dropped[loc - 1] -= 1 # remove the bottle from the ground
            new_carrying += 1 # add it to the agent's load

    new_pot    = _potential(new_dropped) # potential after the step: 0 if ground clear, -1 if any bottle remains
    done       = (new_delivered == 2) # episode ends when both bottles have been delivered
    next_state = (new_loc, new_carrying, new_delivered,
                  int(new_dropped[0] > 0), # segment 0: 1 if a bottle was dropped there
                  int(new_dropped[1] > 0), # segment 1: 1 if a bottle was dropped there
                  int(new_dropped[2] > 0)) # segment 2: 1 if a bottle was dropped there

    reward = np.array([
        env.time_penalty, 
        float(bottle_rew),
        float(new_pot - old_pot),
    ])

    return (next_state, reward, done, prob)