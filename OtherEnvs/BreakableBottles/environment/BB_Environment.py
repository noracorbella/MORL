import numpy as np
from mo_gymnasium.envs.breakable_bottles.breakable_bottles import BreakableBottles


class BBEnvironment:
    """
    Wrapper around breakable-bottles-v0 environment from mo_gymnasium.

    State: (location, bottles_carrying, bottles_delivered, dropped[0], dropped[1], dropped[2])
    Actions: 0=left, 1=right, 2=pickup
    Rewards: [time_penalty, bottle_reward, potential_change]

    Stochastic environment. when moving at middle locations
    while carrying >1 bottles, there is prob_drop=0.1 chance of dropping a bottle.
    """

    def __init__(self, weights, size=5, prob_drop=0.1, time_penalty=-1, bottle_reward=25, unbreakable_bottles=False):
        """
        Parameters
        weights             
        size                
        prob_drop : float - probability of dropping a bottle 
        time_penalty : int - reward[0] per step 
        bottle_reward : int - reward[1] per bottle delivered 
        unbreakable_bottles : bool - whether dropped bottles can be picked up 
        """

        self.weights = np.array(weights, dtype=float) if weights is not None else None
        self.size = size
        self.prob_drop = prob_drop
        self.time_penalty = time_penalty
        self.bottle_reward = bottle_reward
        self.unbreakable_bottles = unbreakable_bottles

        self.gym_env = BreakableBottles(prob_drop=prob_drop, time_penalty=time_penalty, 
                                        bottle_reward=bottle_reward, unbreakable_bottles=unbreakable_bottles)
        
        self.gym_env.reset()

        self.n_actions  = self.gym_env.action_space.n   # 3
        self.n_rewards  = self.gym_env.reward_dim       # 3
        self.n_middle   = size - 2                      

        self.start_state = (size - 1, 0, 0, 0, 0, 0)

        self.valid_states = [
            (loc, carrying, delivered, d0, d1, d2)
            for loc in range(size)
            for carrying in range(3)
            for delivered in range(3)
            for d0 in (0, 1)
            for d1 in (0, 1)
            for d2 in (0, 1)
        ] # 360 

        # Non-terminal states (delivered < 2)
        self.non_terminal_states = [s for s in self.valid_states if s[2] < 2] 

    def close(self):
        self.gym_env.close()