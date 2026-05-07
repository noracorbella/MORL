import numpy as np
import mo_gymnasium as mo_gym


class RGEnvironment:
    """
    Wrapper around resource-gathering-v0 environment from mo_gymnasium.

    State: (row, col, has_gold, has_gem)
    Actions: 0=up, 1=down, 2=left, 3=right
    Rewards: [enemy_penalty, gold_reward, gem_reward]

    Stochastic environment. Stepping onto an enemy cell (E1 or E2)
    kills the agent with probability 0.1.
    """

    def __init__(self, weights):

        self.weights = np.array(weights, dtype=float) if weights is not None else None
        self.gym_env = mo_gym.make("resource-gathering-v0", enemy_kill_prob=0.1)

        self.gym_env.reset()

        inner = self.gym_env.unwrapped
        self.map = inner.map # 5x5 numpy array of cell labels
        self.size = inner.size # 5
        self.n_rows = self.size # 5
        self.n_cols = self.size #5
        self.n_actions = self.gym_env.action_space.n # 4
        self.n_rewards = inner.reward_dim # 3

        self.dir = inner.dir
        self.enemy_kill_prob = inner.enemy_kill_prob

        self.home_pos = tuple(inner.initial_pos.tolist()) # (4, 2)
        self.enemy_cells = {(r, c) for r in range(self.n_rows) for c in range(self.n_cols)
                            if self.map[r][c] in ("E1", "E2")} # {(0,3), (1,2)}

        self.valid_states = [
            (r, c, gold, gem)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            for gold in (0, 1)
            for gem in (0, 1)
        ] # 100 valid states

        self.start_state = (self.home_pos[0], self.home_pos[1], 0, 0) # (4, 2, 0, 0)
    

    def close(self):
        self.gym_env.close()