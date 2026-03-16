import numpy as np
import mo_gymnasium as mo_gym


class DSTEnvironment:
    """
    Wrapper around DeepSeaTreasure-v0 environment from mo_gymnasium.

    State: (row, col)
    Actions: 0=up, 1=down, 2=left, 3=right
    Rewards: [treasure_value, time_penalty]
    """

    ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}

    def __init__(self, weights):

        self.weights = np.array(weights, dtype=float)
        self.gym_env = mo_gym.make("deep-sea-treasure-v0")

        self.gym_env.reset()

        self.sea_map = np.array(self.gym_env.unwrapped.sea_map)
        self.n_rows = self.sea_map.shape[0]
        self.n_cols = self.sea_map.shape[1]
        self.n_actions = self.gym_env.action_space.n
        self.n_rewards = self.gym_env.unwrapped.reward_dim

        self.valid_states = [
            (r, c) for r in range(self.n_rows) for c in range(self.n_cols)
            if self.sea_map[r, c] != -10
        ]

        self.treasure_states = {
            (r, c): self.sea_map[r, c] for r, c in self.valid_states
            if self.sea_map[r, c] > 0
        }

        self.non_terminal_states = [s for s in self.valid_states if s not in self.treasure_states]

        self.start_state = (0, 0)
    

    def close(self):
        self.gym_env.close()