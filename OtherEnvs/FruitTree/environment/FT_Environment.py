import numpy as np
import mo_gymnasium as mo_gym


class FTEnvironment:
    """
    Wrapper around fruit-tree-v0 environment from mo_gymnasium.

    State: (depth, node)
    Actions: 0=left, 1=right
    Rewards: [protein, carbs, fats, vitamins, minerals, water]
             Non-zero only at leaf nodes (depth == tree_depth).

    Fully deterministic
    Configurable tree depth (5, 6, 7)
    """

    def __init__(self, weights, tree_depth=5):

        self.weights = np.array(weights, dtype=float) if weights is not None else None
        self.tree_depth = tree_depth
        self.gym_env    = mo_gym.make("fruit-tree-v0", depth=tree_depth)
        self.gym_env.reset()

        self.n_actions  = self.gym_env.action_space.n   # 2
        self.n_rewards  = self.gym_env.unwrapped.reward_dim  # 6

        self.tree = self.gym_env.unwrapped.tree  # shape (2^(d+1)-1, 6)

        self.valid_states = [
            (d, n)
            for d in range(tree_depth + 1)
            for n in range(2 ** d)
        ] 

        # terminal
        self.leaf_states = {
            (tree_depth, n) for n in range(2 ** tree_depth)
        }

        self.non_terminal_states = [
            s for s in self.valid_states if s not in self.leaf_states
        ]

        self.start_state = (0, 0)
    
    def get_tree_value(self, depth, node):
        """Return the 6D fruit vector at (depth, node). Zero for non-leaves."""
        idx = int(2 ** depth - 1) + node
        return np.array(self.tree[idx], dtype=float)

    def close(self):
        self.gym_env.close()