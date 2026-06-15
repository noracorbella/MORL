import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, '..'))

from Universal import RandomEnv


class RandMOMDPEnvironment:
    """
    Wrapper around the RandomEnv (Universal.py) fully-random Multi-Objective MDP.

    State: integer s in {0, ..., S-1}
    Actions: integer a in {0, ..., A-1} (state-dependent availability via action_mask)
    Rewards: vector of length K, R(k, s, a, s') on transitions

    Stochastic environment: T(s, a, s') gives the successor distribution.
    A transition is terminal when its successor state is in env.terminal_states.

    This mirrors the RGEnvironment interface so the CHVI / LGVI algorithms can be
    reused with only the state encoding changed (integer states instead of grid
    tuples).
    """

    def __init__(self, weights, seed=45, states=100, actions=5,
                 actions_per_state=5, successors=1, objectives=3,
                 terminal_states=1):

        self.weights = np.array(weights, dtype=float) if weights is not None else None

        # Build the underlying random MOMDP. The construction is fully
        # determined by the seed, so every algorithm sees the same environment.
        self.gym_env = RandomEnv(
            seed=seed,
            states=states,
            actions=actions,
            actions_per_state=actions_per_state,
            successors=successors,
            objectives=objectives,
            terminal_states=terminal_states,
        )

        inner = self.gym_env

        self.n_states  = inner.num_states       # S
        self.n_actions = inner.num_actions       # A
        self.n_rewards = inner.num_objectives    # K

        # Tensors describing the dynamics / rewards.
        self.T           = inner.get_transition_tensor()  # (S, A, S)
        self.R           = inner.get_reward_tensor()      # (K, S, A, S)
        self.action_mask = inner.get_action_mask()        # (S, A)

        self.terminal_states = set(int(s) for s in inner.terminal_states)
        self.initial_states  = [int(s) for s in inner.initial_states]

        # States are just the integers 0..S-1. Terminal states are still valid
        # states (the algorithms simply never expand past a terminal transition).
        self.valid_states = list(range(self.n_states))

        # Deterministic start state for hull reporting / single-trajectory tests.
        self.start_state = self.initial_states[0]

    def valid_actions(self, state):
        """Return the list of available action indices for a state."""
        return np.flatnonzero(self.action_mask[state]).tolist()

    def close(self):
        pass


if __name__ == "__main__":
    env = RandMOMDPEnvironment(weights=[1, 1, 1])
    print(f"States: {env.n_states}, Actions: {env.n_actions}, Objectives: {env.n_rewards}")
    print(f"Initial states: {env.initial_states}")
    print(f"Terminal states: {sorted(env.terminal_states)}")
    print(f"Start state: {env.start_state}")
    print(f"Valid actions in start state: {env.valid_actions(env.start_state)}")
