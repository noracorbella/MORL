import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RandomEnv(gym.Env):
    """
    Fully random Multi-Objective Markov Decision Process
    compatible with MO-Gymnasium.
    """

    metadata = {"render_modes": []}

    def __init__(
            self,

            states: int = -1,
            actions: int = -1,
            objectives: int = -1,
            successors: int = -1,
            actions_per_state: int = -1,
            terminal_states: int = -1,
            initial_states: int = -1,
            min_states: int = -1,
            max_states: int = -1,
            min_actions: int = -1,
            max_actions: int = -1,
            min_objectives: int = -1,
            max_objectives: int = -1,
            min_reward: float = -1000,
            max_reward: float = 1000,
            min_successors: int = -1,
            max_successors: int = -1,
            min_actions_per_state: int = -1,
            max_actions_per_state: int = -1,
            min_terminal_states: int = -1,
            max_terminal_states: int = -1,
            min_initial_states: int = -1,
            max_initial_states: int = -1,
            seed: int | None = None,
    ):
        super().__init__()

        # ----------------------------
        # Seeding
        # ----------------------------
        self.rng = np.random.default_rng(seed)

        if states > -1:
            min_states = max_states = states
        if actions > -1:
            min_actions = max_actions = actions
        if objectives > -1:
            min_objectives = max_objectives = objectives
        if successors > -1:
            min_successors = max_successors = successors
        if actions_per_state > -1:
            min_actions_per_state = max_actions_per_state = actions_per_state
        if terminal_states > -1:
            min_terminal_states = max_terminal_states = terminal_states
        if initial_states > -1:
            min_initial_states = max_initial_states = initial_states

        # ----------------------------
        # Resolve unspecified ranges
        # ----------------------------
        def resolve_range(min_val, max_val, low, high):
            if min_val == -1:
                min_val = self.rng.integers(low, high)
            if max_val == -1:
                max_val = self.rng.integers(min_val, high + 1)
            return min_val, max_val

        min_states, max_states = resolve_range(min_states, max_states, 5, 50)
        min_actions, max_actions = resolve_range(min_actions, max_actions, 2, 20)
        min_objectives, max_objectives = resolve_range(min_objectives, max_objectives, 2, 10)

        # ----------------------------
        # Sample global dimensions
        # ----------------------------
        self.num_states = self.rng.integers(min_states, max_states + 1)
        self.num_actions = self.rng.integers(min_actions, max_actions + 1)
        self.num_objectives = self.rng.integers(min_objectives, max_objectives + 1)

        min_successors, max_successors = resolve_range(
            min_successors, max_successors, 1, self.num_states
        )

        min_actions_per_state, max_actions_per_state = resolve_range(
            min_actions_per_state, max_actions_per_state, 1, self.num_actions
        )

        self.min_reward = min_reward
        self.max_reward = max_reward

        min_terminal_states, max_terminal_states = resolve_range(
            min_terminal_states, max_terminal_states, 0, self.num_states // 2
        )
        min_initial_states, max_initial_states = resolve_range(
            min_initial_states, max_initial_states, 1, self.num_states // 2
        )

        # ----------------------------
        # Action availability mask
        # ----------------------------
        self.action_mask = np.zeros((self.num_states, self.num_actions), dtype=bool)

        for s in range(self.num_states):
            k = self.rng.integers(
                min_actions_per_state,
                min(max_actions_per_state, self.num_actions) + 1,
            )
            actions = self.rng.choice(self.num_actions, size=k, replace=False)
            self.action_mask[s, actions] = True

        # ----------------------------
        # Transition tensor T(s,a,s')
        # ----------------------------
        self.T = np.zeros(
            (self.num_states, self.num_actions, self.num_states),
            dtype=np.float64,
        )

        for s in range(self.num_states):
            for a in range(self.num_actions):
                if not self.action_mask[s, a]:
                    continue

                k = self.rng.integers(
                    min_successors,
                    min(max_successors, self.num_states) + 1,
                )
                successors = self.rng.choice(
                    self.num_states, size=k, replace=False
                )
                probs = self.rng.random(k)
                probs /= probs.sum()

                self.T[s, a, successors] = probs

        # ----------------------------
        # Reward tensor R^i(s,a,s')
        # Shape: (objectives, states, actions, states)
        # ----------------------------
        self.R = self.rng.uniform(
            low=self.min_reward,
            high=self.max_reward,
            size=(
                self.num_objectives,
                self.num_states,
                self.num_actions,
                self.num_states,
            ),
        )

        # Mask rewards for invalid transitions
        self.R *= self.T[None, :, :, :]

        # ----------------------------
        # Terminal states
        # ----------------------------
        num_terminal = self.rng.integers(
            min_terminal_states,
            min(max_terminal_states, self.num_states) + 1,
        )
        self.terminal_states = set(
            self.rng.choice(self.num_states, size=num_terminal, replace=False)
        )

        # ----------------------------
        # Initial states
        # ----------------------------
        num_initial = self.rng.integers(
            min_initial_states,
            min(max_initial_states, self.num_states) + 1,
        )
        self.initial_states = self.rng.choice(
            self.num_states, size=num_initial, replace=False
        )

        # ----------------------------
        # Spaces
        # ----------------------------
        self.observation_space = spaces.Discrete(self.num_states)
        self.action_space = spaces.Discrete(self.num_actions)
        self.reward_space = spaces.Box(
            low=self.min_reward,
            high=self.max_reward,
            shape=(self.num_objectives,),
            dtype=np.float64,
        )

        self.state = None

    # ======================================================
    # Core Gymnasium API
    # ======================================================

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.state = int(self.rng.choice(self.initial_states))
        return self.state, {}

    def step(self, action: int):
        if not self.action_mask[self.state, action]:
            raise ValueError(
                f"Action {action} is not available in state {self.state}"
            )

        probs = self.T[self.state, action]
        next_state = int(self.rng.choice(self.num_states, p=probs))

        # Multi-objective reward vector
        reward = self.R[:, self.state, action, next_state]

        terminated = next_state in self.terminal_states
        truncated = False

        self.state = next_state

        return next_state, reward, terminated, truncated, {}

    def render(self):
        """
        Minimal textual render function for debugging and analysis.
        Prints the current state and basic environment information.
        """
        print("========================================")
        print(f"Current state      : {self.state}")

        if self.state is None:
            print("Environment not yet reset.")
            return

        print(f"Terminal state     : {self.state in self.terminal_states}")

        valid_actions = np.flatnonzero(self.action_mask[self.state])
        print(f"Valid actions      : {valid_actions.tolist()}")

        if self.state in self.terminal_states:
            print("Episode has terminated.")

    # ======================================================
    # Optional helpers
    # ======================================================
    def sample_action(self, state: int) -> int:
        """
        Samples a valid action uniformly at random for a given state,
        respecting the state-dependent action mask.
        """
        valid_actions = np.flatnonzero(self.action_mask[state])

        if valid_actions.size == 0:
            raise RuntimeError(
                f"No valid actions available for state {state}"
            )

        return int(self.rng.choice(valid_actions))

    def get_transition_tensor(self):
        return self.T.copy()

    def get_reward_tensor(self):
        return self.R.copy()

    def get_action_mask(self):
        return self.action_mask.copy()


if __name__ == "__main__":

    env = RandomEnv(seed=45, states=100, actions=5, actions_per_state=5, successors=1, objectives=3, terminal_states=1)

    print(env.get_transition_tensor()[0, 1])

    print("The generated environment has : ", env.observation_space.n, " states, ",
          env.action_space.n, " actions, ",
          env.reward_space.shape[0], " objectives",
          len(env.initial_states), " initial states",
          len(env.terminal_states), " terminal states.")

    done = False
    obs, information = env.reset()

    max_steps = 100

    for _ in range(max_steps):
        obs, r, done, trunked, information = env.step(env.sample_action(obs))
        print("Next state : ", obs, ". Reward obtained : ", r)
        env.render()
        if done:
            break
