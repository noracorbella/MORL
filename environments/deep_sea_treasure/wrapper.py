"""
MOEnv wrapper for the Deep Sea Treasure environments (standard, concave, mirrored).

This adapts mo-gymnasium's Deep Sea Treasure environments to the :class:`MOEnv`
interface. Deep Sea Treasure (DST) is a deterministic grid MOMDP with two
objectives, ``[treasure_value, time_penalty]``. The submarine starts at a fixed
cell and moves up/down/left/right reaching a treasure cell ends the episode.

The same wrapper serves all three DST variants, which differ only in their sea
map (and, for the mirrored variant, their start cell). Which one is used is 
chosen by the ``env_id`` argument. Everything else (state set, terminal
rule, dynamics) is derived from the map.

The transition source is a``step_from`` trick: to find the outcome of a 
``(state, action)`` pair we reset the underlying gym environment, physically 
navigate to the state, and step the action. Navigation reads the start cell 
from ``reset()`` and moves horizontally (left or right) to the target column 
before descending, so it handles a start that is not ``(0, 0)``. Every outcome is 
cached on first access, so :meth:`transitions` is cheap to call repeatedly.
"""

import numpy as np
import mo_gymnasium as mo_gym

from morl.core.env_interface import MOEnv


# Default discount factor
DEFAULT_GAMMA = 0.99

# Default environment id
DEFAULT_ENV_ID = "deep-sea-treasure-v0"

# Value of sea-floor (invalid, unreachable) cells in the DST map.
SEA_FLOOR = -10


def _navigate_to(gym_env, target_row, target_col):
    """
    Reset ``gym_env`` and drive it to ``(target_row, target_col)``.

    Reads the start cell from ``reset()`` and moves horizontally to the target
    column (RIGHT/LEFT), then DOWN to the target row.
    """
    obs, _ = gym_env.reset()
    start_row, start_col = int(obs[0]), int(obs[1])

    if target_col > start_col:
        steps, move = target_col - start_col, 3  # right
    else:
        steps, move = start_col - target_col, 2  # left
    for _ in range(steps):
        obs, reward, terminated, truncated, info = gym_env.step(move)
        if terminated or truncated:
            raise RuntimeError(
                f"Episode ended unexpectedly while navigating to "
                f"({target_row}, {target_col})"
            )

    for _ in range(target_row - start_row):
        obs, reward, terminated, truncated, info = gym_env.step(1)  # down
        if terminated or truncated:
            raise RuntimeError(
                f"Episode ended unexpectedly while navigating to "
                f"({target_row}, {target_col})"
            )


def _step_from(gym_env, row, col, action):
    """
    Navigate to ``(row, col)`` and execute ``action`` from there.

    Returns ``(next_state, reward_vector, done)``.
    """
    _navigate_to(gym_env, row, col)
    obs, reward_vect, terminated, truncated, info = gym_env.step(action)
    return (
        (int(obs[0]), int(obs[1])),
        np.array(reward_vect, dtype=float),
        bool(terminated or truncated),
    )


class DeepSeaTreasureEnv(MOEnv):
    """
    Deep Sea Treasure as an :class:`MOEnv`.

    States are ``(row, col)`` integer tuples.
    Terminal states are the treasure cells (map value ``> 0``); the
    sea-floor cells (map value ``-10``) are excluded from the state set entirely.
    Actions are the integers ``0..3`` (up/down/left/right) on non terminal
    states and none on terminal states.
    """

    def __init__(self, gamma=DEFAULT_GAMMA, env_id=DEFAULT_ENV_ID):
        self.env_id = env_id  
        self.gym_env = mo_gym.make(env_id)
        obs, _ = self.gym_env.reset()

        self.sea_map = np.array(self.gym_env.unwrapped.sea_map)
        self.n_rows, self.n_cols = self.sea_map.shape
        self.n_actions = self.gym_env.action_space.n

        self.n_objectives = int(self.gym_env.unwrapped.reward_dim)
        self.gamma = float(gamma)

        # Valid cells are everything that is not sea floor.
        self._states = [
            (r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.sea_map[r, c] != SEA_FLOOR
        ]

        # Treasure cells (map value > 0) are terminal.
        self._terminal_states = {
            (r, c) for (r, c) in self._states if self.sea_map[r, c] > 0
        }

        # The start cell is read from reset (it is not (0, 0) for the mirrored map).
        self.start_state = (int(obs[0]), int(obs[1]))

        # Cache of transition distributions to keep transitions() cheap on repeated calls.
        self._transition_cache = {}

    def states(self):
        """
        Return all valid states (water and treasure cells), as a list of
        ``(row, col)`` tuples.
        """
        return list(self._states)

    def actions(self, state):
        """
        Return ``[0, 1, 2, 3]`` on non-terminal states and ``[]`` on
        terminal (treasure) states.
        """
        if self.is_terminal(state):
            return []
        return list(range(self.n_actions))

    def transitions(self, state, action):
        """
        Return the transition distribution for ``action`` in ``state`` 
        as the single-element list ``[(1.0, next_state, reward_vector)]``.

        The outcome is computed once via the ``step_from`` and cached,
        so repeated calls are cheap.
        """
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached

        row, col = state
        next_state, reward_vect, _done = _step_from(self.gym_env, row, col, action)
        outcomes = [(1.0, next_state, reward_vect)]
        self._transition_cache[key] = outcomes
        return outcomes

    def is_terminal(self, state):
        """
        Return ``True`` for treasure cells and ``False`` otherwise.
        """
        return state in self._terminal_states

    def close(self):
        """
        Close the gym environment.
        """
        self.gym_env.close()

    # -- rendering (NOT part of the MOEnv contract) -------------

    def render_policy(self, policy, n_episodes=2, max_steps=50, pause=0.4):
        """
        Visualise ``policy`` in the Deep Sea Treasure pygame window.

        It opens a ``render_mode="human"`` gym environment, then from the start 
        repeatedly takes ``policy[(row, col)]``, steps, and renders each frame until the
        episode terminates or ``max_steps`` is reached. ``policy`` is the unified dict 
        (``(row, col) -> action``).
        """
        import time

        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        render_env = mo_gym.make(self.env_id, render_mode="human")
        try:
            for episode in range(n_episodes):
                obs, _ = render_env.reset()
                render_env.render()
                state = (int(obs[0]), int(obs[1]))
                print(f"Episode {episode + 1} | start {state}")
                done = False
                step = 0
                while step < max_steps and not done:
                    step += 1
                    action = policy[state]
                    obs, reward, terminated, truncated, _ = render_env.step(action)
                    render_env.render()
                    time.sleep(pause)
                    done = terminated or truncated
                    state = (int(obs[0]), int(obs[1]))

                    print(f" t={step:2d} | state={state} | "
                          f"action={action_names[action]} | reward={reward}")
                print(f"  ended at {state} after {step} steps")
        finally:
            render_env.close()
