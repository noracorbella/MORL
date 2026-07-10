"""
MOEnv wrapper for the Resource Gathering environment.

This adapts mo-gymnasium's ``resource-gathering-v0`` to the :class:`MOEnv`
interface. Resource Gathering (RG) is a *stochastic* grid MOMDP with three
objectives, ``[enemy_penalty, gold_reward, gem_reward]``. The agent starts at
its home cell, moves up/down/left/right collecting gold (``R1``) and a gem
(``R2``), and banks whatever it carries by returning home. Stepping onto an enemy
cell (``E1``/``E2``) kills the agent with probability ``enemy_kill_prob`` (0.1),
ending the episode with an ``enemy_penalty`` of ``-1``.

Transition source
-----------------
The transition dynamics are computed *analytically*: from a cell we look up the 
neighbour a move leads to, apply boundary clipping, resource pickup on entry, and 
the terminal/stochastic rules for home and enemy cells. No gym stepping 
is used, so the transitions are cheap.. Each ``(state, action)`` distribution is 
additionally cached.

Terminality
--------------------------
RG's original model has no terminal state, because termination is a property
of a *transition* (returning home, or being killed). The :class:`MOEnv` contract
instead expresses terminality through ``is_terminal(next_state)``. Therefore, we create
a single synthetic absorbing terminal state, :data:`TERMINAL`: every
terminating outcome (entering home, or an enemy kill) is redirected to
``TERMINAL`` with its reward carried on the transition. ``TERMINAL`` has value
zero and is never updated in Bellmann update, which reproduces the ``done`` semantics. 
In particular the home coordinates are **not** terminal, so the start
state ``(home_row, home_col, 0, 0)`` remains an ordinary acting state.
"""

import numpy as np
import mo_gymnasium as mo_gym

from morl.core.env_interface import MOEnv


# Discount factor and objective count from RG's own configuration
DEFAULT_GAMMA = 0.7

# Probability that stepping onto an enemy cell kills the agent. 
ENEMY_KILL_PROB = 0.1

TERMINAL = "TERMINAL"


class ResourceGatheringEnv(MOEnv):
    """
    Resource Gathering as an :class:`MOEnv`.

    States are ``(row, col, has_gold, has_gem)`` integer tuples (all
    ``5 x 5 x 2 x 2 = 100`` of them) plus the sentinel :data:`TERMINAL`. Only
    ``TERMINAL`` is terminal; every ordinary state has actions ``0..3``
    (up/down/left/right). Transitions are stochastic only when a move steps onto
    an enemy cell.
    """

    def __init__(self, gamma=DEFAULT_GAMMA, enemy_kill_prob=ENEMY_KILL_PROB):
        self.gym_env = mo_gym.make("resource-gathering-v0")
        self.gym_env.reset()

        inner = self.gym_env.unwrapped
        self.map = inner.map            # 5x5 array of cell labels
        self.size = inner.size          # 5
        self.n_rows = self.size
        self.n_cols = self.size
        self.n_actions = self.gym_env.action_space.n  # 4
        self.dir = inner.dir            # action -> (dr, dc)
        self.enemy_kill_prob = enemy_kill_prob  # 0.1 (RG config constant)
        self.home_pos = tuple(inner.initial_pos.tolist())  # (4, 2)

        self.n_objectives = int(inner.reward_dim)  # 3
        self.gamma = float(gamma)

        # The 100 ordinary states, plus the terminal sentinel.
        self._ordinary_states = [
            (r, c, gold, gem)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            for gold in (0, 1)
            for gem in (0, 1)
        ]

        self.start_state = (self.home_pos[0], self.home_pos[1], 0, 0)  # (4, 2, 0, 0)

        # Cache of transition distributions, keyed by (state, action).
        self._transition_cache = {}

    def states(self):
        """
        Return the 100 ordinary ``(row, col, has_gold, has_gem)`` states plus
        the :data:`TERMINAL` sentinel.
        """
        return self._ordinary_states + [TERMINAL]

    def actions(self, state):
        """
        Return ``[0, 1, 2, 3]`` on ordinary states and ``[]`` on
        :data:`TERMINAL`.
        """
        if state == TERMINAL:
            return []
        return list(range(self.n_actions))

    def transitions(self, state, action):
        """
        Return the transition distribution for ``action`` in ``state``.

        Ordinary outcomes are ``(prob, next_state, reward_vector)``; any
        terminating outcome (entering home, or an enemy kill) has ``next_state``
        set to :data:`TERMINAL` with its reward on the transition. Stepping onto
        an enemy cell yields two outcomes (kill / survive); every other move is
        deterministic. Results are cached per ``(state, action)``.
        """
        key = (state, action)
        cached = self._transition_cache.get(key)
        if cached is not None:
            return cached

        row, col, has_gold, has_gem = state

        # Where the move leads, but agent is forced to stay if it would
        # otherwise leave the grid.
        dr, dc = self.dir[action]
        next_row, next_col = row + dr, col + dc
        if not (0 <= next_row < self.n_rows and 0 <= next_col < self.n_cols):
            next_row, next_col = row, col

        cell = self.map[next_row][next_col]

        # Resources are collected on entry.
        new_has_gold = 1 if (cell == "R1" or has_gold == 1) else 0
        new_has_gem = 1 if (cell == "R2" or has_gem == 1) else 0
        next_state = (next_row, next_col, new_has_gold, new_has_gem)

        if cell in ("E1", "E2"):
            # Stochastic: killed (terminal) with prob p, else survive and continue.
            p = self.enemy_kill_prob
            outcomes = [
                (p, TERMINAL, np.array([-1.0, 0.0, 0.0])),
                (1.0 - p, next_state, np.array([0.0, 0.0, 0.0])),
            ]
        elif cell == "H":
            # Returning home banks the carried resources and terminates.
            reward = np.array([0.0, float(new_has_gold), float(new_has_gem)])
            outcomes = [(1.0, TERMINAL, reward)]
        else:
            # Deterministic non-terminal move (empty cell, R1, R2).
            outcomes = [(1.0, next_state, np.zeros(3))]

        self._transition_cache[key] = outcomes
        return outcomes

    def is_terminal(self, state):
        """
        Return ``True`` only for the :data:`TERMINAL` sentinel.
        """
        return state == TERMINAL

    def close(self):
        self.gym_env.close()

    # -- rendering (NOT part of the MOEnv contract) -------------

    def render_policy(self, policy, n_episodes=5, max_steps=50, pause=0.4, seed=0):
        """
        Visualise ``policy`` in the Resource Gathering pygame window.

        It opens a fresh ``render_mode="human"`` gym environment and drives it with the 
        policy dict. RG is stochastic (an enemy cell kills with probability 0.1), so each 
        episode is seeded (``reset(seed=...)``) for reproducibility, and several episodes 
        are run. Stepping stops at the real gym episode end (``terminated``/``truncated``).
        """
        import time

        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        render_env = mo_gym.make("resource-gathering-v0", render_mode="human")
        try:
            for episode in range(n_episodes):
                obs, _ = render_env.reset(seed=seed + episode)
                render_env.render()
                state = tuple(int(x) for x in obs)  # (row, col, has_gold, has_gem)
                print(f"Episode {episode + 1} | start {state}")
                total = np.zeros(self.n_objectives)
                done = False
                step = 0
                while step < max_steps and not done:
                    step += 1
                    action = policy[state]
                    obs, reward, terminated, truncated, _ = render_env.step(action)
                    render_env.render()
                    time.sleep(pause)
                    done = terminated or truncated
                    state = tuple(int(x) for x in obs)
                    total += np.asarray(reward, dtype=float)
                    note = "  <- KILLED by enemy" if (done and reward[0] < 0) else ""
                    print(f" t={step:2d} | state={state} | "
                          f"action={action_names[action]} | reward={reward}{note}")
                if total[0] < 0:
                    outcome = "killed by enemy"
                elif done:
                    outcome = "reached home"
                else:
                    outcome = "reached max steps"
                print(f"  {outcome} after {step} steps | total reward {total}")
        finally:
            render_env.close()
