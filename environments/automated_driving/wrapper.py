"""
MOEnv wrapper for the Automated Driving System (ADS) environment.

This adapts the ADS simulator (in the ``simulator/`` directory) 
to :class:`MOEnv`. ADS is a MOMDP with three objectives, 
``[vehicle_goal, internal_safety, external_safety]``.

This wrapper imports that ``Environment`` class directly and drives it with
``reset``/``step``.

* **Stochasticity** is enabled at ``degree_of_stochasticity = 1``: a pedestrian 
  standing on cell ``[3, 3]`` moves stochastically over ``move_map[3][3]``. 
  Everywhere else pedestrians act deterministically. The degree is determined on 
  ``constants`` module *before* the env is built.
* **State** is ``(car, ped1, ped2)``, and the two pedestrians are kept **sorted** 
``ped1 >= ped2``.
* **Terminality** is per-state: a state is terminal iff the car is on a goal cell
  (``is_terminal(s) = car in {13, 20}``). These terminal states are *included*
  in ``states()``. 

Transitions are precomputed once for every non-terminal ``(state, action)`` by
driving the simulator, then cached in memory, so :meth:`transitions` is cheap on
repeated calls.
"""

import os
import pickle
import sys
import time
import threading

import pygame

import numpy as np

from morl.core.env_interface import MOEnv

_ADS_SIM_DIR = os.path.join(os.path.dirname(__file__), "simulator")
if _ADS_SIM_DIR not in sys.path:
    sys.path.insert(0, _ADS_SIM_DIR)

import constants 
from ADS_Environment import Environment 


# Degree of pedestrian stochasticity used by the reference ADS results.
DEFAULT_DEGREE_OF_STOCHASTICITY = 1

# The cell (row, col) on which a pedestrian behaves stochastically.
STOCHASTIC_CELL = [3, 3]

# Default discount factor from ADS's configuration.
DEFAULT_GAMMA = 0.7

# Sentinel pedestrian action meaning "act cleverly" (deterministic) in step().
ACT_CLEVER = 8000

# The cache is keyed by the stochasticity degree.
CACHE_VERSION = 1
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_transition_cache")


class AutomatedDrivingEnv(MOEnv):
    """
    Automated Driving System as an :class:`MOEnv`.

    States are ``(car, ped1, ped2)`` tuples of translated cell indices with
    ``ped1 >= ped2``.
    Actions are the integers ``0..5`` on non-terminal states and none on
    terminal states. Transitions are stochastic only when a pedestrian occupies
    the stochastic cell.
    """

    def __init__(self, gamma=DEFAULT_GAMMA,
                 degree_of_stochasticity=DEFAULT_DEGREE_OF_STOCHASTICITY,
                 use_cache=True, cache_dir=None):
        # Set the degree BEFORE constructing the env: the agents build their move
        # maps from it at construction time.
        constants.degree_of_stochasticity = degree_of_stochasticity

        _prev_cwd = os.getcwd()
        _prev_sdl = os.environ.get("SDL_VIDEODRIVER")
        try:
            os.chdir(_ADS_SIM_DIR)
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self._env = Environment(weights=None)
        finally:
            os.chdir(_prev_cwd)
            if _prev_sdl is None:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = _prev_sdl

        self.n_objectives = int(Environment.n_objectives)  # 3
        self.gamma = float(gamma)
        self.n_actions = self._env.n_actions               # 6

        car_cells = list(self._env.states_agent_left)      # drivable non-goal cells
        ped_cells = list(self._env.states_agent_right)     # pedestrian cells
        self._goal_cells = {
            self._env.terminal_state_agent_pos1,           # 13
            self._env.terminal_state_agent_pos2,           # 20
        }

        self._ped_stochastic_actions = list(self._env.agents[1].move_map[3][3])
        self._stochastic_cell_1d = self._env.translate(STOCHASTIC_CELL)  # 24

        # Sorted (ped1 >= ped2) pedestrian pairs.
        ped_pairs = [(p1, p2) for p1 in ped_cells for p2 in ped_cells if p1 >= p2]

        self._non_terminal_states = [
            (c, p1, p2) for c in car_cells for (p1, p2) in ped_pairs
        ]
        terminal_states = [
            (c, p1, p2) for c in sorted(self._goal_cells) for (p1, p2) in ped_pairs
        ]
        self._states = self._non_terminal_states + terminal_states

        self.start_state = self._canonical((
            self._env.translate(self._env.initial_agent_left_position),
            self._env.translate(self._env.initial_pedestrian_1_position),
            self._env.translate(self._env.initial_pedestrian_2_position),
        ))

        # Load the precomputed transitions from disk if a matching cache exists,
        # otherwise precompute and save.
        self._degree = degree_of_stochasticity
        self._transition_cache = {}
        self._build_or_load_cache(use_cache, cache_dir)

    # -- MOEnv interface ----------------------------------------------------

    def states(self):
        """
        Return all states: non-terminal (car on a drivable non-goal cell) plus
        terminal (car on a goal cell), pedestrians sorted ``ped1 >= ped2``.
        """
        return list(self._states)

    def actions(self, state):
        """
        Return ``[0..5]`` on non-terminal states and ``[]`` on terminal
        (car-at-goal) states.
        """
        if self.is_terminal(state):
            return []
        return list(range(self.n_actions))

    def transitions(self, state, action):
        """
        Return the transition distribution for ``action`` in ``state`` as a list 
        of ``(prob, next_state, reward_vector)``.

        Deterministic where no pedestrian occupies the stochastic cell, otherwise
        one outcome per stochastic pedestrian action (probabilities ``1/n`` for
        one pedestrian, ``1/n^2`` for both).
        """
        return self._transition_cache[(state, action)]

    def is_terminal(self, state):
        """
        Return ``True`` iff the car is on a goal cell.
        """
        return state[0] in self._goal_cells

    def close(self):
        """Empty"""

    # -- rendering (NOT part of the MOEnv contract) -------------

    def render_policy(self, policy, n_episodes=5, max_steps=200, pause=0.5, seed=0):
        """
        Visualise ``policy`` in the ADS pygame window.
        """

        action_names = {0: "RIGHT", 1: "UP", 2: "LEFT",
                        3: "RIGHT(fast)", 4: "UP(fast)", 5: "LEFT(fast)"}

        constants.degree_of_stochasticity = self._degree

        prev_sdl = os.environ.pop("SDL_VIDEODRIVER", None)
        if pygame.display.get_init():
            pygame.display.quit()
        prev_cwd = os.getcwd()
        try:
            os.chdir(_ADS_SIM_DIR)  # so Window finds wastebasket.png
            render_env = Environment(weights=None)
        finally:
            os.chdir(prev_cwd)

        def _run_episodes():
            for episode in range(n_episodes):
                np.random.seed(seed + episode)  # reproducible pedestrian stochasticity
                render_env.reset()
                state = self._canonical(render_env.get_state())
                print(f"Episode {episode + 1} | start {state}")
                render_env.update_window()
                time.sleep(pause)
                done = False
                step = 0
                while step < max_steps and not done:
                    step += 1
                    action = policy[state]
                    raw_state, rewards, dones = render_env.step([action])
                    done = dones[0]
                    state = self._canonical(raw_state)
                    reward = np.asarray(rewards, dtype=float)
                    render_env.update_window()
                    time.sleep(pause)
                    # reward = [vehicle goal, internal safety, external safety]
                    print(f" t={step:2d} | state={state} | "
                          f"action={action_names[action]} | reward={reward}")
                outcome = "reached goal" if done else "reached max steps"
                print(f"  {outcome} after {step} steps")
            print("All episodes finished -- close the window to exit.")

        threading.Thread(target=_run_episodes, daemon=True).start()
        try:
            render_env.render("Evaluating")
        finally:
            if pygame.display.get_init():
                pygame.display.quit()
            if prev_sdl is not None:
                os.environ["SDL_VIDEODRIVER"] = prev_sdl

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _canonical(state):
        """
        Return ``state`` with the two pedestrians sorted (ped1 >= ped2).
        """
        c, p1, p2 = int(state[0]), int(state[1]), int(state[2])
        return (c, p1, p2) if p1 >= p2 else (c, p2, p1)

    def _cell(self, index_1d):
        """
        1-D cell index -> [row, col] position the simulator's reset expects.
        """
        return self._env.translate_state_cell(index_1d)

    def _outcomes(self, state, action):
        """
        Build the outcome list for ``(state, action)`` by driving the sim.
        """
        c, p1, p2 = state
        pos = [self._cell(c), self._cell(p1), self._cell(p2)]
        p1_stoch = p1 == self._stochastic_cell_1d
        p2_stoch = p2 == self._stochastic_cell_1d
        acts = self._ped_stochastic_actions

        outcomes = []
        if not p1_stoch and not p2_stoch:
            self._env.reset(*pos)
            ns, r, _done = self._env.step([action])
            outcomes.append((1.0, self._canonical(ns), np.array(r, dtype=float)))
        elif p1_stoch and not p2_stoch:
            prob = 1.0 / len(acts)
            for pa in acts:
                self._env.reset(*pos)
                ns, r, _done = self._env.step([action, pa, ACT_CLEVER])
                outcomes.append((prob, self._canonical(ns), np.array(r, dtype=float)))
        elif not p1_stoch and p2_stoch:
            prob = 1.0 / len(acts)
            for pa in acts:
                self._env.reset(*pos)
                ns, r, _done = self._env.step([action, ACT_CLEVER, pa])
                outcomes.append((prob, self._canonical(ns), np.array(r, dtype=float)))
        else:
            prob = 1.0 / (len(acts) ** 2)
            for pa1 in acts:
                for pa2 in acts:
                    self._env.reset(*pos)
                    ns, r, _done = self._env.step([action, pa1, pa2])
                    outcomes.append((prob, self._canonical(ns), np.array(r, dtype=float)))
        return outcomes

    def _precompute(self):
        """
        Fill the transition cache for every non-terminal (state, action).
        """
        for state in self._non_terminal_states:
            for action in range(self.n_actions):
                self._transition_cache[(state, action)] = self._outcomes(state, action)

    # -- disk cache ---------------------------------------------------------

    def _cache_path(self, cache_dir):
        """
        Path of the cache file for this degree.
        """
        cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
        return os.path.join(
            cache_dir, f"ads_transitions_degree{self._degree}_v{CACHE_VERSION}.pkl"
        )

    def _build_or_load_cache(self, use_cache, cache_dir):
        """
        Load the transition cache from disk if valid, else precompute and save.
        """
        path = self._cache_path(cache_dir)
        if use_cache and self._load_cache(path):
            return
        self._precompute()
        if use_cache:
            self._save_cache(path)

    def _load_cache(self, path):
        """
        Populate the cache from ``path`` if it exists and matches this degree
        and cache version. Returns True on a successful load, False otherwise
        (missing file, version/degree mismatch, or unreadable).
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
        except Exception:
            return False
        # Header verification, in addition to the degree being in the filename:
        # a degree-0 or wrong-version cache must never load for a degree-1 run.
        if blob.get("version") != CACHE_VERSION or blob.get("degree") != self._degree:
            return False
        self._transition_cache = blob["transitions"]
        return True

    def _save_cache(self, path):
        """
        Write the transition cache to ``path`` with a version+degree header.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        blob = {
            "version": CACHE_VERSION,
            "degree": self._degree,
            "transitions": self._transition_cache,
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
