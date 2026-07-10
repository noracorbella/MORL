"""
Abstract interface for multi-objective MDPs (MOMDPs).

This module defines :class:`MOEnv`, the abstraction class that all the 
algorithms are built on. The four algorithms (VI, CHVI, LexVI, LHVI) 
never touch a concrete environment.
"""

from abc import ABC, abstractmethod


class MOEnv(ABC):
    """
    Finite multi-objective Markov Decision Process (MOMDP).

    A ``MOEnv`` describes a finite MOMDP by its state set, its per-state action
    set, and a vector-valued transition model. Rewards are vectors with one
    component per objective. There is no scalarisation.

    Subclasses must implement :meth:`states`, :meth:`actions`,
    :meth:`transitions` and :meth:`is_terminal`, and must set the two attributes
    described below. The algorithms rely on the guarantees stated here and a
    subclass that violates them may produce incorrect results.

    Required attributes
    -------------------
    n_objectives : int
        The number of objectives, i.e. the length of every reward vector
        returned by :meth:`transitions`.
    gamma : float
        The discount factor applied to future rewards, in the range
        ``0 <= gamma <= 1``. Constant for the lifetime of the environment.

    States
    ------
    A state is a canonical object that must be usable as a dictionary
    key and as an element of a set. Typically, they are tuples of integers 
    (for example ``(row, col)``). Algorithms store per-state values,
    hulls and policies in dictionaries keyed by these objects.

    Terminal states
    ---------------
    Terminality is a property of a *state*, not of a transition. A terminal
    state is an absorbing end-of-episode state at which the episode stops and no
    further reward is accrued. :meth:`states` returns *all* states, terminal
    ones included, and :meth:`is_terminal` is what distinguishes them.

    :meth:`actions` and :meth:`transitions` are only guaranteed to be meaningful
    on non-terminal states. Algorithms must not apply a Bellman update
    to terminal states: the value of a terminal state is the zero vector by
    definition, and the reward for entering a terminal state is carried on the
    transition that leads into it. On a terminal state :meth:`actions` may return
    an empty list, and :meth:`transitions` need not be defined.

    Determinism
    -----------
    A deterministic transition is simply a transition distribution with a single
    outcome of probability ``1.0``: ``[(1.0, next_state, reward_vector)]``. 
    Algorithms handle deterministic and stochastic environments with the same code
    path.
    """

    n_objectives: int
    gamma: float

    @abstractmethod
    def states(self):
        """
        Return all states of the MOMDP.

        Returns
        -------
        iterable
            An iterable of every state in the environment, including terminal
            states. The collection must be finite, and every ``next_state``
            produced by :meth:`transitions` must be one of these states.
        """
        raise NotImplementedError

    @abstractmethod
    def actions(self, state):
        """
        Return the actions available in ``state``.

        Parameters
        ----------
        state
            A state previously returned by :meth:`states`.

        Returns
        -------
        iterable
            An iterable of the actions available in ``state``. The set of
            actions for a non-terminal state must be non-empty and stable
            across calls.

            For a terminal state this may return an empty iterable, since
            algorithms never query actions on terminal states.
        """
        raise NotImplementedError

    @abstractmethod
    def transitions(self, state, action):
        """
        Return the transition distribution for taking ``action`` in ``state``.

        This is a vector-valued transition model of the MOMDP.

        Parameters
        ----------
        state
            A non-terminal state previously returned by :meth:`states`.
        action
            An action previously returned by ``actions(state)``.

        Returns
        -------
        list of (float, state, sequence of float)
            A list of ``(prob, next_state, reward_vector)`` triples describing
            the distribution over outcomes. The list must satisfy all of:

            * Every ``prob`` is non-negative and the probabilities sum to
              ``1.0`` (up to floating-point tolerance).
            * Every ``next_state`` is one of the states returned by
              :meth:`states`.
            * Every ``reward_vector`` has length :attr:`n_objectives`. It is the
              immediate vector reward received on this transition, including the
              reward for entering a terminal ``next_state``.

            A deterministic transition is expressed as 
            ``[(1.0, next_state, reward_vector)]``.
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state):
        """
        Return whether ``state`` is a terminal state.

        Parameters
        ----------
        state
            A state previously returned by :meth:`states`.

        Returns
        -------
        bool
            ``True`` if ``state`` is terminal and ``False`` otherwise. 
            The value of terminal states is the zero vector and the 
            reward for reaching them is carried on the incoming transition.
        """
        raise NotImplementedError
