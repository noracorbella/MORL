"""Abstract interface for multi-objective MDPs (MOMDPs).

This module defines :class:`MOEnv`, the single abstraction every algorithm in
this repository depends on. The four algorithms (VI, CHVI, LexVI, LHVI) are
written against :class:`MOEnv` alone and never touch a concrete environment.

To apply the algorithms to a new multi-objective environment you implement one
subclass of :class:`MOEnv` that adapts your environment's dynamics to the
contract documented below. Nothing beyond this contract is assumed by the
algorithms, so if your subclass satisfies it the algorithms will run unchanged.
"""

from abc import ABC, abstractmethod


class MOEnv(ABC):
    """A finite multi-objective Markov Decision Process (MOMDP).

    An ``MOEnv`` describes a finite MOMDP by its state set, its per-state action
    set, and a vector-valued transition model. Rewards are vectors with one
    component per objective; there is no scalarisation baked into the interface,
    so a single environment instance can be reused by scalar (VI), convex-hull
    (CHVI) and lexicographic (LexVI, LHVI) algorithms alike.

    Subclasses must implement :meth:`states`, :meth:`actions`,
    :meth:`transitions` and :meth:`is_terminal`, and must set the two attributes
    described below. The algorithms rely on the guarantees stated here; a
    subclass that violates them may produce incorrect results.

    Required attributes
    -------------------
    n_objectives : int
        The number of objectives, i.e. the length of every reward vector
        returned by :meth:`transitions`. Must be a positive integer and constant
        for the lifetime of the environment.
    gamma : float
        The discount factor applied to future rewards, in the range
        ``0 <= gamma <= 1``. Constant for the lifetime of the environment.

    States
    ------
    A state is any hashable, canonical object: it must be usable as a dictionary
    key and as an element of a set, and two states that represent the same
    situation must compare equal and hash equal. Tuples of integers (for example
    ``(row, col)``) are a typical choice. Algorithms store per-state values,
    hulls and policies in dictionaries keyed by these objects, so equal
    situations must always be represented by the identical canonical value.

    Terminal states
    ---------------
    Terminality is a property of a *state*, not of a transition. A terminal
    state is an absorbing end-of-episode state at which the episode stops and no
    further reward is accrued. :meth:`states` returns *all* states, terminal
    ones included, and :meth:`is_terminal` is what distinguishes them.

    :meth:`actions` and :meth:`transitions` are only guaranteed to be meaningful
    on non-terminal states. Algorithms must not back up (apply a Bellman update
    to) terminal states: the value of a terminal state is the zero vector by
    definition, and the reward for entering a terminal state is carried on the
    transition that leads into it. On a terminal state :meth:`actions` may return
    an empty list, and :meth:`transitions` need not be defined.

    Determinism
    -----------
    Determinism is not a separate mode. A deterministic transition is simply a
    transition distribution with a single outcome of probability ``1.0``:
    ``[(1.0, next_state, reward_vector)]``. Algorithms handle deterministic and
    stochastic environments with the same code path, so a deterministic
    environment needs no special treatment beyond returning single-element
    lists from :meth:`transitions`.
    """

    n_objectives: int
    gamma: float

    @abstractmethod
    def states(self):
        """Return all states of the MOMDP.

        Returns
        -------
        iterable
            An iterable of every state in the environment, including terminal
            states. Each state must be hashable and canonical (see the class
            docstring). The collection must be finite, and every ``next_state``
            produced by :meth:`transitions` must be one of these states.

            The set of states must be stable across calls: repeated calls must
            yield the same states (order need not be preserved).
        """
        raise NotImplementedError

    @abstractmethod
    def actions(self, state):
        """Return the actions available in ``state``.

        Parameters
        ----------
        state
            A state previously returned by :meth:`states`.

        Returns
        -------
        iterable
            An iterable of the actions available in ``state``. Actions may be
            any objects the subclass chooses (integers are typical). The set of
            actions for a given non-terminal state must be non-empty and stable
            across calls.

            For a terminal state this may return an empty iterable, since
            algorithms never query actions on terminal states.
        """
        raise NotImplementedError

    @abstractmethod
    def transitions(self, state, action):
        """Return the transition distribution for taking ``action`` in ``state``.

        This is the vector-valued transition model of the MOMDP. It must be safe
        and cheap to call repeatedly with the same arguments (algorithms call it
        many times per state across iterations); any simulation or computation
        needed to produce the result should be cached internally by the
        subclass so that repeated calls are inexpensive and side-effect free.

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

            A deterministic transition is expressed as the single-element list
            ``[(1.0, next_state, reward_vector)]`` (see the class docstring).
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state):
        """Return whether ``state`` is a terminal (absorbing) state.

        Parameters
        ----------
        state
            A state previously returned by :meth:`states`.

        Returns
        -------
        bool
            ``True`` if ``state`` is terminal and ``False`` otherwise. Terminal
            states are never backed up by the algorithms; their value is the
            zero vector and the reward for reaching them is carried on the
            incoming transition (see the class docstring).
        """
        raise NotImplementedError
