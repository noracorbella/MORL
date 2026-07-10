"""
This is a reusable contract checker for :class:`MOEnv` implementations.

This module verifies that an environment satisfies the :class:`MOEnv` contract. It makes 
no reference to any particular environment or to any reference implementation. It is the 
tool to use when writing a wrapper for a new MOMDP environment: implement
:class:`MOEnv`, then call :func:`check_moenv_contract` to confirm the algorithms will accept it.

The entry point is :func:`check_moenv_contract`.
"""

import numpy as np

from morl.core.env_interface import MOEnv


def check_moenv_contract(env: MOEnv, prob_tol=1e-9, verbose=True):
    """
    Check that ``env`` satisfies the :class:`MOEnv` contract.

    Runs the following checks and prints PASS/FAIL line:

    1. ``n_objectives`` is a positive integer and ``gamma`` is a float in
       ``[0, 1]``.
    2. ``states()`` is finite, non-empty, every state is hashable, and repeated
       calls return the same set of states.
    3. ``is_terminal(state)`` returns a ``bool`` for every state, and at least
       one state is non-terminal.
    4. For every non-terminal state, ``actions(state)`` is non-empty.
    5. For every non-terminal state and available action, ``transitions``
       returns a distribution with non negative probabilities and sum to
       ``1.0`` (within ``prob_tol``), with all ``next_state`` values in
       ``states()``, and with all ``reward_vector`` values length
       ``n_objectives``.
    6. ``transitions`` is stable: a second call with the same arguments yields
       the same probabilities, next states and rewards.

    Parameters
    ----------
    env : MOEnv
        The environment to check.
    prob_tol : float, optional
        Tolerance used when checking that transition probabilities sum to 1.0.
    verbose : bool, optional
        If true, print a report. If false, run silently and only
        return the boolean result.

    Returns
    -------
    bool
        ``True`` if every check passed, ``False`` otherwise.
    """
    results = []  # (name, passed, detail)

    # --- Check 1: attributes -----------------------------------------------
    n_obj = getattr(env, "n_objectives", None)
    gamma = getattr(env, "gamma", None)
    obj_ok = isinstance(n_obj, (int, np.integer)) and int(n_obj) > 0
    gamma_ok = isinstance(gamma, (float, int, np.floating)) and 0.0 <= float(gamma) <= 1.0
    results.append((
        "n_objectives is a positive int and gamma is in [0, 1]",
        obj_ok and gamma_ok,
        f"n_objectives={n_obj!r}, gamma={gamma!r}",
    ))
    if not (obj_ok and gamma_ok):
        # Later checks depend on these. 
        return _report(results, verbose)
    n_obj = int(n_obj)

    # --- Check 2: states() finite, non-empty, stable -------------
    states_1 = list(env.states())
    detail = ""
    states_ok = len(states_1) > 0
    if not states_ok:
        detail = "states() is empty"
    else:
        try:
            set_1 = set(states_1)
        except TypeError as e:
            states_ok = False
            detail = f"states are not hashable: {e}"
        else:
            set_2 = set(env.states())
            if set_1 != set_2:
                states_ok = False
                detail = "states() is not stable across calls"
            else:
                detail = f"{len(set_1)} states, hashable and stable"
    results.append(("states() is finite, non-empty, hashable and stable",
                    states_ok, detail))
    if not states_ok:
        return _report(results, verbose)

    all_states = set(states_1)

    # --- Check 3: is_terminal returns bool; some non-terminal exists -------
    non_bool = [s for s in states_1 if not isinstance(env.is_terminal(s), (bool, np.bool_))]
    non_terminal = [s for s in states_1 if not env.is_terminal(s)]
    check3_ok = not non_bool and len(non_terminal) > 0
    if non_bool:
        detail = f"{len(non_bool)} states where is_terminal is not a bool"
    elif not non_terminal:
        detail = "every state is terminal; nothing to solve"
    else:
        detail = f"{len(non_terminal)} non-terminal, {len(states_1) - len(non_terminal)} terminal"
    results.append(("is_terminal returns bool; at least one non-terminal state",
                    check3_ok, detail))
    if not check3_ok:
        return _report(results, verbose)

    # --- Check 4: actions() non-empty on non-terminal states ---------------
    empty_action_states = [s for s in non_terminal if len(list(env.actions(s))) == 0]
    results.append((
        "actions() is non-empty on every non-terminal state",
        not empty_action_states,
        f"{len(empty_action_states)} non-terminal states with no actions"
        if empty_action_states else "all non-terminal states have actions",
    ))

    # --- Check 5: transition distribution contract -------------------------
    prob_bad, state_bad, reward_bad = [], [], []
    n_pairs = 0
    for s in non_terminal:
        for a in env.actions(s):
            n_pairs += 1
            dist = env.transitions(s, a)
            probs = [p for (p, _ns, _r) in dist]
            if any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > prob_tol:
                prob_bad.append((s, a, probs))
            for (_p, ns, r) in dist:
                if ns not in all_states:
                    state_bad.append((s, a, ns))
                if len(np.asarray(r)) != n_obj:
                    reward_bad.append((s, a, np.asarray(r).shape))
    results.append((
        "transition probabilities non-negative and sum to 1.0",
        not prob_bad,
        f"{len(prob_bad)} offending (state, action) pairs" if prob_bad
        else f"all {n_pairs} (state, action) pairs OK",
    ))
    results.append((
        "every next_state is in states()",
        not state_bad,
        f"{len(state_bad)} next_states outside states()" if state_bad
        else "all next_states are valid states",
    ))
    results.append((
        f"every reward_vector has length n_objectives (={n_obj})",
        not reward_bad,
        f"{len(reward_bad)} reward vectors of wrong length" if reward_bad
        else "all reward vectors correctly sized",
    ))

    # --- Check 6: transitions() stable across calls ------------------------
    unstable = []
    for s in non_terminal:
        for a in env.actions(s):
            d1 = env.transitions(s, a)
            d2 = env.transitions(s, a)
            if not _same_distribution(d1, d2):
                unstable.append((s, a))
    results.append((
        "transitions() is stable across repeated calls",
        not unstable,
        f"{len(unstable)} (state, action) pairs changed between calls"
        if unstable else "all transitions stable",
    ))

    return _report(results, verbose)


def _same_distribution(d1, d2):
    """True if two transition distributions are element-wise equal."""
    if len(d1) != len(d2):
        return False
    for (p1, s1, r1), (p2, s2, r2) in zip(d1, d2):
        if p1 != p2 or s1 != s2:
            return False
        if not np.array_equal(np.asarray(r1), np.asarray(r2)):
            return False
    return True


def _report(results, verbose):
    """Print the report if verbose and return whether all checks passed."""
    all_passed = all(passed for _name, passed, _detail in results)
    if verbose:
        print("MOEnv contract checks")
        print("=" * 60)
        for name, passed, detail in results:
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")
            print(f"        {detail}")
        print("=" * 60)
        print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return all_passed
