"""Unbias noise.

See [this document](swap_unbiasing.md).
"""

import functools
import math
import sys

import numpy as np
import scipy

MAX_SWAP_LEVEL = 4 - 2 * math.sqrt(3) - sys.float_info.epsilon

SWAP_START_STATE = np.array([0, 0, 0, 0, 0, 0, 0, 1])
# This is the probability distribution of the Markov Chain.
# We start at the starting state with a probability of one.

SWAP_LEVENSHTEIN_VALUE = np.array([1, 0, 0, 2, 0, 2, 0, 0])
# This is the vector of the value of each state of the Markov Chain.


# We do not use @functools.lru_cache on the following function.
# Caching would not be efficient, since its purpose is to be called through a solver.
def __compute_expected_cer_from_noise_level(p: float, N: int) -> float:
    """Compute the expected CER from an uncorrected (biased) p for a string of length N.

    Args:
        p: Uncorrected (biased) noise level,
            that is probability to swap an unswapped character
        N: The length of the string.

    Returns:
        The expected CER (in the sense of the
            [expected value](https://en.wikipedia.org/wiki/Expected_value))
    """
    p = float(p)
    q = 1 - p
    # The transition matrix of the Markov Chain:
    P = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, p, q, 0, 0, 0],
            [p, q, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, p, q, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, p, q, 0, 0, 0],
            [0, 0, 0, 0, 0, p, q, 0],
        ]
    )
    # Initialize the state:
    state = SWAP_START_STATE
    # Initialize the value of Levenshtein (this value should be zero):
    levenshtein = state @ SWAP_LEVENSHTEIN_VALUE

    for _ in range(N - 1):
        # We compute the probability distribution of the Markov Chain
        # for the next iteration:
        state = state @ P
        levenshtein += state @ SWAP_LEVENSHTEIN_VALUE
    return levenshtein / N


@functools.cache
def __compute_noise_level_from_expected_cer(cer: float, N: int) -> float:
    """Compute the noise level we have to pass as input in order to get.

    This is the real "unbias_swap" function. The other one is just a wrapper.

    Args:
        cer: The Character Error Rate we want to have.
        N: The length of the string.

    Returns:
        Unbiased probability
    """
    return float(
        scipy.optimize.fsolve(
            lambda x: __compute_expected_cer_from_noise_level(float(x[0]), N) - cer,
            [0],
        )[0]
    )


@functools.lru_cache
def unbias_swap(p: float, N: int, natural_language_swap_correction: float) -> float:
    """Re-compute p to take unbiasing into account.

    See doc for [more details](swap_unbiasing.md).

    Args:
        p: Input probability. The user want the expectation of the Character Error Rate
            to tend to this value.
        N: The length of the string.
        natural_language_swap_correction: A correction factor to take into account the
            fact that natural language is not random.

    Returns:
        Unbiased probability, using an approximation formula for strings that are too
            long.
    """
    # To avoid some "math domain error" later, in the case when a former unbiasing
    # made p > MAX_SWAP_LEVEL, we need to force it at the max level:
    p = min(p, MAX_SWAP_LEVEL) * natural_language_swap_correction
    # Whatever, N = nchar = 0 anyway, so returning p or something else does not matter:
    if N == 0:
        return p
    if p == 0:
        return 0
    # We have a formula if N is too long:
    if N > 50:
        return (2 - p) / 2 - math.sqrt((p**2) - (8 * p) + 4) / 2
    return __compute_noise_level_from_expected_cer(p, N)


def unbias_split_into_words(p: float, text: list[str]) -> float:
    """Unbias probability to take into account the absence of spaces when splitting.

    We need to apply noise word by word in order to have an output list with the same
    length as the input.
    If we applied noise word by word we will decrease the effective error rate
    since no noise will be added on spaces between words.
    So we increase the probability in order to compensate this loss


    Args:
        p: Input probability. The user want the expectation of the Character Error Rate
            to tend to this value.
        text: Text on which we unbias.

    Returns:
        Unbiased probability
    """
    n_chars = sum(map(len, text))
    n_spaces = len(text) - 1
    return p * (1 + n_spaces / n_chars)


def unbias_several_action(p: float, n_actions: int) -> float:
    """Unbias probability to remove the bias due to the successive actions.

    If applied N times, the probability to change something will become.

    p_effective = (1 - (1 - noise_level) ** N)

    so we have to invert it to have p_effective = p.

    Notice that at first order of the Taylor expansion, this becomes
    p = p / len(self.actions)

    Args:
        p: Input probability. The user want the expectation of the Character Error Rate
            to tend to this value.
        n_actions: Number of actions.

    Returns:
        Unbiased probability
    """
    p_effective: float = 1.0 - (1.0 - p) ** (1 / n_actions)
    return p_effective
