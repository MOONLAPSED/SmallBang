#!/usr/bin/env python3
# stdlib-only example: classical 1-bit bit-flip channel and reversible dilation

import random
import itertools
from typing import Tuple, List, Dict

# --------------------------
# Deterministic maps on 1 bit
# --------------------------
def id_map(x: int) -> int:
    return x

def not_map(x: int) -> int:
    return 1 - x

DETERMINISTIC_FUNCS = [id_map, not_map]

# --------------------------
# Build channel (Gamma)
# --------------------------
def build_bitflip_channel(p: float) -> List[List[float]]:
    """Return 2x2 column-stochastic matrix Gamma[y][x]."""
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1]")
    # rows are destination y = 0,1; columns are origin x = 0,1
    return [[1 - p, p],
            [p,     1 - p]]

# --------------------------
# Decompose as mixture of deterministic maps
# --------------------------
def classical_decomposition_bitflip(p: float):
    """
    For bitflip, decomposition is trivial:
      with prob (1-p) apply id_map, with prob p apply not_map.
    Returns list of (weight, function).
    """
    return [(1.0 - p, id_map), (p, not_map)]

# --------------------------
# Reversible dilation (system bit + env bit)
# --------------------------
# We'll represent states as tuples (x, k) where x in {0,1} is system,
# k in {0,1} is environment choice index (0 => id, 1 => not).
# The reversible map U applies f_k to the system and leaves k unchanged:
#   U: (x, k) -> ( f_k(x), k )
#
# As a permutation on 4 states this is deterministic and reversible.

def apply_dilation_U(state: Tuple[int, int]) -> Tuple[int, int]:
    x, k = state
    f = DETERMINISTIC_FUNCS[k]
    return (f(x), k)

# --------------------------
# Given env distribution q_k, induce system channel by marginalizing
# --------------------------
def induced_channel_from_dilation(env_dist: List[float]) -> List[List[float]]:
    """
    env_dist[k] = probability environment is in state k prior to U.
    Returns 2x2 Gamma[y][x].
    """
    K = len(env_dist)
    Gamma = [[0.0, 0.0], [0.0, 0.0]]  # rows y, cols x
    for x in (0,1):
        for k in range(K):
            y = DETERMINISTIC_FUNCS[k](x)
            Gamma[y][x] += env_dist[k]
    # columns should already sum to 1 if env_dist is normalized over k
    return Gamma

# --------------------------
# Sampling utilities (oracle simulation)
# --------------------------
def sample_channel(Gamma: List[List[float]], x: int) -> int:
    """Sample y ~ P(y|x) given column-stochastic Gamma[y][x]."""
    r = random.random()
    p0 = Gamma[0][x]
    return 0 if r <= p0 else 1

def sample_dilation(env_dist: List[float], x: int) -> int:
    """Sample by first sampling env k, then applying deterministic f_k."""
    # pick k according to env_dist
    r = random.random()
    c = 0.0
    for k, q in enumerate(env_dist):
        c += q
        if r <= c:
            return DETERMINISTIC_FUNCS[k](x)
    return DETERMINISTIC_FUNCS[len(env_dist)-1](x)

# --------------------------
# Small demos
# --------------------------
if __name__ == "__main__":
    p = 0.1  # bit-flip prob
    Gamma = build_bitflip_channel(p)
    print("Gamma (rows y, cols x):")
    print(Gamma)

    dec = classical_decomposition_bitflip(p)
    print("Classical decomposition (weight, func names):", [(w, f.__name__) for (w, f) in dec])

    # env: choose k=0 (id) with prob 1-p, k=1 (not) with prob p
    env_dist = [1.0 - p, p]
    G_from_dilation = induced_channel_from_dilation(env_dist)
    print("Induced channel from dilation:", G_from_dilation)

    # Quick Monte Carlo check that sampling from dilation == sampling from Gamma
    N = 20000
    counts_via_gamma = {0:0, 1:0}
    counts_via_dilation = {0:0, 1:0}
    x = 1  # origin bit
    for _ in range(N):
        y1 = sample_channel(Gamma, x)
        y2 = sample_dilation(env_dist, x)
        counts_via_gamma[y1] += 1
        counts_via_dilation[y2] += 1
    print("empirical via Gamma:", counts_via_gamma)
    print("empirical via dilation:", counts_via_dilation)
