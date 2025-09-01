#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
msc_qsd.py

A standard-library-only Python 3.12+ implementation of core MSC/QSD primitives:
  - ContinuedFraction for high-precision irrationals
  - MorphicComplex: reversible complex numbers tied to π via Euler’s identity
  - HilbertSpace & QuantumState: Hilbert-space semantics
  - Q kernel: epigenetic probabilistic runtimes

All classes use @dataclass where appropriate and avoid any third-party dependencies.
"""

import math
import json
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generic, TypeVar, Union



# Type variables
T = TypeVar('T')  # Static structure
V = TypeVar('V')  # Dynamic value
C = TypeVar('C', bound=Callable)  # Computational, imaginary-part or periodicity


class OperatorType(Enum):
    """Fundamental operation types in our computational universe."""
    COMPOSITION = auto()   # Function composition (f >> g)
    TENSOR      = auto()   # Tensor product (⊗)
    DIRECT_SUM  = auto()   # Direct sum (⊕)
    OUTER       = auto()   # Outer product (|ψ⟩⟨φ|)
    ADJOINT     = auto()   # Hermitian adjoint (†)
    MEASUREMENT = auto()   # Quantum measurement (⟨M|ψ⟩)


@dataclass
class ContinuedFraction:
    """
    Represent an irrational via its continued-fraction coefficients.
    Evaluate to float on demand with controlled precision.
    """
    integer_part: int
    coefficients: list[int]
    max_terms: int = 100

    def evaluate(self, terms: int | None = None) -> float:
        n = self.max_terms if terms is None else min(terms, self.max_terms)
        coeffs = self.coefficients[:n]
        val = 0.0
        for a in reversed(coeffs):
            val = 1.0 / (a + val)
        return self.integer_part + val

    def __add__(self, other: Union['ContinuedFraction', int, float]) -> float:
        return self.evaluate() + (other.evaluate() if isinstance(other, ContinuedFraction) else other)

    def __mul__(self, other: Union['ContinuedFraction', int, float]) -> float:
        return self.evaluate() * (other.evaluate() if isinstance(other, ContinuedFraction) else other)

    def __repr__(self) -> str:
        c = self.coefficients
        snippet = c[:5] if len(c) > 5 else c
        return f"ContinuedFraction({self.integer_part}, {snippet}...)"


# Precomputed continued fractions for π and e
PI_CF = ContinuedFraction(3, [
    7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1, 14, 2, 1, 1, 2, 2, 2, 2, 1,
    84, 2, 1, 1, 15, 3, 13, 1, 4, 2, 6, 6, 99, 1, 2, 2, 6, 3, 5, 1,
    1, 6, 8, 1, 7, 1, 2, 3, 7, 1, 2, 1, 1, 12, 1, 1, 1, 3, 1, 1, 8,
    1, 1, 2, 1, 6, 1, 1, 5, 2, 2, 3, 1, 2, 4, 4, 16, 1, 161, 45, 1,
    22, 1, 2, 2, 1, 4, 1, 2, 24, 1, 2, 1, 3, 1, 2, 1
])

E_CF = ContinuedFraction(2, [
    1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10, 1, 1, 12, 1, 1, 14,
    1, 1, 16, 1, 1, 18, 1, 1, 20, 1, 1, 22, 1, 1, 24, 1, 1, 26, 1,
    1, 28, 1, 1, 30, 1, 1, 32, 1, 1, 34, 1, 1, 36, 1, 1, 38, 1, 1,
    40, 1, 1, 42, 1, 1, 44, 1, 1, 46, 1, 1, 48, 1, 1, 50, 1, 1, 52,
    1, 1, 54, 1, 1, 56, 1, 1, 58, 1, 1, 60, 1, 1, 62, 1, 1, 64, 1, 1, 66
])


@dataclass
class MorphicComplex:
    """
    Complex number where computation is reversible and context-tracked.
    Euler's identity e^(iπ) = -1 underlies the implementation.
    """
    real: float = 0.0
    imag: float = 0.0
    precision: int = 32
    history: list[str] = field(default_factory=list)

    @classmethod
    def from_euler(cls, r: float, theta: float) -> 'MorphicComplex':
        """Construct via polar coords using Euler's formula."""
        return cls(r * math.cos(theta), r * math.sin(theta))

    def to_euler(self) -> tuple[float, float]:
        r = math.hypot(self.real, self.imag)
        theta = math.atan2(self.imag, self.real)
        return r, theta

    def conjugate(self) -> 'MorphicComplex':
        self.history.append("conjugate")
        return MorphicComplex(self.real, -self.imag, self.precision, self.history.copy())

    def __add__(self, other: Union['MorphicComplex', float, int]) -> 'MorphicComplex':
        if isinstance(other, MorphicComplex):
            r = self.real + other.real
            i = self.imag + other.imag
        else:
            r = self.real + float(other)
            i = self.imag
        self.history.append(f"add({other})")
        return MorphicComplex(r, i, self.precision, self.history.copy())

    def __mul__(self, other: Union['MorphicComplex', float, int]) -> 'MorphicComplex':
        if isinstance(other, MorphicComplex):
            r = self.real * other.real - self.imag * other.imag
            i = self.real * other.imag + self.imag * other.real
        else:
            r = self.real * float(other)
            i = self.imag * float(other)
        self.history.append(f"mul({other})")
        return MorphicComplex(r, i, self.precision, self.history.copy())

    def inner(self, other: 'MorphicComplex') -> float:
        """Hilbert-space inner product: ⟨self|other⟩ = conj(self)·other."""
        conj = self.conjugate()
        prod = conj * other
        return prod.real

    def to_dict(self) -> dict[str, Any]:
        r, theta = self.to_euler()
        return {
            "real": self.real,
            "imag": self.imag,
            "r": r,
            "theta": theta,
            "history": self.history.copy()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def __repr__(self) -> str:
        if abs(self.imag) < 1e-12:
            return f"{self.real:.6g}"
        if abs(self.real) < 1e-12:
            return f"{self.imag:.6g}i"
        sign = '+' if self.imag >= 0 else '-'
        return f"{self.real:.6g} {sign} {abs(self.imag):.6g}i"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MorphicComplex):
            return False
        return (math.isclose(self.real, other.real, rel_tol=1e-12)
                and math.isclose(self.imag, other.imag, rel_tol=1e-12))


class HilbertSpace:
    """Finite-dimensional Hilbert space over MorphicComplex coordinates."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        # Standard basis vectors
        self.basis = [
            [MorphicComplex(1.0 if i == j else 0.0, 0.0) for j in range(dimension)]
            for i in range(dimension)
        ]

    def inner(self, v1: list[MorphicComplex], v2: list[MorphicComplex]) -> MorphicComplex:
        if len(v1) != self.dimension or len(v2) != self.dimension:
            raise ValueError("Vectors must match space dimension")
        acc = MorphicComplex(0.0, 0.0)
        for a, b in zip(v1, v2):
            acc = acc + a.conjugate() * b
        return acc

    def norm(self, v: list[MorphicComplex]) -> float:
        val = self.inner(v, v).real
        return math.sqrt(max(val, 0.0))

    def orthogonalize(self, vectors: list[list[MorphicComplex]]) -> list[list[MorphicComplex]]:
        """Gram–Schmidt process."""
        orth: list[list[MorphicComplex]] = []
        for v in vectors:
            w = v.copy()
            for u in orth:
                coeff = self.inner(w, u).real / max(self.inner(u, u).real, 1e-12)
                w = [wi + (ui * -coeff) for wi, ui in zip(w, u)]
            if self.norm(w) > 1e-12:
                norm = self.norm(w)
                orth.append([wi * (1.0 / norm) for wi in w])
        return orth

    def project(self, v: list[MorphicComplex], subspace: list[list[MorphicComplex]]) -> list[MorphicComplex]:
        """Project v onto the span of subspace."""
        proj = [MorphicComplex(0.0, 0.0) for _ in range(self.dimension)]
        for b in subspace:
            coeff = self.inner(v, b).real / max(self.inner(b, b).real, 1e-12)
            proj = [pi + (bi * coeff) for pi, bi in zip(proj, b)]
        return proj


class QuantumState:
    """State vector in a Hilbert space, with measurement semantics."""

    def __init__(self, amplitudes: list[MorphicComplex], space: HilbertSpace):
        if len(amplitudes) != space.dimension:
            raise ValueError("Amplitude length must match space dimension")
        self.space = space
        self.amps = amplitudes
        self.normalize()

    def normalize(self) -> None:
        n = self.space.norm(self.amps)
        if n > 0:
            self.amps = [ai * (1.0 / n) for ai in self.amps]

    def measure(self) -> int:
        """Collapse to one basis state according to |amplitude|² distribution."""
        probs = [(ai.real * ai.real + ai.imag * ai.imag) for ai in self.amps]
        total = sum(probs)
        if total <= 0:
            return 0
        r = random.random() * total
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r <= cum:
                return i
        return len(probs) - 1

    def superpose(self, other: 'QuantumState', a: MorphicComplex, b: MorphicComplex) -> 'QuantumState':
        if self.space.dimension != other.space.dimension:
            raise ValueError("Spaces must match")
        amps = [a * x + b * y for x, y in zip(self.amps, other.amps)]
        return QuantumState(amps, self.space)

    def __repr__(self) -> str:
        return f"<QuantumState {self.amps!r}>"


@dataclass
class Q:
    """
    Epigenetic kernel: a probabilistic runtime with novelty (ψ) and inertia (π) operators.
    """
    state: float
    ψ: Callable[[float], float]
    π: Callable[[float], float]
    history: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.history.append(self.state)

    def free_energy(self, prior: float = 1.0) -> float:
        p = abs(self.state)
        return p * math.log((p + 1e-12) / (abs(prior) + 1e-12))

    def normalize(self, prior: float = 1.0) -> None:
        total = self.ψ(self.state) + self.π(self.state)
        norm = abs(total)
        fe = self.free_energy(prior)
        if norm > 0:
            self.state = (total / norm) * (1 - fe)
        else:
            self.state = prior
        self.history.append(self.state)

    def evolve(self) -> 'Q':
        new = Q(self.ψ(self.state) + self.π(self.state), self.ψ, self.π, self.history.copy())
        new.normalize()
        return new

    def entangle(self, other: 'Q') -> 'Q':
        s = (self.state + other.state) / 2
        ψ_comb = lambda x: (self.ψ(x) + other.ψ(x)) / 2
        π_comb = lambda x: (self.π(x) + other.π(x)) / 2
        return Q(s, ψ_comb, π_comb, self.history + other.history)

    def self_modify(self, modifier: Callable[[Callable, Callable, list[float]], tuple[Callable, Callable]]) -> None:
        self.ψ, self.π = modifier(self.ψ, self.π, self.history)


# ===== Demo routines =====
if __name__ == "__main__":
    # ContinuedFraction demo
    print("π ≈", PI_CF.evaluate(20), " vs math.pi =", math.pi)
    print("e ≈", E_CF.evaluate(4), " vs math.e =", math.e)
    print('e has noncommuting float behavior evident, due to being limited to 4, vs π\'s 20, total-allowed decimal places for calculation.')
    # MorphicComplex demo
    c1 = MorphicComplex(3, 4)
    c2 = MorphicComplex(1, -2)
    print("c1 + c2 =", c1 + c2)
    print("c1 * c2 =", c1 * c2)
    print("Inner ⟨c1|c2⟩ =", c1.inner(c2))
    # Hilbert & QuantumState demo
    hs = HilbertSpace(2)
    psi = QuantumState([MorphicComplex(1,0), MorphicComplex(1,0)], hs)
    print("Measured index:", psi.measure())
    # Q kernel demo
    def novel(x): return x * (1 + 0.1 * math.sin(x))
    def inertia(x): return x * 0.95
    q = Q(1.0, novel, inertia)
    for _ in range(5):
        q = q.evolve()
        print("Q state:", q.state)
