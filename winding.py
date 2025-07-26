from typing import NamedTuple
from functools import reduce
from math import log2
import math
from functools import reduce
from operator import xor

# —— 1. Simple 8-bit float emulation —— #
def quantize_8bit(value, bits_exp=3, bits_man=4):
    """
    Pack a positive float into 1-bit sign | bits_exp exponent | bits_man mantissa
    using a bias of (2^(bits_exp-1)-1). Returns the raw 8-bit integer.
    """
    assert value >= 0, "Only positive values in this demo"
    bias = (1 << (bits_exp - 1)) - 1

    # handle zero
    if value == 0:
        return 0

    # find exponent e so 1.0 <= m < 2.0
    e = math.floor(math.log(value, 2))
    m = value / (2 ** e)

    # clamp exponent
    e_biased = max(0, min((1 << bits_exp) - 1, e + bias))

    # mantissa fractional part
    man = round((m - 1.0) * (1 << bits_man))
    man = max(0, min((1 << bits_man) - 1, man))

    return (e_biased << bits_man) | man  # sign=0 implicitly in high bit

def popcount(x): return bin(x).count('1')

class ByteWord(NamedTuple):
    raw: int  # 0–255

    @property
    def C(self): return (self.raw >> 7) & 1
    @property
    def V(self): return (self.raw >> 4) & 0b111
    @property
    def T(self): return self.raw & 0b1111

    def is_null(self): return self.C == 0 and self.V == 0 and self.T == 0

    def winding_vector(self):
        w1 = (self.T >> 1) & 1
        w2 = self.T & 1
        return (w1, w2)

    def entropy(self):
        """Algorithmic entropy via compression proxy."""
        return log2(len(bin(self.raw).replace("0b", "").rstrip("0") or "1"))

    def evolve_with(self, other):
        """Sparse-unitary evolution: XOR-only."""
        return ByteWord(self.raw ^ other.raw)
    def __repr__(self):
        return (f"ByteWord(raw=0x{self.raw:02X}, "
                f"C={self.C}, V={self.V:03b}, T={self.T:04b})")

    def xor_popcount(self, mask):
        """
        Self-adjoint XOR-popcount operator:
        popcount(raw XOR mask), returns new integer state.
        """
        x = self.raw ^ (mask & 0xFF)
        return bin(x).count("1")
    def deputize(self):
        """Lazy evaluation: promote next V bit if C == 0."""
        if self.C == 1:
            return self
        v_bits = [(self.V >> i) & 1 for i in reversed(range(3))]
        for i, bit in enumerate(v_bits):
            if bit:
                # Promote this bit to C, zero rest
                new_raw = (1 << 7) | (bit << (6 - i)) | (self.T & 0b1111)
                return ByteWord(new_raw)
        return ByteWord(0)  # null-state

    def __str__(self):
        return f"ByteWord(raw=0b{self.raw:08b}, C={self.C}, V={self.V}, T={self.T})"

# --- Example ---


# —— 3. Build “arbitrarily small transcendental objects” —— #
pi_raw = quantize_8bit(math.pi)    # ≈ 3.14159 → small 8-bit float
e_raw  = quantize_8bit(math.e)     # ≈ 2.71828 → small 8-bit float

bw_pi = ByteWord(pi_raw)
bw_e  = ByteWord(e_raw)

print("Quantized constants as ByteWords:")
print(" ", bw_pi)
print(" ", bw_e)

# —— 4. Morphogenetic step: XOR-popcount across an array —— #
# Let's form a sparse bit-vector over F₂ from our two consts
sparse_vec = [bw_pi.raw, bw_e.raw, bw_pi.raw ^ bw_e.raw]

# Define a simple mask (e.g. 0b10110010)
mask = 0b10110010

print("\nApplying self-adjoint XOR-popcount operator:")
for i, word in enumerate(sparse_vec, 1):
    count = ByteWord(word).xor_popcount(mask)
    print(f"  word[{i}] = 0x{word:02X}, popcount(word⊕mask) = {count}")

# —— 5. A very crude “non-Markovian” lift: cumulative XOR cascade —— #
print("\nNon-Markovian XOR cascade (cumulative):")
cascade = reduce(lambda acc, w: acc ^ w, sparse_vec, 0)
bw_cascade = ByteWord(cascade)
print("  cascade raw =", f"0x{cascade:02X}", "→", bw_cascade)

bw1 = ByteWord(0b10111011)  # C=1, V=011, T=1011
bw2 = ByteWord(0b01010101)  # C=0, V=101, T=0101

print("Original:", bw1)
print("Winding vector:", bw1.winding_vector())
print("Entropy:", round(bw1.entropy(), 2))

print("\nEvolved state:")
evolved = bw1.evolve_with(bw2)
print(evolved)

print("\nDeputized bw2:")
print(bw2.deputize())
