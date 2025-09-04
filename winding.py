from __future__ import annotations
# <a href="https://github.com/MOONLAPSED/SmallBang">Morphological Source Code: SmallBang</a> © 2025 by Moonlapsed:MOONLAPSED@GMAIL.COM CC BY 
from dataclasses import dataclass
from abc import ABC, abstractmethod
import hashlib
import re
import enum
from enum import Enum, IntEnum, StrEnum, IntFlag, auto
from typing import TypeVar, Generic, List, Tuple, Callable, Dict, Set, Type, Any, NamedTuple
from functools import reduce
from math import log2
import math
from datetime import datetime, timedelta
from functools import reduce
from operator import xor
import ctypes
"""Core Operators:

Composition (@): Sequential application of operations
Tensor Product (*): Parallel combination of operations
Direct Sum (+): Alternative pathways of computation
Adjoint (†): Reversal/dual of operations

Algebraic Properties:

Associativity: (A @ B) @ C = A @ (B @ C)
Distributivity: A * (B + C) = (A * B) + (A * C)
Adjoint rules: (A @ B)† = B† @ A†"""

T = TypeVar('T')  # Type structure
V = TypeVar('V')  # Value space
R = TypeVar('R')  # Result type

# Covariant/contravariant type variables for advanced type modeling
T_co = TypeVar('T_co', covariant=True)  # Covariant Type structure
V_co = TypeVar('V_co', covariant=True)  # Covariant Value space
C_co = TypeVar('C_co', bound=Callable[..., Any], covariant=True)  # Covariant Control space

T_anti = TypeVar('T_anti', contravariant=True)  # Contravariant Type structure
V_anti = TypeVar('V_anti', contravariant=True)  # Contravariant Value space
C_anti = TypeVar('C_anti', bound=Callable[..., Any], contravariant=True)  # Contravariant Computation space

# BYTE type for byte-level operations
BYTE = TypeVar("BYTE", bound="BYTE_WORD")

# Core Enums and Constants

class Morphology(enum.Enum):
    """
    Represents the floor morphic state of a BYTE_WORD.
    
    C = 0: Floor morphic state (stable, low-energy)
    C = 1: Dynamic or high-energy state
    
    The control bit (C) indicates whether other holoicons can point to this holoicon:
    - DYNAMIC (1): Other holoicons CAN point to this holoicon
    - MORPHIC (0): Other holoicons CANNOT point to this holoicon
    
    This ontology maps to thermodynamic character: intensive & extensive.
    A 'quine' (self-instantiated runtime) is a low-energy, intensive system,
    while a dynamic holoicon is a high-energy, extensive system inherently
    tied to its environment.
    """
    MORPHIC = 0      # Stable, low-energy state
    DYNAMIC = 1      # High-energy, potentially transformative state
    
    # Fundamental computational orientation and symmetry
    MARKOVIAN = -1    # Forward-evolving, irreversible
    NON_MARKOVIAN = math.e  # Reversible, with memory


class QuantumState(enum.Enum):
    """Represents a computational state that tracks its quantum-like properties."""
    SUPERPOSITION = 1   # Known by handle only
    ENTANGLED = 2       # Referenced but not loaded
    COLLAPSED = 4       # Fully materialized
    DECOHERENT = 8      # Garbage collected


class WordSize(enum.IntEnum):
    """Standardized computational word sizes"""
    BYTE = 1     # 8-bit
    SHORT = 2    # 16-bit
    INT = 4      # 32-bit
    LONG = 8     # 64-bit

# Complex Number with Morphic Properties

class MorphicComplex:
    """Represents a complex number with morphic properties."""
    def __init__(self, real: float, imag: float):
        self.real = real
        self.imag = imag
    
    def conjugate(self) -> 'MorphicComplex':
        """Return the complex conjugate."""
        return MorphicComplex(self.real, -self.imag)
    
    def __add__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(self.real + other.real, self.imag + other.imag)
    
    def __mul__(self, other: 'MorphicComplex') -> 'MorphicComplex':
        return MorphicComplex(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    
    def __repr__(self) -> str:
        if self.imag == 0:
            return f"{self.real}"
        elif self.real == 0:
            return f"{self.imag}j"
        else:
            sign = "+" if self.imag >= 0 else ""
            return f"{self.real}{sign}{self.imag}j"

# Base Byte-level Representation

class BYTE_WORD:
    """Basic 8-bit word representation."""
    def __init__(self, value: int = 0):
        if not 0 <= value <= 255:
            raise ValueError("BYTE_WORD value must be between 0 and 255")
        self.value = value

    def __repr__(self) -> str:
        return f"BYTE_WORD(value={self.value:08b})"
    

class WordAlignment(IntEnum):
    UNALIGNED = 1
    WORD = 2
    DWORD = 4
    QWORD = 8
    CACHE_LINE = 64
    PAGE = 4096

class TorusWinding(IntEnum):
    """
    Binary torus winding states (2 bits):
        NULL  = 0b00 → no twist
        W1    = 0b01 → twist on axis-1
        W2    = 0b10 → twist on axis-2
        W12   = 0b11 → twist on both axes
    """
    NULL = 0b00
    W1   = 0b01
    W2   = 0b10
    W12  = 0b11


class ByteWord:
    """
    An 8-bit Morphological Atom, split into:

      < C | V  V  V | a  a | w  w >
       └─┬─┘  └────┘  └┬┘  └┬┘
        captain  value aux  winding

    C (1b), VVV (3b), A1 (1b), A2 (1b), ww (2b)
    """
    __slots__ = ("raw", "captain", "value", "aux1", "aux2", "torus_raw", "w1", "w2", "winding")

    def __init__(self, raw: int):
        if not 0 <= raw <= 0xFF:
            raise ValueError("ByteWord must be in 0–255 range.")
        self.raw = raw
        self._parse_fields()

    def _parse_fields(self) -> None:
        self.captain   = bool(self.raw & 0b1000_0000)
        self.value     = (self.raw & 0b0111_0000) >> 4
        self.aux1      = bool(self.raw & 0b0000_1000)
        self.aux2      = bool(self.raw & 0b0000_0100)
        self.torus_raw =  self.raw & 0b0000_0011
        self.w1        = bool(self.torus_raw & 0b01)
        self.w2        = bool(self.torus_raw & 0b10)
        self.winding   = TorusWinding(self.torus_raw)

    @staticmethod
    def build(captain: int, value: int, aux1: int, aux2: int, winding: TorusWinding) -> "ByteWord":
        raw = ((captain & 1) << 7) | ((value & 0b111) << 4) | ((aux1 & 1) << 3) | ((aux2 & 1) << 2) | (int(winding) & 0b11)
        return ByteWord(raw)

    def deputize(self) -> "ByteWord":
        """Promote highest V-bit to captain if C=0. Leaves ww/A1/A2 intact."""
        if self.captain:
            return self
        new_raw = ((self.value & 0b100) << 4) | ((self.value & 0b011) << 5) \
                  | (int(self.aux1) << 3) | (int(self.aux2) << 2) \
                  | self.torus_raw
        return ByteWord(new_raw)

    def xor_cascade(self, other: "ByteWord") -> "ByteWord":
        """Unitary XOR on winding bits, preserving high fields and aux bits."""
        nw1 = self.w1 ^ other.w1
        nw2 = self.w2 ^ other.w2
        new_torus = (int(nw2) << 1) | int(nw1)
        new_raw = (self.raw & 0b1111_1100) | new_torus
        return ByteWord(new_raw)

    def is_null(self) -> bool:
        return (not self.captain) and (self.winding == TorusWinding.NULL)

    def __repr__(self) -> str:
        return (
            f"ByteWord(C={int(self.captain)}, V={self.value:03b}, "
            f"A1={int(self.aux1)}, A2={int(self.aux2)}, "
            f"w1={int(self.w1)}, w2={int(self.w2)}, raw=0x{self.raw:02x})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic tokenization for ANSI/Latin-1
# ──────────────────────────────────────────────────────────────────────────────

_ANSI_RE = re.compile(
    r"""(
        [A-Za-z_][A-Za-z0-9_]* |     # ident
        \d+\.\d+ |                   # float-like (still treated as bytes; no FP math)
        \d+ |                        # int-like
        \s+ |                        # whitespace
        .                            # single char fallback
    )""",
    re.VERBOSE,
)

def _to_latin1_bytes(s: str) -> bytes:
    """Strict Latin-1 to guarantee 0..255 domain. Raises on non-ANSI."""
    return s.encode("latin-1", errors="strict")

def tokenize_ansi(s: str) -> List[bytes]:
    """Split into byte-tokens while preserving whitespace and punctuation."""
    tokens: List[bytes] = []
    for m in _ANSI_RE.finditer(s):
        tok = m.group(0)
        tokens.append(_to_latin1_bytes(tok))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# User-definable semantics (influence aux bits, captain, winding)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TokenClass:
    name: str
    code3: int  # 0..7 → goes into VVV


def default_token_classifier(tok: bytes) -> TokenClass:
    """Map token → small class set (VVV). Deterministic & reversible enough for analysis."""
    t = tok
    if not t:
        return TokenClass("empty", 0)
    if t.isspace():
        return TokenClass("space", 1)
    if t.isalpha() or (t[:1].isalpha() and all(ch in b"_0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for ch in t)):
        return TokenClass("ident", 2)
    if all(48 <= b <= 57 for b in t):  # digits
        return TokenClass("int", 3)
    # float-ish (no FP operations — just classification)
    if b"." in t and all((48 <= b <= 57) or b == 46 for b in t):
        return TokenClass("float", 4)
    # groups of punctuation
    if all(not chr(b).isalnum() and not chr(b).isspace() for b in t):
        return TokenClass("punct", 5)
    # fallback: raw bytes chunk
    return TokenClass("bytes", 6)


# Hooks decide A1/A2/C/ww given token metadata + digest stream position/state.
AuxHook = Callable[[TokenClass, bytes, int, int], Tuple[int, int]]          # -> (A1, A2)
CaptainHook = Callable[[TokenClass, bytes, int], int]                       # -> C
WindingHook = Callable[[TokenClass, bytes, int, int], TorusWinding]         # -> ww


def default_aux_hook(cls: TokenClass, tok: bytes, ix_in_token: int, global_ix: int) -> Tuple[int, int]:
    """
    Example: A1 = parity of token length, A2 = parity of byte popcount at this position.
    Pure integer; stable.
    """
    a1 = (len(tok) & 1)
    b = tok[ix_in_token] if ix_in_token < len(tok) else 0
    pop = (b & 1) + ((b >> 1) & 1) + ((b >> 2) & 1) + ((b >> 3) & 1) + ((b >> 4) & 1) + ((b >> 5) & 1) + ((b >> 6) & 1) + ((b >> 7) & 1)
    a2 = pop & 1
    return a1, a2


def default_captain_hook(cls: TokenClass, tok: bytes, global_ix: int) -> int:
    """Make first byte of each token a captain; others not."""
    return 1 if global_ix == 0 else 0


def default_winding_hook(cls: TokenClass, tok: bytes, ix_in_token: int, rolling_state: int) -> TorusWinding:
    """
    Simple rolling state → winding mapping.
    rolling_state is a 2-bit register updated by the precompiler.
    """
    return TorusWinding(rolling_state & 0b11)


# ──────────────────────────────────────────────────────────────────────────────
# Precompiler
# ──────────────────────────────────────────────────────────────────────────────

class MorphoPrecompiler:
    """
    Deterministic, float-free mapping:
        ANSI/Latin-1 string  ──tokenize──> tokens ──SHAKE-256 XOF──> ByteWords
    * Each token emits a 1-ByteWord header then a body stream derived from XOF.
    * User semantics (hooks) adjust A1/A2/C/ww without breaking determinism.
    """

    def __init__(
        self,
        *,
        salt: bytes = b"MORPHOSEMANTIC-1",
        classifier: Callable[[bytes], TokenClass] = default_token_classifier,
        aux_hook: AuxHook = default_aux_hook,
        captain_hook: CaptainHook = default_captain_hook,
        winding_hook: WindingHook = default_winding_hook,
        header_marker_value: int = 0b111,  # VVV for header ByteWord
    ):
        self.salt = bytes(salt)
        self.classifier = classifier
        self.aux_hook = aux_hook
        self.captain_hook = captain_hook
        self.winding_hook = winding_hook
        self.header_marker_value = header_marker_value & 0b111

    @staticmethod
    def _shake(seed: bytes) -> hashlib._shake256:  # type: ignore[attr-defined]
        return hashlib.shake_256(seed)

    @staticmethod
    def _u16(b: bytes) -> int:
        return (b[0] << 8) | b[1]

    def _token_digest_stream(self, tok: bytes, token_index: int, rolling_state: int) -> Iterable[int]:
        """
        XOF stream → infinite bytes for this token. We use:
            SHAKE-256(salt || len(tok) || tok || token_index || rolling_state)
        """
        sep = b"|"
        idx_bytes = token_index.to_bytes(4, "big", signed=False)
        rs_bytes  = rolling_state.to_bytes(2, "big", signed=False)
        seed = self.salt + sep + len(tok).to_bytes(2, "big") + sep + tok + sep + idx_bytes + sep + rs_bytes
        sh = self._shake(seed)
        while True:
            # Draw 16 bytes at a time for efficiency; yield byte by byte
            block = sh.digest(16)
            for b in block:
                yield b

    def compile(self, text: str) -> List[ByteWord]:
        """
        Compile Latin-1/ANSI string into a ByteWord tape.
        Deterministic, portable, pure integer (no float).
        """
        tokens = tokenize_ansi(text)  # bytes[]
        tape: List[ByteWord] = []

        # 2-bit rolling state drives default winding, updated deterministically
        rolling_state = 0

        for t_index, tok in enumerate(tokens):
            cls = self.classifier(tok)

            # ── Header ByteWord (C=1, VVV=header_marker_value, A1/A2 captured from hooks, ww from rolling state)
            a1_h, a2_h = self.aux_hook(cls, tok, 0, 0)
            ww_h = self.winding_hook(cls, tok, 0, rolling_state)
            header = ByteWord.build(
                captain=1,
                value=self.header_marker_value,
                aux1=a1_h,
                aux2=a2_h,
                winding=ww_h
            )
            tape.append(header)

            # ── Body stream (one ByteWord per digest byte)
            #     Map digest byte → fields:
            #       VVV = (b >> 5)
            #       A1  = ((b >> 4) & 1) xor (cls.code3 & 1)
            #       A2  = ((b >> 3) & 1)
            #       ww  = (rolling_state ^ (b & 0b11)) & 0b11
            #       C   = captain_hook(...) (1 for first body byte if you want), else 0
            digest = self._token_digest_stream(tok, t_index, rolling_state)
            # emit exactly len(tok) body words (1:1 makes downstream indexing simple)
            for ix, b in zip(range(len(tok)), digest):
                vvv = (b >> 5) & 0b111
                a1_body, a2_body = self.aux_hook(cls, tok, ix, ix)
                a1 = a1_body ^ (cls.code3 & 1) ^ ((b >> 4) & 1)
                a2 = a2_body ^ ((b >> 3) & 1)
                ww  = (rolling_state ^ (b & 0b11)) & 0b11
                c   = self.captain_hook(cls, tok, ix)

                bw = ByteWord.build(
                    captain=c,
                    value=vvv,
                    aux1=a1 & 1,
                    aux2=a2 & 1,
                    winding=TorusWinding(ww)
                )
                tape.append(bw)

                # Update rolling state (pure integer, small & fast):
                # mix in token class, byte value parity, and position
                parity = (bin(b).count("1") & 1)
                rolling_state = (rolling_state + cls.code3 + parity + (ix & 3)) & 0b11

            # Small, deterministic state bump per token boundary:
            rolling_state = (rolling_state + (len(tok) & 3)) & 0b11

        return tape


# ──────────────────────────────────────────────────────────────────────────────
# Disassembler (debug/inspection)
# ──────────────────────────────────────────────────────────────────────────────

def pretty_tape(tape: List[ByteWord]) -> str:
    out: List[str] = []
    for i, bw in enumerate(tape):
        out.append(
            f"{i:04d}: C={int(bw.captain)} VVV={bw.value} A1={int(bw.aux1)} A2={int(bw.aux2)} ww={bw.winding.name} raw=0x{bw.raw:02x}"
        )
    return "\n".join(out)

#############################################
# PyWord: Aligned Memory Management
#############################################


class PyWord(Generic[T]):
    """
    [[PyWord]] represents a word-sized value optimized for CPython.
    It manages alignment according to the system's memory model and
    provides conversion between Python and C types.
    """
    __slots__ = ('_value', '_alignment', '_arch', '_mem_model')

    def __init__(self,
                 value: Union[int, bytes, bytearray, array.array],
                 alignment: WordAlignment = WordAlignment.WORD):
        self._mem_model = MemoryModel.get_system_info()
        self._arch = ProcessorArchitecture.current()
        self._alignment = alignment
        aligned_size = self._calculate_aligned_size()
        self._value = self._allocate_aligned(aligned_size)
        self._store_value(value)

    def _calculate_aligned_size(self) -> int:
        base_size = max(self._mem_model.word_size,
                        ctypes.sizeof(ctypes.c_size_t))
        return (base_size + self._alignment - 1) & ~(self._alignment - 1)

    def _allocate_aligned(self, size: int) -> ctypes.Array:
        class AlignedArray(ctypes.Structure):
            _pack_ = self._alignment
            _fields_ = [("data", ctypes.c_char * size)]
        return AlignedArray()

    def _store_value(self, value: Union[int, bytes, bytearray, array.array]) -> None:
        if isinstance(value, int):
            if self._arch in (ProcessorArchitecture.X86_64, ProcessorArchitecture.ARM64, ProcessorArchitecture.RISCV64):
                c_val = ctypes.c_uint64(value)
            else:
                c_val = ctypes.c_uint32(value)
            ctypes.memmove(ctypes.addressof(self._value),
                           ctypes.addressof(c_val), ctypes.sizeof(c_val))
        else:
            value_bytes = memoryview(value).tobytes()
            ctypes.memmove(ctypes.addressof(self._value),
                           value_bytes, len(value_bytes))

    def get_raw_pointer(self) -> int:
        return ctypes.addressof(self._value)

    def as_memoryview(self) -> memoryview:
        return memoryview(self._value)

    def as_buffer(self) -> ctypes.Array:
        return (ctypes.c_char * self._calculate_aligned_size()).from_buffer(self._value)

    @property
    def alignment(self) -> int:
        return self._alignment

    @property
    def architecture(self) -> ProcessorArchitecture:
        return self._arch

    def __int__(self) -> int:
        if isinstance(self._value, ctypes.Array):
            return int.from_bytes(self._value.data, sys.byteorder)
        return int.from_bytes(self._value.tobytes(), sys.byteorder)

    def __bytes__(self) -> bytes:
        if isinstance(self._value, ctypes.Array):
            return bytes(self._value.data)
        return self._value.tobytes()


class PyWordCache:
    """LRU Cache for [[PyWord]] objects to minimize allocations."""







class ProcessorFeatures(IntFlag):
    BASIC = auto()
    SSE = auto()
    AVX = auto()
    AVX2 = auto()
    AVX512 = auto()
    NEON = auto()
    SVE = auto()
    RVV = auto()  # RISC-V Vector Extensions
    AMX = auto()  # Advanced Matrix Extensions

    @classmethod
    def detect_features(cls) -> 'ProcessorFeatures':
        features = cls.BASIC
        try:
            if platform.machine().lower() in ('x86_64', 'amd64', 'x86', 'i386'):
                if sys.platform == 'win32':
                    import winreg
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                         r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
                    identifier = winreg.QueryValueEx(
                        key, 'ProcessorNameString')[0]
                else:
                    with open('/proc/cpuinfo') as f:
                        identifier = next(line.split(
                            ':')[1] for line in f if 'model name' in line)
                identifier = identifier.lower()
                if 'avx512' in identifier:
                    features |= cls.AVX512
                if 'avx2' in identifier:
                    features |= cls.AVX2
                if 'avx' in identifier:
                    features |= cls.AVX
                if 'sse' in identifier:
                    features |= cls.SSE
            elif platform.machine().lower().startswith('arm'):
                if sys.platform == 'darwin':  # Apple Silicon
                    features |= cls.NEON
                else:
                    with open('/proc/cpuinfo') as f:
                        content = f.read().lower()
                        if 'neon' in content:
                            features |= cls.NEON
                        if 'sve' in content:
                            features |= cls.SVE
        except Exception:
            pass
        return features


@dataclass
class RegisterSet:
    gp_registers: int
    vector_registers: int
    register_width: int
    vector_width: int

    @classmethod
    def detect_current(cls) -> 'RegisterSet':
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return cls(gp_registers=16, vector_registers=32, register_width=64, vector_width=512)
        elif machine.startswith('arm64'):
            return cls(gp_registers=31, vector_registers=32, register_width=64, vector_width=128)
        else:
            return cls(gp_registers=8, vector_registers=8, register_width=32, vector_width=128)


class ProcessorArchitecture(IntEnum):
    X86 = auto()
    X86_64 = auto()
    ARM32 = auto()
    ARM64 = auto()
    RISCV32 = auto()
    RISCV64 = auto()

    @classmethod
    def current(cls) -> 'ProcessorArchitecture':
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            return cls.X86_64
        elif machine in ('x86', 'i386', 'i686'):
            return cls.X86
        elif machine.startswith('arm'):
            return cls.ARM64 if sys.maxsize > 2**32 else cls.ARM32
        elif machine.startswith('riscv'):
            return cls.RISCV64 if sys.maxsize > 2**32 else cls.RISCV32
        raise ValueError(f"Unsupported architecture: {machine}")


@dataclass
class MemoryModel:
    ptr_size: int = ctypes.sizeof(ctypes.c_void_p)
    word_size: int = ctypes.sizeof(ctypes.c_size_t)
    cache_line_size: int = 64
    page_size: int = 4096

    @classmethod
    def get_system_info(cls) -> 'MemoryModel':
        try:
            with open('/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size') as f:
                cache_line_size = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            cache_line_size = 64
        return cls(
            ptr_size=ctypes.sizeof(ctypes.c_void_p),
            word_size=ctypes.sizeof(ctypes.c_size_t),
            cache_line_size=cache_line_size,
            page_size=cls.page_size
        )


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

# Example ANSI strings (Latin-1). Any non-Latin-1 char will raise.
samples = [
    "SELECT * FROM users WHERE id=42;",
    "sum(x) / 3.14159 + y_2",
    "  \t\n-- punctuation !!! ??? ;;",
]

pre = MorphoPrecompiler()

for s in samples:
    print("=" * 80)
    print("INPUT:", repr(s))
    tape = pre.compile(s)
    print(f"ByteWords emitted: {len(tape)}")
    # Print the first 24 words to keep it readable
    print(pretty_tape(tape[:24]))
