/* 8-bit Posit Epistemic ISA - ANSI C99
 * Minimal thermodynamic computing primitives
 * No external dependencies, pure bit manipulation
 <a href="https://github.com/MOONLAPSED/SmallBang">Morphological Source Code: SmallBang</a> © 2025 by Moonlapsed:MOONLAPSED@GMAIL.COM CC BY 
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>  /* for size_t and NULL */
#include <stdlib.h>  /* for malloc/free */

/* 8-bit posit format: sign(1) + regime(variable) + fraction(remaining)
 * No exponent field for maximum simplicity
 */
typedef uint8_t posit8_t;

/* Epistemic set with null-byte membership delimiters */
typedef struct {
    posit8_t* elements;
    uint8_t* membership_map;  /* 0xFF = element, 0x00 = delimiter */
    size_t capacity;
    size_t count;
} episet_t;

/* Thermodynamic cost accounting */
typedef struct {
    uint32_t landauer_ops;    /* Count of irreversible operations */
    uint32_t quine_moments;   /* Count of epistemic commitments */
    double thermal_cost;      /* Accumulated kT ln(2) */
} thermal_ledger_t;

/* Core posit8 constants */
#define POSIT8_ZERO     0x00
#define POSIT8_ONE      0x40  /* regime=01, frac=000000 */
#define POSIT8_INF      0x80  /* NaR (Not a Real) */
#define POSIT8_SIGN_BIT 0x80

/* Extract sign bit */
static inline bool posit8_sign(posit8_t p) {
    return (p & POSIT8_SIGN_BIT) != 0;
}

/* Two's complement for negative posits */
static inline posit8_t posit8_negate(posit8_t p) {
    return (p == POSIT8_INF) ? POSIT8_INF : (~p + 1);
}

/* Count leading bits in regime (after sign) */
static inline int posit8_regime_length(posit8_t p) {
    if (p == POSIT8_ZERO || p == POSIT8_INF) return 0;
    
    uint8_t abs_p = posit8_sign(p) ? posit8_negate(p) : p;
    uint8_t regime_bits = abs_p << 1;  /* Skip sign bit */
    
    /* Count consecutive identical bits starting from MSB */
    int len = 1;
    bool first_bit = (regime_bits & 0x80) != 0;
    regime_bits <<= 1;
    
    for (int i = 1; i < 7; i++) {
        bool current_bit = (regime_bits & 0x80) != 0;
        if (current_bit != first_bit) break;
        len++;
        regime_bits <<= 1;
    }
    
    return len;
}

/* Extract regime value (k) */
static inline int posit8_regime_value(posit8_t p) {
    if (p == POSIT8_ZERO || p == POSIT8_INF) return 0;
    
    uint8_t abs_p = posit8_sign(p) ? posit8_negate(p) : p;
    uint8_t regime_bits = abs_p << 1;  /* Skip sign bit */
    bool regime_sign = (regime_bits & 0x80) != 0;
    int regime_len = posit8_regime_length(p);
    
    return regime_sign ? (regime_len - 1) : -(regime_len);
}

/* Extract fraction bits */
static inline uint8_t posit8_fraction(posit8_t p) {
    if (p == POSIT8_ZERO || p == POSIT8_INF) return 0;
    
    uint8_t abs_p = posit8_sign(p) ? posit8_negate(p) : p;
    int regime_len = posit8_regime_length(p);
    int frac_bits = 7 - regime_len - 1;  /* 7 bits after sign, -1 for regime terminator */
    
    if (frac_bits <= 0) return 0;
    
    uint8_t frac_mask = (1 << frac_bits) - 1;
    return abs_p & frac_mask;
}

/* Convert posit8 to approximate double (for debugging/interfacing) */
double posit8_to_double(posit8_t p) {
    if (p == POSIT8_ZERO) return 0.0;
    if (p == POSIT8_INF) return 1.0/0.0;  /* inf */
    
    bool sign = posit8_sign(p);
    int regime = posit8_regime_value(p);
    uint8_t frac = posit8_fraction(p);
    int frac_bits = 7 - posit8_regime_length(p) - 1;
    
    /* useed = 2^2 = 4 for 8-bit posit with es=0 */
    double scale = 1.0;
    for (int i = 0; i < (regime < 0 ? -regime : regime); i++) {
        scale = (regime < 0) ? scale / 4.0 : scale * 4.0;
    }
    
    /* Add implicit leading 1 and fraction */
    double mantissa = 1.0;
    if (frac_bits > 0) {
        mantissa += (double)frac / (1 << frac_bits);
    }
    
    double result = scale * mantissa;
    return sign ? -result : result;
}

/* EPISTEMIC OPERATIONS */

/* Record a Quine moment - irreversible epistemic commitment */
void record_quine_moment(thermal_ledger_t* ledger) {
    ledger->quine_moments++;
    ledger->landauer_ops++;
    ledger->thermal_cost += 1.0;  /* kT ln(2) = 1 unit */
}

/* Epistemic addition with thermodynamic accounting */
posit8_t posit8_add_epistemic(posit8_t a, posit8_t b, thermal_ledger_t* ledger) {
    record_quine_moment(ledger);
    
    /* Simplified addition - would need full implementation */
    if (a == POSIT8_ZERO) return b;
    if (b == POSIT8_ZERO) return a;
    if (a == POSIT8_INF || b == POSIT8_INF) return POSIT8_INF;
    
    /* TODO: Implement proper posit addition algorithm */
    /* For now, return approximate result */
    return POSIT8_ONE;  /* Placeholder */
}

/* Initialize epistemic set */
episet_t* episet_create(size_t capacity) {
    episet_t* set = malloc(sizeof(episet_t));
    if (!set) return NULL;
    
    set->elements = malloc(capacity * sizeof(posit8_t));
    set->membership_map = malloc(capacity * sizeof(uint8_t));
    set->capacity = capacity;
    set->count = 0;
    
    if (!set->elements || !set->membership_map) {
        free(set->elements);
        free(set->membership_map);
        free(set);
        return NULL;
    }
    
    return set;
}

/* Add element to epistemic set with null-byte delimiter */
bool episet_add(episet_t* set, posit8_t element, thermal_ledger_t* ledger) {
    if (set->count >= set->capacity - 1) return false;
    
    record_quine_moment(ledger);  /* Set membership is epistemic commitment */
    
    set->elements[set->count] = element;
    set->membership_map[set->count] = 0xFF;  /* Element marker */
    set->count++;
    
    /* Add null delimiter */
    set->elements[set->count] = POSIT8_ZERO;
    set->membership_map[set->count] = 0x00;  /* Delimiter marker */
    set->count++;
    
    return true;
}

/* Bytecode VM opcodes for epistemic operations */
typedef enum {
    OP_POSIT_ADD = 0x01,
    OP_POSIT_MUL = 0x02,
    OP_SET_UNION = 0x10,
    OP_SET_INTERSECT = 0x11,
    OP_QUINE_MOMENT = 0x20,
    OP_THERMAL_READ = 0x30,
    OP_HALT = 0xFF
} epistemic_opcode_t;

/* Minimal bytecode interpreter */
typedef struct {
    uint8_t* code;
    size_t pc;           /* program counter */
    posit8_t stack[256]; /* operand stack */
    int sp;              /* stack pointer */
    thermal_ledger_t ledger;
} epistemic_vm_t;

/* Execute one bytecode instruction */
bool vm_step(epistemic_vm_t* vm) {
    if (vm->pc >= 65536) return false;  /* code limit */
    
    epistemic_opcode_t op = (epistemic_opcode_t)vm->code[vm->pc++];
    
    switch (op) {
        case OP_POSIT_ADD:
            if (vm->sp < 2) return false;
            vm->sp--;
            vm->stack[vm->sp-1] = posit8_add_epistemic(
                vm->stack[vm->sp-1], 
                vm->stack[vm->sp], 
                &vm->ledger
            );
            break;
            
        case OP_QUINE_MOMENT:
            record_quine_moment(&vm->ledger);
            break;
            
        case OP_HALT:
            return false;
            
        default:
            return false;  /* Invalid opcode */
    }
    
    return true;
}

/* Initialize the epistemic virtual machine */
epistemic_vm_t* vm_create(uint8_t* bytecode) {
    epistemic_vm_t* vm;
    vm = (epistemic_vm_t*)malloc(sizeof(epistemic_vm_t));
    if (!vm) return NULL;
    
    vm->code = bytecode;
    vm->pc = 0;
    vm->sp = 0;
    vm->ledger.landauer_ops = 0;
    vm->ledger.quine_moments = 0;
    vm->ledger.thermal_cost = 0.0;
    
    return vm;
}