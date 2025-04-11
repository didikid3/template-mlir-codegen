#pragma once

#include <tuple>

#define LOAD_FRAME_PTR_ASM "mov %0, x0"
#define REGISTER_RETURN_ASM "mov $0, {0}"
constexpr const char* REGISTERS[] = {
    /*operands: */ "X1", "X2", "X3", "X4", "X5", "X6", "X7",
    /*cache only: */ "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", /*X16, x17 are our sketch registers */};
constexpr const char* FRAME_REGISTER = "X0";
constexpr int TAIL_CALL_INSTR_SIZE = 4;
constexpr int REGION_CALL_INSTR_SIZE = 4;
constexpr int OPERAND_REGISTERS = 7;
constexpr int CACHE_REGISTERS = sizeof(REGISTERS) / sizeof(char*);
constexpr bool SUPPORT_SWITCH_TABLE = false;
constexpr uint32_t MAX_IMM_SIZE = (1ul << 12) - 1;

template <uint32_t>
struct PatchInfo
{
    template <typename... T>
    static int size(T...) { return 0; }
    template <typename... T>
    static void patch(T...) {}
};

#define INSTRUCTION(NAME, INSTR, RIGHT, LEFT)          \
    constexpr uint32_t NAME = INSTR;                   \
    template <>                                        \
    struct PatchInfo<NAME>                             \
    {                                                  \
        template <typename... T>                       \
        static int size(T...) { return 4; }            \
        static void patch(uint32_t* ptr, uint32_t val) \
        {                                              \
            *ptr = NAME | ((val >> RIGHT) << LEFT);    \
        }                                              \
    };
constexpr uint32_t MOV_CONST_TO_SCRATCH2 = 0;
template <>
struct PatchInfo<MOV_CONST_TO_SCRATCH2>
{
    template <typename... T>
    static int size(T...) { return 16; }
    static void patch(uint32_t* ptr, uint64_t val)
    {
        // movz x16, #imm
        // movk x16, #imm, LSL #16
        // movk x16, #imm, LSL #32
        // movk x16, #imm, LSL #64
        const uint16_t* p = reinterpret_cast<uint16_t*>(&val);
        ptr[0] = 0xd2800010 | (static_cast<uint32_t>(p[0]) << 5);
        ptr[1] = 0xf2a00010 | (static_cast<uint32_t>(p[1]) << 5);
        ptr[2] = 0xf2c00010 | (static_cast<uint32_t>(p[2]) << 5);
        ptr[3] = 0xf2e00010 | (static_cast<uint32_t>(p[3]) << 5);
    }
};
constexpr uint32_t STACK_SLOT_ADDRESS = 1;
template <>
struct PatchInfo<STACK_SLOT_ADDRESS>
{
    static int size(uint32_t offset)
    {
        return offset > MAX_IMM_SIZE ? 8 : 4;
    }
    static void patch(uint32_t* ptr, uint32_t offset)
    {
        // // add x20, x19, #imm
        // ptr[0] = 0x91000274 | ((offset & 0b111111111111) << 10);
        // add x1, x0, #imm
        ptr[0] = 0x91000001 | ((offset & 0b111111111111) << 10);
        if (offset > MAX_IMM_SIZE) {
            ASSURE(offset, <=, (1 << 24) - 1);
            // // add x20, x20, #imm
            // ptr[1] = 0x91000294 | (((offset >> 12) & 0b111111111111) << 10) | (1u << 22);
            // add x1, x1, #imm
            ptr[1] = 0x91000021 | (((offset >> 12) & 0b111111111111) << 10) | (1u << 22);
        }
    }
};
constexpr uint32_t MOV_STACK_TO_SCRATCH_ZX = 2;
template <>
struct PatchInfo<MOV_STACK_TO_SCRATCH_ZX>
{
    static int size(uint32_t offset)
    {
        return offset > MAX_IMM_SIZE ? 8 : 4;
    }
    static void patch(uint32_t* ptr, uint32_t offset)
    {
        if (offset > MAX_IMM_SIZE) {
            ASSURE(offset, <=, (1ul << 17) - 1);
            // movz x16, #imm
            ptr[0] = 0xd2800010 | (offset << 5);
            // // ldrb w17, [x19, x16]
            // ptr[1] = 0x38706a71;
            // ldrb w17, [x0, x16]
            ptr[1] = 0x38706811;
        } else {
            // // ldrb w17, [x19, #imm]
            // ptr[0] = 0x39400271 | (offset << 10);
            // ldrb w17, [x0, #imm]
            ptr[0] = 0x39400011 | (offset << 10);
        }
    }
};
constexpr uint32_t MOV_SCRATCH_TO_STACK_BYTE = 3;
template <>
struct PatchInfo<MOV_SCRATCH_TO_STACK_BYTE>
{
    static int size(uint32_t offset)
    {
        return offset > MAX_IMM_SIZE ? 8 : 4;
    }
    static void patch(uint32_t* ptr, uint32_t offset)
    {
        if (offset > MAX_IMM_SIZE) {
            ASSURE(offset, <=, (1ul << 17) - 1);
            // movz x16, #imm
            ptr[0] = 0xd2800010 | (offset << 5);
            // // strb w17, [x19, x16]
            // ptr[1] = 0x38306a71;
            // strb w17, [x0, x16]
            ptr[1] = 0x38306811;
        } else {
            // // strb w17, [x19, #imm]
            // ptr[0] = 0x39000271 | (offset << 10);
            // strb w17, [x0, #imm]
            ptr[0] = 0x39000011 | (offset << 10);
        }
    }
};
// INSTRUCTION(TBZ_FIRST_ARGUMENT, 0x36000014, 2, 5); // tbz x20, #0, .label
INSTRUCTION(TBZ_FIRST_ARGUMENT, 0x36000001, 2, 5); // tbz x1, #0, .label
// INSTRUCTION(MOV_STACK_TO_SCRATCH, 0xf9400271, 3, 10); // ldr x17, [x19, #imm]
INSTRUCTION(MOV_STACK_TO_SCRATCH, 0xf9400011, 3, 10); // ldr x17, [x0, #imm]
// INSTRUCTION(MOV_SCRATCH_TO_STACK, 0xf9000271, 3, 10); // str x17, [x19, #imm]
INSTRUCTION(MOV_SCRATCH_TO_STACK, 0xf9000011, 3, 10); // str x17, [x0, #imm]
INSTRUCTION(SUB_SCRATCH, 0xd1000231, 0, 10); // sub x17, x17, #imm
INSTRUCTION(CMP_SCRATCH, 0xf100023f, 0, 10); // cmp x17, #imm
INSTRUCTION(ADD_SCRATCH_1_2, 0x0b100231, 0, 0); // add x17, x17, x16
INSTRUCTION(CMP_SCRATCH_1_2, 0xeb10023f, 0, 0); // cmp x17, x16
// INSTRUCTION(SET_FIRST_ARG_LE, 0x9a9fc7f4, 0, 0); // cset x20, le
INSTRUCTION(SET_FIRST_ARG_LE, 0x9a9fc7e1, 0, 0); // cset x1, le
// INSTRUCTION(SET_FIRST_ARG_EQ, 0x9a9f17f4, 0, 0); // cset x20, eq
INSTRUCTION(SET_FIRST_ARG_EQ, 0x9a9f17e1, 0, 0); // cset x1, eq
INSTRUCTION(LEA_SCRATCH, 0x10000011, 0, 0); // adr x17, #imm
INSTRUCTION(MOV_CONST_TO_SCRATCH_BYTE, 0xd2800011, 0, 5); // movz x17, #imm

#define ASM0(ACCESS, CODE)                                           \
    do {                                                             \
        auto* loc = ACCESS.memory(cgDepth, PatchInfo<CODE>::size()); \
        PatchInfo<CODE>::patch(reinterpret_cast<uint32_t*>(loc), 0); \
    } while (false)
#define ASM1(ACCESS, CODE, PATCH)                                         \
    do {                                                                  \
        auto* loc = ACCESS.memory(cgDepth, PatchInfo<CODE>::size(PATCH)); \
        PatchInfo<CODE>::patch(reinterpret_cast<uint32_t*>(loc), PATCH);  \
    } while (false)
#define ASM(...) SELECT_VA(__VA_ARGS__, ASM1, ASM1, ASM0)(jit, __VA_ARGS__)
#define ASM_THIS(...) SELECT_VA(__VA_ARGS__, ASM1, ASM1, ASM0)((*this), __VA_ARGS__)

typedef std::bitset<CACHE_REGISTERS> ClobberMask;
constexpr ClobberMask CLOBBER_ALL((1u << CACHE_REGISTERS) - 1);
constexpr ClobberMask CLOBBER_NONE(0b0000000000);

// as the scratch registers are not part of the cached ones,
// clobbering is just like clobbering none
constexpr ClobberMask CLOBBER_SCRATCH_1 = CLOBBER_NONE;
constexpr ClobberMask CLOBBER_SCRATCH_2 = CLOBBER_NONE;
static ClobberMask CLOBBER_SCRATCH = CLOBBER_SCRATCH_1 | CLOBBER_SCRATCH_2;
