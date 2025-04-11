#pragma once

#define ASM0(ACCESS, CODE)                          \
    do {                                            \
        ACCESS.insert(cgDepth, sizeof(CODE), CODE); \
    } while (false)
#define ASM1(ACCESS, CODE, PATCH)                   \
    do {                                            \
        ACCESS.insert(cgDepth, sizeof(CODE), CODE); \
        decltype(PATCH) tmp = PATCH;                \
        ACCESS.insert(cgDepth, sizeof(tmp), &tmp);  \
    } while (false)
#define ASM(...) SELECT_VA(__VA_ARGS__, ASM1, ASM1, ASM0)(jit, __VA_ARGS__)
#define ASM_THIS(...) SELECT_VA(__VA_ARGS__, ASM1, ASM1, ASM0)((*this), __VA_ARGS__)

constexpr uint8_t MOV_CONST_TO_SCRATCH_BYTE[] = {0xb0}; // mov al, 0x00
constexpr uint8_t MOV_CONST_TO_SCRATCH[] = {0x48, 0xb8}; // mov rax, 0x0000
constexpr uint8_t MOV_CONST_TO_SCRATCH2[] = {0x48, 0xb9}; // mov rcx, 0x0000
// constexpr uint8_t MOV_STACK_TO_SCRATCH[] = {0x49, 0x8b, 0x85}; // mov rax, [r13 + 0x0000]
constexpr uint8_t MOV_STACK_TO_SCRATCH[] = {0x48, 0x8b, 0x87}; // mov rax, [rdi + 0x0000]
// constexpr uint8_t MOV_STACK_TO_SCRATCH_ZX[] = {0x49, 0x0f, 0xb6, 0x85}; // movzx rax, [r13 + 0x0000]
constexpr uint8_t MOV_STACK_TO_SCRATCH_ZX[] = {0x48, 0x0f, 0xb6, 0x87}; // movzx rax, [rdi + 0x0000]
// constexpr uint8_t MOV_SCRATCH_TO_STACK[] = {0x49, 0x89, 0x85}; // mov [r13 + 0x0000], rax
constexpr uint8_t MOV_SCRATCH_TO_STACK[] = {0x48, 0x89, 0x87}; // mov [rdi + 0x0000], rax
// constexpr uint8_t MOV_SCRATCH_TO_STACK_BYTE[] = {0x41, 0x88, 0x85}; // mov [r13 + 0x0000], al
constexpr uint8_t MOV_SCRATCH_TO_STACK_BYTE[] = {0x88, 0x87}; // mov [rdi + 0x0000], al
// constexpr uint8_t TEST_FIRST_ARGUMENT[] = {0x40, 0xf6, 0xc5, 0x01}; // test bpl, 1
constexpr uint8_t TEST_FIRST_ARGUMENT[] = {0x40, 0xf6, 0xc6, 0x01}; // test sil, 1
// constexpr uint8_t STACK_SLOT_ADDRESS[] = {0x49, 0x8d, 0xad}; // lea rbp, [r13 + 0x0000]
constexpr uint8_t STACK_SLOT_ADDRESS[] = {0x48, 0x8d, 0xb7}; // lea rsi, [rdi + 0x0000]
constexpr uint8_t SUB_SCRATCH[] = {0x48, 0x2d}; // sub rax, 0x0000
constexpr uint8_t ADD_SCRATCH_1_2[] = {0x48, 0x01, 0xc8}; // add rax, rcx
constexpr uint8_t CMP_SCRATCH[] = {0x48, 0x3d}; // cmp rax, 0x0000
constexpr uint8_t CMP_SCRATCH_1_2[] = {0x48, 0x39, 0xc8}; // cmp rax, rcx
constexpr uint8_t LEA_SCRATCH[] = {0x48, 0x8d, 0x05}; // lea rax, [rip + 0x0000]
constexpr uint8_t LEA_SCRATCH2[] = {0x48, 0x8d, 0x0d}; // lea rcx, [rip + 0x0000]
constexpr uint8_t MOV_OFFSETED_TO_SCRATCH[] = {0x48, 0x63, 0x04, 0x81}; // movsxd rax, dword ptr [rcx + rax*4]
constexpr uint8_t JMP_SCRATCH[] = {0xff, 0xe0}; // jmp *rax

#define LOAD_FRAME_PTR_ASM "mov %%rdi,%0"
#define REGISTER_RETURN_ASM "mov %{0}, $0"
// constexpr const char* REGISTERS[] = {"RBP", "R12", "RBX", "R14", "RSI", "RDI", "R8", "R9", "R15", "RDX"};
constexpr const char* REGISTERS[] = {
    /*operands: */ "RSI", "RDX", "RCX", "R8", "R9",
    /*cache only: */ "R10", "R11", "R12", "RAX", "R13", "R14", "R15", "RBX"};

constexpr int TAIL_CALL_INSTR_SIZE = 5;
constexpr int REGION_CALL_INSTR_SIZE = 5;
constexpr int OPERAND_REGISTERS = 4;
constexpr int CACHE_REGISTERS = sizeof(REGISTERS) / sizeof(char*);

constexpr bool SUPPORT_SWITCH_TABLE = true;
constexpr uint32_t MAX_IMM_SIZE = INT32_MAX;

typedef std::bitset<CACHE_REGISTERS> ClobberMask;
constexpr ClobberMask CLOBBER_ALL((1u << CACHE_REGISTERS) - 1);
constexpr ClobberMask CLOBBER_NONE(0b0000000000);

constexpr ClobberMask CLOBBER_SCRATCH_1 = CLOBBER_NONE;
constexpr ClobberMask CLOBBER_SCRATCH_2 = 0b00100;
static ClobberMask CLOBBER_SCRATCH = CLOBBER_SCRATCH_1 | CLOBBER_SCRATCH_2;
