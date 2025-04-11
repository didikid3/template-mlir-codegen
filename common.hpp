#pragma once

#include "llvm/Support/raw_ostream.h"
#include <bitset>
#include <csignal>
#ifdef INSTRUMENT
#include <chrono>
#endif

#define NO_EVICT 1
#define EVICTION_ROUND_ROBIN 2
#ifdef EVICTION_STRATEGY
#define TMP_REGISTER_COPY
#endif
#define MAX_ALIGN 64
// #define LESS_TEMPLATES
#ifdef FIXED_STACKSIZE
#define MAX_STACKSIZE FIXED_STACKSIZE
#else
#define MAX_STACKSIZE INT32_MAX
#endif
#define MAX_STACK_SLOTS (MAX_STACKSIZE / 8)
#define STACK_SAFETY_REGION 8

#if defined(__aarch64__) && defined(LESS_TEMPLATES)
#error "less templates is currently not supported on aarch64"
#endif

// enable instrumentation of certain components
#ifdef INSTRUMENT
struct ScopeTimer
{
    static uint64_t templateLookup;
    static uint64_t globalPreparation;
    static uint64_t copyAndPatching;
    static uint64_t valueStorage;
    static uint64_t templateGeneration;

    static void reset()
    {
        templateLookup = 0;
        globalPreparation = 0;
        copyAndPatching = 0;
        valueStorage = 0;
        templateGeneration = 0;
    }

    static void print()
    {
        llvm::outs() << "============= Timings ============="
                     << "\n";
        llvm::outs() << "TemplateLookup    : " << templateLookup << "ns\n";
        llvm::outs() << "GlobalPreparation : " << globalPreparation << "ns\n";
        llvm::outs() << "CopyAndPatch      : " << copyAndPatching << "ns\n";
        llvm::outs() << "ValueStorage      : " << valueStorage << "ns\n";
        llvm::outs() << "TemplateGeneration: " << templateGeneration << "ns\n";
        llvm::outs() << "\n";
    }

    ScopeTimer(uint64_t* v)
        : a(std::chrono::steady_clock::now())
        , var(v)
    {
    }
    ~ScopeTimer()
    {
        auto b = std::chrono::steady_clock::now();
        *var += std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    }
    const decltype(std::chrono::steady_clock::now()) a;
    uint64_t* var;
};
#define MEASURE(var) ScopeTimer timer##__LINE__(&var)
#define RESET_TIMERS()       \
    do {                     \
        ScopeTimer::reset(); \
    } while (false)
#define PRINT_TIMERS()       \
    do {                     \
        ScopeTimer::print(); \
    } while (false)
#else
#define MEASURE(...)
#define RESET_TIMERS(...)
#define PRINT_TIMERS(...)
#endif

// tracing levels:
// - 0: no outputs (even disable output of unreachable/assure)
// - 1: only assures/unreachable
// - 2: + info
// - 3: + debug
extern int TRACING_LEVEL;
extern bool COMPILE_ONLY;
extern int RUNS;
extern bool TEMPLATE_CONST_EVAL;
extern bool DUMP_CODE;
extern bool NO_TEMPLATE_OUTLINES;
extern bool TIME_FLAG;
extern int LLVM_CMODEL;
extern std::string INPUT;
extern bool MAIN;
extern bool LLVM_OPT;

template <typename T>
inline T UnalignedRead(char* src)
{
    T ret;
    memcpy(&ret, src, sizeof(T));
    return ret;
}

template <typename T>
inline void UnalignedWrite(char* dst, T value)
{
    memcpy(dst, &value, sizeof(T));
}

template <typename T>
inline void UnalignedAddAndWriteback(char* addr, T value)
{
    T old = UnalignedRead<T>(addr);
    UnalignedWrite<T>(addr, old + value);
}

enum class TEMPLATE_KIND : uint8_t
{
    STACK_ARGS = 0,
    STACK_PATCHED = 1,
    REGISTERS = 2
};

inline bool IsRegisterCC([[maybe_unused]] TEMPLATE_KIND kind)
{
#if FORCE_STACK_TEMPLATE
    return false;
#else
    return kind == TEMPLATE_KIND::REGISTERS;
#endif
}

inline bool hasReturnOffset([[maybe_unused]] TEMPLATE_KIND kind)
{
#if FORCE_STACK_TEMPLATE
    return true;
#else
    return !IsRegisterCC(kind);
#endif
}

#define TRACE_INTERNAL(file, line, level, OUTPUT)                                 \
    do {                                                                          \
        if (TRACING_LEVEL > level)                                                \
            llvm::outs() << "[" << file << ":" << line << "] " << OUTPUT << "\n"; \
    } while (false)

#define ASSURE_INTERNAL(file, line, X, OP, Y, ...)                                                                                                 \
    do {                                                                                                                                           \
        if (!((X)OP(Y))) [[unlikely]] {                                                                                                            \
            TRACE_INTERNAL(file, line, 0, "expected: " << #X << " " << #OP << " " << #Y << " but lhs: " << (X) << " and rhs: " << (Y)__VA_ARGS__); \
            std::raise(SIGINT);                                                                                                                    \
        }                                                                                                                                          \
    } while (false)

#define UNREACHABLE_INTERNAL(file, line, ...)                                  \
    do {                                                                       \
        TRACE_INTERNAL(file, line, 0, "unreachable was reached " __VA_ARGS__); \
        std::raise(SIGINT);                                                    \
        __builtin_unreachable();                                               \
    } while (false)

#ifdef CODEGEN_RELEASE_BUILD
#define RELEASE

#define INFO(...)
#define DEBUG(...)
#define ASSURE(...)
#define UNREACHABLE(...) __builtin_unreachable()

#else
#define NORELEASE

#define INFO(OUTPUT) TRACE_INTERNAL(__FILE__, __LINE__, 1, OUTPUT)
#define DEBUG(OUTPUT) TRACE_INTERNAL(__FILE__, __LINE__, 2, OUTPUT)
#define ASSURE(...) ASSURE_INTERNAL(__FILE__, __LINE__, __VA_ARGS__)
#define UNREACHABLE(...) UNREACHABLE_INTERNAL(__FILE__, __LINE__, __VA_ARGS__)

#endif

#define ASSURE_TRUE(X, ...) ASSURE(static_cast<bool>(X), ==, true, __VA_ARGS__)
#define ASSURE_FALSE(X, ...) ASSURE(static_cast<bool>(X), ==, false, __VA_ARGS__)
#define REL_ASSURE(...) ASSURE_INTERNAL(__FILE__, __LINE__, __VA_ARGS__)

#if defined(NORELEASE) || (defined(VTUNE_SUPPORT) && !defined(VTUNE_COMPILE))
#define LOGGING
#endif
