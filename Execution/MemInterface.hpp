#pragma once

#include "common.hpp"

#include <numeric>
#include <tuple>
#include <vector>

#ifdef VTUNE_SUPPORT
#include "jitprofiling.h"
#endif

class ExecutableMemoryInterface
{
public:
    ExecutableMemoryInterface();
    ~ExecutableMemoryInterface();

    char* memory(uint8_t depth, uint32_t size = 0);
    char* insert(uint8_t depth, uint32_t size, const void* src);
    char* insert(uint8_t depth, std::byte c);
    char* empty(uint8_t depth, uint32_t size);

    void align(uint8_t depth);

    void make_memory_executable();

    void flushICache();

    void setCodeSize(uint8_t depth, size_t size);
    void logTemplate(
        uint8_t depth,
        uint32_t size,
        const std::string& str,
        uint32_t templateId = static_cast<uint32_t>(-1));

protected:
    template <typename>
    friend class Executor;
    // required to be called before using the other memory operations
    char* setup(size_t extraSize);
    void dumpMemoryToFile() const;
    static constexpr uint8_t MAX_DEPTH = 7;
    size_t SIZES[MAX_DEPTH] = {
        4 * 4096,
        4 * 4096,
        4 * 4096,
        4 * 4096,
        4096,
        4096,
        4096};
    size_t BASE_OFFSETS[MAX_DEPTH + 1];
    size_t SIZE;

    char* m_memRegion;
    size_t m_memoryOffsets[MAX_DEPTH];
};

class WithLogging
    : public ExecutableMemoryInterface
{
public:
    void logTemplate(
        uint8_t depth,
        uint32_t size,
        const std::string& str,
        uint32_t templateId = static_cast<uint32_t>(-1),
        uint32_t line = static_cast<uint32_t>(-1));
    void logTemplate(
        void* addr,
        uint32_t size,
        const std::string& str,
        uint32_t templateId = static_cast<uint32_t>(-1),
        uint32_t line = static_cast<uint32_t>(-1));

protected:
    void dump();
#ifdef VTUNE_SUPPORT
    std::vector<LineNumberInfo> generateLineInfo() const;
#endif
private:
    void dumpMappingToFile() const;
    std::vector<std::tuple<void*, uint32_t, std::string, uint32_t, uint32_t>> m_entries;
};
