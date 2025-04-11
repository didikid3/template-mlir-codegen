#pragma once
#include "common.hpp"

#include "llvm/ADT/StringMap.h"

#include <vector>

struct DataSection
{
    char* data;
    uint32_t size;
    uint32_t align;
};
class DataSectionProvider
{
public:
    // not copyable but movable
    DataSectionProvider(DataSectionProvider&) = delete;
    DataSectionProvider(const DataSectionProvider&) = delete;
    DataSectionProvider(DataSectionProvider&&) = default;
    DataSectionProvider() = default;

    ~DataSectionProvider();

    /**
     * @brief Allocation of memory for data sections
     * @param size section size
     * @param alignment section alignment
     * @return pointer to the allocated (read + write) memory
     */
    DataSection alloc(size_t size, size_t alignment);

    // Add symbol pointing to last allocated data section
    void addSymbol(std::string name);
    const llvm::StringMap<uint64_t>& symbols() const;

    // copy sections to target, writes exactly `getDataSize()` bytes
    void copySections(char* target) const;
    size_t getDataSize() const;

private:
    size_t m_totalDataSize = 0;
    // symbols with an offset to the beginning
    llvm::StringMap<uint64_t> m_symbols;
    std::vector<DataSection> m_storedChunks;
};

class CodeSectionProvider
{
public:
    // not copyable but movable
    CodeSectionProvider(CodeSectionProvider&) = delete;
    CodeSectionProvider(const CodeSectionProvider&) = delete;
    CodeSectionProvider(CodeSectionProvider&&) = default;
    CodeSectionProvider() = default;

    ~CodeSectionProvider();

    /**
     * @brief Allocation of memory for templates
     * @param size code size
     * @return pointer to the allocated (executable if possible + read + write) memory
     */
    char* alloc(size_t size);

private:
    static constexpr size_t ALLOC_CHUNK_SIZE = 4096 * 4ul; // 16k

    std::vector<char*> m_storedChunks;
    size_t m_storedInLastChunk = ALLOC_CHUNK_SIZE;
};
