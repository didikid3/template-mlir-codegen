#include "Preparation/SectionProvider.hpp"

#include <sys/mman.h>

DataSection DataSectionProvider::alloc(size_t size, size_t align)
{
    char* addr = new char[size];
    ASSURE_TRUE(addr != nullptr);
    m_storedChunks.push_back(
        DataSection{addr,
            static_cast<uint32_t>(size),
            static_cast<uint32_t>(align)});
    if (align > 0) {
        ASSURE(align, &&, ((align & (align - 1)) == 0));
        // align the total size and afterward add the real size
        m_totalDataSize = (m_totalDataSize + align - 1) & -align;
    }
    m_totalDataSize += size;
    return DataSection{addr, static_cast<uint32_t>(size), static_cast<uint32_t>(align)};
}

DataSectionProvider::~DataSectionProvider()
{
    for (const auto sec : m_storedChunks) {
        delete[] sec.data;
    }
}

void DataSectionProvider::addSymbol(std::string name)
{
    ASSURE_FALSE(m_storedChunks.empty());
    const auto base = m_totalDataSize - m_storedChunks.back().size;
    [[maybe_unused]] const auto inserted = m_symbols.try_emplace(name, base).second;
    ASSURE_TRUE(inserted, << " duplicate section: " << name);
}

const llvm::StringMap<uint64_t>& DataSectionProvider::symbols() const
{
    return m_symbols;
}

void DataSectionProvider::copySections(char* target) const
{
    ASSURE(reinterpret_cast<uint64_t>(target) % MAX_ALIGN, ==, 0, << " unaligned start pointer");
    uint32_t offset = 0;
    for (const auto sec : m_storedChunks) {
        offset = (offset + sec.align - 1) & -sec.align;
        memcpy(target + offset, sec.data, sec.size);
    }
}

size_t DataSectionProvider::getDataSize() const
{
    return m_totalDataSize;
}

CodeSectionProvider::~CodeSectionProvider()
{
    for (char* chunk : m_storedChunks) {
        [[maybe_unused]] int err = munmap(chunk, ALLOC_CHUNK_SIZE);
        ASSURE(err, ==, 0);
    }
}

char* CodeSectionProvider::alloc(size_t size)
{
    ASSURE(size, <=, ALLOC_CHUNK_SIZE);
    if (size <= ALLOC_CHUNK_SIZE - m_storedInLastChunk) {
        char* ptr = m_storedChunks.back() + m_storedInLastChunk;
        m_storedInLastChunk += size;
        return ptr;
    } else {
        m_storedInLastChunk = size;
        auto memRegion = mmap(0, ALLOC_CHUNK_SIZE,
#ifndef NO_MAP_WX
            PROT_READ | PROT_WRITE | PROT_EXEC,
#else
            PROT_READ | PROT_WRITE,
#endif
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

        ASSURE(memRegion, !=, (void*)-1);
        return m_storedChunks.emplace_back(reinterpret_cast<char*>(memRegion));
    }
}
