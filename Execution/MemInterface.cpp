#include "Execution/MemInterface.hpp"
#include "TemplateCompiler.hpp"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

ExecutableMemoryInterface::ExecutableMemoryInterface()
{
    m_memRegion = nullptr;
}

char* ExecutableMemoryInterface::setup(size_t additionalSize)
{
    SIZE = 0;
    uint8_t i = 0;
    for (; i < MAX_DEPTH; i++) {
        BASE_OFFSETS[i] = SIZE;
        m_memoryOffsets[i] = SIZE;
        SIZE += SIZES[i];
    }
    BASE_OFFSETS[i] = SIZE;
    SIZE += additionalSize;

    m_memRegion = reinterpret_cast<char*>(mmap(0, SIZE,
#ifndef NO_MAP_WX
        PROT_READ | PROT_WRITE | PROT_EXEC,
#else
        PROT_READ | PROT_WRITE,
#endif
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSURE(m_memRegion, !=, (char*)-1);
    return m_memRegion + BASE_OFFSETS[i];
}

void WithLogging::dump()
{
    INFO("Program memory usage: " << [&] {
        std::stringstream result;
        for (uint8_t i = 0; i < MAX_DEPTH; i++) {
            if (i != 0)
                result << ", ";
            result << "(" << static_cast<uint32_t>(i) << ":" << (m_memoryOffsets[i] - BASE_OFFSETS[i]) << ")";
        }
        return result.str();
    }());
    if (!DUMP_CODE)
        return;
    std::sort(m_entries.begin(), m_entries.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
#ifdef VTUNE_SUPPORT
    dumpMappingToFile();
    return;
#else
    INFO((dumpMappingToFile(), "Dumped op to addr mapping to file!"));
#endif
    const auto targetTriple = compiler::getTargetTriple();
    const llvm::Target* target = compiler::getTarget();

    llvm::MCTargetOptions mcOptions;
    std::unique_ptr<llvm::MCRegisterInfo> MRI(target->createMCRegInfo(targetTriple));
    std::unique_ptr<llvm::MCInstrInfo> MCII(target->createMCInstrInfo());
    std::unique_ptr<llvm::MCAsmInfo> MAI(target->createMCAsmInfo(*MRI, targetTriple, mcOptions));
    std::unique_ptr<llvm::MCSubtargetInfo> STI(target->createMCSubtargetInfo(targetTriple, llvm::sys::getHostCPUName(), ""));
    std::unique_ptr<llvm::MCRegisterInfo> RI(target->createMCRegInfo(targetTriple));
    llvm::MCInstPrinter* IP = target->createMCInstPrinter(STI->getTargetTriple(), 0, *MAI, *MCII, *MRI);
    IP->setPrintImmHex(true);
    llvm::MCContext mcContext(STI->getTargetTriple(), MAI.get(), MRI.get(), STI.get());
    std::unique_ptr<llvm::MCObjectFileInfo> MOFI(target->createMCObjectFileInfo(mcContext, true, true));
    mcContext.setObjectFileInfo(MOFI.get());
    std::unique_ptr<llvm::MCStreamer> STR(target->createAsmStreamer(mcContext,
        std::make_unique<llvm::formatted_raw_ostream>(llvm::outs()),
        false,
        false,
        IP,
        std::unique_ptr<llvm::MCCodeEmitter>(),
        std::unique_ptr<llvm::MCAsmBackend>(target->createMCAsmBackend(*STI, *MRI, mcOptions)),
        false));

    auto disass = target->createMCDisassembler(*STI, mcContext);

    STR->initSections(false, *STI);

    llvm::ArrayRef<uint8_t> binaryCode(reinterpret_cast<uint8_t*>(m_memRegion), SIZE);
    uint32_t lastEnd = 0;
    for (const auto& t : m_entries) {
        llvm::MCInst instr;
        uint64_t s = 0;

        const auto templateSize = std::get<1>(t);
        std::string substring = "###############  " + std::get<2>(t);

        const auto offset = reinterpret_cast<uint64_t>(std::get<0>(t)) - reinterpret_cast<uint64_t>(m_memRegion);
        if (offset != lastEnd) {
            llvm::outs() << "\t\t[..." << (offset - lastEnd) << "...]\n";
        }
        lastEnd = offset + templateSize;

        if (std::get<3>(t) != static_cast<uint32_t>(-1))
            substring += "(" + std::to_string(std::get<3>(t)) + ")";
        substring += "  ";
        if (!NO_TEMPLATE_OUTLINES)
            llvm::outs() << substring + std::string(50 - substring.size(), '#') << "\n";
        uint32_t i = 0;
        for (; i < templateSize; i += s) {
            disass->getInstruction(instr, s, binaryCode.slice(offset + i), reinterpret_cast<uint64_t>(m_memRegion) + offset + i, llvm::outs());
            llvm::outs() << reinterpret_cast<void*>(reinterpret_cast<uint64_t>(m_memRegion) + offset + i) << "\t";
            STR->emitInstruction(instr, *STI);
        }
        ASSURE(i, ==, templateSize);
    }
}

ExecutableMemoryInterface::~ExecutableMemoryInterface()
{
    [[maybe_unused]] int err = munmap(m_memRegion, SIZE);
    ASSURE(err, ==, 0);
}

void ExecutableMemoryInterface::setCodeSize(uint8_t depth, size_t size)
{
    ASSURE_TRUE(m_memRegion == nullptr, << " code size can only be set BEFORE allocation");
    ASSURE(depth, <, MAX_DEPTH);
    SIZES[depth] = size;
}

char* ExecutableMemoryInterface::memory(uint8_t depth, uint32_t size)
{
#ifdef __aarch64__
    ASSURE(size % 4, ==, 0);
#endif
    ASSURE(depth, <, MAX_DEPTH, << static_cast<uint32_t>(depth));
    ASSURE_TRUE(m_memRegion != nullptr);
    auto* loc = m_memRegion + m_memoryOffsets[depth];
    m_memoryOffsets[depth] += size;
    ASSURE(m_memoryOffsets[depth], <, BASE_OFFSETS[depth + 1], << " depth: " << static_cast<uint32_t>(depth));
    return loc;
}
char* ExecutableMemoryInterface::insert(uint8_t depth, uint32_t size, const void* src)
{
    auto* loc = memory(depth, size);
    memcpy(loc, src, size);
    return loc;
}

char* ExecutableMemoryInterface::insert(uint8_t depth, std::byte b)
{
#ifdef __aarch64__
    UNREACHABLE();
#endif
    return insert(depth, 1, &b);
}

char* ExecutableMemoryInterface::empty(uint8_t depth, uint32_t size)
{
    auto* loc = memory(depth, size);
    memset(loc, 0, size);
    return loc;
}

void ExecutableMemoryInterface::align(uint8_t depth)
{
    ASSURE(depth, <, MAX_DEPTH);
    m_memoryOffsets[depth] |= 0b11;
    m_memoryOffsets[depth] += 1; // align 4 byte
}

void ExecutableMemoryInterface::make_memory_executable()
{
    ASSURE_TRUE(m_memRegion != nullptr);
    [[maybe_unused]] int result = mprotect(m_memRegion, SIZE, PROT_READ | PROT_EXEC);
    ASSURE(result, !=, -1);
}

void ExecutableMemoryInterface::flushICache()
{
    llvm::sys::Memory::InvalidateInstructionCache(m_memRegion, SIZE);
}

void WithLogging::logTemplate(
    uint8_t depth,
    uint32_t size,
    const std::string& str,
    uint32_t templateId,
    uint32_t line)
{
    auto* addr = reinterpret_cast<void*>(memory(depth));
    logTemplate(addr, size, str, templateId, line);
}

void WithLogging::logTemplate(
    void* addr,
    uint32_t size,
    const std::string& str,
    uint32_t templateId,
    uint32_t line)
{
    DEBUG("Addr: " << addr << " : " << str << " - "
                   << "line: " << line << " - " << templateId);
    m_entries.emplace_back(addr, size, str, templateId, line);
}

void ExecutableMemoryInterface::dumpMemoryToFile() const
{
    ASSURE_TRUE(m_memRegion != nullptr);
    auto outputFile = std::fstream("code.dump", std::ios::out | std::ios::binary);
    outputFile.write(m_memRegion, SIZE);
    outputFile.close();
}

void WithLogging::dumpMappingToFile() const
{
    auto template2Addr = std::ofstream("template2Addr.txt");
    for (const auto& entry : m_entries) {
        template2Addr << std::get<0>(entry) << ":" << std::get<2>(entry) << "(" << std::get<3>(entry) << ")"
                      << " size: " << std::get<1>(entry) << " - " << std::get<4>(entry) << "\n";
    }
    template2Addr.close();
}

#ifdef VTUNE_SUPPORT
namespace {
uint32_t copyFile(std::ofstream& out)
{
    auto in = std::ifstream(INPUT);
    uint32_t lineCount = 0;
    for (std::string line; std::getline(in, line);) {
        out << line << "\n";
        ++lineCount;
    }
    return lineCount;
}
}

std::vector<LineNumberInfo> WithLogging::generateLineInfo() const
{
    std::vector<LineNumberInfo> info;
    std::unordered_map<std::string, uint32_t> unknownLines;
    auto out = std::ofstream("info.vtune");
    uint32_t linesInFile = copyFile(out) + 1;
    for (const auto& entry : m_entries) {
        const auto offset = reinterpret_cast<uint64_t>(std::get<0>(entry)) - reinterpret_cast<uint64_t>(m_memRegion);
        ASSURE(offset, <=, UINT32_MAX);
        auto line = std::get<4>(entry);
        if (line == static_cast<uint32_t>(-1)) {
            auto it = unknownLines.emplace(std::get<2>(entry), linesInFile + unknownLines.size());
            line = it.first->second;
            if (it.second) {
                out << std::get<2>(entry) << "\n";
            }
        }
        info.push_back(LineNumberInfo{static_cast<uint32_t>(offset), line});
    }
    out.close();
    return info;
}
#endif
