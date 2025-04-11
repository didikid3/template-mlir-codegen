#include "Preparation/BinaryParser.hpp"
#include "TemplateLibrary.hpp"
#include "common.hpp"

#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <dlfcn.h>

void BinaryParser::resolveSymbol(int64_t addendInfo)
{
    const auto symNameStr = llvm::cantFail(relocation.getSymbol()->getName());
    const auto relocType = relocation.getType();
    const uint32_t offset = static_cast<uint32_t>(relocation.getOffset());
    const bool withGOT = (relocType == llvm::ELF::R_X86_64_GOTPCREL
        || relocType == llvm::ELF::R_X86_64_GOTPCRELX
        || relocType == llvm::ELF::R_X86_64_REX_GOTPCRELX);
    [[maybe_unused]] const bool isSupportedType = (relocType == llvm::ELF::R_X86_64_PLT32 // relative to pc
        || relocType == llvm::ELF::R_X86_64_PC32 // relative to pc
        || relocType == llvm::ELF::R_X86_64_32 // absolute
        || relocType == llvm::ELF::R_X86_64_32S
        || relocType == llvm::ELF::R_X86_64_GOTPCREL // relative address to got entry
        || relocType == llvm::ELF::R_X86_64_GOTPCRELX
        || relocType == llvm::ELF::R_X86_64_REX_GOTPCRELX
        || relocType == llvm::ELF::R_X86_64_64);
    ASSURE_TRUE(isSupportedType);
    [[maybe_unused]] const bool is64Bit = relocType == llvm::ELF::R_X86_64_64;
    char* loc = const_cast<char*>(codeSecs.back().data()) + offset;
    auto write = [=](int64_t nr) {
        ASSURE(nr, <=, INT32_MAX);
        ASSURE(nr, >=, INT32_MIN);
        auto addr = static_cast<int32_t>(nr);
        UnalignedWrite<int32_t>(loc, addr);
    };

    ASSURE(isAnyPlaceholder(symNameStr)
            && !isIndexPlaceholder(symNameStr)
            && !isRegionPlaceholder(symNameStr)
            && !isNextPlaceholder(symNameStr),
        ||, !is64Bit);
    // only calls can be 32bit relocations
    if (isRegionPlaceholder(symNameStr)) {
        // split into substencil
        flushFragment(offset - 1, REGION_CALL_INSTR_SIZE);
        regionOrder.push_back(static_cast<uint8_t>(getPlaceholderNr(symNameStr)));
    } else if (isNextPlaceholder(symNameStr)) {
        jumps.emplace_back(InternalJump::Continue(
            fragments.size(),
            static_cast<uint32_t>(offset + totalSize)));
    } else if (isIndexPlaceholder(symNameStr)) {
        ASSURE(relocType, !=, llvm::ELF::R_X86_64_PC32,
            << " invalid relocation for index placeholder:" << currentTemplateNr);
        write(addendInfo);
        pp.emplace_back(
            PatchPoint{
                offset + totalSize,
                static_cast<uint8_t>(fragments.size()),
                PatchPoint::x86_32bit},
            getPlaceholderNr(symNameStr));
    } else if (isAnyPlaceholder(symNameStr)) {
        ASSURE_TRUE(is64Bit);
        ASSURE(addendInfo, ==, 0);
        pp.emplace_back(
            PatchPoint{
                offset + totalSize,
                static_cast<uint8_t>(fragments.size()),
                PatchPoint::x86_64bit},
            getPlaceholderNr(symNameStr));
    } else if (isFractionTemplateSection(symNameStr)) {
        const auto target = std::stoi(symNameStr.substr(symNameStr.find_last_of(".") + 1).data());
        jumps.emplace_back(InternalJump{
            static_cast<uint32_t>(offset + totalSize),
            static_cast<uint8_t>(fragments.size()),
            static_cast<uint8_t>(target)});
    } else {
        // no absolute symbols are supposed to occur here...
        ASSURE(relocType, !=, llvm::ELF::R_X86_64_32);
        ASSURE(relocType, !=, llvm::ELF::R_X86_64_32S);
        if (void* address = dlsym(RTLD_DEFAULT, symNameStr.str().c_str())) {
            // dynamic symbol
            write(addendInfo);
            calls.emplace_back(
                reinterpret_cast<void*>(address),
                static_cast<uint8_t>(fragments.size()),
                static_cast<uint32_t>(offset + totalSize),
                withGOT);
        } else if (library.symbols().contains(formatDataSectionRef(symNameStr, fileId))) {
            // data section (.bss / .rodata)
            write(addendInfo);
            calls.emplace_back(
                formatDataSectionRef(symNameStr, fileId),
                static_cast<uint8_t>(fragments.size()),
                static_cast<uint32_t>(offset + totalSize),
                withGOT);
        } else {
            DEBUG("Didn't link " << symNameStr << " yet!");
            write(addendInfo);
            calls.emplace_back(
                symNameStr.str(),
                static_cast<uint8_t>(fragments.size()),
                static_cast<uint32_t>(offset + totalSize),
                withGOT);
        }
    }
}
