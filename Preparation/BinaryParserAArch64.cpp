#include "Preparation/BinaryParser.hpp"
#include "TemplateLibrary.hpp"
#include "common.hpp"

#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <dlfcn.h>

void BinaryParser::resolveSymbol([[maybe_unused]] int64_t addendInfo)
{
    const auto symNameStr = llvm::cantFail(relocation.getSymbol()->getName());
    const auto relocType = relocation.getType();
    const uint32_t offset = static_cast<size_t>(relocation.getOffset());
    auto checkPlaceholderForConstant = [&] {
        ASSURE_TRUE(isAnyPlaceholder(symNameStr)
                && !isNextPlaceholder(symNameStr)
                && !isRegionPlaceholder(symNameStr),
            << " Symbol: " << symNameStr);
    };
    ASSURE(offset % 4, ==, 0, << "everything should be aligned, as it always points to the beginning of an instruction");
    auto PLACE_NOP = [&](size_t off) {
        reinterpret_cast<uint32_t*>(const_cast<char*>(codeSecs.back().data()))[off / 4] = static_cast<uint32_t>(0xd503201f);
    };
    ASSURE(addendInfo, ==, 0, << " non-zero addend for symbol " << symNameStr);
    auto patchpoint = [&](uint8_t kind) {
        ASSURE(addendInfo, ==, 0, << " non-zero addend for symbol " << symNameStr);
        if (isIndexPlaceholder(symNameStr)) {
            if (kind == PatchPoint::aarch64_mov1 || kind == PatchPoint::aarch64_mov2 || kind == PatchPoint::aarch64_mov3) {
                PLACE_NOP(offset);
                return;
            } else if (kind == PatchPoint::aarch64_mov0) {
                // ensure that it is a movz
                reinterpret_cast<uint32_t*>(const_cast<char*>(codeSecs.back().data()))[offset / 4] &= ~(1 << 19);
            }
        }
        pp.emplace_back(PatchPoint{
                            offset + totalSize,
                            static_cast<uint8_t>(fragments.size()),
                            kind},
            getPlaceholderNr(symNameStr));
    };
    switch (relocType) {
        case llvm::ELF::R_AARCH64_MOVW_UABS_G0_NC:
        case llvm::ELF::R_AARCH64_MOVW_UABS_G0:
        case llvm::ELF::R_AARCH64_MOVW_SABS_G0: // bits 0  - 15
        {
            checkPlaceholderForConstant();
            patchpoint(PatchPoint::aarch64_mov0);
            break;
        }
        case llvm::ELF::R_AARCH64_MOVW_UABS_G1_NC:
        case llvm::ELF::R_AARCH64_MOVW_UABS_G1:
        case llvm::ELF::R_AARCH64_MOVW_SABS_G1: // bits 16 - 31
        {
            checkPlaceholderForConstant();
            patchpoint(PatchPoint::aarch64_mov1);
            break;
        }
        case llvm::ELF::R_AARCH64_MOVW_UABS_G2_NC:
        case llvm::ELF::R_AARCH64_MOVW_UABS_G2:
        case llvm::ELF::R_AARCH64_MOVW_SABS_G2: // bits 32 - 47
        {
            checkPlaceholderForConstant();
            patchpoint(PatchPoint::aarch64_mov2);
            break;
        }
        case llvm::ELF::R_AARCH64_MOVW_UABS_G3: // bits 48 - 63
        {
            checkPlaceholderForConstant();
            patchpoint(PatchPoint::aarch64_mov3);
            break;
        }
        case llvm::ELF::R_AARCH64_JUMP26: // pc relative jump bits 2 - 27
        case llvm::ELF::R_AARCH64_CALL26: // pc relative call bits 2 - 27
        {
            if (isNextPlaceholder(symNameStr)) {
                ASSURE(relocType, ==, llvm::ELF::R_AARCH64_JUMP26, << " next call that is no tail-call");
                jumps.emplace_back(InternalJump::Continue(
                    fragments.size(),
                    static_cast<uint32_t>(offset + totalSize)));
            } else if (isRegionPlaceholder(symNameStr)) {
                flushFragment(offset, REGION_CALL_INSTR_SIZE);
                regionOrder.push_back(static_cast<uint8_t>(getPlaceholderNr(symNameStr)));
            } else if (isAnyPlaceholder(symNameStr)) {
                patchpoint(PatchPoint::aarch64_branch);
            } else {
                if (void* address = dlsym(RTLD_DEFAULT, symNameStr.data())) {
                    // dynamic symbol
                    calls.emplace_back(
                        reinterpret_cast<void*>(address),
                        static_cast<uint8_t>(fragments.size()),
                        offset,
                        false);
                } else if (library.symbols().contains(formatDataSectionRef(symNameStr, fileId))) {
                    // data section (.bss / .rodata)
                    UNREACHABLE("unimplemented");
                } else {
                    DEBUG("Didn't link " << symNameStr << " yet!");
                    calls.emplace_back(
                        symNameStr.str(),
                        static_cast<uint8_t>(fragments.size()),
                        offset,
                        false);
                }
            }
            break;
        }
        case llvm::ELF::R_AARCH64_ADR_PREL_PG_HI21:
        case llvm::ELF::R_AARCH64_ADR_PREL_PG_HI21_NC: // pc realtive (but page aligned...)
        case llvm::ELF::R_AARCH64_ADR_GOT_PAGE: {
            PLACE_NOP(offset);
            break;
        }
        case llvm::ELF::R_AARCH64_ADD_ABS_LO12_NC:
            ASSURE_TRUE(isIndexPlaceholder(symNameStr), << " Symbol: " << symNameStr);
            [[fallthrough]];
        case llvm::ELF::R_AARCH64_LD64_GOT_LO12_NC:
            if (isIndexPlaceholder(symNameStr)) {
                // instead of generating a got entry we replace the instructions with a
                // mov instruction, which will save on GOT entry creation
                // keep last 5 bits as it contains the target register
                uint32_t* instr = reinterpret_cast<uint32_t*>(const_cast<char*>(codeSecs.back().data()));
                instr[offset / 4] = static_cast<uint32_t>(0b11010010100000000000000000000000) | (instr[offset / 4] & 0b11111);
                pp.emplace_back(PatchPoint{
                                    offset + totalSize,
                                    static_cast<uint8_t>(fragments.size()),
                                    PatchPoint::aarch64_mov0},
                    getPlaceholderNr(symNameStr));
            } else {
                // symbols... (e.g. adder)
                DEBUG("Didn't link " << symNameStr << " yet!");
                // use a ldr to load the actual value from a got entry
                uint32_t* instr = reinterpret_cast<uint32_t*>(const_cast<char*>(codeSecs.back().data()));
                instr[offset / 4] = static_cast<uint32_t>(0b01011000000000000000000000000000) | (instr[offset / 4] & 0b11111);
                calls.emplace_back(
                    symNameStr.str(),
                    static_cast<uint8_t>(fragments.size()),
                    offset + totalSize,
                    true);
            }
            break;
        default:
            INFO("unimplemented: " << relocType << " symbol: " << symNameStr);
    }
}
