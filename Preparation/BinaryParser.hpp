#pragma once

#include "Preparation/Utils.hpp"
#include "Templates/Template.hpp"

#include "llvm/ADT/StringMap.h"
#include "llvm/Object/ObjectFile.h"

#include <bitset>
#include <map>
#include <vector>

class TemplateLibrary;

/**
 * @brief Parser for reading template binary files and parsing them into their in-memory representation
 *
 * @details
 *  For each file the following happens:
 *  #1 Go through data sections, allocate them in the library and collect the symbols pointing into them
 *  #2 Go through code sections and parse templates.
 *      !! There is no 1 to 1 mapping between section and templates due to our basic block splitting on x86!!
 *      Instead we go through the sections combine them across borders but split them at region calls into fragments.
 *      We also have to account for that during offset computation into the single fragments.
 */
class BinaryParser
{
public:
    using RelocationMap = std::map<llvm::object::SectionRef, std::vector<llvm::object::RelocationRef>>;
    using SymbolMap = std::multimap<llvm::object::SectionRef, std::string>;

    BinaryParser(TemplateLibrary& lib);

    void operator()(llvm::object::SectionRef sec);
    void operator()(const std::string& filename, ClobberMask clobbers);

private:
    // architecture dependent implementation
    void resolveSymbol(int64_t addendInfo);

    void flushFragment(uint32_t offset, uint32_t skipNext);
    void flushTemplate();

    // constant infos:
    RelocationMap relocationMap;
    llvm::object::ObjectFile* obj;
    TemplateLibrary& library;
    ClobberMask clobbersForCurrentFile;

    // internal state:
    // if the fragment starts in the middle of a section, this contains the offset
    uint32_t offsetInitial = 0;

    // at the end of a section is contains the total size of code in the current fragment.
    // At any point in time, it can be used as addend to an offset into the current section
    // in order to get the offset into the current fragment).
    // If we just split a section, totalSize can also be negative
    int32_t totalSize = 0;

    llvm::object::RelocationRef relocation;
    llvm::object::RelocationRef nextRelocation;

    int currentTemplateKind = -1;
    int currentTemplateNr = -1;
    bool canInline = false;
    int fileId = -1;

    // intermediate buffer for sections of the current fragment
    std::vector<llvm::StringRef> codeSecs{};

    // information filled during parsing and move to the template once `flush` is called:
    // - labels always starts with {0, 0}, indicating fragment 0 offset 0 is first label
    // - others are empty on beginning
    std::vector<Loc> labels{};
    std::vector<Code> fragments{};
    std::vector<std::pair<PatchPoint, uint8_t>> pp{};
    std::vector<InternalJump> jumps{};
    std::vector<ExternalCall> calls{};
    std::vector<uint8_t> regionOrder{};
};
