#include "Preparation/BinaryParser.hpp"
#include "TemplateLibrary.hpp"
#include "common.hpp"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include <dlfcn.h>

namespace {
int64_t GetRelocationInfo(const llvm::object::ELFObjectFileBase* obj,
    const llvm::object::RelocationRef& rel)
{
    auto getRelocInfoImpl = [&](auto* obj) {
        return llvm::cantFail(obj->getRelocationAddend(rel.getRawDataRefImpl()));
    };
    return llvm::TypeSwitch<const llvm::object::ELFObjectFileBase*, int64_t>(obj)
        .Case<llvm::object::ELF32LEObjectFile>(getRelocInfoImpl)
        .Case<llvm::object::ELF64LEObjectFile>(getRelocInfoImpl)
        .Case<llvm::object::ELF32BEObjectFile>(getRelocInfoImpl)
        .Case<llvm::object::ELF64BEObjectFile>(getRelocInfoImpl);
}

BinaryParser::SymbolMap GetSymbolMap(llvm::object::ObjectFile const& obj)
{
    BinaryParser::SymbolMap ret;
    for (auto sym : obj.symbols()) {
        const auto symName = llvm::cantFail(sym.getName());
        auto sec = llvm::cantFail(sym.getSection());
        if (sec == obj.section_end())
            continue;
        if (!isAnyPlaceholder(symName) && (sec->isData() || sec->isBSS())) {
            ret.emplace(*sec, symName.str());
        }
    }
    return ret;
}

BinaryParser::RelocationMap GetRelocationMap(llvm::object::ObjectFile const& obj)
{
    BinaryParser::RelocationMap ret;
    for (auto sec : obj.sections()) {
        auto reloc = llvm::cantFail(sec.getRelocatedSection());
        if (reloc == obj.section_end())
            continue;

        std::vector<llvm::object::RelocationRef>& vec = ret[*reloc];
        vec.insert(vec.end(), sec.relocation_begin(), sec.relocation_end());
        // Sort relocations by address.
        llvm::stable_sort(vec, [](auto a, auto b) {
            return a.getOffset() < b.getOffset();
        });
    }

    return ret;
}
}

BinaryParser::BinaryParser(
    TemplateLibrary& lib)
    : obj(nullptr)
    , library(lib)
    , labels({{0, 0}})
{
}

void BinaryParser::operator()(llvm::object::SectionRef sec)
{
    auto name = llvm::cantFail(sec.getName());
    const auto [stencilNr, kind] = parseSectionName(name);
    canInline = !name.starts_with(kPrefixSectionName);

    if ((currentTemplateKind != kind || currentTemplateNr != stencilNr) && (!codeSecs.empty() || !fragments.empty())) {
        flushTemplate();
    }
    currentTemplateKind = kind;
    currentTemplateNr = stencilNr;

    if (isFractionTemplateSection(name)) {
        ASSURE(totalSize, >=, 0);
        labels.emplace_back(static_cast<uint32_t>(totalSize), static_cast<uint8_t>(fragments.size()));
    }
    codeSecs.emplace_back(llvm::cantFail(sec.getContents()));

    const auto& relocs = relocationMap[sec];
    for (auto relocSymIt = relocs.begin(); relocSymIt != relocs.end();) {
        relocation = *relocSymIt;
        relocSymIt++;
        if (relocSymIt != relocs.end()) {
            nextRelocation = *relocSymIt;
        } else {
            nextRelocation = llvm::object::RelocationRef();
        }
        const auto* elf = llvm::dyn_cast<llvm::object::ELFObjectFileBase>(obj);
        ASSURE(elf, !=, reinterpret_cast<decltype(elf)>(0));

        resolveSymbol(GetRelocationInfo(elf, relocation));
    }

    ASSURE_FALSE(codeSecs.empty());
    totalSize += codeSecs.back().size();
}

void BinaryParser::operator()(const std::string& filename, ClobberMask clobbers)
{
    static int invocations = 0;
    fileId = invocations++;
    clobbersForCurrentFile = clobbers;
    ASSURE(offsetInitial, ==, 0);
    ASSURE(totalSize, ==, 0);
    currentTemplateKind = -1;
    currentTemplateNr = -1;
    canInline = false;
    ASSURE_TRUE(codeSecs.empty());
    ASSURE_TRUE(fragments.empty());
    ASSURE_TRUE(pp.empty());
    ASSURE_TRUE(jumps.empty());
    ASSURE_TRUE(calls.empty());
    ASSURE_TRUE(regionOrder.empty());
    // only labels is missing as it is always size == 1 with the 0, 0 element in it

    auto memoryBufferOrError = llvm::WritableMemoryBuffer::getFile(filename);
    REL_ASSURE(static_cast<bool>(memoryBufferOrError), ==, true, << " unable to open binary file");
    const auto binary = llvm::cantFail(
        llvm::object::ObjectFile::createELFObjectFile(**memoryBufferOrError));
    ASSURE_TRUE(llvm::isa<llvm::object::ObjectFile>(binary));
    obj = llvm::dyn_cast<llvm::object::ObjectFile>(binary.get());
    auto symsPerSection = GetSymbolMap(*obj);
    for (auto sec : obj->sections()) {
        if ((sec.isData() || sec.isBSS())
            && !llvm::cantFail(sec.getContents()).empty()
            && !llvm::cantFail(sec.getName()).equals(".eh_frame")
            && symsPerSection.count(sec)) {
            const auto secName = llvm::cantFail(sec.getName());
            const auto code = llvm::cantFail(sec.getContents());
            DataSection dataSec = library.DataSectionProvider::alloc(code.size(), sec.getAlignment().value());
            memcpy(dataSec.data, code.data(), code.size());
            library.addSymbol(formatDataSectionRef(secName, fileId));

            const auto [b, e] = symsPerSection.equal_range(sec);
            for (auto it = b; it != e; ++it) {
                if (it->second != secName) {
                    library.addSymbol(formatDataSectionRef(it->second, fileId));
                }
            }
        }
    }
    relocationMap = GetRelocationMap(*obj);
    for (auto sec : obj->sections()) {
        if (sec.isText()) {
            auto name = llvm::cantFail(sec.getName());
            if (isTemplateSection(name)) {
                operator()(sec);
            }
        }
    }

    if (!codeSecs.empty() || !fragments.empty()) {
        flushTemplate();
    }
}

void BinaryParser::flushFragment(uint32_t offset, uint32_t skipNext)
{
    if (totalSize + offset <= 0) {
        // this case appears if the region call is the last instruction of a template (e.g. a function terminated by an unreachable)
        // here we additionally append a template containing a single nop byte
#if defined(__x86_64__)
        char* loc = library.CodeSectionProvider::alloc(1);
        UnalignedWrite<uint8_t>(loc, 0x90);
        fragments.emplace_back(Code{loc, 1});
#elif defined(__aarch64__)
        char* loc = library.CodeSectionProvider::alloc(4);
        UnalignedWrite<uint32_t>(loc, 0xd503201f);
        fragments.emplace_back(Code{loc, 4});
#endif
        offsetInitial = 0;
        return;
    }
    char* loc = library.CodeSectionProvider::alloc(totalSize + offset);
    auto it = codeSecs.begin();
    ASSURE_FALSE(codeSecs.empty());
    if (codeSecs.size() == 1) {
        memcpy(loc, it->data() + offsetInitial, totalSize + offset);
    } else {
        memcpy(loc, it->data() + offsetInitial, it->size() - offsetInitial);
        uint32_t off = it->size() - offsetInitial;
        it++;
        for (; it != (codeSecs.end() - 1); it++) {
            memcpy(loc + off, it->data(), it->size());
            off += it->size();
        }
        memcpy(loc + off, it->data(), totalSize + offset - off);
        codeSecs = std::vector({codeSecs.back()});
    }

    fragments.emplace_back(Code{loc, totalSize + offset});

    offsetInitial = offset + skipNext;
    totalSize = -offsetInitial;
}

void BinaryParser::flushTemplate()
{
    if (!codeSecs.empty()) {
        flushFragment(0, 0);
        codeSecs.clear();
    }

    std::stable_sort(pp.begin(), pp.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) < std::get<1>(b);
    });
    std::vector<PatchPoint> ppNoNumber;
    for (size_t i = 0; i < pp.size(); i++) {
        if (i == 0) {
            auto& ppNoNum = ppNoNumber.emplace_back(pp[i].first);
            ppNoNum.duplicate = true;
        } else if (pp[i - 1].second == pp[i].second) {
            auto& ppNoNum = ppNoNumber.emplace_back(pp[i].first);
            ppNoNum.duplicate = true;
        } else if ((pp[i - 1].second + 1) == pp[i].second) {
            auto& ppNoNum = ppNoNumber.emplace_back(pp[i].first);
            ppNoNum.duplicate = false;
        } else {
            const auto diff = pp[i].second - pp[i - 1].second;
            ASSURE(diff, >, 1);
            for (int j = 0; j < (diff - 1); j++) {
                auto& ppNoNum = ppNoNumber.emplace_back(pp[i].first);
                ppNoNum.duplicate = false;
            }
        }
    }
    pp.clear();
    ASSURE_FALSE(fragments.empty(), << currentTemplateNr);
    library.addBinaryTemplate(
        currentTemplateNr,
        static_cast<TEMPLATE_KIND>(currentTemplateKind),
        Template{
            std::move(fragments),
            std::move(ppNoNumber),
            std::move(jumps),
            std::move(calls),
            std::move(labels),
            std::move(regionOrder),
            clobbersForCurrentFile,
            canInline});
    ASSURE_TRUE(fragments.empty());
    labels.emplace_back(0, 0);
    canInline = false;
}
