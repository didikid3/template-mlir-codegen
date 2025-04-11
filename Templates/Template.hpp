#pragma once

#include "Preparation/Utils.hpp"
#include "common.hpp"

#include <bitset>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

struct Code
{
    size_t instantiate(char* targetLoc) const;
    char* location;
    uint32_t size;
};

/**
 * @brief location inside a binary template using fragment index and offset inside the fragment
 */
struct Loc
{
    Loc(uint32_t off, uint8_t frag)
        : offset(off)
        , fragment(frag)
    {
    }
    uint32_t offset;
    // some trickery to allow further members without padding in between
    [[no_unique_address]] uint8_t fragment;
};

/**
 * @brief fixup location inside the binary template to place arbitrary constants
 *
 * @details
 * Marks location inside the template with the fragment index and the offset inside the fragment
 * It indicates the kind of relocation required with the `kind` attribute
 *
 * The `duplicate` member is used during patching, which is described in more detail in `Template::fixupPatchpoints`
 */
struct PatchPoint
    : public Loc
{
    static constexpr uint8_t x86_64bit = 0;
    static constexpr uint8_t x86_32bit = 1;

    static constexpr uint8_t aarch64_mov0 = 0;
    static constexpr uint8_t aarch64_mov1 = 1;
    static constexpr uint8_t aarch64_mov2 = 2;
    static constexpr uint8_t aarch64_mov3 = 3;
    static constexpr uint8_t aarch64_branch = 4;

    PatchPoint(uint32_t off, uint8_t frag, uint8_t k, bool dup = false)
        : Loc(off, frag)
        , kind(k)
        , duplicate(dup)
    {
    }

    // this could ne stuffed into one byte but
    // due to the alignment size is anyways 8
    uint8_t kind;
    bool duplicate;

    void patch(char* addr, uint64_t arg) const;
};

/**
 * @brief fixup location inside the binary for jumps between labels inside the template
 * or to the end of the template (continuation)
 */
struct InternalJump
    : public Loc
{
    static constexpr uint8_t NEXT_TARGET = UINT8_MAX;
    InternalJump(uint32_t off, uint8_t frag, uint8_t tar)
        : Loc(off, frag)
        , target(tar)
    {
    }

    static InternalJump Continue(uint8_t fragment, uint32_t offset)
    {
        return InternalJump{offset, fragment, NEXT_TARGET};
    }

    bool isTailCall(size_t size) const
    {
        return target == NEXT_TARGET && offset == size - 4;
    }

    uint8_t target;
};

/**
 * @brief fixup location for calls/jumps outside the template
 *
 * @details
 * If we know the address of the target already (it was resolved via dlsym during binary parsing),
 * it is contained in `addr` (and indicated by `hasAddr`).
 * Otherwise (e.g. it is another jit compiled function) `addr` points to a string containing the symbol name
 *
 * `isIndirect` indicate, whether it is supposed to be a GOT load.
 */
class ExternalCall
    : public Loc
{
public:
    ExternalCall() = delete;
    ExternalCall(ExternalCall&) = delete;
    ExternalCall(const ExternalCall&) = delete;

    ExternalCall(ExternalCall&& rhs)
        : Loc(rhs.offset, rhs.fragment)
        , hasAddr(rhs.hasAddr)
        , isIndirect(rhs.isIndirect)
        , addr(rhs.addr)
    {
        // reset the address for the moved-from object
        // otherwise this might lead to a double free
        rhs.addr = nullptr;
    }

    ExternalCall(
        const std::string& symName,
        uint8_t fragment,
        uint32_t offset,
        bool isIndirect = false)
        : Loc(offset, fragment)
        , hasAddr(false)
        , isIndirect(isIndirect)
        , addr(new std::string(symName))
    {
    }

    ExternalCall(
        void* addr,
        uint8_t fragment,
        uint32_t offset,
        bool isIndirect = false)
        : Loc(offset, fragment)
        , hasAddr(true)
        , isIndirect(isIndirect)
        , addr(addr)
    {
    }

    ~ExternalCall()
    {
        if (!hasAddr && addr != nullptr) {
            delete reinterpret_cast<std::string*>(addr);
        }
    }

    // private:
    friend class Template;

    bool hasAddr : 1;
    bool isIndirect : 1;
    void* addr;
};

class Template
{
public:
    std::vector<Code> fragments;
    std::vector<Loc> labels;
    const bool canInline = false;

    std::vector<PatchPoint> patchpoints;
    std::vector<InternalJump> internalJumps;
    std::vector<ExternalCall> externalCalls;
    // the order of the regions inside the binary template
    // (as it might happen, that the regions are no longer contained in the order they were abstracted)
    // so far we only support ascending or descending and NO arbitrary permutations
    std::vector<uint8_t> regionOrder;
    ClobberMask clobberMask;

    Template() = default;
    Template(
        std::vector<Code>&& code,
        std::vector<PatchPoint>&& pps,
        std::vector<InternalJump>&& inJump,
        std::vector<ExternalCall>&& extCalls,
        std::vector<Loc>&& labels,
        std::vector<uint8_t>&& regs,
        ClobberMask clobbers,
        bool canInline);

    // not copyable, only movable
    Template(Template&& rhs) = default;
    Template(const Template& rhs) = delete;
    Template(Template& rhs) = delete;

    template <typename... Args>
    size_t instantiateAndPatch(char* targetLoc, Args... args) const
    {
        MEASURE(ScopeTimer::copyAndPatching);
        ASSURE(fragments.size(), ==, 1);
        const auto size = instantiate(targetLoc, 0);
        if constexpr (sizeof...(args) > 0)
            fixupPatchpoints(targetLoc, std::forward<Args>(args)...);

        return size;
    }
    size_t instantiate(char* targetLoc, uint32_t indx) const;

    void* location() const;
    bool isReadyForExecution() const;
    void executeInplace(uint64_t* stackframe, const std::vector<uint64_t>& args) const;
    bool requiresStackAlign() const;
    bool isValid() const;
    bool breaksInlining() const;
    const std::vector<uint8_t>& regions() const;
    uint32_t size(uint8_t i = 0) const;

    void patch(
        const std::vector<char*>& addresses,
        const std::vector<uint64_t>& args,
        std::function<char*(void*)> createThunk,
        std::function<char*(void*)> createEntry,
        std::function<void*(const std::string&, char*)> resolveSym) const;

private:
    template <typename... Args, typename = std::enable_if_t<(... && std::is_same<uint64_t, Args>::value) && (sizeof...(Args) > 0)>>
    void fixupPatchpoints([[maybe_unused]] char* addr, Args... args) const
    {
#if defined(__x86_64__)
        ASSURE(patchpoints.size(), ==, sizeof...(args));
        [[maybe_unused]] int i = 0;
        ([&] {
            patchpoints[i].patch(addr, args);
            ++i;
        }(),
            ...);
#elif defined(__aarch64__)
        const uint64_t arguments[sizeof...(args)] = {args...};
        uint32_t i = 0;
        for (auto& pp : patchpoints) {
            if (!pp.duplicate)
                i++;

            ASSURE(i, <, sizeof...(args));
            const auto arg = arguments[i];
            pp.patch(addr, arg);
        }
#endif
    }
    /**
     * @brief fix up the patch point locations by placing the addresses into the template
     * @details
     * We iterate the patch points and place the current argument into the fixup location.
     * If the patch point is not a duplicate (`duplicate`), the argument iterator is advanced.
     *
     * This imposes the following requirements to the stored patch points:
     *  - they MUST be ordered the same way as the arguments are (ordered ascending to the indices given during abstraction)
     *  - every argument MUST be matched by a patch point (to drop an argument, the patch point location must be overwritten afterwards, e.g.: by the next patch point)
     *
     * @param addresses the start addresses for each fragment
     * @param args the collected argument to be patched
     */
    void fixupPatchpoints(
        const std::vector<char*>& addresses,
        const std::vector<uint64_t>& args) const;
    void fixupInternalJumps(
        const std::vector<char*>& addresses) const;
    void fixupExternalCalls(
        const std::vector<char*>& addresses,
        std::function<char*(void*)> createThunk,
        std::function<char*(void*)> createEntry,
        std::function<void*(const std::string&, char*)> resolveSym) const;

    void trim();

    template <typename>
    friend struct CustomPattern;
};

std::ostream& operator<<(std::ostream& os, const Template& stencil);
std::ostream& operator<<(std::ostream& os, const Code& stencil);

uint64_t invokeFromVector(void* function, const std::vector<uint64_t>& args, bool considerMain = true);
