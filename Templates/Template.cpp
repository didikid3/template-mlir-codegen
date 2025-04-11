
#include "Templates/Template.hpp"
#include "common.hpp"

#include <iomanip>
#include <iostream>
#include <unordered_set>

size_t Code::instantiate(char* targetLoc) const
{
    MEASURE(ScopeTimer::copyAndPatching);
    memcpy(targetLoc, location, size);
    return size;
}

void Template::trim()
{
    ASSURE_FALSE(fragments.empty());
    auto& code = fragments.back();
#ifdef __x86_64__
    // if it ends with an ret instruction, do not try to trim the end
    if (code.location[code.size - 1] == (char)0xc3) {
        return;
    }
#endif

    // eliminate any tail jumps to the next template
    // only exception is, if the template only consists of a tail-call (internal template)
    if (code.size > TAIL_CALL_INSTR_SIZE
        && !internalJumps.empty()
        && internalJumps.back().isTailCall(code.size)) {
        internalJumps.pop_back();
        code.size -= TAIL_CALL_INSTR_SIZE;
    }
}

std::ostream& operator<<(std::ostream& os, const Code& stencil)
{
    os << "===========\n";
    for (unsigned i = 0; i < stencil.size; i++) {
        os << std::setfill('0') << std::setw(2) << std::right << std::hex << static_cast<uint32_t>(reinterpret_cast<const uint8_t*>(stencil.location)[i]) << " ";
    }
    os << "\n===========\n";
    return os;
}

std::ostream& operator<<(std::ostream& os, const Template& stencil)
{
    for (auto& x : stencil.fragments) {
        os << x;
    }
    return os;
}

void PatchPoint::patch(char* addr, uint64_t arg) const
{
    switch (kind) {
#if defined(__x86_64__)
        case PatchPoint::x86_64bit:
            UnalignedAddAndWriteback<uint64_t>(addr + offset, arg);
            break;
        case PatchPoint::x86_32bit:
            ASSURE(kind, ==, PatchPoint::x86_32bit);
            ASSURE(arg, <=, UINT32_MAX);
            UnalignedAddAndWriteback<uint32_t>(addr + offset, static_cast<uint32_t>(arg));
            break;
#elif defined(__aarch64__)
        case PatchPoint::aarch64_mov0:
        case PatchPoint::aarch64_mov1:
        case PatchPoint::aarch64_mov2:
        case PatchPoint::aarch64_mov3: {
            const auto shift = kind * 16;
            const auto subelement = static_cast<uint32_t>(static_cast<uint16_t>(arg >> shift)) << 5;
            uint32_t* instr = reinterpret_cast<uint32_t*>(addr);
            instr[offset / 4] |= subelement;
            break;
        }
        case PatchPoint::aarch64_branch: {
            const auto patchValue = static_cast<int64_t>(arg) - reinterpret_cast<int64_t>(addr + 4);
            ASSURE(patchValue, <=, (1 << 27));
            ASSURE(patchValue, >=, -(1 << 27));
            uint32_t* instr = reinterpret_cast<uint32_t*>(addr);
            instr[offset / 4] |= ((static_cast<uint32_t>(patchValue) >> 2) & ((1 << 26) - 1));
            break;
        }
#endif
        default:
            UNREACHABLE();
    }
}

Template::Template(
    std::vector<Code>&& fragments_,
    std::vector<PatchPoint>&& pps,
    std::vector<InternalJump>&& inJump,
    std::vector<ExternalCall>&& extCalls,
    std::vector<Loc>&& labels_,
    std::vector<uint8_t>&& regionOrder_,
    ClobberMask clobbers,
    bool canInline_)
    : fragments(std::move(fragments_))
    , labels(std::move(labels_))
    , canInline(canInline_)
    , patchpoints(std::move(pps))
    , internalJumps(std::move(inJump))
    , externalCalls(std::move(extCalls))
    , regionOrder(std::move(regionOrder_))
    , clobberMask(clobbers)
{
    trim();
    // the hack with the global sections requires to enumerate all fragments,
    // therefore more than 10 havn't been testet yet... might work out of the box
    ASSURE(fragments.size(), <=, 10);
    ASSURE((std::unordered_set<uint8_t>(regionOrder.cbegin(), regionOrder.cend()).size()), ==, regionOrder.size());
}

bool Template::isReadyForExecution() const
{
    return fragments.size() == 1 && patchpoints.empty() && internalJumps.empty() && externalCalls.empty();
}

void Template::executeInplace(uint64_t* stackframe, const std::vector<uint64_t>& args) const
{
    std::vector<uint64_t> allArgs;
    allArgs.reserve(args.size() + 1);
    allArgs.push_back(reinterpret_cast<uint64_t>(stackframe));
    allArgs.insert(allArgs.cend(), args.cbegin(), args.cend());
    invokeFromVector(location(), allArgs, false);
}

bool Template::requiresStackAlign() const
{
    return !externalCalls.empty();
}

bool Template::isValid() const
{
    return !fragments.empty();
}

bool Template::breaksInlining() const
{
    return !canInline && fragments.size() > 1;
}

const std::vector<uint8_t>& Template::regions() const
{
    // we require the regions to be sorted in either ascending or descending order as this
    // simplifies handling of the arguments significantly
    ASSURE(std::is_sorted(regionOrder.cbegin(), regionOrder.cend()),
        ||, std::is_sorted(regionOrder.rbegin(), regionOrder.rend()));
    ASSURE(regionOrder.empty(), ||, (regionOrder.front() == 0 || regionOrder.back() == 0));
    return regionOrder;
}

uint32_t Template::size(uint8_t i) const
{
    return fragments[i].size;
}

size_t Template::instantiate(char* targetLoc, uint32_t indx) const
{
    ASSURE(indx, <, fragments.size());
    return fragments[indx].instantiate(targetLoc);
}

void* Template::location() const
{
    ASSURE(fragments.size(), ==, 1);
    ASSURE_TRUE(isReadyForExecution());
    return reinterpret_cast<void*>(fragments[0].location);
}

void Template::patch(
    const std::vector<char*>& addresses,
    const std::vector<uint64_t>& args,
    std::function<char*(void*)> createThunk,
    std::function<char*(void*)> createEntry,
    std::function<void*(const std::string&, char*)> resolveSym) const
{
    MEASURE(ScopeTimer::copyAndPatching);
    ASSURE(fragments.size() + 1, ==, addresses.size()); // one additional address for continuation
    fixupInternalJumps(addresses);
    fixupPatchpoints(addresses, args);
    fixupExternalCalls(
        addresses,
        createThunk,
        createEntry,
        resolveSym);
}

void Template::fixupInternalJumps(
    const std::vector<char*>& addresses) const
{
    for (const auto jmp : internalJumps) {
        ASSURE(jmp.fragment, <, addresses.size());
        const auto addr = [&]() {
            if (jmp.target == InternalJump::NEXT_TARGET) {
                return addresses.back();
            } else {
                ASSURE(jmp.target, <=, labels.size());
                const auto label = labels[jmp.target];
                ASSURE(label.fragment, <=, addresses.size());
                return addresses[label.fragment] + label.offset;
            }
        }();
        auto relAddr = reinterpret_cast<int64_t>(addr) - reinterpret_cast<int64_t>(addresses[jmp.fragment] + jmp.offset);
#if defined(__x86_64__)
        relAddr -= 4; // as the offset is relative to the "end" of the instruction, substract the size
        if (relAddr == 0) {
            ASSURE(addresses[jmp.fragment][jmp.offset - 1], ==, (char)0xe9);
            // this mainly targets the cases when we split up the code using -basic-block-sections and now stick it back together
            // in those cases jumps with a distance of zero might occur -> make them a nop
            addresses[jmp.fragment][jmp.offset - 1] = (char)0x0f;
            addresses[jmp.fragment][jmp.offset - 0] = (char)0x1f;
            addresses[jmp.fragment][jmp.offset + 1] = (char)0x44;
            addresses[jmp.fragment][jmp.offset + 2] = (char)0x00;
            addresses[jmp.fragment][jmp.offset + 3] = (char)0x00;
        } else {
            ASSURE(relAddr, >=, INT32_MIN);
            ASSURE(relAddr, <=, INT32_MAX);
            UnalignedWrite(
                addresses[jmp.fragment] + jmp.offset,
                static_cast<int32_t>(relAddr));
        }
#elif defined(__aarch64__)
        // on aarch64 we only support tail-calls, which haven't been optimized (cases like if-else etc.)
        ASSURE(jmp.target, ==, InternalJump::NEXT_TARGET);
        ASSURE(relAddr, >=, (-(1 << 27)));
        ASSURE(relAddr, <=, (1 << 27));
        uint32_t* instr = reinterpret_cast<uint32_t*>(addresses[jmp.fragment]);
        instr[jmp.offset / 4] |= (static_cast<uint32_t>(relAddr) >> 2) & ((1 << 26) - 1);
#endif
    }
}

void Template::fixupExternalCalls(
    const std::vector<char*>& addresses,
    [[maybe_unused]] std::function<char*(void*)> createThunk,
    std::function<char*(void*)> createEntry,
    std::function<void*(const std::string&, char*)> resolveSym) const
{
    for (const auto& call : externalCalls) {
        ASSURE(static_cast<uint32_t>(call.fragment), <, addresses.size());
        auto addr = call.hasAddr ? call.addr : resolveSym(*reinterpret_cast<std::string*>(call.addr), addresses[call.fragment] + call.offset);
        if (!call.hasAddr && addr == nullptr)
            continue;
        addr = call.isIndirect ? createEntry(addr) : addr;
        auto relAddr = reinterpret_cast<int64_t>(addr) - reinterpret_cast<int64_t>(addresses[call.fragment] + call.offset);
#if defined(__x86_64__)
        const auto addend = UnalignedRead<int32_t>(addresses[call.fragment] + call.offset);
        relAddr += addend;
        if (INT32_MIN <= relAddr && relAddr <= INT32_MAX) {
            UnalignedWrite(
                addresses[call.fragment] + call.offset,
                static_cast<int32_t>(relAddr));
        } else {
            ASSURE_FALSE(call.isIndirect);
            relAddr = reinterpret_cast<int64_t>(createThunk(addr)) - reinterpret_cast<int64_t>(addresses[call.fragment] + call.offset) + addend;
            ASSURE(relAddr, >=, INT32_MIN);
            ASSURE(relAddr, <=, INT32_MAX);
            UnalignedWrite(
                addresses[call.fragment] + call.offset,
                static_cast<int32_t>(relAddr));
        }
#elif defined(__aarch64__)
        auto instr = reinterpret_cast<uint32_t*>(addresses[call.fragment]);
        if (call.isIndirect) {
            ASSURE(relAddr, >=, (-(1 << 20)));
            ASSURE(relAddr, <=, (1 << 20));
            instr[call.offset / 4] |= ((static_cast<uint32_t>(relAddr) >> 2) & ((1u << 19) - 1)) << 5;
        } else {
            if (-(1 << 27) <= relAddr && relAddr <= (1 << 27)) {
                instr[call.offset / 4] |= (static_cast<uint32_t>(relAddr) >> 2) & ((1u << 26) - 1);
            } else {
                relAddr = reinterpret_cast<int64_t>(createThunk(addr)) - reinterpret_cast<int64_t>(addresses[call.fragment] + call.offset);
                ASSURE(relAddr, >=, (-(1 << 27)));
                ASSURE(relAddr, <=, (1 << 27));
                instr[call.offset / 4] |= (static_cast<uint32_t>(relAddr) >> 2) & ((1u << 26) - 1);
            }
        }
#endif
    }
}

void Template::fixupPatchpoints(
    const std::vector<char*>& addresses,
    const std::vector<uint64_t>& args) const
{
    auto arg = args.cbegin();
    for (auto& pp : patchpoints) {
        if (!pp.duplicate)
            arg++;
        ASSURE(pp.fragment, <, addresses.size());
        ASSURE_TRUE(arg != args.cend(), << " arguments: " << args.size());
        pp.patch(addresses[pp.fragment], *arg);
    }
}

uint64_t invokeFromVector(void* function, const std::vector<uint64_t>& args, bool considerMain)
{
    uint64_t result = 0;
    const auto invoke = [&]<typename... Args>(Args... args) {
        typedef uint64_t (*DynFunc)(Args...);
        result = reinterpret_cast<DynFunc>(function)(args...);
    };

    if (considerMain && MAIN) {
        ASSURE(args.size(), ==, 2);
        invoke(static_cast<int>(args[0]), reinterpret_cast<char*>(args[1]));
        return result;
    }

    auto invokeImpl = [&]<size_t... Indx>(std::index_sequence<Indx...>) {
        invoke(args[Indx]...);
    };
    if (args.size() == 0)
        invokeImpl(std::make_index_sequence<0>{});
    else if (args.size() == 1)
        invokeImpl(std::make_index_sequence<1>{});
    else if (args.size() == 2)
        invokeImpl(std::make_index_sequence<2>{});
    else if (args.size() == 3)
        invokeImpl(std::make_index_sequence<3>{});
    else if (args.size() == 4)
        invokeImpl(std::make_index_sequence<4>{});
    else if (args.size() == 5)
        invokeImpl(std::make_index_sequence<5>{});
    else
        UNREACHABLE("too many arguments to call");
    return result;
}
