#include "Execution/JIT.hpp"
#include "Templates/PredefinedPatterns.hpp"
#include "common.hpp"

#include <numeric>
#include <sys/stat.h>

#ifdef VTUNE_SUPPORT
#include "jitprofiling.h"
#endif

#define TEMPLATE(depth, size, str, data) \
    do {                                 \
        LOG(depth, size, str);           \
        insert(depth, size, data);       \
    } while (false)

#define TEMPLATE_FRAGMENT(index)                                            \
    do {                                                                    \
        regionLocs.push_back(memory(codeDepth));                            \
        const auto templateSize = binaryTemplate.size(index);               \
        LOG(codeDepth, templateSize, op->getName().getStringRef().str(),    \
            templateId,                                                     \
            llvm::isa<mlir::FileLineColLoc>(op->getLoc())                   \
                ? llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine()  \
                : 0);                                                       \
        binaryTemplate.instantiate(memory(codeDepth, templateSize), index); \
        applyClobberMask(binaryTemplate.clobberMask);                       \
    } while (false)

namespace {
void PRINT_FRAME()
{
    uint64_t* sf;
    asm(LOAD_FRAME_PTR_ASM
        : "=r"(sf));
    llvm::outs() << "STACKFRAME CALL: ";
    for (uint8_t i = 0; i < 128; i++) {
        llvm::outs() << sf[i] << "|";
    }
    llvm::outs() << "\n";
}

void PRINT_VALUE(uint64_t value)
{
    llvm::outs() << "VALUE:" << value << "\n";
}

template <typename T>
void preorder(mlir::Block* block, T callback)
{
    llvm::DenseSet<mlir::Block*> visited;
    llvm::SmallVector<mlir::Block*> stack;
    stack.push_back(block);
    while (!stack.empty()) {
        auto next = stack.back();
        stack.pop_back();
        callback(next);
        for (auto s : next->getSuccessors()) {
            if (visited.insert(s).second) {
                stack.push_back(s);
            }
        }
    }
}
}

JIT::JIT(TemplateLibrary& lib)
    : RegisterExecutor("JIT")
    , m_library(lib)
    , m_returnOffset(0)
{
    registerSymbol("PRINT_STACKFRAME", reinterpret_cast<void*>(&PRINT_FRAME));
    registerSymbol("PRINT", reinterpret_cast<void*>(&PRINT_VALUE));
    registerSymbol("memmove", reinterpret_cast<void*>(&memmove));
    registerSymbol("memcpy", reinterpret_cast<void*>(&memcpy));
    registerSymbol("memset", reinterpret_cast<void*>(&memset));
    registerSymbol("strcmp", reinterpret_cast<void*>(&strcmp));
    registerSymbol("free", reinterpret_cast<void*>(&free));
    registerSymbol("calloc", reinterpret_cast<void*>(&calloc));
    registerSymbol("fstat", reinterpret_cast<void*>(&fstat));
    registerSymbol("lstat", reinterpret_cast<void*>(&lstat));
    registerSymbol("stat", reinterpret_cast<void*>(&stat));
    registerSymbol("_fopen", reinterpret_cast<void*>(&fopen));
}

void JIT::registerSymbol(llvm::StringRef name, void* function)
{
    m_symbols.insertFixupAddress(name, function);
}

void JIT::prepareImpl(mlir::ModuleOp* mod)
{
    {
        // insert the data section at the end of our allocated pages (in order for them to be in range of 32 bit)
        char* endPtr = setup(m_library.getDataSize() + MAX_ALIGN - 1);
        endPtr = reinterpret_cast<char*>((reinterpret_cast<uint64_t>(endPtr) + MAX_ALIGN - 1) & -MAX_ALIGN);
        // copy over all section memory
        m_library.copySections(endPtr);
        // register all symbols
        for (const auto& [name, offset] : m_library.DataSectionProvider::symbols()) {
            registerSymbol(name, reinterpret_cast<char*>(endPtr) + offset);
        }
    }
    ASSURE_TRUE(mod->lookupSymbol<mlir::FunctionOpInterface>("main"));
    ASSURE(m_argumentSizes.size(), ==, mod->lookupSymbol<mlir::FunctionOpInterface>("main").getNumArguments());

    {
        MEASURE(ScopeTimer::globalPreparation);
        // generate code for globals (things happening before main)
        mod->walk([&](mlir::LLVM::GlobalOp global) {
            prepareOp(global, 0, 0);
        });
        ASSURE(memory(0), ==, m_memRegion, << " globals should not generate code");
    }

    // this only resets temporary variables - globals are stored in m_symbols, which is kept
    reset();

    for (auto f : mod->getOps<mlir::FunctionOpInterface>()) {
        if (!f.isExternal()) {
            registerSymbol(f.getName(), reinterpret_cast<void*>(memory(0)));
            prepareOp(f, 0, 1);
            // reset variables again
            reset();
        }
    }

    m_blocks.assureAllSymbolsResolved();
    m_symbols.assureAllSymbolsResolved();

    flushICache();
#ifdef NO_MAP_WX
    make_memory_executable();
#endif

#ifdef VTUNE_SUPPORT
    const char* name = "jit-code";
#ifndef VTUNE_COMPILE
    auto info = generateLineInfo();
    std::string filename = "info.vtune";
#endif
    iJIT_Method_Load jmethod;
    jmethod.method_id = iJIT_GetNewMethodID();
    jmethod.method_name = const_cast<char*>(name);
    jmethod.method_load_address = m_memRegion;
    jmethod.method_size = SIZE;
#ifndef VTUNE_COMPILE
    jmethod.source_file_name = const_cast<char*>(filename.c_str());
    jmethod.line_number_size = info.size();
    jmethod.line_number_table = info.data();
#endif
    jmethod.class_file_name = nullptr;
    jmethod.class_id = 0;
    iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, (void*)&jmethod);
#endif
#ifdef LOGGING
    dump();
#endif
}

void JIT::prepareOp(mlir::Operation* op, size_t codeDepth, size_t sourceNestingDepth)
{
    CustomCodegen::Codegen(
        SkipCodegen::Skip(llvm::TypeSwitch<mlir::Operation*, void>(op)),
        *this,
        sourceNestingDepth,
        codeDepth)
        .Default([&](mlir::Operation* op) {
            prepareOpDefault(op, codeDepth, sourceNestingDepth);
        });
}

void JIT::prepareOpDefault(mlir::Operation* op, size_t codeDepth, size_t sourceNestingDepth)
{
    const auto libraryIt = m_library.find(*op);
    if (libraryIt == m_library.end()) {
        INFO("Skip: " << op->getName().getStringRef());
        return;
    }

    const auto templateId = libraryIt->second;
    if (m_library.isConstant(templateId)) {
        const auto [returnSizes, stackmap, constant] = m_library.getConstant(templateId);
        INFO("JIT " << op->getName().getStringRef() << " Stencil: " << templateId << "(constant)");
        size_t offset = 0;
        for (size_t i = 0; i < returnSizes.size(); i++) {
            const auto size = returnSizes[i];
            ASSURE(offset, <, constant.size());
            setConstant(op->getResult(i), size, const_cast<uint64_t*>(constant.data() + offset));
            offset += size.slots();
        }
        return;
    }

    const auto [returnSizes, stackmaps, binaryTemplate] = m_library.getTemplate(templateId);
    const bool isVarArgFun = llvm::isa<mlir::LLVM::LLVMFuncOp>(op) && llvm::cast<mlir::LLVM::LLVMFuncOp>(op).isVarArg();
    std::vector<uint64_t> args;
    args.reserve(/* operands */ op->getNumOperands() + /* return value */ 1 + /* stackmaps */ op->getNumRegions());
    std::vector<char*> regionLocs;
    regionLocs.reserve(op->getNumRegions() + 2);

    const auto kind = prepareOpCall(op, codeDepth, args);
    ASSURE(isVarArgFun, ||, binaryTemplate.fragments.size() == std::accumulate(op->getRegions().begin(), op->getRegions().end(), 1ul, [](size_t acc, mlir::Region& reg) {
        return acc + (reg.empty() ? 0 : 1);
    }),
        << templateId);
    ASSURE(!isVarArgFun, ||, binaryTemplate.fragments.size() == 1);
    uint32_t i = 0;
    const auto& regionOrdering = binaryTemplate.regions();
    // we can check whether the regions are ordered in ascending order just by looking at the first element, as we know,
    // that all regions must be represented somewhere
    const auto regionsInAscendingOrder = regionOrdering.empty() || regionOrdering.front() == 0;
    ASSURE(regionsInAscendingOrder, ||, (regionOrdering.empty() || regionOrdering.back() == 0));

    auto insertIt = args.cend();
    const auto inputArgs = args.size();
    for (uint32_t regionNr : regionOrdering) {
        ASSURE(regionNr, <, op->getNumRegions());
        auto& region = op->getRegion(regionNr);
        if (region.empty() || isVarArgFun)
            continue;

        ASSURE(i, <, stackmaps.size());
        const auto& stackmap = stackmaps[i];
        const bool requiresStack = [&] {
            uint32_t registerArgumentsSize = 0;
            for (auto elem : stackmap) {
                if (elem.slots() > MAX_SIZE_REGISTER)
                    return true;
                registerArgumentsSize += elem.slots();
            }
            return registerArgumentsSize > OPERAND_REGISTERS;
        }();
        if ((hasReturnOffset(kind) && region.getNumArguments()) || requiresStack) {
            // at least one argument in the region ... otherwise we would have dropped the
            // unrealized cast and with it the argument as well
            insertIt = args.insert(insertIt, stackOffset() * 8) + 1;
        }

        // ----------------
        TEMPLATE_FRAGMENT(i);
        // ----------------

        // recursive call to create the region
        const bool isSingleBlockRegion = region.getBlocks().size() == 1;
        // we can not inline regions with multiple blocks as this might imply multiple exits
        const auto inlineRegion = !binaryTemplate.breaksInlining() && isSingleBlockRegion;
        const auto regionDepth = inlineRegion ? codeDepth : codeDepth + 1;
        const auto regionSourceNestingDepth = sourceNestingDepth + 1;
        if (!inlineRegion) {
            const void* callAddr = memory(regionDepth);
#if defined(__x86_64__)
            const int64_t relAddr = reinterpret_cast<int64_t>(callAddr) - reinterpret_cast<int64_t>(memory(codeDepth)) - 5;
            ASSURE(relAddr, >=, INT32_MIN);
            ASSURE(relAddr, <=, INT32_MAX);
            const auto relAddr32 = static_cast<int32_t>(relAddr);
            // call 0x000000
            const uint8_t CODE[] = {0xe8, 0, 0, 0, 0};
            TEMPLATE(codeDepth, 5, "regionCall", CODE);
            UnalignedWrite(memory(codeDepth) - 4, relAddr32);
#elif defined(__aarch64__)
            const int64_t relAddr = reinterpret_cast<int64_t>(callAddr) - reinterpret_cast<int64_t>(memory(codeDepth));
            ASSURE(relAddr, >=, (-(1 << 27)));
            ASSURE(relAddr, <=, (1 << 27));
            uint32_t instr = (0b10010100000000000000000000000000) | ((static_cast<uint32_t>(relAddr) >> 2) & ((1 << 26) - 1));
            TEMPLATE(codeDepth, 4, "regionCall", &instr);
            const uint32_t storeRetAddr = 0xf81f0ffe; // str x30, [sp, #-16]!
            TEMPLATE(codeDepth + 1, 4, "saveReturn", &storeRetAddr);
#endif
        } else {
            // we have to at least simulate the "misalignment" of the stack upon entry of the other functions
            // x86 guarantees a 16 byte alignment at call site (+ 8 byte beeing pushed by the call instruction)
            // by just removing the call entirely, we would miss out on the additional 8 byte (breaking the abi)
            // -> we add a push here and add a pop at the end of each region
            const uint8_t CODE[] = {0x48, 0x83, 0xec, 0x08}; // sub rsp, 8
            TEMPLATE(codeDepth, 4, "regionEntry", CODE);
        }

        // generate code for the region:
        {
            const StackFrame stackframe(
                *this,
                stackmap,
                region,
                prepareRegion(
                    regionDepth,
                    kind,
                    stackmap,
                    region.getArguments()));
            preorder(&region.front(), [&](mlir::Block* block) {
                onBlockStart(regionDepth, block);
                for (auto& o : *block) {
                    prepareOp(&o, regionDepth, regionSourceNestingDepth);
                    if ((isSingleBlockRegion && o.hasTrait<mlir::OpTrait::IsTerminator>())
                        || (!isSingleBlockRegion && o.hasTrait<mlir::OpTrait::ReturnLike>())) {
                        mlir::Operation* terminator = &o;
                        ASSURE_TRUE(terminator->hasTrait<mlir::OpTrait::IsTerminator>());
                        const auto& operands = terminator->getOperands();
                        const auto position = (static_cast<uint8_t>(kind) - 2) * 2 + op->getNumOperands() + 1;
                        if (position < OPERAND_REGISTERS
                            && IsRegisterCC(kind)
                            && operands.size() == 1
                            && getStorage(operands.front()).size().slots() == 1) {
                            // in this special case just move it "on top of register"
                            const auto storage = getStorage(operands.front());
                            moveTo(regionDepth, storage, Storage::Register(position));
                        } else {
                            forceOnStack(
                                regionDepth,
                                stackOffset(),
                                terminator->getOperands(),
                                args,
                                insertIt,
                                false);
                        }
                        if (!inlineRegion) {
#ifdef __aarch64__
                            const uint32_t CODE = 0xf84107fe; // ldr x30, [sp], #16
                            TEMPLATE(codeDepth + 1, 4, "restoreReturn", &CODE);
#endif
                            Materializer<RegionTerminator>()(*this, regionDepth);
                        } else {
                            const uint8_t CODE[] = {0x48, 0x83, 0xc4, 0x08}; // add rsp, 8
                            TEMPLATE(codeDepth, 4, "regionExit", CODE);
                        }
                    }
                }
            });
        }
        i += 1;
        insertIt = regionsInAscendingOrder ? args.cend() : args.cbegin() + inputArgs;
    }

    // ----------------
    TEMPLATE_FRAGMENT(i);
    // ----------------

    // add address for "next" jumps (that haven't been optimized yet)
    regionLocs.push_back(memory(codeDepth));

    prepareReturnValues(op, codeDepth, kind, args, returnSizes);

#ifndef FIXED_STACKSIZE
    if (libraryIt->first.needStack()) [[unlikely]] {
        const auto stacksize = m_maxStackSpace * 8 + MAX_ALIGN + /*additional storage for tmps*/ STACK_SAFETY_REGION;
        const auto alignedStackSize = (stacksize + (MAX_ALIGN - 1)) & -MAX_ALIGN;
        args.push_back(alignedStackSize);
        // compare to all non-duplicate patch points (start with one as the first element is always duplicate)
        ASSURE(args.size(), ==, std::accumulate(binaryTemplate.patchpoints.cbegin(), binaryTemplate.patchpoints.cend(), 1ul, [](const auto agg, const auto pp) {
            return agg + (pp.duplicate ? 0 : 1);
        }),
            << " Id: " << templateId);
    }
#endif

    binaryTemplate.patch(
        regionLocs,
        args,
        [&](void* originalAddr) {
            const auto addr = memory(codeDepth + 1);
            JIT::Materializer<ExternalThunk>()(*this, codeDepth + 1, reinterpret_cast<uint64_t>(originalAddr));
            return addr;
        },
        [&](void* originalAddr) {
            align(codeDepth + 1);
            const auto addr = memory(codeDepth + 1);
            insert(codeDepth + 1, sizeof(addr), &originalAddr);
            return addr;
        },
        [&](const std::string& name, char* location) {
            void* addr = m_symbols.getOrNull(name);
            if (addr == nullptr) {
                DEBUG("didn't resolve " << name << " yet!");
#if defined(__x86_64__)
                if (reinterpret_cast<uint8_t*>(location)[-2] == 0x8b
                    && reinterpret_cast<uint8_t*>(location)[-3] == 0x48) {
                    // in case of a move instruction, the relocation was a GOT entry, but as we now know, that it is
                    // a close-range symbol (generated by us), we cant turn the mov into an lea
                    reinterpret_cast<uint8_t*>(location)[-2] = 0x8d;
                } else {
                    // must be a direct call
                    ASSURE(static_cast<uint32_t>(reinterpret_cast<uint8_t*>(location)[-1]), ==, 0xe8, << " " << name << " - " << templateId);
                }
                ASSURE(UnalignedRead<int32_t>(location), ==, -4);
#elif defined(__aarch64__)
                // in case of a ldr instruction, the relocation was a GOT entry, but as we now know, that it is
                // a close-range symbol (generated by us), we cant turn the ldr into an adr
                if ((*reinterpret_cast<uint32_t*>(location) & 0b01011000000000000000000000000000) == 0b01011000000000000000000000000000) {
                    [[maybe_unused]] const auto previousNopIsInFragment = [&]() -> bool {
                        uint32_t* nopslot = reinterpret_cast<uint32_t*>(location);
                        while (*nopslot != 0xd503201f)
                            nopslot--;
                        auto it = std::lower_bound(regionLocs.cbegin(), regionLocs.cend(),
                            reinterpret_cast<char*>(nopslot), [](const auto a, const auto b) {
                                return a <= b;
                            });
                        ASSURE_TRUE(it != regionLocs.cend());
                        ASSURE_TRUE(it != regionLocs.cbegin());
                        return reinterpret_cast<char*>(nopslot) >= *(it - 1);
                    };
                    ASSURE_TRUE(previousNopIsInFragment());
                    *reinterpret_cast<uint32_t*>(location) &= 0b00010000000000000000000000011111;
                }
#endif
                m_symbols.insertFixupLocation(name, location);
            }
            return addr;
        });
}

void JIT::onBlockStart(uint8_t depth, mlir::Block* block)
{
    const auto blockAddr = memory(depth);
    m_blocks.insertFixupAddress(block, blockAddr);

    for (auto& arg : block->getArguments()) {
        // reserve a stack slot for the variable if not already present
        getOrCreateSlot(arg);
    }

    if (block->getSinglePredecessor() == nullptr
        || block->getSinglePredecessor() != block->getPrevNode()) {
        applyClobberMask(CLOBBER_ALL);
    }
}

void JIT::insertBlockAddress(char* location, mlir::Block* block)
{
#ifdef __x86_64__
    UnalignedWrite(location, static_cast<int32_t>(-4u));
#endif
    m_blocks.insertFixupLocation(block, location);
}

uint64_t JIT::runImpl(mlir::ModuleOp*)
{
    INFO("Saved loads from stack: " << savedStackLoads << " of total " << stackLoads);
    ASSURE_TRUE(m_symbols.getOrNull("main") != nullptr);
    return invoke(m_symbols.getOrNull("main"));
}

void JIT::forceOnStack(
    size_t codeDepth,
    uint32_t stackOffset,
    const mlir::ValueRange& operands,
    std::vector<uint64_t>& args,
    std::vector<uint64_t>::const_iterator argsInsertPos,
    bool reuseStack)
{
    for (const auto& o : operands) {
        const auto storage = getStorage(o);
        const auto size = storage.size();
        auto position = !storage.isStack() ? stackOffset : storage.getOffset();
        if (size.align() > 8) {
            position = (position + MAX_ALIGN - 1) & -MAX_ALIGN;
        }

        stackOffset = !storage.isStack() ? position + size.slots() : stackOffset;
        if (!storage.isStack()) {
            moveTo(codeDepth, storage, Storage::Stack(position, size));
        }

        argsInsertPos = args.insert(argsInsertPos, position * 8) + 1;
        if (reuseStack
            && o.hasOneUse()
            && o.getDefiningOp() != nullptr
            && o.getDefiningOp()->getBlock() == o.getUsers().begin()->getBlock()) {
            clearSlot(storage);
        }
    }
    m_maxStackSpace = std::max(m_maxStackSpace, stackOffset);
}

void JIT::moveTo(uint32_t cgDepth, const Storage& from, const Storage& to)
{
    ASSURE_FALSE(to.isConstant());
    // use a switch to speed up the cascading if-else
    const auto size = from.size();
    const auto combinedMask = (from.mask() << 1) | to.mask();
    ASSURE(combinedMask, <=, 5);
    switch (combinedMask) {
        case 0: // from stack to stack
            ASSURE(from.getOffset(), !=, to.getOffset());
            for (uint32_t i = 0; i < size.slots(); i++) {
#ifdef __x86_64__
                LOG(cgDepth,
                    sizeof(MOV_STACK_TO_SCRATCH) + sizeof(MOV_SCRATCH_TO_STACK) + 8,
                    "CopyOnStack");
                ASM_THIS(MOV_STACK_TO_SCRATCH, static_cast<uint32_t>((from.getOffset() + i) * 8));
                ASM_THIS(MOV_SCRATCH_TO_STACK, static_cast<uint32_t>((to.getOffset() + i) * 8));
#else
                Materializer<CopyOnStack>()(*this, cgDepth, static_cast<uint64_t>(from.getOffset() + i) * 8, static_cast<uint64_t>(to.getOffset() + i) * 8);
#endif
            }
            break;
        case 1: // from stack to register
            for (uint32_t i = 0; i < size.slots(); i++) {
                Materializer<MoveStackToRegister>(static_cast<size_t>(to.getOffset() + i))(*this, cgDepth, static_cast<uint64_t>(from.getOffset() + i) * 8);
            }
            break;
        case 2: // from register to stack
            for (uint32_t i = 0; i < size.slots(); i++) {
                Materializer<MoveRegisterToStack>(from.getOffset() + i)(*this, cgDepth, static_cast<uint64_t>(to.getOffset() + i) * 8);
            }
            break;
        case 3: // from register to register
            ASSURE(static_cast<uint32_t>(size.slots()), ==, 1);
            if (from.getOffset() != to.getOffset())
                Materializer<MoveBetweenRegisters>(from.getOffset(), to.getOffset())(*this, cgDepth);
            break;
        case 4: // from constant to stack
            for (uint32_t i = 0; i < size.slots(); i++) {
#ifdef __x86_64__
                LOG(cgDepth,
                    sizeof(MOV_CONST_TO_SCRATCH) + sizeof(MOV_SCRATCH_TO_STACK) + 12,
                    "ConstantToStack");
                ASM_THIS(MOV_CONST_TO_SCRATCH, from.getAddress()[i]);
                ASM_THIS(MOV_SCRATCH_TO_STACK, static_cast<uint32_t>(to.getOffset() + i) * 8);
#else
                Materializer<MoveConstantToStack>()(*this, cgDepth, static_cast<uint64_t>(to.getOffset() + i) * 8, from.getAddress()[i]);
#endif
            }
            break;
        case 5: // from constant to register
            for (uint32_t i = 0; i < size.slots(); i++) {
                Materializer<MoveConstantToRegister>(to.getOffset() + i)(*this, cgDepth, from.getAddress()[i]);
            }
            break;
        default:
            UNREACHABLE();
    }
}

void JIT::moveTo(uint32_t depth, const mlir::Value& val, const Storage& to)
{
    moveTo(depth, getStorage(val), to);
}
