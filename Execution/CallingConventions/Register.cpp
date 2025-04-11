#include "Execution/JIT.hpp"
#include "Templates/PredefinedPatterns.hpp"
#include "common.hpp"

#include <numeric>

TEMPLATE_KIND JIT::prepareOpCall(
    mlir::Operation* op,
    size_t codeDepth,
    std::vector<uint64_t>& args,
    bool reuseStack)
{
    uint8_t registerOffset = 1;
    uint32_t currentStackOffset = stackOffset();
    for (const auto& o : op->getOperands()) {
        const auto storage = getStorage(o);
        const auto size = storage.size();

        if (size.slots() <= MAX_SIZE_REGISTER && registerOffset + size.slots() <= OPERAND_REGISTERS + 1) {
            if (size.bytes() <= 8 && storage.isStack() && isValid(storage.tmpRegister(), o)) {
#ifndef RELEASE
                savedStackLoads += 1;
#endif
                moveTo(codeDepth, Storage::Register(storage.tmpRegister()), Storage::Register(registerOffset));
            } else if (!storage.isRegister() || storage.getOffset() != registerOffset) {
                moveTo(codeDepth, storage, Storage::Register(registerOffset));
            }
#ifndef RELEASE
            stackLoads += 1;
#endif
            registerOffset += size.slots();
        } else {
            uint32_t position = storage.isConstant() ? currentStackOffset : storage.getOffset();
            if (!storage.isStack()) {
                ASSURE_TRUE(storage.isConstant());
                // we might have to align the actual position...
                if (size.align() > 8) {
                    position = (position + MAX_ALIGN - 1) & -MAX_ALIGN;
                }
                currentStackOffset = position + size.slots();
                moveTo(codeDepth, storage, Storage::Stack(position, size));
            }
            args.push_back(position * 8);
        }
        if (reuseStack
            && o.hasOneUse()
            && o.getDefiningOp() != nullptr
            && o.getDefiningOp()->getBlock() == o.getUsers().begin()->getBlock()) {
            clearSlot(storage);
        }
    }
    m_maxStackSpace = std::max(m_maxStackSpace, currentStackOffset);
    return TEMPLATE_KIND::REGISTERS;
}

void JIT::prepareReturnValues(
    mlir::Operation* op,
    size_t codeDepth,
    TEMPLATE_KIND,
    std::vector<uint64_t>& args,
    const ResultSizes& returnSizes)
{
    // put the results back to the stack
    ASSURE(op->getNumResults(), ==, returnSizes.size());
    // the total size of return registers should not exceed 9 as after that, the values are no longer
    // placed into the registers as expected
    ASSURE(std::accumulate(returnSizes.begin(), returnSizes.end(), 0ul, [](auto acc, auto size) {
        const auto regSize = size.slots();
        return acc + (regSize > MAX_SIZE_REGISTER ? 0 : regSize);
    }),
        <=, OPERAND_REGISTERS);
    uint8_t registerOffset = 1;
    for (size_t i = 0; i < returnSizes.size(); i++) {
        const auto r = op->getResult(i);
        const auto size = returnSizes[i];
#ifdef OPTIMIZED_VARIABLE_PASSING
        // a redundant store of a values requires:
        // - the value to have a single use
        // - the user is the next op
        // - it is the first operands of the user
        // - the value size is <= MAX_SIZE_REGISTER
        // - only a single operand
        // - the user is no CondBr op
        const bool redundantStore = i == 0
            && r.hasOneUse()
            && *r.user_begin() == op->getNextNode()
            && *r.user_begin()->operand_begin() == r
            && size.slots() <= MAX_SIZE_REGISTER
            && op->getNumResults() == 1
            && !llvm::isa<mlir::LLVM::SwitchOp>(*r.user_begin())
            && !llvm::isa<mlir::LLVM::CondBrOp>(*r.user_begin())
            && !llvm::isa<mlir::LLVM::BrOp>(*r.user_begin());
        ASSURE(!redundantStore, ||, r.user_begin()->getBlock() == op->getBlock());
        if (redundantStore) {
            setStorage(r, Storage::Register(registerOffset, size));
            return;
        }
#endif
        auto newStorage = getOrCreateSlot(r, size);
        if (size.slots() <= MAX_SIZE_REGISTER) {
            moveTo(codeDepth, Storage::Register(registerOffset, size), newStorage);
#ifdef TMP_REGISTER_COPY
            if (size.slots() <= 1) {
                // scatter somewhere into a scratch register
                auto tmp = tmpRegister(r);
                if (tmp != 0) {
                    ASSURE_TRUE(newStorage.isStack());
                    newStorage = Storage::Stack(newStorage.getOffset(), TypeSize(64), tmp);
                    moveTo(codeDepth, Storage::Register(registerOffset), Storage::Register(tmp));
                    set(tmp, r);
                }
            }
#endif
            registerOffset += size.slots();
        }
        setStorage(r, newStorage, true);

        if (i == 0)
            args.push_back(static_cast<uint64_t>(newStorage.getOffset()) * 8);
    }
}

JIT::Storage JIT::prepareRegion(
    size_t depth,
    TEMPLATE_KIND,
    const StackMap& stackmap,
    [[maybe_unused]] const mlir::ValueRange& args)
{
    ASSURE(args.size(), ==, stackmap.size());

    uint32_t registerOffset = 1;
    uint32_t stackOffsetTmp = stackOffset();
    for (size_t i = 0; i < stackmap.size(); i++) {
        const auto size = stackmap[i];
        if (size.slots() <= MAX_SIZE_REGISTER && registerOffset + size.slots() <= OPERAND_REGISTERS + 1) {
            moveTo(depth, Storage::Register(registerOffset, size), Storage::Stack(stackOffsetTmp, size));
            registerOffset += size.slots();
        }
        stackOffsetTmp += size.slots();
    }
    return Storage::Stack(stackOffset(), TypeSize(0));
}
