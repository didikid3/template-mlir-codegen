#include "Execution/JIT.hpp"
#include "Templates/PredefinedPatterns.hpp"
#include "common.hpp"

#include <numeric>

TEMPLATE_KIND JIT::prepareOpCall(
    mlir::Operation* op,
    size_t codeDepth,
    std::vector<uint64_t>& args,
    bool)
{
    forceOnStack(codeDepth, stackOffset(), op->getOperands(), args, args.cend());
    return TEMPLATE_KIND::STACK_PATCHED;
}

void JIT::prepareReturnValues(
    mlir::Operation* op,
    size_t,
    [[maybe_unused]] TEMPLATE_KIND kind,
    std::vector<uint64_t>& args,
    const ResultSizes& returnSizes)
{
    // prepare return values
    ASSURE_TRUE(hasReturnOffset(kind));
    ASSURE(op->getNumResults(), ==, returnSizes.size());
    for (size_t i = 0; i < returnSizes.size(); i++) {
        const auto r = op->getResult(i);
        const auto newStorage = getOrCreateSlot(r, returnSizes[i]);
        setStorage(r, newStorage);

        if (i == 0)
            args.push_back(static_cast<uint64_t>(newStorage.getOffset()) * 8);
    }
}

JIT::Storage JIT::prepareRegion(
    size_t,
    TEMPLATE_KIND,
    const StackMap&,
    const mlir::ValueRange&)
{
    return Storage::Stack(stackOffset(), TypeSize(0));
}
