#include "Execution/JIT.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include "Templates/Custom/Utils.hpp"

void CustomPattern<mlir::arith::ConstantOp>::Codegen(
    mlir::arith::ConstantOp& op,
    JIT& jit,
    uint16_t,
    uint16_t)
{
    auto [val, size] = fillConstantFromAttr(jit, op.getValue(), op.getResult().getType());
    jit.setConstant(op.getResult(), size, std::move(val));
}

#ifdef LESS_TEMPLATES
void CustomPattern<mlir::func::CallOp>::Codegen(
    mlir::func::CallOp& op,
    JIT& jit,
    uint16_t sourceDepth,
    uint16_t cgDepth)
{
    genericCallOp(op, jit, sourceDepth, cgDepth);
}
#endif
