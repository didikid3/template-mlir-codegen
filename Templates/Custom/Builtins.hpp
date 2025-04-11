class JIT;
namespace mlir::func {
class ReturnOp;
}

namespace mlir::arith {
class ConstantOp;
}

template <>
struct CustomPattern<mlir::arith::ConstantOp>
{
    static void Codegen(mlir::arith::ConstantOp& op, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth);
};

template <>
struct CustomPattern<mlir::func::CallOp>
{
    static void Codegen(mlir::func::CallOp& op, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth);
};
