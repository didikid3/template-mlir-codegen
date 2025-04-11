#define CUSTOM_CODEGEN(opT)                                                             \
    template <>                                                                         \
    struct CustomPattern<opT>                                                           \
    {                                                                                   \
        static void Codegen(opT& op, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth); \
    }

class JIT;
class TemplateBuilder;
namespace mlir::LLVM {
class BrOp;
class CondBrOp;
class ConstantOp;
class UndefOp;
class ReturnOp;
class AllocaOp;
class GlobalOp;
class AddressOfOp;
class InsertValueOp;
class SwitchOp;
class CallOp;
}
CUSTOM_CODEGEN(mlir::LLVM::BrOp);
CUSTOM_CODEGEN(mlir::LLVM::CondBrOp);
CUSTOM_CODEGEN(mlir::LLVM::ConstantOp);
CUSTOM_CODEGEN(mlir::LLVM::UndefOp);
CUSTOM_CODEGEN(mlir::LLVM::AllocaOp);
CUSTOM_CODEGEN(mlir::LLVM::GlobalOp);
CUSTOM_CODEGEN(mlir::LLVM::AddressOfOp);
CUSTOM_CODEGEN(mlir::LLVM::InsertValueOp);
CUSTOM_CODEGEN(mlir::LLVM::SwitchOp);
CUSTOM_CODEGEN(mlir::LLVM::CallOp);

#undef CUSTOM_CODEGEN
