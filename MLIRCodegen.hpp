#pragma once

#include "Execution/Executor.hpp"
#include "TemplateBuilder.hpp"
#include "TemplateLibrary.hpp"
#include "common.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"

using LowerT = Abstraction::IRLoweringFunction;

template <typename Executor>
TimedResult run(
    mlir::ModuleOp mod,
    const std::string& libFolder,
    const LowerT& lowering,
    const std::vector<uint64_t>& codeSizes,
    const std::vector<uint64_t>& args)
{
    mlir::OwningOpRef<mlir::ModuleOp> modCloned = mod.clone();
    TemplateLibrary lib = TemplateLibrary::FromFolder(libFolder, *mod.getContext());
    TemplateBuilder builder(lib, lowering);
    modCloned->walk([&](mlir::Operation* op) {
        if ((llvm::isa<mlir::LLVM::CallOp>(op) && llvm::cast<mlir::LLVM::CallOp>(op).getCallee().value_or("").starts_with("llvm.memmove"))
#ifdef LINGODB_LLVM
            || llvm::isa<mlir::LLVM::MemsetOp>(op)
            || llvm::isa<mlir::LLVM::MemmoveOp>(op)
            || llvm::isa<mlir::LLVM::MemcpyOp>(op)
            || llvm::isa<mlir::LLVM::AbsOp>(op)
            || llvm::isa<mlir::LLVM::Prefetch>(op)
            || llvm::isa<mlir::LLVM::CountLeadingZerosOp>(op)
            || llvm::isa<mlir::LLVM::CountTrailingZerosOp>(op)
#endif
        ) {
            if (lib.contains(*op)) {
                return;
            }
            std::unordered_set<uint8_t> immargs = {3};
            if (llvm::isa<mlir::LLVM::AbsOp>(op)
                || llvm::isa<mlir::LLVM::CountLeadingZerosOp>(op)
                || llvm::isa<mlir::LLVM::CountTrailingZerosOp>(op))
                immargs = {1};
            if (llvm::isa<mlir::LLVM::Prefetch>(op))
                immargs = {1, 2, 3};
            builder.addOpWithSignature(
                *op,
                Abstraction::FromOp(lowering, *op, immargs),
                Signature(*op));
        }
    });
    builder.addOpsFromModule(*modCloned.operator->());
    auto newExecutor = [&] {
        Executor executor{lib};
        for (uint8_t i = 0; i < codeSizes.size(); i++) {
            executor.setCodeSize(i, codeSizes[i]);
        }
        return executor;
    };

    if (COMPILE_ONLY && RUNS > 1) {
        for (int i = 0; i < (RUNS - 1); i++) {
            newExecutor()(&mod, args);
        }
    }
    return newExecutor()(&mod, args);
}

template <typename Executor, typename... Args>
TimedResult run(
    mlir::ModuleOp mod,
    const std::vector<uint64_t>& args)
{
    Executor executor;
    return executor(&mod, args);
}
