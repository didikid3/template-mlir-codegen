#pragma once
#include "Preparation/Utils.hpp"
#include "common.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include <functional>
#include <unordered_set>
#include <vector>

namespace mlir::LLVM {
class LLVMFuncOp;
}

/**
 * @brief Semantics of an mlir instruction distilled into a module
 *
 * @details
 * Contains the module for representing the semantics lowered to LLVMIR but still containing
 * unrealized-conversion-casts for inputs/outputs.
 * Main function is named `kInternalTemplateFunction`.
 * Intended workflow:
 *  #1 create an abstraction for a particular op (using `FromOp`)
 *  #2 call `getTemplates` to derive respective modules for certain template kinds
 *  #3 use `take...()` to take over the stackmap and result sizes to pass them into the TemplateLibrary
 */
class Abstraction
{
public:
    using IRLoweringFunction = std::function<void(mlir::Operation& op)>;

    static Abstraction FromOp(
        const IRLoweringFunction& lowering,
        mlir::Operation& op,
        const std::unordered_set<uint8_t>& constantOperands);

    mlir::ModuleOp getTemplates(
        size_t id,
        const std::vector<TEMPLATE_KIND>& kinds);

    StackMaps&& takeStackMaps();
    ResultSizes&& takeResultSizes();

    mlir::ModuleOp& getModule();
    bool isNopCast() const;

    static void defaultLowering(mlir::Operation& op);
    static void quickLowering(mlir::Operation& op);

private:
    Abstraction(mlir::ModuleOp&& mod, ResultSizes&& sizes, bool isFunc);
    Abstraction();

    static Abstraction createAbstraction(
        const IRLoweringFunction& lowering,
        mlir::Operation& op,
        const std::unordered_set<uint8_t>& constantOperands);
    static StackMaps propagateChanges(
        mlir::LLVM::LLVMFuncOp& in,
        TEMPLATE_KIND kind,
        bool isFuncPattern = false);
    static StackMaps stackCCasArgs(mlir::LLVM::LLVMFuncOp& stencilFunc, mlir::OpBuilder& builder, size_t id);
    static StackMaps stackCCasGlobals(mlir::LLVM::LLVMFuncOp& stencilFunc, mlir::OpBuilder& builder, size_t id, bool isFuncPattern);
    static StackMaps registerCC(mlir::LLVM::LLVMFuncOp& stencilFunc, mlir::OpBuilder& builder, size_t id, bool isFuncPattern);

    mlir::ModuleOp m_module;
    // stackmaps are overwritten by each call to `getTemplates` implying:
    // - it is not valid until the first `getTemplates` call
    // - it reflects the stackmap for the latest derived template kind
    StackMaps m_stackmaps;
    ResultSizes m_resultSizes;
    bool m_isFunction;
    bool m_isNopCast;
};
