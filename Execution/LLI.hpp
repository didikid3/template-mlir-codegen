#pragma once

#include "Executor.hpp"

#include "llvm/ExecutionEngine/ExecutionEngine.h"

namespace mlir {
class ModuleOp;
}

std::unique_ptr<llvm::ExecutionEngine> getInterpreter(
    std::unique_ptr<llvm::Module>&& llvmModule,
    std::string* errorStr = nullptr);

class LLI
    : public Executor<LLI>
{
public:
    LLI();

    void prepareImpl(llvm::Module* mod);
    uint64_t runImpl(mlir::ModuleOp* mod);

private:
    llvm::LLVMContext llvmContext;

    std::unique_ptr<llvm::ExecutionEngine> m_executionEngine;
    llvm::Function* m_entryFunc;

    std::string m_errorMsg;
};
