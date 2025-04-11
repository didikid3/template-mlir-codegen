#include "Execution/LLI.hpp"
#include "common.hpp"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/Interpreter.h"

std::unique_ptr<llvm::ExecutionEngine> getInterpreter(
    std::unique_ptr<llvm::Module>&& llvmModule,
    std::string* errorStr)
{
    llvm::EngineBuilder builder(std::move(llvmModule));
    builder.setErrorStr(errorStr);
    builder.setEngineKind(llvm::EngineKind::Interpreter);
    builder.setOptLevel(llvm::CodeGenOpt::None);

    auto execEngine = std::unique_ptr<llvm::ExecutionEngine>(builder.create());
    ASSURE_TRUE(execEngine, << *errorStr);
    return execEngine;
}

LLI::LLI()
    : Executor("LLI")
{
}

void LLI::prepareImpl(llvm::Module* mod)
{
    m_entryFunc = mod->getFunction("main");
    ASSURE_TRUE(m_entryFunc != nullptr);
    m_executionEngine = getInterpreter(std::move(m_llvmModule), &m_errorMsg);
}

uint64_t LLI::runImpl(mlir::ModuleOp*)
{
    std::vector<llvm::GenericValue> args(m_argumentVector.size());
    for (unsigned i = 0; i < args.size(); i++) {
        const auto t = m_entryFunc->getFunctionType()->getParamType(i);
        if (llvm::isa<llvm::IntegerType>(t))
            args[i].IntVal = llvm::APInt(t->getIntegerBitWidth(), m_argumentVector[i]);
        else
            args[i].IntVal = llvm::APInt(64, m_argumentVector[i]);
    }
    // calling some external dependencies (e.g. malloc) requires FFI support (LLVM_ENABLE_FFI=ON)
    return m_executionEngine->runFunction(m_entryFunc, args).IntVal.getZExtValue();
}
