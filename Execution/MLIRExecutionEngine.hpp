#pragma once
#include "Executor.hpp"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"

class MLIRExecEngine : public Executor<MLIRExecEngine>
{
public:
    explicit MLIRExecEngine(int optLevel);

    void prepareImpl(llvm::Module*);
    uint64_t runImpl(mlir::ModuleOp*);

private:
    const int optLevel;
    std::unique_ptr<llvm::orc::LLJIT> lljit;
    std::string functionName;
    void* function;
};

class MLIRExecEngineO0
    : public MLIRExecEngine
{
public:
    MLIRExecEngineO0()
        : MLIRExecEngine(0)
    {
    }
};

class MLIRExecEngineO1
    : public MLIRExecEngine
{
public:
    MLIRExecEngineO1()
        : MLIRExecEngine(1)
    {
    }
};

class MLIRExecEngineO2
    : public MLIRExecEngine
{
public:
    MLIRExecEngineO2()
        : MLIRExecEngine(2)
    {
    }
};

class MLIRExecEngineO3
    : public MLIRExecEngine
{
public:
    MLIRExecEngineO3()
        : MLIRExecEngine(3)
    {
    }
};
