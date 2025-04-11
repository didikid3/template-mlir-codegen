#pragma once

#include "Execution/Executor.hpp"
#include "Execution/MemInterface.hpp"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "Preparation/Abstraction.hpp"

class Baseline
    : public Executor<Baseline>
    , public ExecutableMemoryInterface
{
public:
    Baseline()
        : Executor<Baseline>("Baseline")
    {
    }

    void prepareImpl(mlir::ModuleOp* mod)
    {
        setup(0);
        auto func = mod->lookupSymbol<mlir::FunctionOpInterface>("main");
        REL_ASSURE(static_cast<bool>(func), ==, true);
        std::function<void(mlir::Operation*)> processOp;
        processOp = [&](mlir::Operation* op) {
            for (auto& region : op->getRegions()) {
                for (auto& block : region) {
                    for (auto& op : block) {
                        REL_ASSURE(op.hasAttr("__dummy__"), ==, false);
                    }
                }
            }
        };

        processOp(mod->operator mlir::Operation*());
    }

    uint64_t runImpl(mlir::ModuleOp*)
    {
        return 0;
    }
};
