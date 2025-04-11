#pragma once

#include "Preparation/Utils.hpp"
#include "common.hpp"

#include "llvm/IR/Module.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include <bitset>
#include <vector>

namespace llvm {
class Target;
class raw_pwrite_stream;
}

namespace compiler {
ClobberMask compileMLIRToBinary(
    mlir::ModuleOp& module,
    std::string filename,
    const std::vector<size_t>& regionStencils);
ClobberMask emitFile(
    llvm::Module& llvmModule,
    llvm::TargetMachine& targetMachine,
    llvm::raw_pwrite_stream& dest,
    llvm::CodeGenFileType fileType);
std::unique_ptr<llvm::TargetMachine> getTargetMachine(llvm::CodeModel::Model, llvm::TargetOptions);
const llvm::Target* getTarget();
const char* getTargetTriple();
} // namespace compiler
