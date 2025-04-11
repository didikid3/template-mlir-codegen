#include "Execution/MLIRExecutionEngine.hpp"
#include "TemplateCompiler.hpp"

#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/Support/Error.h"

MLIRExecEngine::MLIRExecEngine(int optLevel)
    : Executor("MLIRExecEngineOpt" + std::to_string(optLevel))
    , optLevel(optLevel)
{
}

void MLIRExecEngine::prepareImpl(llvm::Module* mod)
{
    functionName = "main";
#if defined(__x86_64__)
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86AsmParser();
#elif defined(__aarch64__)
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64AsmParser();
#endif

    if (LLVM_OPT && optLevel > 0) {
        auto ok = mlir::makeOptimizingTransformer(
            optLevel,
            0,
            nullptr)(mod);
        REL_ASSURE(static_cast<bool>(ok), ==, false);
    }

    llvm::TargetOptions targetOptions;
    targetOptions.EnableFastISel = (optLevel == 0);

    auto targetMachineBuilder = llvm::cantFail(llvm::orc::JITTargetMachineBuilder::detectHost());
    if (LLVM_CMODEL == 0)
        targetMachineBuilder.setCodeModel(llvm::CodeModel::Small);
    else if (LLVM_CMODEL == 1)
        targetMachineBuilder.setCodeModel(llvm::CodeModel::Medium);
    else
        targetMachineBuilder.setCodeModel(llvm::CodeModel::Large);
    targetMachineBuilder.setCodeGenOptLevel(llvm::CodeGenOpt::getLevel(optLevel).value());
    targetMachineBuilder.setOptions(std::move(targetOptions));
    if (DUMP_CODE) {
        auto targetMachine = llvm::cantFail(targetMachineBuilder.createTargetMachine());
        mod->setTargetTriple(targetMachine->getTargetTriple().normalize());
        mod->setDataLayout(targetMachine->createDataLayout());
        compiler::emitFile(*mod, *targetMachine, llvm::outs(), llvm::CGFT_AssemblyFile);
    }

    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(targetMachineBuilder));
    builder.setObjectLinkingLayerCreator(
        [&](llvm::orc::ExecutionSession& execSession, const llvm::Triple&) {
            return std::make_unique<llvm::orc::ObjectLinkingLayer>(
                execSession, llvm::cantFail(llvm::jitlink::InProcessMemoryManager::Create()));
        });
    lljit = llvm::cantFail(builder.create());
    // remove the data layout to not run into any conflicts
    mod->setDataLayout("");
    llvm::cantFail(lljit->addIRModule(llvm::orc::ThreadSafeModule(std::move(m_llvmModule), std::move(m_llvmContext))));
    // Resolve symbols that are statically linked in the current process.
    llvm::orc::JITDylib& mainJD = lljit->getMainJITDylib();
    mainJD.addGenerator(llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess('\0')));
    function = llvm::cantFail(lljit->lookup(functionName)).toPtr<void*>();
}

uint64_t MLIRExecEngine::runImpl(mlir::ModuleOp*)
{
    return invoke(function);
}
