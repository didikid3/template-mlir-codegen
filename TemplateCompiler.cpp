#include "TemplateCompiler.hpp"
#include "Preparation/Abstraction.hpp"
#include "Preparation/BinaryParser.hpp"

#include "mlir/ExecutionEngine/OptUtils.h"
#ifndef LINGODB_LLVM
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#endif
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/AsmPrinterHandler.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#ifdef LINGODB_LLVM
#include "llvm/MC/SubtargetFeature.h"
#else
#include "llvm/TargetParser/SubtargetFeature.h"
#endif

namespace {
class MachineFunctionInfoExtractor
    : public llvm::AsmPrinterHandler
{
    ClobberMask& m_clobberMask;
    std::array<llvm::MCRegister, CACHE_REGISTERS> m_regs;
    const llvm::MCRegisterInfo& m_regInfo;

public:
    MachineFunctionInfoExtractor(ClobberMask& clobberMask, llvm::MCRegisterInfo& regInfo)
        : m_clobberMask(clobberMask)
        , m_regInfo(regInfo)
    {
        for (unsigned int i = 0; i < regInfo.getNumRegs(); i++) {
            auto name = regInfo.getName(i);
            const auto it = std::find_if(std::begin(REGISTERS), std::end(REGISTERS), [&](const char* str) {
                return std::strcmp(str, name) == 0;
            });
            if (it != std::end(REGISTERS)) {
                m_regs[it - std::begin(REGISTERS)] = i;
            }
        }
        ASSURE_TRUE(std::all_of(m_regs.cbegin(), m_regs.cend(), std::mem_fn(&llvm::MCRegister::isValid)));
    }
    virtual ~MachineFunctionInfoExtractor() {}
    virtual void setSymbolSize(const llvm::MCSymbol*, uint64_t) override {}
    virtual void beginModule(llvm::Module*) override {}
    virtual void endModule() override {}
    virtual void beginFunction(const llvm::MachineFunction*) override {}
    virtual void endFunction([[maybe_unused]] const llvm::MachineFunction* MF) override
    {
        DEBUG(MF->getName() << ":" << m_clobberMask.to_string());
    }
    virtual void beginInstruction(const llvm::MachineInstr* MI) override
    {
        ASSURE_TRUE(MI->getParent()->getParent()->getName().starts_with("stencil"));
        if (!MI->getParent()->getParent()->getName().ends_with("_2")) {
            return;
        }
        if (MI->isCall()) {
            const auto op = MI->getOperand(0);
            if (!op.isGlobal() || !isAnyPlaceholder(op.getGlobal()->getName()))
                m_clobberMask = CLOBBER_ALL;
            return;
        }

        for (auto def : MI->operands()) {
            if (!(def.isReg() && def.isDef()))
                continue;
            for (auto reg : m_regInfo.sub_and_superregs_inclusive(def.getReg())) {
                auto it = std::find(m_regs.cbegin(), m_regs.cend(), reg);
                if (it != m_regs.cend()) {
                    m_clobberMask.set(it - m_regs.cbegin());
                }
            }
        }
    }
    virtual void endInstruction() override {}
};
}

namespace compiler {
bool fixupLLVMIR(llvm::Module& llvmModule, bool)
{
    llvm::LLVMContext& llvmContext = llvmModule.getContext();

    auto* smallRangeMDNode = llvm::MDNode::get(llvmContext,
        {llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(llvm::Type::getInt64Ty(llvmContext), llvm::APInt(64, 0))),
            llvm::ConstantAsMetadata::get(llvm::Constant::getIntegerValue(llvm::Type::getInt64Ty(llvmContext), llvm::APInt(64, INT32_MAX)))});

    bool requires64BitReloc = false;
    for (auto& global : llvmModule.globals()) {
        if (isIndexPlaceholder(global.getName())) {
            global.setMetadata(llvm::LLVMContext::MD_absolute_symbol, smallRangeMDNode);
        } else if (isAnyPlaceholder(global.getName())) {
            requires64BitReloc = true;
        } else if (global.isConstantUsed() && global.isPrivateLinkage(global.getLinkage())) {
            requires64BitReloc = true;
        }
    }

    // set the attributes again as mlir fails in doing so...
    for (auto& func : llvmModule.functions()) {
        if (isAnyPlaceholder(func.getName()) || isTemplateFunction(func.getName())) {
            if (func.hasFnAttribute("NO_ADJUST"))
                continue;
            // this leads to invalid code in some cases (overwrites $rbp ...):
            // # if (!isFuncTemplate && isTemplateFunction(func.getName())) {
            // #     func.addFnAttr(llvm::Attribute::get(llvmContext, "no_callee_saved_registers"));
            // # }
            std::vector<llvm::Instruction*> deadInstructions;
            for (auto inst = llvm::inst_begin(func); inst != llvm::inst_end(func); inst++) {
                if (auto callInst = llvm::dyn_cast<llvm::CallInst>(&*inst)) {
                    if (!callInst->getCalledOperand())
                        continue;
                    const auto calledFuncName = callInst->getCalledOperand()->getName();
                    if (isNextPlaceholder(calledFuncName)) {
                        callInst->setCallingConv(llvm::CallingConv::CALLING_CONVENTION);
                        callInst->setTailCall(true);
                        callInst->setTailCallKind(llvm::CallInst::TailCallKind::TCK_MustTail);
                    } else if (isRegionPlaceholder(calledFuncName)) {
                        callInst->getCalledFunction()->setCannotDuplicate();
                        callInst->setCallingConv(llvm::CallingConv::CALLING_CONVENTION);
                        callInst->addFnAttr(llvm::Attribute::get(llvmContext, "no_callee_saved_registers"));
                    } else if (isAnyPlaceholder(calledFuncName)) {
                        callInst->setCallingConv(llvm::CallingConv::CALLING_CONVENTION);
                    }
                }
            }
        }
    }
    return requires64BitReloc;
}

const char* getTargetTriple()
{
#if defined(__x86_64__)
    return "x86_64-unknown-linux-elf";
#elif defined(__aarch64__) && defined(__APPLE__)
    return "aarch64-unknown-linux-elf";
    // return "aarch64-unknown-darwin-elf";
#elif defined(__aarch64__) && defined(__linux__)
    return "aarch64-unknown-linux-elf";
#endif
}

const llvm::Target* getTarget()
{
#if defined(__x86_64__)
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86AsmParser();
    LLVMInitializeX86Disassembler();
#elif defined(__aarch64__)
    LLVMInitializeAArch64AsmPrinter();
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64AsmParser();
    LLVMInitializeAArch64Disassembler();
#endif
    std::string err;
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(getTargetTriple(), err);
    ASSURE(target, !=, reinterpret_cast<decltype(target)>(0), << err);
    return target;
}

std::unique_ptr<llvm::TargetMachine> getTargetMachine(llvm::CodeModel::Model cmodel, llvm::TargetOptions opt)
{
    const llvm::Target* target = getTarget();
    const auto cpu = llvm::sys::getHostCPUName();
    llvm::SubtargetFeatures features;
    {
        llvm::StringMap<bool> hostFeatures;
        if (llvm::sys::getHostCPUFeatures(hostFeatures))
            for (const auto& [feature, enabled] : hostFeatures)
                features.AddFeature(feature, enabled);
    }

    return std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            getTargetTriple(),
            cpu,
            features.getString(),
            opt,
            llvm::Reloc::Model::Static,
            cmodel,
            llvm::CodeGenOpt::Aggressive));
}

ClobberMask compileMLIRToBinary(
    mlir::ModuleOp& module,
    std::string file,
    [[maybe_unused]] const std::vector<size_t>& regionStencils)
{
    llvm::LLVMContext llvmContext;
#ifndef LINGODB_LLVM
    mlir::registerBuiltinDialectTranslation(*module.getContext());
#endif
    mlir::registerLLVMDialectTranslation(*module.getContext());
    const bool withStack = std::any_of(
        module.getOps<mlir::LLVM::LLVMFuncOp>().begin(),
        module.getOps<mlir::LLVM::LLVMFuncOp>().end(),
        [](const auto func) {
            return func->hasAttr(kNeedsStackIndicator);
        });
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    ASSURE_TRUE(llvmModule, << " " << module);

    llvm::TargetOptions opt;
    opt.FunctionSections = true; // -ffunction-sections
    opt.UniqueBasicBlockSectionNames = true; // -funique-basic-block-section-names

    // some fixup as MLIR has only restricted capabilities for setting the attributes etc.

    // with small code model, LLVM generates rip relative relocations for the stackframe size
    // this can be fixed, by using the medium code model in those cases
    const bool requiresMediumCM = fixupLLVMIR(*llvmModule, withStack) || withStack;

#ifdef __x86_64__
    std::stringstream bbSectionList;
    for (auto id : regionStencils) {
        for (uint8_t i = 0; i < static_cast<uint8_t>(TEMPLATE_KIND::REGISTERS); i++) {
            bbSectionList << "!" << formatFunction(id) << "_" << static_cast<uint32_t>(i + 1) << "\n";
            for (unsigned j = 0; j < 10; j++) {
                // make all section global in order to enforce relocation entry emission
                std::stringstream s;
                s << ".globl stencil" << id << "_" << static_cast<uint32_t>(i + 1) << ".__part." << j;
                const auto tmpString = s.str();
                llvmModule->appendModuleInlineAsm(tmpString);
            }
        }
    }
    opt.BBSections = llvm::BasicBlockSection::List; // -fbasic-block-sections=list
    const auto tmpString = bbSectionList.str();
    opt.BBSectionsFuncListBuf = llvm::MemoryBuffer::getMemBuffer(tmpString); // filled with sections that have regions
#endif

#if defined(__x86_64__)
    const auto biggerCM = llvm::CodeModel::Medium;
#elif defined(__aarch64__)
    const auto biggerCM = llvm::CodeModel::Large;
#endif
    llvm::CodeModel::Model cmodel = llvm::CodeModel::Small;
    cmodel = requiresMediumCM ? biggerCM : cmodel;
    cmodel = module->hasAttr("custom.cmodel-large") ? llvm::CodeModel::Large : cmodel;
    DEBUG("CodeModel: " << cmodel);
    auto targetMachine = getTargetMachine(cmodel, opt);
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().normalize());
    llvmModule->setDataLayout(targetMachine->createDataLayout());
    ASSURE_FALSE(llvm::verifyModule(*llvmModule, &llvm::outs()), << *llvmModule);

    auto ok = mlir::makeOptimizingTransformer(
        3,
        0,
        nullptr)(llvmModule.get());
    REL_ASSURE(static_cast<bool>(ok), ==, false);
    DEBUG("LLVM Module: " << *llvmModule);

    std::error_code ec;
    llvm::raw_fd_ostream dest(file, ec, llvm::sys::fs::OF_None);
    ASSURE_FALSE(ec, << " Can't open file: " << ec.message());

    auto clobbers = emitFile(*llvmModule, *targetMachine, dest, llvm::CGFT_ObjectFile);
    dest.close();
    return clobbers;
}

ClobberMask emitFile(
    llvm::Module& llvmModule,
    llvm::TargetMachine& targetMachine,
    llvm::raw_pwrite_stream& dest,
    llvm::CodeGenFileType fileType)
{
    llvm::legacy::PassManager PM;

    // well... without RTTI i don't really see another option...
    llvm::LLVMTargetMachine* llvmTargetMachine = static_cast<llvm::LLVMTargetMachine*>(&targetMachine);

    auto* MMIWP = new llvm::MachineModuleInfoWrapperPass(llvmTargetMachine);
    auto* PassConfig = llvmTargetMachine->createPassConfig(PM);
    PassConfig->setDisableVerify(true);
    PM.add(PassConfig);
    PM.add(MMIWP);
    PassConfig->addISelPasses();
    PassConfig->addMachinePasses();
    PassConfig->setInitialized();

    std::unique_ptr<llvm::MCStreamer> MCStreamer = llvm::cantFail(llvmTargetMachine->createMCStreamer(
        dest,
        nullptr,
        fileType,
        MMIWP->getMMI().getContext()));

    auto* Printer = targetMachine.getTarget().createAsmPrinter(targetMachine, std::move(MCStreamer));
    std::unique_ptr<llvm::MCRegisterInfo> regInfo(targetMachine.getTarget().createMCRegInfo(getTargetTriple()));
    ClobberMask clobbers{0};
    if (fileType != llvm::CGFT_AssemblyFile)
        Printer->addAsmPrinterHandler(llvm::AsmPrinter::HandlerInfo(
            std::make_unique<MachineFunctionInfoExtractor>(clobbers, *regInfo),
            "", "", "", ""));
    ASSURE_TRUE(Printer != nullptr);
    PM.add(Printer);
    PM.add(llvm::createFreeMachineFunctionPass());
    PM.run(llvmModule);
    dest.flush();
    return clobbers;
}
} // namespace compiler
