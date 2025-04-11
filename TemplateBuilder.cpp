#include "TemplateBuilder.hpp"
#include "Preparation/CustomPass.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include "TemplateCompiler.hpp"

#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

TemplateBuilder::TemplateBuilder(TemplateLibrary& library, const Abstraction::IRLoweringFunction& lowering)
    : m_library(library)
    , m_parser(m_library)
    , m_lowering(lowering)
{
}

void TemplateBuilder::addOpWithSignature(
    mlir::Operation& op,
    Abstraction abstraction,
    Signature&& signature)
{
    const auto id = m_library.m_templateStorage.size();
    ASSURE_FALSE(abstraction.isNopCast());
    INFO("templateId " << id << ", " << op.getName().getStringRef() << "(" << op.getLoc() << ")");

    std::vector<TEMPLATE_KIND> kinds;
    // stacksave (and other intrinsics) should no be constant evaluated...
    ASSURE(!signature.isConstexpr(), ||, !op.getName().getStringRef().starts_with("llvm.intr."));
    if (TEMPLATE_CONST_EVAL && signature.isConstexpr())
        kinds.push_back(TEMPLATE_KIND::STACK_ARGS);
#ifdef FORCE_STACK_TEMPLATE
    const auto k = TEMPLATE_KIND::STACK_PATCHED;
    kinds.push_back(k);
#else
    const auto k = TEMPLATE_KIND::REGISTERS;
    kinds.push_back(k);
#endif
    mlir::OwningOpRef<mlir::ModuleOp> mod = abstraction.getTemplates(id,
        llvm::isa<mlir::FunctionOpInterface>(op) ? std::vector<TEMPLATE_KIND>{k} : kinds);
    m_library.insert(
        std::move(signature),
        abstraction.takeResultSizes(),
        abstraction.takeStackMaps(),
        abstraction.getModule());

    std::string filename = m_library.getName() + "/stencil" + std::to_string(id) + ".o";
    // compile templates -> update afterwards (potentially turn them into a constant)
    const auto isVarArgFun = llvm::isa<mlir::LLVM::LLVMFuncOp>(op) && llvm::cast<mlir::LLVM::LLVMFuncOp>(op).isVarArg();
    const auto clobbers = compiler::compileMLIRToBinary(
        *mod.operator->(),
        filename,
        op.getNumRegions() > 0 && !isVarArgFun ? std::vector<size_t>{id} : std::vector<size_t>{});
    m_parser(filename, clobbers);
}

void TemplateBuilder::addOp(mlir::Operation& op)
{
    using F = mlir::FunctionOpInterface;
    if (llvm::isa<F>(op) && llvm::cast<F>(op).isExternal()) {
        return;
    }

    if (m_library.contains(op)) {
        return;
    }

    addOpWithSignature(
        op,
        Abstraction::FromOp(m_lowering, op, {}),
        Signature(op));
}

void TemplateBuilder::addOpsFromModule(mlir::ModuleOp& mod)
{
    MEASURE(ScopeTimer::templateGeneration);
    mod.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
        return NoPatternGeneration::Skip(llvm::TypeSwitch<mlir::Operation*, mlir::WalkResult>(op))
            .Default([&](mlir::Operation* op) {
                if (!op->hasTrait<mlir::OpTrait::IsTerminator>() && !llvm::isa<mlir::ModuleOp>(op))
                    addOp(*op);
                return mlir::WalkResult::advance();
            });
    });
}

void TemplateBuilder::addOpsFromFile(const std::string& filename)
{
    auto& context = m_library.getContext();
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

    mlir::OwningOpRef<mlir::ModuleOp> mod;
    mod = mlir::parseSourceFile<mlir::ModuleOp>(filename, sourceMgr, mlir::ParserConfig{&context});
    REL_ASSURE(static_cast<bool>(mod), ==, true);

    mlir::ModuleOp moduleOp = *mod;
    addOpsFromModule(moduleOp);
}

mlir::MLIRContext& TemplateBuilder::getContext()
{
    return m_library.getContext();
}

TemplateLibrary::IdType TemplateBuilder::reserveInternal()
{
    const auto id = m_library.m_templateStorage.size();
    m_library.m_templateStorage.emplace_back(
        ResultSizes(),
        StackMaps(),
        mlir::ModuleOp(),
        false);
    return id;
}
void TemplateBuilder::compileInternal(
    TemplateLibrary::IdType id,
    mlir::ModuleOp& mod)
{
    std::string filename = m_library.getName() + "/stencil" + std::to_string(id) + ".o";
    const auto clobbers = compiler::compileMLIRToBinary(
        mod,
        filename,
        {});
    m_parser(filename, clobbers);
}
