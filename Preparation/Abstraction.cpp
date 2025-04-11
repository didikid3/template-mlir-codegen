#include "Preparation/Abstraction.hpp"
#include "Preparation/CustomPass.hpp"
#include "Preparation/Utils.hpp"
#include "Templates/Signature.hpp"
#include "common.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"

#include "mlir/Pass/PassManager.h"

#include <numeric>

namespace {
class PatternRewriterWrapper
    : public mlir::PatternRewriter
{
public:
    explicit PatternRewriterWrapper(mlir::MLIRContext* ctx)
        : mlir::PatternRewriter(ctx)
    {
    }
};
}

void Abstraction::defaultLowering(mlir::Operation& op)
{
    mlir::PassManager pm{op.getContext()};
    pm.enableVerifier(true);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(ToLLVMLoweringPass::create());
    pm.addPass(mlir::createCSEPass());

    [[maybe_unused]] auto fail = failed(pm.run(&op));
    REL_ASSURE(fail, ==, false, << op);
}

void Abstraction::quickLowering(mlir::Operation& op)
{
    mlir::PassManager pm{op.getContext()};
    pm.enableVerifier(false);
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(ToLLVMLoweringPass::create());
    [[maybe_unused]] auto fail = failed(pm.run(&op));
    REL_ASSURE(fail, ==, false);
}

Abstraction
Abstraction::createAbstraction(
    const IRLoweringFunction& lowering,
    mlir::Operation& op,
    const std::unordered_set<uint8_t>& constantOperands)
{
    auto originalMod = op.getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder builder(op.getContext());
    mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(moduleOp.getBody());

    const auto ptr = stackT(builder);
    const bool isFuncInput = llvm::isa<mlir::FunctionOpInterface>(op);
    const bool isVarArgFun = isFuncInput && llvm::isa<mlir::LLVM::LLVMFuncOp>(op) && llvm::cast<mlir::LLVM::LLVMFuncOp>(op).isVarArg();

    // we can avoid copying all the function definitions into the module, if the op is from the LLVM dialect
    // because this tells us that lowering will not suddenly reference a function
    if (!llvm::isa<mlir::LLVM::LLVMDialect>(op.getDialect())
        || llvm::isa<mlir::LLVM::CallOp>(op)
        || llvm::isa<mlir::LLVM::AddressOfOp>(op)
        || isVarArgFun) {
        for (auto f : originalMod.getOps<mlir::FunctionOpInterface>()) {
            if (!isFuncInput || mlir::SymbolTable::getSymbolName(&op).strref() != f.getName()) {
                auto clone = builder.cloneWithoutRegions(f);
                mlir::SymbolTable::setSymbolVisibility(clone, mlir::SymbolTable::Visibility::Private);
                if (llvm::isa<mlir::LLVM::LLVMFuncOp>(clone) && !f.isExternal() && clone.isExternal()) {
                    llvm::cast<mlir::LLVM::LLVMFuncOp>(clone).setLinkage(mlir::LLVM::Linkage::External);
                    llvm::cast<mlir::LLVM::LLVMFuncOp>(clone).removeComdatAttr();
                }
                if (llvm::isa<mlir::func::FuncOp>(clone)) {
                    clone->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(builder.getContext(), mlir::LLVM::Linkage::External));
                }
            }
        }
    }

    mlir::FunctionOpInterface funcOp = [&]() -> mlir::FunctionOpInterface {
        if (isVarArgFun)
            return llvm::cast<mlir::FunctionOpInterface>(builder.clone(op));
        else if (isFuncInput)
            return llvm::cast<mlir::FunctionOpInterface>(builder.cloneWithoutRegions(op));
        else
            return builder.create<mlir::func::FuncOp>(
                builder.getUnknownLoc(),
                kInternalTemplateFunction,
                builder.getFunctionType({}, ptr));
    }();
    if (isVarArgFun) {
        // clone the globals
        for (auto global : originalMod.getOps<mlir::LLVM::GlobalOp>()) {
            builder.clone(*global.operator mlir::Operation*());
        }
    }
    if (llvm::isa<mlir::LLVM::LLVMFuncOp>(funcOp)) {
        llvm::cast<mlir::LLVM::LLVMFuncOp>(funcOp).setLinkage(mlir::LLVM::Linkage::External);
        llvm::cast<mlir::LLVM::LLVMFuncOp>(funcOp).removeComdatAttr();
    }
    if (llvm::isa<mlir::func::FuncOp>(funcOp)) {
        funcOp->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(builder.getContext(), mlir::LLVM::Linkage::External));
    }
    mlir::SymbolTable::setSymbolName(funcOp, kInternalTemplateFunction);

    if (!isVarArgFun) {
        auto* bodyBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(bodyBlock);
    }

    uint8_t nrInputs = 0;
    std::vector<mlir::Type> newOperandTypes;
    std::vector<mlir::Value> newOperands;
    for (uint8_t i = 0; i < op.getNumOperands(); i++) {
        if (!constantOperands.count(i)) {
            newOperandTypes.push_back(op.getOperand(i).getType());
        }
    }
    nrInputs += newOperandTypes.size();
    if (!newOperandTypes.empty()) {
        auto newInOperands = builder.create<mlir::UnrealizedConversionCastOp>(
                                        builder.getUnknownLoc(),
                                        newOperandTypes,
                                        std::vector<mlir::Value>{},
                                        builder.getNamedAttr(kInputNumAttr, builder.getI8IntegerAttr(0)))
                                 .getResults();
        auto it = newInOperands.begin();
        for (uint8_t i = 0; i < op.getNumOperands(); i++) {
            auto operand = op.getOperand(i);
            if (constantOperands.count(i)) {
                ASSURE_TRUE(operand.getDefiningOp()->hasTrait<mlir::OpTrait::ConstantLike>());
                newOperands.push_back(builder.clone(*operand.getDefiningOp())->getResult(0));
            } else {
                ASSURE_TRUE(it != newInOperands.end());
                newOperands.push_back(*(it++));
            }
        }
    }

    mlir::Operation* clonedOp;
    if (!isFuncInput) {
        clonedOp = builder.cloneWithoutRegions(op);
        clonedOp->setOperands(newOperands);
        ASSURE(
            (llvm::DenseSet<mlir::Value>(clonedOp->operand_begin(), clonedOp->operand_end()).size()),
            ==,
            clonedOp->getNumOperands());
    } else {
        clonedOp = funcOp;
    }
#ifdef LESS_TEMPLATES
    if (llvm::isa<mlir::LLVM::CallOp>(*clonedOp)) {
        auto callee = llvm::cast<mlir::LLVM::CallOp>(*clonedOp).getCallee();
        if (callee.has_value()) {
            llvm::cast<mlir::LLVM::CallOp>(*clonedOp).setCallee(kDummyFunctionName);
            for (auto func : clonedOp->getParentOfType<mlir::ModuleOp>().getOps<mlir::LLVM::LLVMFuncOp>()) {
                if (func.getSymName() == *callee) {
                    func.setSymName(kDummyFunctionName);
                    break;
                }
            }
        }
    } else if (llvm::isa<mlir::func::CallOp>(*clonedOp)) {
        auto callee = llvm::cast<mlir::func::CallOp>(*clonedOp).getCallee();
        llvm::cast<mlir::func::CallOp>(*clonedOp).setCallee(kDummyFunctionName);
        for (auto func : clonedOp->getParentOfType<mlir::ModuleOp>().getOps<mlir::func::FuncOp>()) {
            if (func.getSymName() == callee) {
                func.setSymName(kDummyFunctionName);
                break;
            }
        }
    }
#endif
    ASSURE(op.getNumRegions(), ==, clonedOp->getNumRegions());
    for (size_t i = 0; i < clonedOp->getNumRegions(); i++) {
        auto& region = clonedOp->getRegion(i);
        auto& oldRegion = op.getRegion(i);
        if (oldRegion.empty() || isVarArgFun)
            continue;

        mlir::OpBuilder regionBuilder(region);
        mlir::Block* block;
        if (!isFuncInput) {
            block = regionBuilder.createBlock(&region, {},
                oldRegion.getArgumentTypes(),
                std::vector(oldRegion.getNumArguments(), regionBuilder.getUnknownLoc()));
        } else {
            block = &*region.getBlocks().begin();
        }
        ASSURE(block->getNumArguments(), ==, oldRegion.getNumArguments());

        auto func = mlir::func::FuncOp::create(builder.getUnknownLoc(),
            formatRegionCall(i),
            mlir::FunctionType::get(op.getContext(), ptr, {}));
        func.setVisibility(mlir::SymbolTable::Visibility::Private);
        moduleOp.insert(moduleOp.begin(), func);

        auto cast = regionBuilder.create<mlir::UnrealizedConversionCastOp>(
            builder.getUnknownLoc(),
            ptr,
            block->getArguments(),
            std::vector{builder.getNamedAttr(kInputNumAttr, builder.getI8IntegerAttr(nrInputs++)), builder.getNamedAttr(kIsStoreIndicator, builder.getUnitAttr())});

        // call to other stencils
        regionBuilder.create<mlir::func::CallOp>(regionBuilder.getUnknownLoc(), func, cast.getResult(0));

        mlir::Operation* terminator = nullptr;
        for (auto& o : oldRegion.getOps()) {
            if (isFuncInput && o.hasTrait<mlir::OpTrait::ReturnLike>()) {
                terminator = &o;
            } else if (!isFuncInput && o.hasTrait<mlir::OpTrait::IsTerminator>()) {
                ASSURE(terminator, ==, reinterpret_cast<decltype(terminator)>(0ul), << "multiple terminators found\n"
                                                                                    << op);
                terminator = &o;
            }
        }
        if (terminator == nullptr) {
            // this might happen in cases where an unreachable is used to terminate the region instead
            ASSURE_TRUE(isFuncInput);
            ASSURE_TRUE(llvm::isa<mlir::LLVM::LLVMFuncOp>(op));
            regionBuilder.create<mlir::LLVM::UnreachableOp>(builder.getUnknownLoc());
            break;
        }
        ASSURE(terminator, !=, reinterpret_cast<decltype(terminator)>(0ul));
        ASSURE(terminator->getNumRegions(), ==, 0u, << " no regions for terminator instructions allowed");

        if (terminator->getNumOperands()) {
            auto terminatorOps = regionBuilder.create<mlir::UnrealizedConversionCastOp>(
                regionBuilder.getUnknownLoc(),
                terminator->getOperandTypes(),
                std::vector<mlir::Value>{},
                std::vector{
                    regionBuilder.getNamedAttr(kInputNumAttr, regionBuilder.getI8IntegerAttr(nrInputs)),
                    regionBuilder.getNamedAttr(kFromRegionIndicator, regionBuilder.getUnitAttr())});
            if (terminator->getNumOperands() == 1)
                terminatorOps->setAttr(kOperandCountAttr, regionBuilder.getI32IntegerAttr(op.getNumOperands()));

            nrInputs += terminator->getNumOperands();
            regionBuilder.cloneWithoutRegions(*terminator)->setOperands(terminatorOps.getResults());
        } else {
            regionBuilder.cloneWithoutRegions(*terminator);
        }
    }

    ResultSizes returnTypeSizes;
    if (!isFuncInput) {
        const auto returnNr = nrInputs++;
        const bool isLLVMPtrReturn = clonedOp->getResults().size() == 1 && clonedOp->getResult(0).getType() == ptr;
        std::vector<mlir::NamedAttribute> attrs{
            builder.getNamedAttr(kInputNumAttr, builder.getI8IntegerAttr(returnNr)),
            builder.getNamedAttr(kIsStoreIndicator, builder.getUnitAttr())};
        if (isLLVMPtrReturn) {
            attrs.push_back(
                builder.getNamedAttr(kDummyIndicator, builder.getUnitAttr()));
        }
        auto ret = builder.create<mlir::UnrealizedConversionCastOp>(
            builder.getUnknownLoc(),
            ptr,
            isLLVMPtrReturn ? std::vector<mlir::Value>({clonedOp->getResult(0), clonedOp->getResult(0)}) : std::vector<mlir::Value>(clonedOp->getResults().begin(), clonedOp->getResults().end()),
            std::move(attrs));
        builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), ret.getResults());
        //======== lowering =========//
        lowering(*moduleOp.operator mlir::Operation*());
        //===========================//
        if (isLLVMPtrReturn) {
            // in this case we can be sure about the type
            returnTypeSizes.push_back(TypeSize(getBitSize(ptr)));
        } else {
            ASSURE(ret->getAttrOfType<mlir::IntegerAttr>(kInputNumAttr).getInt(), ==, returnNr);
            for (auto o : ret.getOperands()) {
                auto op = o.getDefiningOp();
                mlir::Type t = o.getType();
                [[maybe_unused]] bool passThrough = false;
                while (op != nullptr
                    && !mlir::LLVM::isCompatibleType(t)
                    && llvm::isa<mlir::UnrealizedConversionCastOp>(op)) {
                    if (op->hasAttr(kInputNumAttr)) {
                        ASSURE_FALSE(op->hasAttr(kIsStoreIndicator));
                        // pass through from input parameter
                        passThrough = true;
                        ASSURE(op->getNumOperands(), ==, 0);
                        // go through the whole use-tree in an effort to find a compatible type
                        {
                            t = findLLVMCompatibleType(o);
                        }
                        break;
                    } else {
                        ASSURE(op->getNumOperands(), >, 0);
                        t = op->getOperand(0).getType();
                        o = op->getOperand(0);
                        op = o.getDefiningOp();
                    }
                }
                if ((t == nullptr || !mlir::LLVM::isCompatibleType(t))) {
                    // in rare cases (higher level bitcasts) it can happen, that inputs are directly connected to outputs
                    // we detect the case as nop bitcast (during runtime only forwarding the input without any computations)
                    ASSURE_TRUE(passThrough);
                    // for those cases, we do not allow any control flow...
                    ASSURE(ret->getParentOfType<mlir::LLVM::LLVMFuncOp>().getFunctionBody().getBlocks().size(), ==, 1);
                    return Abstraction{};
                } else {
                    returnTypeSizes.push_back(TypeSize::FromType(t));
                }
            }
        }
    } else {
        //======== lowering =========//
        lowering(*moduleOp.operator mlir::Operation*());
        //===========================//
    }
    ASSURE(returnTypeSizes.size(), ==, op.getNumResults());
    return Abstraction{
        std::move(moduleOp),
        std::move(returnTypeSizes),
        llvm::isa<mlir::FunctionOpInterface>(op)};
}

StackMaps Abstraction::propagateChanges(
    mlir::LLVM::LLVMFuncOp& in,
    TEMPLATE_KIND kind,
    bool isFuncPattern)
{
    {
        PatternRewriterWrapper rewriter(in.getContext());
        LoadInputFromStackRewriteOp load(in.getContext(), kind);
        StoreRegionArgsToStackRewriteOp store(in.getContext(), kind);

        const auto size = [&] {
            return std::accumulate(in.getOps<mlir::UnrealizedConversionCastOp>().begin(), in.getOps<mlir::UnrealizedConversionCastOp>().end(), 0ul, [](const auto acc, const auto& op) {
                return acc + (op->hasAttr(kInputNumAttr) ? 1 : 0);
            });
        };
        unsigned long numUnrealizedConversions;

        do {
            numUnrealizedConversions = size();
            for (auto op : in.getOps<mlir::UnrealizedConversionCastOp>()) {
                if (op->hasAttr(kInputNumAttr)) {
                    rewriter.setInsertionPoint(op);
                    if (op->hasAttr(kIsStoreIndicator))
                        std::ignore = store.matchAndRewrite(op, rewriter);
                    else
                        std::ignore = load.matchAndRewrite(op, rewriter);
                    break;
                }
            }
            // repeat until we reach a fixpoint
        } while (numUnrealizedConversions > size());
        ASSURE(size(), ==, 0);
    }

    {
        mlir::RewritePatternSet patterns(in.getContext());
        mlir::populateReconcileUnrealizedCastsPatterns(patterns);
        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        auto ok = mlir::applyPatternsAndFoldGreedily(in, frozenPatterns).succeeded();
        REL_ASSURE(ok, ==, true);
    }

    if (!isFuncPattern) {
        mlir::RewritePatternSet patterns(in.getContext());
        mlir::populateReconcileUnrealizedCastsPatterns(patterns);
        patterns.add<TailCallRewriteOp>(in.getContext(), kind);
        mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
        auto ok = mlir::applyPatternsAndFoldGreedily(in, frozenPatterns).succeeded();
        REL_ASSURE(ok, ==, true);
    }
    StackMaps stackmaps;
    for (auto op : in.getOps<mlir::LLVM::CallOp>()) {
        if (op->hasAttr(kStackmapAttr)) {
            StackMap& stackmap = stackmaps.emplace_back();
            for (auto attr : op->getAttrOfType<mlir::ArrayAttr>(kStackmapAttr).getValue()) {
                stackmap.push_back(TypeSize::FromAttr(attr));
            }
        }
    }
    DEBUG(in->getParentOfType<mlir::ModuleOp>());
    // make the index range of the placeholder dense again (it might have happened, that e.g. a placeholder from the
    // middle does no longer exist, as it is passed by register)
    std::vector<uint16_t> ids;
    for (auto global : in->getParentOfType<mlir::ModuleOp>().getOps<mlir::LLVM::GlobalOp>()) {
        const auto symNameStr = global.getSymName();
        if (isIndexPlaceholder(symNameStr)) {
            ASSURE(symNameStr.count('_'), ==, 6);
            if (std::find(ids.cbegin(), ids.cend(), getPlaceholderNr(symNameStr)) == ids.cend())
                ids.push_back(getPlaceholderNr(symNameStr));
        }
    }
    std::sort(ids.begin(), ids.end());
    for (auto global : in->getParentOfType<mlir::ModuleOp>().getOps<mlir::LLVM::GlobalOp>()) {
        const auto symNameStr = global.getSymName();
        if (isIndexPlaceholder(symNameStr)) {
            ASSURE(symNameStr.count('_'), ==, 6);
            ASSURE_TRUE(std::find(ids.cbegin(), ids.cend(), getPlaceholderNr(symNameStr)) != ids.cend());
            const auto newName = symNameStr.substr(0, symNameStr.find_last_of("_")).str()
                + "_"
                + std::to_string(std::find(ids.cbegin(), ids.cend(), getPlaceholderNr(symNameStr)) - ids.cbegin());
            global.setSymName(newName);
            for (auto addrOf : in.getOps<mlir::LLVM::AddressOfOp>()) {
                if (addrOf.getGlobalName() == symNameStr && !addrOf->hasAttr("updated")) {
                    addrOf->setAttr("updated", mlir::UnitAttr::get(in.getContext()));
                    addrOf.setGlobalName(newName);
                }
            }
        }
    }
    return stackmaps;
}

StackMaps Abstraction::stackCCasArgs(mlir::LLVM::LLVMFuncOp& func, mlir::OpBuilder& builder, size_t id)
{
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto stencilFunc = llvm::cast<mlir::LLVM::LLVMFuncOp>(builder.clone(*func.operator->()));

    const auto ptr = stackT(builder);
    stencilFunc.setName(formatFunction(id, static_cast<uint8_t>(TEMPLATE_KIND::STACK_ARGS)));
    stencilFunc.setFunctionTypeAttr(mlir::TypeAttr::get(mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), {ptr})));
    stencilFunc.getBody().addArgument(ptr, builder.getUnknownLoc());

    return propagateChanges(stencilFunc, TEMPLATE_KIND::STACK_ARGS);
}

StackMaps Abstraction::stackCCasGlobals(mlir::LLVM::LLVMFuncOp& func, mlir::OpBuilder& builder, size_t id, bool isFuncPattern)
{
    ASSURE_TRUE(llvm::isa<mlir::ModuleOp>(builder.getInsertionBlock()->getParentOp()));

    mlir::OpBuilder::InsertionGuard guard(builder);
    auto globalBuilder = mlir::OpBuilder::atBlockBegin(builder.getInsertionBlock());
    auto stencilFunc = llvm::cast<mlir::LLVM::LLVMFuncOp>(globalBuilder.clone(*func.operator->()));

    const auto ptr = stackT(builder);
    mlir::Block* bodyBlock = &*stencilFunc.getBlocks().begin();
    builder.setInsertionPointToStart(bodyBlock);

    stencilFunc.setName(formatFunction(id, static_cast<uint8_t>(TEMPLATE_KIND::STACK_PATCHED)));

    if (isFuncPattern) {
        stencilFunc->setAttr(kNeedsStackIndicator, builder.getUnitAttr());
    } else {
        const auto functionType = builder.getType<mlir::LLVM::LLVMFunctionType>(
            builder.getType<mlir::LLVM::LLVMVoidType>(),
            mlir::ArrayRef<mlir::Type>({ptr}),
            false);
        stencilFunc.setFunctionTypeAttr(mlir::TypeAttr::get(functionType));
        stencilFunc.getBody().addArguments(mlir::TypeRange({ptr}), std::vector(1, builder.getUnknownLoc()));
        stencilFunc.setCConv(mlir::LLVM::CConv::CALLING_CONVENTION);
        stencilFunc->setAttr(kTailCallIndicator, builder.getUnitAttr());
    }
    return propagateChanges(stencilFunc, TEMPLATE_KIND::STACK_PATCHED, isFuncPattern);
}

StackMaps Abstraction::registerCC(mlir::LLVM::LLVMFuncOp& func, mlir::OpBuilder& builder, size_t id, bool isFuncPattern)
{
    ASSURE_TRUE(llvm::isa<mlir::ModuleOp>(builder.getInsertionBlock()->getParentOp()));

    const auto ptr = stackT(builder);
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto globalBuilder = mlir::OpBuilder::atBlockBegin(builder.getInsertionBlock());
    auto stencilFunc = llvm::cast<mlir::LLVM::LLVMFuncOp>(globalBuilder.clone(*func.operator->()));

    if (isFuncPattern) {
        stencilFunc->setAttr(kNeedsStackIndicator, builder.getUnitAttr());
    } else {
        stencilFunc.getBody().addArgument(ptr, builder.getUnknownLoc());
        std::vector<mlir::Type> argTypes{ptr};
        stencilFunc.setFunctionTypeAttr(mlir::TypeAttr::get(mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(builder.getContext()), argTypes)));
        stencilFunc.setCConv(mlir::LLVM::CConv::CALLING_CONVENTION);
    }

    stencilFunc.setName(formatFunction(id, static_cast<uint8_t>(TEMPLATE_KIND::REGISTERS)));
    stencilFunc->setAttr(kTailCallIndicator, builder.getUnitAttr());

    return propagateChanges(stencilFunc, TEMPLATE_KIND::REGISTERS, isFuncPattern);
}

Abstraction::Abstraction(
    mlir::ModuleOp&& mod,
    ResultSizes&& sizes,
    bool isFunction)
    : m_module(std::move(mod))
    , m_resultSizes(std::move(sizes))
    , m_isFunction(isFunction)
    , m_isNopCast(false)
{
}

Abstraction::Abstraction()
    : m_module(nullptr)
    , m_resultSizes()
    , m_isFunction(false)
    , m_isNopCast(true)
{
}

Abstraction Abstraction::FromOp(
    const IRLoweringFunction& lowering,
    mlir::Operation& op,
    const std::unordered_set<uint8_t>& constantOperands)
{
    auto abstraction = createAbstraction(lowering, op, constantOperands);
    return abstraction;
}

mlir::ModuleOp Abstraction::getTemplates(
    size_t id,
    const std::vector<TEMPLATE_KIND>& kinds)
{
    mlir::PassManager verifier(m_module.getContext());
    verifier.enableVerifier(true);
    mlir::OpBuilder builder(m_module.getContext());
    mlir::ModuleOp moduleOp = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(moduleOp.getBody());

    ASSURE_FALSE(m_module.getOps<mlir::LLVM::LLVMFuncOp>().empty());
    for (auto& o : m_module.getOps()) {
        if (auto func = llvm::dyn_cast<mlir::LLVM::LLVMFuncOp>(o)) {
            if (func.getName().equals(kInternalTemplateFunction)) {
                for (const auto kind : kinds) {
                    switch (kind) {
                        case TEMPLATE_KIND::STACK_ARGS:
                            ASSURE_FALSE(m_isFunction);
                            m_stackmaps = stackCCasArgs(func, builder, id);
                            break;
                        case TEMPLATE_KIND::STACK_PATCHED:
                            m_stackmaps = stackCCasGlobals(func, builder, id, m_isFunction);
                            break;
                        case TEMPLATE_KIND::REGISTERS:
                            m_stackmaps = registerCC(func, builder, id, m_isFunction);
                            break;
                    };
                }
            } else if (isAnyPlaceholder(func.getName())) {
                auto cloned = llvm::cast<mlir::LLVM::LLVMFuncOp>(builder.clone(*func));
                cloned.setCConv(mlir::LLVM::CConv::CALLING_CONVENTION);
            } else {
                builder.clone(*func);
            }
        } else {
            builder.clone(o);
        }
    }

    [[maybe_unused]] auto fail = failed(verifier.run(moduleOp));
    ASSURE_FALSE(fail);

    return moduleOp;
}

StackMaps&& Abstraction::takeStackMaps()
{
    return std::move(m_stackmaps);
}

ResultSizes&& Abstraction::takeResultSizes()
{
    return std::move(m_resultSizes);
}

mlir::ModuleOp& Abstraction::getModule()
{
    return m_module;
}

bool Abstraction::isNopCast() const
{
    return m_isNopCast;
}
