#include "Preparation/CustomPass.hpp"
#include "Preparation/Utils.hpp"

#include "TemplateCompiler.hpp"
#include "common.hpp"

#include "mlir/Analysis/DataLayoutAnalysis.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace {
using namespace mlir::LLVM;

bool isPassedInGPRegisters(const llvm::Type* type)
{
    if (type->isIntOrPtrTy()) {
        return true;
    } else if (type->isFloatingPointTy() || type->isVectorTy()) {
        return false;
    } else {
        ASSURE_TRUE(type->isAggregateType());
        return std::all_of(
            type->subtype_begin(),
            type->subtype_end(),
            isPassedInGPRegisters);
    }
}
bool isPassedInGPRegisters(mlir::Type type)
{
    llvm::LLVMContext llvmContext;
    auto llvmTypeTranslator = TypeToLLVMIRTranslator(llvmContext);
    auto llvmType = llvmTypeTranslator.translateType(type);
    return isPassedInGPRegisters(llvmType);
}

class Helper
{
    // dense i8-array attribute to help keeping function parameter in
    // the expected order according to their placeholder numbers
    static constexpr const char* kArgNumbersAttr = "argNumbers";

public:
    Helper(mlir::RewriterBase& builder_, mlir::Operation& op_, TEMPLATE_KIND kind_)
        : builder(builder_)
        , op(op_)
        , kind(kind_)
        , function(op.getParentOfType<LLVMFuncOp>())
    {
        ASSURE_TRUE(function);
        if (!function->hasAttr(kArgNumbersAttr)) {
            function->setAttr(kArgNumbersAttr, builder.getDenseI8ArrayAttr({}));
        }
    }

    template <typename T, typename... Args>
    T create(Args... args)
    {
        return builder.create<T>(op.getLoc(), std::forward<Args>(args)...);
    }

    mlir::Type stackT() { return ::stackT(builder); }
    mlir::Type offsetT() { return builder.getI64Type(); }

    mlir::Value stack()
    {
        if (function->hasAttr(kNeedsStackIndicator)) {
            for (auto& alloca : function.getOps()) {
                if (alloca.hasAttr("stack"))
                    return alloca.getResult(0);
            }
            ASSURE_FALSE(function.empty());
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&function.front());
#ifdef FIXED_STACKSIZE
            auto size = create<ConstantOp>(builder.getI64Type(), MAX_STACKSIZE);
            auto alloc = create<AllocaOp>(stackT(), size, 0);
            // we manually do the aligning as otherwise llvm generates invalid code...
            auto stackPtrInt = create<PtrToIntOp>(builder.getI64Type(), alloc);
            auto constNegAlign = create<ConstantOp>(builder.getI64Type(), -MAX_ALIGN);
            auto constAlignMinusOne = create<ConstantOp>(builder.getI64Type(), MAX_ALIGN - 1);
            auto alignedStackPtrInt = create<AndOp>(create<AddOp>(stackPtrInt, constAlignMinusOne), constNegAlign);
            auto stackPtr = create<IntToPtrOp>(stackT(), alignedStackPtrInt);
#else
            auto size = placeholder(INT16_MAX);
            auto stackPtr = create<AllocaOp>(
                stackT(), size, MAX_ALIGN);
#endif
            stackPtr->setAttr("stack", builder.getUnitAttr());
            return stackPtr;
        } else {
            ASSURE(function.getArgument(0).getType(), ==, stackT());
            return function.getArgument(0);
        }
    }

    uint32_t sizeFactor()
    {
        const auto factor = 64 / getBitSize(llvm::cast<LLVMPointerType>(stackT()).getElementType());
        ASSURE(factor, ==, 8);
        return factor;
    }

    uint32_t id()
    {
        return getTemplateFunctionNr(function.getSymName());
    }

    TEMPLATE_KIND loadKind()
    {
        return (function->hasAttr(kNeedsStackIndicator) || op.hasAttr(kFromRegionIndicator)) ? TEMPLATE_KIND::STACK_PATCHED : kind;
    }

    mlir::Value placeholder(size_t num)
    {
        ASSURE(num, <=, UINT16_MAX);
        std::string symbolName = formatIndexPlaceholder(id(), static_cast<uint8_t>(kind), num);
        GlobalOp global;
        if (auto* symRef = mlir::SymbolTable::lookupSymbolIn(function->getParentOp(), symbolName)) {
            ASSURE_TRUE(llvm::isa<GlobalOp>(*symRef));
            global = llvm::cast<GlobalOp>(*symRef);
        } else {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(function);
            global = create<GlobalOp>(
                builder.getI8Type(),
                false,
                Linkage::ExternWeak,
                symbolName,
                builder.getZeroAttr(builder.getI8Type()),
                1);
        }
        auto addrOf = create<AddressOfOp>(global);
        return create<PtrToIntOp>(offsetT(), addrOf);
    }

    mlir::Value load(mlir::Value offset, mlir::Type type)
    {
        ASSURE(offset.getType(), ==, offsetT());
        auto gepOp = create<GEPOp>(stackT(), stack(), offset);
        auto loadAddr = create<BitcastOp>(LLVMPointerType::get(type), gepOp);
        return create<LoadOp>(loadAddr);
    }

    void store(mlir::Value offset, uint64_t additionalOffInSlots, mlir::Value val)
    {
        auto constOffset = create<ConstantOp>(offsetT(), additionalOffInSlots * sizeFactor());
        auto addOp = create<AddOp>(offsetT(), offset, constOffset);
        auto gepOp = create<GEPOp>(stackT(), stack(), addOp.getResult());
        auto castAddr = create<BitcastOp>(LLVMPointerType::get(val.getType()), gepOp);
        create<StoreOp>(val, castAddr);
    }

    StackMap storeAll(mlir::Value offset, const llvm::SmallVector<mlir::Value>& values)
    {
        ASSURE(offset.getType(), ==, offsetT());
        StackMap offsets;
        uint32_t slotOffset = 0;
        for (size_t i = 0; i < values.size(); i++) {
            store(offset, slotOffset, values[i]);
            auto& size = offsets.emplace_back(TypeSize::FromType(values[i].getType()));
            slotOffset += size.slots();
        }

        return offsets;
    }

    CallOp regionCall(CallOp callOp, const llvm::ArrayRef<mlir::Value>& args)
    {
        const auto numberSubstring = callOp.getCallee().value().substr(21);
        ASSURE_TRUE(callOp.getCallee().value().contains("region"));
        const auto num = std::atoi(numberSubstring.str().c_str());

        std::vector<mlir::Type> types;
        std::vector<mlir::Value> values;
        for (auto arg : args) {
            // as this is not a tail-call is is sufficient to make the types match in size
            // but the signature does not have to match the surrounding
            const auto passableType = getGPPassableType(arg.getType());
            values.push_back(cast(arg, passableType));
            types.push_back(passableType);
        }
        const auto functionType = LLVMFunctionType::get(LLVMVoidType::get(builder.getContext()), types);

        LLVMFuncOp func;
        {
            mlir::OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPoint(function);
            func = create<LLVMFuncOp>(
                formatRegionCall(id(), static_cast<uint8_t>(kind), num), functionType);
            func.setVisibility(mlir::SymbolTable::Visibility::Private);
            func.setCConv(CConv::CALLING_CONVENTION);
        }
        return builder.replaceOpWithNewOp<CallOp>(callOp, func, values);
    }

    mlir::Type getGPPassableType(mlir::Type t)
    {
        // instead of using the target type directly we use an "same sized" integer
        // afterwards using bitcast to retrieve the correct type
        // this is required to avoid different calling convention registers for some type e.g. vector/float
        // TODO: think about vector in aggregate types...
        if (isPassedInGPRegisters(t)) {
            return t;
        } else {
            ASSURE_FALSE((t.isa<LLVMStructType, LLVMArrayType>()));
            return builder.getIntegerType(getBitSize(t));
        }
    }

    std::vector<mlir::Value> castForABI(mlir::Value value)
    {
        auto type = value.getType();
        if (type.isa<LLVMStructType, LLVMArrayType>() || getBitSize(type) > 64) {
            return destructAggregate(value);
        }
        return {cast(value, builder.getI64Type())};
    }

    mlir::Value cast(mlir::Value value, mlir::Type type)
    {
        if (value.getType() == type) {
            return value;
        } else if (value.getType().isa<mlir::IntegerType, mlir::VectorType, mlir::FloatType>()
            && type.isa<mlir::IntegerType, mlir::VectorType, mlir::FloatType>()
            && getBitSize(value.getType()) == getBitSize(type)) {
            return create<BitcastOp>(type, value);
        } else if (type.isa<mlir::IntegerType>()
            && value.getType().isa<mlir::IntegerType>()
            && getBitSize(value.getType()) > getBitSize(type)) {
            return create<TruncOp>(type, value);
        } else if (type.isa<mlir::IntegerType>()
            && value.getType().isa<mlir::IntegerType>()
            && getBitSize(value.getType()) < getBitSize(type)) {
            return create<ZExtOp>(type, value);
        } else if (type.isInteger(64) && value.getType().isa<LLVMPointerType>()) {
            return create<PtrToIntOp>(type, value);
        } else if (type.isa<LLVMPointerType>() && value.getType().isInteger(64)) {
            return create<IntToPtrOp>(type, value);
        } else if (value.getType().isa<mlir::FloatType>()
            && type.isa<mlir::IntegerType>()
            && getBitSize(value.getType()) != getBitSize(type)) {
            return cast(cast(value, builder.getIntegerType(value.getType().getIntOrFloatBitWidth())), type);
        } else if (value.getType().isa<mlir::IntegerType>()
            && type.isa<mlir::FloatType>()
            && getBitSize(value.getType()) != getBitSize(type)) {
            return cast(cast(value, builder.getIntegerType(type.getIntOrFloatBitWidth())), type);
        } else if (value.getType().isa<mlir::VectorType>()
            && llvm::cast<mlir::VectorType>(value.getType()).getElementType().isa<mlir::IntegerType>()) {
            return cast(cast(value, builder.getIntegerType(getBitSize(value.getType()))), type);
        } else if (type.isa<mlir::VectorType>()
            && llvm::cast<mlir::VectorType>(type).getElementType().isa<mlir::IntegerType>()) {
            return cast(cast(value, builder.getIntegerType(getBitSize(type))), type);
        } else {
            UNREACHABLE("unknown conversion from " << value.getType() << " to " << type);
        }
    }

    std::vector<mlir::Value> destructAggregate(mlir::Value val)
    {
        auto type = val.getType();
        std::vector<mlir::Value> returnValues;
        if (type.isa<LLVMArrayType>()) {
            LLVMArrayType arrayType = type.cast<LLVMArrayType>();
            for (unsigned i = 0; i < arrayType.getNumElements(); i++) {
                auto subValues = castForABI(create<ExtractValueOp>(val, i));
                returnValues.insert(returnValues.end(), subValues.begin(), subValues.end());
            }
        } else if (type.isa<LLVMStructType>()) {
            auto subtypes = type.cast<LLVMStructType>().getBody();
            uint32_t i = 0;
            for (auto subtype : subtypes) {
                auto subValues = castForABI(create<ExtractValueOp>(subtype, val, i));
                returnValues.insert(returnValues.end(), subValues.begin(), subValues.end());
                i++;
            }
        } else if (type.isa<mlir::FloatType>()) {
            ASSURE(getBitSize(type), ==, 80);
            auto i = create<BitcastOp>(builder.getIntegerType(80), val);
            return destructAggregate(create<ZExtOp>(builder.getIntegerType(128), i));
        } else if (type.isa<LLVMFixedVectorType>()) {
            ASSURE_TRUE((llvm::cast<LLVMFixedVectorType>(type).getElementType().isa<LLVMPointerType>()));
            ASSURE(getBitSize(type) % 64, ==, 0, << " so far we only support multiples of 64 bits");
            const int64_t vecSize = getBitSize(type) / 64;
            for (unsigned i = 0; i < vecSize; i++) {
                auto constPos = create<ConstantOp>(builder.getI8Type(), i);
                returnValues.push_back(cast(create<ExtractElementOp>(val, constPos), builder.getI64Type()));
            }
        } else {
            ASSURE_TRUE((type.isa<mlir::VectorType, mlir::IntegerType>()), << type);
            ASSURE(getBitSize(type) % 64, ==, 0, << " so far we only support multiples of 64 bits");
            const int64_t vecSize = getBitSize(type) / 64;
            auto vector = create<BitcastOp>(
                mlir::VectorType::get({vecSize}, builder.getI64Type()), val);
            for (unsigned i = 0; i < vecSize; i++) {
                auto constPos = create<ConstantOp>(builder.getI8Type(), i);
                returnValues.push_back(create<ExtractElementOp>(vector, constPos));
            }
        }

        return returnValues;
    }

    mlir::Value constructAggregateFromFunctionParams(mlir::Type type, uint8_t indx)
    {
        if (type.isa<LLVMArrayType>()) {
            mlir::Value constructedValue = create<UndefOp>(type);
            LLVMArrayType arrayType = type.cast<LLVMArrayType>();
            for (unsigned i = 0; i < arrayType.getNumElements(); i++) {
                constructedValue = create<InsertValueOp>(constructedValue,
                    addFunctionParameter(arrayType.getElementType(), indx),
                    builder.getDenseI64ArrayAttr({i}));
            }
            return constructedValue;
        } else if (type.isa<LLVMStructType>()) {
            mlir::Value constructedValue = create<UndefOp>(type);
            auto subtypes = type.cast<LLVMStructType>().getBody();
            uint32_t i = 0;
            for (auto subtype : subtypes) {
                constructedValue = create<InsertValueOp>(constructedValue,
                    addFunctionParameter(subtype, indx),
                    builder.getDenseI64ArrayAttr({i}));
                i++;
            }
            return constructedValue;
        } else if (type.isa<mlir::FloatType>()) {
            ASSURE(getBitSize(type), ==, 80);
            auto vec = constructAggregateFromFunctionParams(builder.getIntegerType(128), indx);
            auto i80 = create<TruncOp>(builder.getIntegerType(80), vec);
            return create<BitcastOp>(type, i80);
        } else if (type.isa<LLVMFixedVectorType>()) {
            auto elementType = llvm::cast<LLVMFixedVectorType>(type).getElementType();
            ASSURE_TRUE((elementType.isa<LLVMPointerType>()));
            ASSURE(getBitSize(type) % 64, ==, 0, << " so far we only support multiples of 64 bits");
            const int64_t vecSize = getBitSize(type) / 64;
            mlir::Value constructedValue = create<UndefOp>(type);
            for (unsigned i = 0; i < vecSize; i++) {
                auto constPos = create<ConstantOp>(builder.getI8Type(), i);
                constructedValue = create<InsertElementOp>(constructedValue,
                    cast(addFunctionParameter(builder.getI64Type(), indx), elementType), constPos);
            }
            return constructedValue;
        } else {
            ASSURE_TRUE((type.isa<mlir::VectorType, mlir::IntegerType>()));
            ASSURE(getBitSize(type) % 64, ==, 0, << " so far we only support multiples of 64 bits");
            const int64_t vecSize = getBitSize(type) / 64;
            mlir::Value constructedValue = create<UndefOp>(
                mlir::VectorType::get({vecSize}, builder.getI64Type()));
            for (unsigned i = 0; i < vecSize; i++) {
                auto constPos = create<ConstantOp>(builder.getI8Type(), i);
                constructedValue = create<InsertElementOp>(constructedValue,
                    addFunctionParameter(builder.getI64Type(), indx), constPos);
            }
            return cast(constructedValue, type);
        }
    }

    mlir::Value
    addFunctionParameter(mlir::Type type, uint8_t indx)
    {
        if (type.isa<LLVMStructType, LLVMArrayType>() || getBitSize(type) > 64) {
            return constructAggregateFromFunctionParams(type, indx);
        }
        mlir::Type argType = builder.getI64Type();

        // add parameters can be added in any order, we have to ensure that they are ordered according to their indices
        // for that we also maintain a (sorted) array of the already inserted indices
        auto indices = function->getAttrOfType<mlir::DenseI8ArrayAttr>(kArgNumbersAttr).asArrayRef().vec();
        auto it = std::lower_bound(
            indices.begin(), indices.end(), indx, [](const auto elem, const auto value) {
                return elem <= value;
            });
        uint32_t insertIndex = static_cast<uint32_t>(std::distance(indices.begin(), it)) + 1;
        indices.insert(it, indx);
        ASSURE(insertIndex, >=, 1);
        ASSURE(insertIndex, <=, function.getBody().getNumArguments());
        auto argTypes = function.getArgumentTypes().vec();
        argTypes.insert(argTypes.begin() + insertIndex, argType);
        auto arg = function.getBody().insertArgument(insertIndex, argType, op.getLoc());
        auto resultType = !function.getResultTypes().empty() ? function.getResultTypes()[0] : builder.getType<mlir::LLVM::LLVMVoidType>();
        function.setFunctionTypeAttr(mlir::TypeAttr::get(LLVMFunctionType::get(resultType, argTypes)));
        function->setAttr(kArgNumbersAttr, builder.getDenseI8ArrayAttr(indices));
        // if not strict abi this is obviously a nop-cast
        return cast(arg, type);
    }

    llvm::LLVMContext llvmContext;
    mlir::RewriterBase& builder;
    mlir::Operation& op;
    TEMPLATE_KIND kind;
    LLVMFuncOp function;
};
}

void ToLLVMLoweringPass::getDependentDialects(mlir::DialectRegistry& registry) const
{
    registry.insert<mlir::LLVM::LLVMDialect>();
}

void ToLLVMLoweringPass::runOnOperation()
{
    const auto& dataLayoutAnalysis = getAnalysis<mlir::DataLayoutAnalysis>();

    mlir::LLVMConversionTarget target(getContext());
    target.addLegalOp<mlir::ModuleOp>();

    mlir::LowerToLLVMOptions options(&getContext(), dataLayoutAnalysis.getAtOrAbove(getOperation()));
    mlir::LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
    typeConverter.addSourceMaterialization([&](mlir::OpBuilder& builder, mlir::FunctionType type, mlir::ValueRange valueRange, mlir::Location loc) {
        auto cast = builder.create<mlir::UnrealizedConversionCastOp>(loc, type, valueRange);
        return cast.getResult(0);
    });

    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateSCFToControlFlowConversionPatterns(patterns);
    mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::index::populateIndexToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateReconcileUnrealizedCastsPatterns(patterns);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> ToLLVMLoweringPass::create()
{
    return std::make_unique<ToLLVMLoweringPass>();
}

TailCallRewriteOp::TailCallRewriteOp(
    mlir::MLIRContext* context,
    TEMPLATE_KIND kind)
    : RewritePattern("llvm.return", mlir::PatternBenefit(123), context)
    , m_kind(kind)
{
}

mlir::LogicalResult TailCallRewriteOp::match(mlir::Operation* op) const
{
    return mlir::LogicalResult::success(op->getNumOperands() > 0);
}

void TailCallRewriteOp::rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const
{
    using namespace mlir::LLVM;

    Helper helper(rewriter, *op, m_kind);
    auto func = helper.function;
    ASSURE_FALSE(func->hasAttr(kNeedsStackIndicator));

    if (func->hasAttr(kTailCallIndicator)) {
        ASSURE(op->getNumOperands(), >=, 1);
        std::vector<mlir::Value> args;
        // the stack parameter is forwarded just as pointer
        // all other arguments might need a conversion
        auto valIt = op->operand_begin();
        args.push_back(*valIt);
        valIt++;
        for (; valIt != op->operand_end(); valIt++) {
            auto castValues = helper.castForABI(*valIt);
            ASSURE_TRUE(std::all_of(castValues.begin(), castValues.end(), [](const auto val) {
                return static_cast<bool>(val);
            }));
            args.insert(args.end(), castValues.begin(), castValues.end());
        }
        // parameter counts must match
        if (args.size() < helper.function.getNumArguments()) {
            auto undef = helper.create<UndefOp>(rewriter.getI64Type());
            args.insert(args.end(), helper.function.getNumArguments() - args.size(), undef);
        }
        while (args.size() != helper.function.getNumArguments()) {
            helper.addFunctionParameter(rewriter.getI64Type(), -1);
        }
        std::vector<mlir::Type> types(args.size());
        std::transform(args.cbegin(), args.cend(), types.begin(), std::mem_fn(&mlir::Value::getType));

        rewriter.setInsertionPoint(func);
        auto tailCall = helper.create<LLVMFuncOp>(
            formatNextCall(helper.id(), static_cast<uint8_t>(m_kind)),
            LLVMFunctionType::get(LLVMVoidType::get(rewriter.getContext()), types, false),
            Linkage::External,
            false,
            CConv::CALLING_CONVENTION);
        rewriter.setInsertionPoint(op);

        helper.create<CallOp>(tailCall, args);
    }

    rewriter.replaceOpWithNewOp<ReturnOp>(op, std::vector<mlir::Value>{});
}

LoadInputFromStackRewriteOp::LoadInputFromStackRewriteOp(mlir::MLIRContext* context, TEMPLATE_KIND kind)
    : RewritePattern("builtin.unrealized_conversion_cast", mlir::PatternBenefit(123), context)
    , m_kind(kind)
{
}

mlir::LogicalResult LoadInputFromStackRewriteOp::match(mlir::Operation* op) const
{
    return mlir::LogicalResult::success(op->hasAttr(kInputNumAttr) && !op->hasAttr(kIsStoreIndicator));
}

void LoadInputFromStackRewriteOp::rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const
{
    using namespace mlir::LLVM;
    Helper helper(rewriter, *op, m_kind);

    ASSURE_FALSE(op->hasAttr(kIsStoreIndicator));
    ASSURE(op->getNumOperands(), ==, 0);

    auto argumentIndex = op->getAttrOfType<mlir::IntegerAttr>(kInputNumAttr).getValue().getZExtValue();
    std::vector<mlir::Value> sourceValuesReplace;
    uint32_t registerArgumentsSize = 0;
    for (auto [resultNr, it] = std::pair{0u, op->result_begin()};
         it != op->result_end();
         it++, resultNr++, argumentIndex++) {
        mlir::Type targetType = findLLVMCompatibleType(*it);
        ASSURE_TRUE(targetType && isCompatibleType(targetType));

        mlir::Value replacedValue;
        switch (helper.loadKind()) {
            case TEMPLATE_KIND::STACK_ARGS: {
                auto arg = helper.addFunctionParameter(helper.offsetT(), argumentIndex);
                replacedValue = helper.load(arg, targetType);
                break;
            }
            case TEMPLATE_KIND::STACK_PATCHED: {
                if (op->hasAttr(kFromRegionIndicator)
                    && op->hasAttr(kOperandCountAttr)
                    && IsRegisterCC(m_kind)
                    && op->getNumResults() == 1) {
                    const auto bitSize = getBitSize(targetType);
                    const auto reg = op->getAttrOfType<mlir::IntegerAttr>(kOperandCountAttr).getValue().getZExtValue();
                    if ((bitSize + 63) / 64 == 1 && reg < OPERAND_REGISTERS) {
                        auto outType = bitSize == 64 ? targetType : rewriter.getI64Type();
                        auto asmOut = helper.create<InlineAsmOp>(
                                                outType,
                                                mlir::ValueRange{},
                                                llvm::formatv(REGISTER_RETURN_ASM, REGISTERS[reg]).str(),
                                                "=r",
                                                true,
                                                false,
                                                rewriter.getAttr<AsmDialectAttr>(AsmDialect::AD_ATT),
                                                rewriter.getAttr<mlir::ArrayAttr>(mlir::ArrayRef<mlir::Attribute>{}))
                                          .getRes();
                        if (bitSize == 64) {
                            replacedValue = asmOut;
                        } else if (llvm::isa<mlir::IntegerType, mlir::FloatType>(targetType)) {
                            replacedValue = helper.cast(asmOut, targetType);
                        } else {
                            UNREACHABLE("unsupported type loaded from register: " << targetType);
                        }
                        break;
                    }
                }
#ifndef FIXED_STACKSIZE
                ASSURE(!op->hasAttr(kFromRegionIndicator), ||, !helper.function->hasAttr(kNeedsStackIndicator),
                    << " llvm generates invalid code, if we try to access the frame"
                    << " again after the call... (tries an invalid $rbp relative reload)\n"
                    << helper.function);
#endif
                replacedValue = helper.load(helper.placeholder(argumentIndex), targetType);
                break;
            }
            case TEMPLATE_KIND::REGISTERS: {
                const auto size = TypeSize::FromType(targetType);
                if (size.slots() <= MAX_SIZE_REGISTER && registerArgumentsSize + size.slots() <= OPERAND_REGISTERS) {
                    replacedValue = helper.addFunctionParameter(targetType, argumentIndex);
                    registerArgumentsSize += size.slots();
                } else {
                    replacedValue = helper.load(helper.placeholder(argumentIndex), targetType);
                }
                break;
            }
        }
        (*it).replaceAllUsesWith(replacedValue);
    }
    ASSURE_TRUE(op->getUses().empty(), << *op << "\n\n"
                                       << helper.function);
    rewriter.eraseOp(op);
}

StoreRegionArgsToStackRewriteOp::StoreRegionArgsToStackRewriteOp(mlir::MLIRContext* context, TEMPLATE_KIND kind)
    : RewritePattern("builtin.unrealized_conversion_cast", mlir::PatternBenefit(123), context)
    , m_kind(kind)
{
}

mlir::LogicalResult StoreRegionArgsToStackRewriteOp::match(mlir::Operation* op) const
{
    return mlir::LogicalResult::success(op->hasAttr(kInputNumAttr) && op->hasAttr(kIsStoreIndicator));
}

void StoreRegionArgsToStackRewriteOp::rewrite(mlir::Operation* op, mlir::PatternRewriter& rewriter) const
{
    using namespace mlir::LLVM;

    Helper helper(rewriter, *op, m_kind);
    auto func = helper.function;
    auto stack = helper.stack();

    if (op->hasAttr(kDummyIndicator)) {
        op->removeAttr(kDummyIndicator);
        ASSURE(op->getNumOperands(), ==, 2);
        ASSURE(op->getOperand(0), ==, op->getOperand(1));
        op->setOperands(op->getOperand(0));
    }

    ASSURE_TRUE(op->hasAttr(kIsStoreIndicator));
    const auto argumentIndex = op->getAttrOfType<mlir::IntegerAttr>(kInputNumAttr).getValue().getZExtValue();

    llvm::SmallVector<mlir::Value> sourceValues(op->operand_begin(), op->operand_end());

    for (unsigned i = 0; i < op->getNumOperands(); i++) {
        mlir::Operation* defOp = op->getOperand(i).getDefiningOp();
        while (defOp != nullptr && !defOp->hasAttr(kInputNumAttr) && !isCompatibleType(sourceValues[i].getType())) {
            ASSURE_TRUE(llvm::isa<mlir::UnrealizedConversionCastOp>(defOp));
            ASSURE(defOp->getNumOperands(), ==, 1);
            sourceValues[i] = defOp->getOperand(0);
            defOp = defOp->getOperand(0).getDefiningOp();
        }
    }
    ASSURE_TRUE(std::all_of(std::begin(sourceValues), std::end(sourceValues),
                    [](const auto& value) { return isCompatibleType(value.getType()); }),
        << helper.function);

    StackMap offsets;
    const auto isFunctionPattern = func->hasAttr(kNeedsStackIndicator);
    const auto forwardingCount = isFunctionPattern ? func.getNumArguments() : 1;
    ASSURE(func.getNumArguments(), >=, forwardingCount);
    std::vector<mlir::Value> forwardingArgs = isFunctionPattern
        ? std::vector<mlir::Value>{stack}
        : std::vector<mlir::Value>(func.getArguments().begin(), func.getArguments().begin() + forwardingCount);

    ASSURE_TRUE(op->getResult(0).hasOneUse());
    auto user = *op->getResult(0).getUsers().begin();
    ASSURE_TRUE((llvm::isa<ReturnOp, CallOp>(user)));
    const bool isRegionCall = llvm::isa<CallOp>(user);

    ASSURE(forwardingArgs.size(), ==, 1);

    if (!sourceValues.empty()) {
        switch (m_kind) {
            case TEMPLATE_KIND::STACK_ARGS: {
                auto arg = helper.addFunctionParameter(helper.offsetT(), argumentIndex);
                offsets = helper.storeAll(arg, sourceValues);
            } break;
            case TEMPLATE_KIND::STACK_PATCHED: {
                auto placeholder = helper.placeholder(argumentIndex);
                offsets = helper.storeAll(placeholder, sourceValues);
            } break;
            case TEMPLATE_KIND::REGISTERS: {
                auto placeholder = helper.placeholder(argumentIndex);
                uint32_t slotOffset = 0;
                uint32_t registerArgumentsSize = 0;
                for (size_t i = 0; i < sourceValues.size(); i++) {
                    const auto size = TypeSize::FromType(sourceValues[i].getType());
                    if (size.slots() <= MAX_SIZE_REGISTER && registerArgumentsSize + size.slots() <= OPERAND_REGISTERS) {
                        forwardingArgs.push_back(sourceValues[i]);
                        registerArgumentsSize += size.slots();
                    } else
                        helper.store(placeholder, slotOffset, sourceValues[i]);
                    offsets.emplace_back(size);
                    slotOffset += size.slots();
                }
            } break;
        }
    }

    ASSURE(forwardingArgs.size(), >=, 1);
    if (isRegionCall) {
        auto call = helper.regionCall(
            llvm::cast<CallOp>(user),
            forwardingArgs);
        std::vector<mlir::Attribute> typeSizeAttributes;
        for (const auto typeSize : offsets) {
            typeSizeAttributes.push_back(typeSize.ToAttr(rewriter));
        }
        call->setAttr(kStackmapAttr, rewriter.getArrayAttr(typeSizeAttributes));
        rewriter.replaceOp(op, stack);
    } else {
        user->setOperands(forwardingArgs);
        rewriter.eraseOp(op);
    }
}
