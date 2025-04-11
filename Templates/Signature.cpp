#include "Templates/Signature.hpp"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

Signature::Signature(mlir::Operation& op)
    : m_opId(op.getRegisteredInfo()->getTypeID())
    , m_propertyHash(op.hashProperties())
    , m_properties(op.getPropertiesStorage())
    , m_operands(op.getOperandTypes().begin(), op.getOperandTypes().end())
    , m_results(op.getResultTypes().begin(), op.getResultTypes().end())
    , m_numRegions(op.getNumRegions())
    , m_mightBeConstant(
          op.getOperands().empty()
          && op.getNumRegions() == 0
          && mlir::isMemoryEffectFree(&op))
    , m_needStackArg(
          llvm::isa<mlir::FunctionOpInterface>(op)
          && !(llvm::isa<mlir::LLVM::LLVMFuncOp>(op)
              && llvm::cast<mlir::LLVM::LLVMFuncOp>(op).isVarArg()))

{
}

Signature::Signature(mlir::TypeID opId,
    llvm::hash_code propertyHash,
    mlir::OpaqueProperties properties,
    std::vector<mlir::Type> operands,
    std::vector<mlir::Type> results,
    uint8_t numRegions,
    bool constant,
    bool stack)
    : m_opId(opId)
    , m_propertyHash(propertyHash)
    , m_properties(properties)
    , m_operands(operands)
    , m_results(results)
    , m_numRegions(numRegions)
    , m_mightBeConstant(constant)
    , m_needStackArg(stack)
{
}

bool Signature::needStack() const
{
    return m_needStackArg;
}

bool Signature::isConstexpr() const
{
    return m_mightBeConstant;
}

mlir::TypeID Signature::id() const
{
    return m_opId;
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, std::pair<const Signature&, mlir::MLIRContext&> data)
{
    auto& context = data.second;
    const auto& s = data.first;
    const auto it = std::find_if(
        context.getRegisteredOperations().begin(),
        context.getRegisteredOperations().end(),
        [&](auto opname) {
            return opname.getTypeID() == s.m_opId;
        });
    ASSURE(it, !=, context.getRegisteredOperations().end());
    os << it->getStringRef();
    os << "(";
    for (const auto& t : s.m_operands) {
        os << t << ",";
    }
    os << ")->(";
    for (const auto& t : s.m_results) {
        os << t << ",";
    }
    os << ")";

    return os;
}

namespace {
#ifdef LESS_TEMPLATES
bool comparePropertiesTyped(
    mlir::LLVM::LLVMFuncOp::Properties lhs,
    mlir::LLVM::LLVMFuncOp::Properties rhs)
{
    if (llvm::cast<mlir::LLVM::LLVMFunctionType>(rhs.getFunctionType().getValue()).isVarArg())
        // vararg functions are unique and therefore should also be named after their "real" name
        return rhs.getSymName() == lhs.getSymName();
    return rhs.getCConv() == lhs.getCConv()
        && rhs.getAlignment() == lhs.getAlignment()
        && rhs.getArgAttrs() == lhs.getArgAttrs()
        && rhs.getArmLocallyStreaming() == lhs.getArmLocallyStreaming()
        && rhs.getArmStreaming() == lhs.getArmStreaming()
        // && rhs.getComdat() == lhs.getComdat()
        // && rhs.getDsoLocal() == lhs.getDsoLocal()
        && rhs.getFunctionEntryCount() == lhs.getFunctionEntryCount()
        && rhs.getFunctionType() == lhs.getFunctionType()
        && rhs.getGarbageCollector() == lhs.getGarbageCollector()
        // && rhs.getLinkage() == lhs.getLinkage()
        && rhs.getMemory() == lhs.getMemory()
        && rhs.getPassthrough() == lhs.getPassthrough()
        && rhs.getPersonality() == lhs.getPersonality()
        && rhs.getResAttrs() == lhs.getResAttrs()
        && rhs.getSection() == lhs.getSection();
    // && rhs.getSymName() == lhs.getSymName()
    // && rhs.getUnnamedAddr() == lhs.getUnnamedAddr()
    // && rhs.getVisibility() == lhs.getVisibility()
}

bool comparePropertiesTyped(
    mlir::LLVM::CallOp::Properties lhs,
    mlir::LLVM::CallOp::Properties rhs)
{
    return
        // rhs.access_groups == this->access_groups &&
        // rhs.alias_scopes == this->alias_scopes &&
        // rhs.branch_weights == this->branch_weights &&
        // rhs.noalias_scopes == this->noalias_scopes &&
        // rhs.tbaa == this->tbaa &&
        // the callee does not matter it is just important, that it exists
        static_cast<bool>(rhs.getCallee()) == static_cast<bool>(lhs.getCallee())
        && rhs.getFastmathFlags() == lhs.getFastmathFlags();
}

bool comparePropertiesTyped(
    mlir::func::CallOp::Properties lhs,
    mlir::func::CallOp::Properties rhs)
{
    return
        // the callee does not matter it is just important, that it exists
        static_cast<bool>(rhs.getCallee()) == static_cast<bool>(lhs.getCallee());
}
#endif

bool comparePropertiesTyped([[maybe_unused]] mlir::TypeID typeID, mlir::OpaqueProperties lhs, mlir::OpaqueProperties rhs)
{
#ifdef LESS_TEMPLATES
    if (typeID == mlir::TypeID::get<mlir::LLVM::LLVMFuncOp>()) {
        auto& _lhs = *lhs.as<mlir::LLVM::LLVMFuncOp::Properties*>();
        auto& _rhs = *rhs.as<mlir::LLVM::LLVMFuncOp::Properties*>();
        return comparePropertiesTyped(_lhs, _rhs);
    } else if (typeID == mlir::TypeID::get<mlir::LLVM::CallOp>()) {
        auto& _lhs = *lhs.as<mlir::LLVM::CallOp::Properties*>();
        auto& _rhs = *rhs.as<mlir::LLVM::CallOp::Properties*>();
        return comparePropertiesTyped(_lhs, _rhs);
    } else if (typeID == mlir::TypeID::get<mlir::func::CallOp>()) {
        auto& _lhs = *lhs.as<mlir::func::CallOp::Properties*>();
        auto& _rhs = *rhs.as<mlir::func::CallOp::Properties*>();
        return comparePropertiesTyped(_lhs, _rhs);
    }
#endif
    return lhs.as<void*>() == rhs.as<void*>();
}
}

bool operator==(const Signature& lhs, const Signature& rhs)
{
    const auto& inLhs = lhs.m_operands;
    const auto& inRhs = rhs.m_operands;
    const auto& outLhs = lhs.m_results;
    const auto& outRhs = rhs.m_results;
    if (lhs.m_opId != rhs.m_opId
#ifndef LESS_TEMPLATES
        || lhs.m_propertyHash != rhs.m_propertyHash
#endif
        || lhs.m_numRegions != rhs.m_numRegions
        || inLhs.size() != inRhs.size()
        || outLhs.size() != outRhs.size())
        return false;
    return (inLhs == inRhs)
        && (outLhs == outRhs)
        && comparePropertiesTyped(lhs.m_opId, lhs.m_properties, rhs.m_properties);
}

bool Signature::compareProperties(mlir::Operation* op, mlir::OpaqueProperties props)
{
#ifdef LESS_TEMPLATES
    if (llvm::isa<mlir::LLVM::LLVMFuncOp>(op)) {
        auto& lhs = *props.as<mlir::LLVM::LLVMFuncOp::Properties*>();
        auto& rhs = llvm::cast<mlir::LLVM::LLVMFuncOp>(op).getProperties();
        return comparePropertiesTyped(lhs, rhs);
    } else if (llvm::isa<mlir::LLVM::CallOp>(op)) {
        auto& lhs = *props.as<mlir::LLVM::CallOp::Properties*>();
        auto& rhs = llvm::cast<mlir::LLVM::CallOp>(op).getProperties();
        return comparePropertiesTyped(lhs, rhs);
    } else if (llvm::isa<mlir::func::CallOp>(op)) {
        auto& lhs = *props.as<mlir::func::CallOp::Properties*>();
        auto& rhs = llvm::cast<mlir::func::CallOp>(op).getProperties();
        return comparePropertiesTyped(lhs, rhs);
    }
#endif
    return op->getRegisteredInfo()->compareOpProperties(op->getPropertiesStorage(), props);
}

size_t llvm::DenseMapInfo<Signature>::getHashValue(const Signature& sig)
{
    return llvm::hash_combine(
        sig.m_opId
#ifndef LESS_TEMPLATES
        ,
        sig.m_propertyHash
#endif
    );
}

size_t llvm::DenseMapInfo<Signature>::getHashValue(const mlir::Operation& op)
{
    ASSURE_TRUE(const_cast<mlir::Operation&>(op).getRegisteredInfo().has_value());
    return llvm::hash_combine(
        const_cast<mlir::Operation&>(op).getRegisteredInfo()->getTypeID()
#ifndef LESS_TEMPLATES
            ,
        const_cast<mlir::Operation&>(op).hashProperties()
#endif
    );
}

bool llvm::DenseMapInfo<Signature>::isEqual(const mlir::Operation& opConst, const Signature& sig)
{
    auto& op = const_cast<mlir::Operation&>(opConst);
    const auto& props = sig.m_properties;
    const auto& ops = sig.m_operands;
    const auto& res = sig.m_results;
    ASSURE_TRUE(op.getRegisteredInfo().has_value());
    return sig.m_opId == op.getRegisteredInfo()->getTypeID()
        && sig.m_numRegions == op.getNumRegions()
        && ops.size() == op.getNumOperands()
        && res.size() == op.getNumResults()
        // this is safe as we previously already checked the length
        && std::equal(ops.begin(), ops.end(), op.operand_type_begin())
        && std::equal(res.begin(), res.end(), op.result_type_begin())
        && Signature::compareProperties(&op, props);
}
