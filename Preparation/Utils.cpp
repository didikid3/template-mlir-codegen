#include "Preparation/Utils.hpp"

#include "TemplateCompiler.hpp"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

mlir::Type stackT(mlir::Builder& builder)
{
    return mlir::LLVM::LLVMPointerType::get(builder.getI8Type());
}

mlir::DataLayout layout;
void setDataLayout(mlir::ModuleOp mod)
{
    const auto dataLayout = compiler::getTargetMachine(llvm::CodeModel::Small, llvm::TargetOptions{})->createDataLayout();
    auto dataLayoutSpec = mlir::translateDataLayout(dataLayout, mod.getContext());
    // ASSURE_FALSE(mod->hasAttr(mlir::DLTIDialect::kDataLayoutAttrName));
    mod->setAttr(mlir::DLTIDialect::kDataLayoutAttrName, dataLayoutSpec);

    [[maybe_unused]] auto* ctx = mod.getContext();
    std::destroy_at(&layout);
    std::construct_at(&layout, mod);
    // some consistency checks
    // it is important, that the size/alignments etc. are consistent across the MLIR methods and the LLVM equivalents,
    // as we generate the binary code with LLVM (and its assumptions about size and alignment) but rely on the MLIR information
    // during runtime compilation and patching of the code...

    // Due to the implementation of MLIR, there are currently a few restrictions (for which we have to provide a manual work-around):
    // - vectors of i1 are NOT bit-packed (vector sizes in general are a bit off...)
    // - there is no implementation for LLVM vector present at all: "LLVM ERROR: neither the scoping op nor the type class provide data layout information for !llvm.vec<10 x ptr>"
    ASSURE(getBitSize(mlir::LLVM::getFixedVectorType(mlir::LLVM::LLVMPointerType::get(ctx), 10)), ==, 640);
    ASSURE([&] {
        llvm::LLVMContext llvmCtx;
        return dataLayout.getTypeSizeInBits(llvm::FixedVectorType::get(llvm::IntegerType::getInt1Ty(llvmCtx), 6));
    }(),
        ==, 6);
    ASSURE(getBitSize(mlir::VectorType::get({6}, mlir::IntegerType::get(ctx, 1))), ==, 6);
    ASSURE(dataLayout.getABIIntegerTypeAlignment(64).value(), ==, 8);
    ASSURE(layout.getTypeABIAlignment(mlir::IntegerType::get(ctx, 64)), ==, 8);
}

uint32_t getBitSize(mlir::Type type)
{
    ASSURE(type, !=, mlir::Type());
    ASSURE_TRUE(mlir::LLVM::isCompatibleType(type));
    if (type.isa<mlir::LLVM::LLVMFixedVectorType>()) {
        // missing layout spec for those cases...
        auto vecT = llvm::cast<mlir::LLVM::LLVMFixedVectorType>(type);
        return getBitSize(vecT.getElementType()) * vecT.getNumElements();
    } else if (type.isa<mlir::VectorType>()) {
        // mlir returns invalid bit width for i1 vectors
        auto elemType = llvm::cast<mlir::VectorType>(type).getElementType();
        if (elemType.isa<mlir::IntegerType>() && elemType.getIntOrFloatBitWidth() == 1 && llvm::cast<mlir::VectorType>(type).getShape().size() == 1) {
            return llvm::cast<mlir::VectorType>(type).getShape().front();
        }
    }
    return layout.getTypeSizeInBits(type);
}

uint32_t getBitAlignment(mlir::Type type)
{
    ASSURE(type, !=, mlir::Type());
    ASSURE_TRUE(mlir::LLVM::isCompatibleType(type));
    if (type.isa<mlir::LLVM::LLVMFixedVectorType, mlir::VectorType>()) {
        // missing layout spec for those cases...
        return MAX_ALIGN;
    }
    return layout.getTypeABIAlignment(type);
}

uint64_t accumulateOffsets(mlir::Type type, std::vector<int64_t>::const_iterator begin, std::vector<int64_t>::const_iterator end)
{
    using namespace mlir::LLVM;

    static mlir::DataLayout layout;
    if (begin == end) {
        return 0;
    }
    ASSURE_TRUE(type);
    ASSURE_TRUE((type.isa<LLVMArrayType, LLVMStructType, LLVMPointerType>()), << type);
    if (type.isa<LLVMPointerType>()) {
        auto pointerT = llvm::cast<LLVMPointerType>(type);
        ASSURE_FALSE(pointerT.isOpaque(), << pointerT);
        const auto thisOffset = (*begin) * ((getBitSize(pointerT.getElementType()) + 7) / 8);
        return thisOffset + accumulateOffsets(pointerT.getElementType(), begin + 1, end);
    } else if (type.isa<LLVMArrayType>()) {
        auto arrayT = llvm::cast<LLVMArrayType>(type);
        const auto thisOffset = (*begin) * ((getBitSize(arrayT.getElementType()) + 7) / 8);
        return thisOffset + accumulateOffsets(arrayT.getElementType(), begin + 1, end);
    } else {
        auto structT = llvm::cast<LLVMStructType>(type);
        auto thisOffset = 0;
        int64_t i = 0;
        for (; i < *begin; i++) {
            auto t = structT.getBody()[i];
            thisOffset += (getBitSize(t) + 7) / 8;
            if (structT.getBody().size() > static_cast<size_t>(i + 1) && !structT.isPacked()) {
                const auto align = layout.getTypeABIAlignment(structT.getBody()[i + 1]);
                ASSURE(align, &&, ((align & (align - 1)) == 0));
                thisOffset = (thisOffset + align - 1) & -align;
            }
        }
        return thisOffset + accumulateOffsets(structT.getBody()[i], begin + 1, end);
    }
}

mlir::Type findLLVMCompatibleType(mlir::Value val)
{
    if (mlir::LLVM::isCompatibleType(val.getType())) {
        return val.getType();
    }
    mlir::Type type = nullptr;
    for (auto user : val.getUsers()) {
        if (user != nullptr
            && llvm::isa<mlir::UnrealizedConversionCastOp>(user)
            && user->getNumOperands() == 1
            && user->getNumResults() == 1
            && !user->hasAttr(kInputNumAttr)) {
            auto t = findLLVMCompatibleType(user->getResult(0));
            if (t != nullptr && mlir::LLVM::isCompatibleType(t)) {
                // we could break here but instead run to the end and
                // use this as a consistency check
                ASSURE(type == nullptr, ||, t == type);
                type = t;
            }
        }
    }
    return type;
}

std::pair<int, int> parseSectionName(llvm::StringRef str)
{
    if (str.starts_with(kPrefixSectionName)) {
        const auto stencilNr = std::stoi(str.substr(13, str.find_first_of("_") - 13).data());
        const auto kind = std::stoi(str.substr(str.find_first_of("_") + 1).data());
        return {stencilNr, kind};
    } else {
        const auto stencilNr = std::stoi(str.substr(17, str.find_first_of("_") - 17).data());
        const auto kind = std::stoi(str.substr(str.find_first_of("_") + 1).data());
        return {stencilNr, kind};
    }
}

bool isTemplateSection(llvm::StringRef str)
{
    return str.starts_with(kPrefixSectionName) || str.starts_with(kPrefixSplitSectionName);
}
