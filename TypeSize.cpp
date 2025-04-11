#include "TypeSize.hpp"
#include "Preparation/Utils.hpp"

#include "mlir/IR/Builders.h"

TypeSize TypeSize::FromType(mlir::Type type)
{
    return TypeSize(getBitSize(type), getBitAlignment(type));
}

TypeSize TypeSize::FromAttr(mlir::Attribute attr)
{
    ASSURE_TRUE(attr.isa<mlir::DenseIntElementsAttr>());
    const auto data = attr.cast<mlir::DenseIntElementsAttr>().getValues<int32_t>();
    return TypeSize(static_cast<uint32_t>(*data.begin()), static_cast<uint32_t>(*(data.begin() + 1)));
}

mlir::Attribute TypeSize::ToAttr(mlir::OpBuilder& builder) const
{
    return builder.getI32VectorAttr(mlir::ArrayRef<int32_t>({static_cast<int32_t>(bitSize), static_cast<int32_t>(alignment)}));
}

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeSize& typeSize)
{
    os << typeSize.bytes() << " B (" << typeSize.align() << ")";
    return os;
}
bool operator==(const TypeSize& lhs, const TypeSize& rhs)
{
    return lhs.bits() == rhs.bits();
}
