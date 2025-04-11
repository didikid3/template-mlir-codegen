#pragma once

#include "common.hpp"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace mlir {
class OpBuilder;
}

class TypeSize
{
public:
    explicit constexpr TypeSize(uint32_t bits, uint32_t align = 0)
        : bitSize(bits)
        , alignment(align)
    {
    }

    TypeSize() = default;

    static TypeSize FromType(mlir::Type type);
    static TypeSize FromAttr(mlir::Attribute attr);
    mlir::Attribute ToAttr(mlir::OpBuilder& builder) const;

    constexpr uint32_t bits() const { return bitSize; };
    constexpr uint32_t bytes() const { return (bitSize + 7) / 8; };
    constexpr uint32_t slots() const { return (bitSize + 63) / 64; };
    constexpr uint32_t align() const { return alignment; };

private:
    uint32_t bitSize;
    uint32_t alignment;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const TypeSize& typeSize);
bool operator==(const TypeSize& lhs, const TypeSize& rhs);
