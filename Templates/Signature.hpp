#pragma once

#include "common.hpp"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

/**
 * @brief unique signature of an arbitrary mlir operation
 */
struct Signature
{
public:
    explicit Signature(mlir::Operation& op);
    Signature(mlir::TypeID opId,
        llvm::hash_code propertyHash,
        mlir::OpaqueProperties properties,
        std::vector<mlir::Type> operands,
        std::vector<mlir::Type> results,
        uint8_t numRegions,
        bool constant = false,
        bool stack = false);

    bool isConstexpr() const;
    bool needStack() const;
    mlir::TypeID id() const;

    static bool compareProperties(mlir::Operation* op, mlir::OpaqueProperties props);

private:
    friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os, std::pair<const Signature&, mlir::MLIRContext&>);
    friend bool operator==(const Signature& lhs, const Signature& rhs);
    friend class TemplateLibrary;
    template <typename, typename>
    friend class llvm::DenseMapInfo;

    /////////////////////////////
    // used for comparison
    mlir::TypeID m_opId;
    llvm::hash_code m_propertyHash;
    mlir::OpaqueProperties m_properties;
    std::vector<mlir::Type> m_operands;
    std::vector<mlir::Type> m_results;
    uint8_t m_numRegions;
    /////////////////////////////
    bool m_mightBeConstant = false;
    bool m_needStackArg = false;
};
namespace llvm {

// Utilities for hash map usage of a signature
template <>
struct DenseMapInfo<Signature>
{
    static inline Signature getEmptyKey()
    {
        return Signature{
            mlir::TypeID(), llvm::hash_code(), mlir::OpaqueProperties(nullptr), {}, {}, 0};
    };
    static inline Signature getTombstoneKey()
    {
        return Signature{
            mlir::TypeID(), llvm::hash_code(), mlir::OpaqueProperties(nullptr), {}, {}, 1};
    }
    static size_t getHashValue(const Signature& sig);
    static size_t getHashValue(const mlir::Operation& op);
    static bool isEqual(const Signature& lhs, const Signature& rhs) { return lhs == rhs; }
    static bool isEqual(const mlir::Operation& opConst, const Signature& sig);
};
}
