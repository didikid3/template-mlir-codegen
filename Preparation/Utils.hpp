#pragma once

#include "TypeSize.hpp"
#include "common.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/IR/CallingConv.h"
#include "llvm/Support/FormatVariadic.h"

using StackMap = std::vector<TypeSize>;
using ResultSizes = llvm::SmallVector<TypeSize, 2>;
using StackMaps = llvm::SmallVector<StackMap, 2>;

namespace mlir {
class Builder;
}

// sets the data layout used for getBitsSize/getBitAlignment
void setDataLayout(mlir::ModuleOp mod);

mlir::Type stackT(mlir::Builder& builder);

uint32_t getBitSize(mlir::Type type);
uint32_t getBitAlignment(mlir::Type type);

/**
 * @brief Computes the total offset for indexing into the type using the provided indices (similar to gep)
 *
 * @param type base type
 * @param begin start iterator of constant indices
 * @param end end iterator of constant indices
 * @return total offset
 */
uint64_t accumulateOffsets(
    mlir::Type type,
    std::vector<int64_t>::const_iterator begin,
    std::vector<int64_t>::const_iterator end);

/**
 * @brief Starting from the value going through the unrealized-conversion-cast use chains recursively to find an LLVM compatible type
 *
 * @param val Starting value
 * @return an LLVM compatible type or null if none is found
 */
mlir::Type findLLVMCompatibleType(mlir::Value val);

//////////////////////////////////////
// Different constants and formatting methods used in different places
//////////////////////////////////////
constexpr const char* kPrefixSectionName = ".text.stencil";
constexpr const char* kPrefixSplitSectionName = ".text.hot.stencil";
std::pair<int, int> parseSectionName(llvm::StringRef str);
bool isTemplateSection(llvm::StringRef str);
inline bool isFractionTemplateSection(llvm::StringRef str) { return str.contains("__part"); }
inline bool isAnyPlaceholder(llvm::StringRef str) { return str.starts_with("__placeholder"); }
inline bool isIndexPlaceholder(llvm::StringRef str) { return str.starts_with("__placeholder__idx"); }
inline bool isRegionPlaceholder(llvm::StringRef str) { return str.starts_with("__placeholder__region"); }
inline bool isNextPlaceholder(llvm::StringRef str) { return str.starts_with("__placeholder__next"); }
inline bool isTemplateFunction(llvm::StringRef str) { return str.starts_with("stencil"); }
inline int getPlaceholderNr(llvm::StringRef str) { return std::stoi(str.substr(str.find_last_of("_") + 1).data()); }
inline int getTemplateFunctionNr(llvm::StringRef str) { return std::stoi(str.slice(/*stencil*/ 7, str.find_first_of("_")).data()); }

constexpr const char* kInternalTemplateFunction = "__template__";
// integer attribute to give each unrealized-conversion-cast (input/output) a unique number
constexpr const char* kInputNumAttr = "inputNum";
// integer attribute to give the number of operands of the abstracted operation
constexpr const char* kOperandCountAttr = "args";
// array attribute containing the stackmap for this region call
constexpr const char* kStackmapAttr = "stackmap";
// indicate unrealized-conversion-casts, which are supposed to load a value, but the value is coming from a region
constexpr const char* kFromRegionIndicator = "fromRegion";
// indicate unrealized-conversion-casts, which are supposed to store the values. Always has ONE !llvm.ptr result.
constexpr const char* kIsStoreIndicator = "isStore";
// in order to preserve the "result-store" unrealized-conversion-cast in cases where the return type is "actually"
// an llvm.ptr, we mark the cast as dummy and instead pretend to return two pointers (otherwise the cast could be optimized out)
constexpr const char* kDummyIndicator = "dummy";
// indicate a function to require a stack (function patterns)
constexpr const char* kNeedsStackIndicator = "needsStack";
// indicate a function to require a continuation tail call instead of returning
constexpr const char* kTailCallIndicator = "tailCall";
// used as callee/sym name if we generate less templates in func and call patterns
constexpr const char* kDummyFunctionName = "__dummy_does_not_exist__";

#define SELECT_VA(_1, _2, _3, NAME, ...) NAME
#define FORMATTER3(NAME, FMT, T1, T2, T3) \
    inline std::string NAME(T1 _1, T2 _2, T3 _3) { return llvm::formatv(FMT, _1, _2, _3); }
#define FORMATTER2(NAME, FMT, T1, T2) \
    inline std::string NAME(T1 _1, T2 _2) { return llvm::formatv(FMT, _1, _2); }
#define FORMATTER1(NAME, FMT, T1) \
    inline std::string NAME(T1 _1) { return llvm::formatv(FMT, _1); }
#define FORMATTER(NAME, FMT, ...) SELECT_VA(__VA_ARGS__, FORMATTER3, FORMATTER2, FORMATTER1)(NAME, FMT, __VA_ARGS__)

FORMATTER(formatDataSectionRef, "{0}_{1}", llvm::StringRef, int);
FORMATTER(formatFunction, "stencil{0}", size_t);
FORMATTER(formatFunction, "stencil{0}_{1}", size_t, uint8_t);
FORMATTER(formatIndexPlaceholder, "__placeholder__idx{0}_{1}_{2}", size_t, uint8_t, size_t);
FORMATTER(formatNextCall, "__placeholder__next{0}_{1}", size_t, uint8_t);
FORMATTER(formatRegionCall, "__placeholder__region{0}_{1}_{2}", size_t, uint8_t, size_t);
FORMATTER(formatRegionCall, "__placeholder__region{0}", size_t);

#undef FORMATTER

// max size of elements passed in registers
constexpr int MAX_SIZE_REGISTER = 2;
// used calling convention for the abstracted functions
#define CALLING_CONVENTION C

#if defined(__x86_64__)
#include "Preparation/X86.h"
#elif defined(__aarch64__)
#include "Preparation/AArch64.h"
#else
#error "unsupported architecture"
#endif
