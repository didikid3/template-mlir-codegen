#pragma once

#include <cstdlib>
#include <memory>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

#include "Preparation/Abstraction.hpp"

class JIT;
namespace mlir {
class Operation;
class Attribute;
class Type;
}

void returnFromExecution(mlir::Operation* op, JIT& jit, size_t depth);
std::pair<std::unique_ptr<uint64_t[]>, TypeSize> fillConstantFromAttr(JIT& jit, mlir::Attribute attr, mlir::Type targetType);

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to);

template <typename CallOp>
void genericCallOp(CallOp callOp, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth);
