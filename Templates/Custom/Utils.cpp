#include "Templates/Custom/Utils.hpp"
#include "Execution/JIT.hpp"
#include "common.hpp"

#include <dlfcn.h>
#include <limits>

void returnFromExecution(mlir::Operation* op, JIT& jit, size_t depth)
{
    if (op->getNumOperands() > 0) {
        ASSURE(op->getNumOperands(), ==, 1);
        std::vector<uint64_t> tmp{};
        jit.forceOnStack(depth, 0, op->getOperands(), tmp, tmp.cend());
        ASSURE(tmp.front() % 8, ==, 0);
        jit.m_returnOffset = tmp.front() / 8;
    }
#ifdef __aarch64__
    // load the return address from the stack
    // ldp x30, x20, [sp, #160]
    const uint32_t instruction = 0xf94053fe;
    LOG_JIT(depth, 4, "restoreReturn");
    jit.insert(0, 4, &instruction);
#endif
    JIT::Materializer<RegionTerminator>()(jit, depth);
}

void setInteger(void* location, const llvm::APInt& integer, uint8_t width)
{
    auto setChecked = [&](auto t) {
        using T = decltype(t);
        const uint64_t val = integer.getZExtValue();
        REL_ASSURE(val, >=, std::numeric_limits<T>::min());
        REL_ASSURE(val, <=, std::numeric_limits<T>::max());
        *reinterpret_cast<T*>(location) = static_cast<T>(val);
    };

    switch (width) {
        case 1:
        case 4:
        case 8:
            setChecked(static_cast<uint8_t>(0));
            break;
        case 16:
            setChecked(static_cast<uint16_t>(0));
            break;
        case 32:
            setChecked(static_cast<uint32_t>(0));
            break;
        case 64:
            setChecked(static_cast<uint64_t>(0));
            break;
        case 128: {
            auto* ptr = reinterpret_cast<uint64_t*>(location);
            ptr[0] = integer.extractBits(64, 0).getZExtValue();
            ptr[1] = integer.extractBits(64, 64).getZExtValue();
            break;
        }
        default:
            UNREACHABLE("width: " << static_cast<uint32_t>(width));
    }
}

void setIntOrFloatArray(mlir::DenseIntOrFPElementsAttr arrayAttr, mlir::Type elementT, char* insertPtr)
{
    const auto elementSize = (elementT.getIntOrFloatBitWidth() + 7) / 8;
    const bool isInteger = elementT.isa<mlir::IntegerType>();
    const auto elems = arrayAttr.getNumElements();
    auto valueAt = [&](unsigned i) {
        if (arrayAttr.isSplat() && isInteger) {
            return arrayAttr.getSplatValue<mlir::APInt>().getZExtValue();
        } else if (arrayAttr.isSplat() && !isInteger) {
            return arrayAttr.getSplatValue<mlir::APFloat>().bitcastToAPInt().getZExtValue();
        } else if (isInteger) {
            return (*(arrayAttr.value_begin<mlir::APInt>() + i)).getZExtValue();
        } else {
            return (*(arrayAttr.value_begin<mlir::APFloat>() + i)).bitcastToAPInt().getZExtValue();
        }
    };
    for (unsigned i = 0; i < elems; i++) {
        uint64_t tmp = valueAt(i);
        memcpy(insertPtr, &tmp, elementSize);
        insertPtr += elementSize;
    }
}

std::pair<std::unique_ptr<uint64_t[]>, TypeSize> fillConstantFromAttr(JIT& jit, mlir::Attribute attr, mlir::Type targetType)
{
    if (targetType.isa<mlir::IntegerType>() || targetType.isa<mlir::IndexType>()) {
        ASSURE_TRUE(llvm::isa<mlir::IntegerAttr>(attr));
        const auto iAttr = llvm::cast<mlir::IntegerAttr>(attr);
        const auto size = TypeSize(targetType.isa<mlir::IndexType>() ? 64 : targetType.getIntOrFloatBitWidth());
        auto ptr = std::make_unique<uint64_t[]>(size.slots());
        setInteger(
            ptr.get(),
            iAttr.getValue(),
            size.bits());
        return {std::move(ptr), size};
    } else if (targetType.isa<mlir::FloatType>()) {
        ASSURE_TRUE(llvm::isa<mlir::FloatAttr>(attr));
        const auto fAttr = llvm::cast<mlir::FloatAttr>(attr);
        const auto iValue = fAttr.getValue().bitcastToAPInt();
        const auto size = TypeSize(iValue.getBitWidth());
        ASSURE(size.bytes(), <=, 8);
        auto ptr = std::make_unique<uint64_t[]>(size.slots());
        uint64_t tmp = iValue.getZExtValue();
        memcpy(ptr.get(), &tmp, size.bytes());
        return {std::move(ptr), size};
    } else if (targetType.isa<mlir::VectorType>()) {
        auto vecT = llvm::cast<mlir::VectorType>(targetType);
        ASSURE_TRUE((vecT.getElementType().isa<mlir::IntegerType, mlir::FloatType>()));
        const auto bitWidth = vecT.getElementTypeBitWidth();
        ASSURE(bitWidth % 8, ==, 0);
        const auto size = TypeSize(bitWidth * vecT.getNumElements(), MAX_ALIGN);
        auto ptr = std::make_unique<uint64_t[]>(size.slots());

        if (attr.isa<mlir::DenseIntOrFPElementsAttr>()) {
            const auto denseArr = llvm::cast<mlir::DenseIntOrFPElementsAttr>(attr);
            setIntOrFloatArray(denseArr, vecT.getElementType(), reinterpret_cast<char*>(ptr.get()));
            return {std::move(ptr), size};
        }
    } else if (targetType.isa<mlir::LLVM::LLVMArrayType>()) {
        auto arrayT = llvm::cast<mlir::LLVM::LLVMArrayType>(targetType);
        const auto size = TypeSize::FromType(arrayT);
        auto ptr = std::make_unique<uint64_t[]>(size.slots());
        if (attr.isa<mlir::StringAttr>()) {
            auto stringT = llvm::cast<mlir::StringAttr>(attr);
            auto stringValue = stringT.getValue();
            memcpy(ptr.get(), stringValue.data(), stringValue.size());
            return {std::move(ptr), size};
        } else if (attr.isa<mlir::DenseIntOrFPElementsAttr>()) {
            auto arrayAttr = llvm::cast<mlir::DenseIntOrFPElementsAttr>(attr);
            if (arrayT.getElementType().isa<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMArrayType>()) {
                std::function<mlir::Type(mlir::LLVM::LLVMArrayType)> innerType;
                innerType = [&innerType](mlir::LLVM::LLVMArrayType t) {
                    ASSURE_TRUE((t.getElementType().isa<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMArrayType>()));
                    if (t.getElementType().isa<mlir::LLVM::LLVMArrayType>()) {
                        return innerType(llvm::cast<mlir::LLVM::LLVMArrayType>(t.getElementType()));
                    } else {
                        return t.getElementType();
                    }
                };
                auto subElementT = innerType(arrayT);
                setIntOrFloatArray(arrayAttr, subElementT, reinterpret_cast<char*>(ptr.get()));
                return {std::move(ptr), size};
            } else {
                // in some cases also structs are (zero) initialized by an array
                ASSURE_TRUE(arrayAttr.isSplat(), << " " << arrayAttr);
                ASSURE_TRUE(arrayAttr.getElementType().isa<mlir::IntegerType>());
                ASSURE(arrayAttr.getSplatValue<mlir::APInt>().getZExtValue(), ==, 0);
                memset(ptr.get(), 0, size.bytes());
                return {std::move(ptr), size};
            }
        }
    } else if (attr.isa<mlir::SymbolRefAttr>()) {
        ASSURE_TRUE(targetType.isa<mlir::LLVM::LLVMPointerType>());
        auto ptr = std::make_unique<uint64_t[]>(1);
        ptr[0] = 0;
        jit.m_symbols.insertFixupLocation(
            llvm::cast<mlir::SymbolRefAttr>(attr).getLeafReference().str(), {reinterpret_cast<char*>(ptr.get()), true});
        return {std::move(ptr), TypeSize(64)};
    }

    UNREACHABLE("unsupported llvm const op for attribute: " << attr << " to type: " << targetType);
}

std::string ReplaceAll(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

template <typename CallOp>
void genericCallOp(CallOp op, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth)
{
    if constexpr (std::is_same_v<CallOp, mlir::LLVM::CallOp>) {
        if (!op.getCallee().has_value()) {
            return jit.prepareOpDefault(op, cgDepth, sourceDepth);
        }
    }
    llvm::StringRef callee = [&] {
        if constexpr (std::is_same_v<CallOp, mlir::LLVM::CallOp>)
            return *op.getCallee();
        else
            return op.getCallee();
    }();
    const auto libraryIt = jit.m_library.find(*op);
    ASSURE_FALSE(libraryIt == jit.m_library.end());
    const auto templateId = libraryIt->second;
    const auto [returnSizes, stackmaps, binaryTemplate] = jit.m_library.getTemplate(templateId);
    std::vector<uint64_t> args;
    args.reserve(/* operands */ op->getNumOperands() + /* return value */ 1);
    std::vector<char*> regionLocs;
    regionLocs.reserve(2);
    const auto kind = jit.prepareOpCall(op, cgDepth, args);

    regionLocs.push_back(jit.memory(cgDepth));
    const auto templateSize = binaryTemplate.size(0);
    LOG_JIT(cgDepth, templateSize, op->getName().getStringRef().str(),
        templateId,
        llvm::isa<mlir::FileLineColLoc>(op->getLoc())
            ? llvm::cast<mlir::FileLineColLoc>(op->getLoc()).getLine()
            : 0);
    binaryTemplate.instantiate(jit.memory(cgDepth, templateSize), 0);
    jit.applyClobberMask(binaryTemplate.clobberMask);
    regionLocs.push_back(jit.memory(cgDepth));

    jit.prepareReturnValues(op, cgDepth, kind, args, returnSizes);

    binaryTemplate.patch(
        regionLocs,
        args,
        [&](void* originalAddr) {
            const auto addr = jit.memory(cgDepth + 1);
            JIT::Materializer<ExternalThunk>()(jit, cgDepth + 1, reinterpret_cast<uint64_t>(originalAddr));
            return addr;
        },
        [&](void* originalAddr) {
            jit.align(cgDepth + 1);
            const auto addr = jit.memory(cgDepth + 1);
            jit.insert(cgDepth + 1, sizeof(addr), &originalAddr);
            return addr;
        },
        [&]([[maybe_unused]] const std::string& name, char* location) {
            ASSURE(name, ==, kDummyFunctionName);
            void* ptr = dlsym(RTLD_DEFAULT, callee.data());
            if (ptr != nullptr) {
                return ptr;
            }
            void* addr = jit.m_symbols.getOrNull(callee.str());
            if (addr == nullptr) {
                DEBUG("didn't resolve " << name << " yet!");
                if (reinterpret_cast<uint8_t*>(location)[-2] == 0x8b
                    && reinterpret_cast<uint8_t*>(location)[-3] == 0x48) {
                    // in case of a move instruction, the relocation was a GOT entry, but as we now know, that it is
                    // a close-range symbol (generated by use), we cant turn the mov into an lea
                    reinterpret_cast<uint8_t*>(location)[-2] = 0x8d;
                } else {
                    // must be a direct call
                    ASSURE(static_cast<uint32_t>(reinterpret_cast<uint8_t*>(location)[-1]), ==, 0xe8, << name << " - " << templateId);
                }
                ASSURE(UnalignedRead<int32_t>(location), ==, -4);
                jit.m_symbols.insertFixupLocation(callee.str(), location);
            }
            return addr;
        });
}

template void genericCallOp<mlir::func::CallOp>(mlir::func::CallOp callOp, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth);
template void genericCallOp<mlir::LLVM::CallOp>(mlir::LLVM::CallOp callOp, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth);
