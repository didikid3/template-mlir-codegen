#pragma once

#include "Preparation/Abstraction.hpp"
#include "Execution/AddressPatchMap.hpp"
#include "Execution/MemInterface.hpp"
#include "Execution/RegisterExecutor.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include "llvm/ADT/StringMap.h"

#include <numeric>
#include <type_traits>
#include <variant>
#include <vector>

#ifdef LOGGING
#define BASE_CLASS WithLogging
#define LOG(...) logTemplate(__VA_ARGS__);
#define LOG_JIT(...) jit.logTemplate(__VA_ARGS__);
#else
#define BASE_CLASS ExecutableMemoryInterface
#define LOG(...)
#define LOG_JIT(...)
#endif

class TemplateLibrary;

class JIT
    : public RegisterExecutor<JIT>
    , public BASE_CLASS
{
public:
    JIT(TemplateLibrary& manager);

    void registerSymbol(llvm::StringRef name, void* function);

    void moveTo(uint32_t depth, const Storage& from, const Storage& to);
    void moveTo(uint32_t depth, const mlir::Value& val, const Storage& to);

    template <typename T>
    struct Materializer
    {
        const uint32_t templateId;

        template <typename... Args>
        Materializer(Args... signatureArgs)
            : templateId([&] {
                const auto id = T::id(std::forward<Args>(signatureArgs)...);
                ASSURE(id, <, T::TEMPLATE_COUNT, << " " << T::templateName());
                return CustomPatternGeneration::baseId<T>() + id;
            }())
        {
        }

        template <typename... Args>
        const Template& operator()(JIT& jit, size_t depth, Args... args)
        {
            const auto& templateInfo = jit.m_library.getTemplate(templateId);
            const auto& templateCode = std::get<2>(templateInfo);
            const auto size = templateCode.size();
            LOG_JIT(depth, size, T::templateName(), templateId);
            templateCode.instantiateAndPatch(jit.memory(depth, size), std::forward<Args>(args)...);
            jit.applyClobberMask(templateCode.clobberMask);
            return templateCode;
        }

        void instantiate(JIT& jit, size_t depth)
        {
            const auto& templateInfo = jit.m_library.getTemplate(templateId);
            const auto& templateCode = std::get<2>(templateInfo);
            const auto size = templateCode.size();
            LOG_JIT(depth, size, T::templateName(), templateId);
            templateCode.instantiate(jit.memory(depth, size), 0);
            jit.applyClobberMask(templateCode.clobberMask);
        }
    };

    void onBlockStart(uint8_t depth, mlir::Block* block);
    void insertBlockAddress(char* location, mlir::Block* block);

private:
    friend class Executor<JIT>;
    template <typename>
    friend struct CustomPattern;
    friend void returnFromExecution(mlir::Operation*, JIT&, size_t);
    friend std::pair<std::unique_ptr<uint64_t[]>, TypeSize> fillConstantFromAttr(JIT&, mlir::Attribute, mlir::Type);
    template <typename CallOp>
    friend void genericCallOp(CallOp, JIT&, uint16_t, uint16_t);

    void prepareImpl(mlir::ModuleOp* mod);
    uint64_t runImpl(mlir::ModuleOp*);

    void prepareOp(mlir::Operation* op, size_t codeDepth, size_t sourceNestingDepth);
    void prepareOpDefault(mlir::Operation* op, size_t codeDepth, size_t sourceNestingDepth);

    void forceOnStack(
        size_t depth,
        uint32_t stackOffset,
        const mlir::ValueRange& operands,
        std::vector<uint64_t>& args,
        std::vector<uint64_t>::const_iterator argsInsertPos,
        bool reuseStack = true);

    TEMPLATE_KIND prepareOpCall(
        mlir::Operation* op,
        size_t codeDepth,
        std::vector<uint64_t>& args,
        bool reuseStack = true);

    void prepareReturnValues(
        mlir::Operation* op,
        size_t codeDepth,
        TEMPLATE_KIND kind,
        std::vector<uint64_t>& args,
        const ResultSizes& returnSizes);

    Storage prepareRegion(
        size_t codeDepth,
        TEMPLATE_KIND kind,
        const StackMap& stackmap,
        const mlir::ValueRange& regionArgs);

    template <typename T>
    using SymbolMap = llvm::StringMap<T>;
    AddressPatchMap<SymbolMap, llvm::StringRef> m_symbols;
    template <typename T>
    using BlockMap = llvm::DenseMap<mlir::Block*, T>;
    AddressPatchMap<BlockMap, mlir::Block*> m_blocks;
    const TemplateLibrary& m_library;
    uint32_t m_returnOffset;
};
