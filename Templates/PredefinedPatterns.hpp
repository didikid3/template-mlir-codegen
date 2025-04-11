#pragma once

#include "Templates/Custom/Utils.hpp"
#include "common.hpp"

#include "TemplateBuilder.hpp"

#include "llvm/ADT/TypeSwitch.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

template <typename T>
struct CustomPattern
{
};

#ifdef FORCE_STACK_TEMPLATE
constexpr uint8_t kind = 1;
#else
constexpr uint8_t kind = 2;
#endif

inline void parseAndRegisterTemplate(std::string str, TemplateBuilder& builder, [[maybe_unused]] uint32_t expectedId)
{
    const auto id = builder.reserveInternal();
    ASSURE(id, ==, expectedId);
    auto& context = builder.getContext();
    auto mod = mlir::parseSourceString<mlir::ModuleOp>(ReplaceAll(ReplaceAll(str, "[NR]", std::to_string(id)), "[KIND]", std::to_string(kind)), mlir::ParserConfig{&context});
    ASSURE_TRUE(mod, << str);
    auto moduleOp = *mod;
    builder.compileInternal(id, moduleOp);
}

//////////////////
#include "Templates/Custom/Builtins.hpp"
#include "Templates/Custom/Internal.hpp"
#include "Templates/Custom/LLVM.hpp"
//////////////////

template <typename T, typename... TT>
struct CustomPatterns
{
    template <typename SwitchT>
    static SwitchT Skip(SwitchT in)
    {
        auto skipThis = std::move(in.template Case<T>([](T&) {
            // don't do anything - just skip
            return mlir::WalkResult::skip();
        }));
        if constexpr (sizeof...(TT) > 0) {
            return CustomPatterns<TT...>::Skip(std::move(skipThis));
        } else {
            return skipThis;
        }
    }

    static constexpr uint32_t templateCount()
    {
        using LastT = typename decltype((std::type_identity<TT>{}, ...))::type;
        return CustomPattern<LastT>::TEMPLATE_COUNT + baseId<LastT>();
    }

    template <typename LookupT>
    static constexpr uint32_t baseId()
    {
        if constexpr (std::is_same_v<LookupT, T>) {
            return 0;
        } else {
            static_assert(sizeof...(TT) > 0);
            return CustomPattern<T>::TEMPLATE_COUNT + CustomPatterns<TT...>::template baseId<LookupT>();
        }
    }

    static void Generate(TemplateBuilder& builder, uint32_t baseId = 0)
    {
        for (uint32_t i = 0; i < CustomPattern<T>::TEMPLATE_COUNT; i++) {
            CustomPattern<T>::GenerateTemplate(builder, baseId + i, i);
        }
        baseId += CustomPattern<T>::TEMPLATE_COUNT;
        if constexpr (sizeof...(TT) > 0) {
            return CustomPatterns<TT...>::Generate(builder, baseId);
        }
    }

    template <typename SwitchT>
    static SwitchT Codegen(SwitchT in, JIT& jit, uint16_t sourceDepth, uint16_t cgDepth)
    {
        auto thisCase = std::move(in.template Case<T>([&](T& op) {
            CustomPattern<T>::Codegen(op, jit, sourceDepth, cgDepth);
        }));
        if constexpr (sizeof...(TT) > 0) {
            return CustomPatterns<TT...>::Codegen(
                std::move(thisCase), jit, sourceDepth, cgDepth);
        } else {
            return thisCase;
        }
    }
};

using NoPatternGeneration = CustomPatterns<
    mlir::func::ReturnOp,
    mlir::LLVM::ReturnOp,
    mlir::linalg::YieldOp,
    mlir::scf::YieldOp,
    mlir::scf::ConditionOp,
#ifndef LINGODB_LLVM
    mlir::LLVM::NoAliasScopeDeclOp,
#endif
    mlir::LLVM::InsertValueOp,
    mlir::LLVM::ComdatOp,
    mlir::LLVM::LifetimeStartOp,
    mlir::LLVM::LifetimeEndOp,
    mlir::LLVM::VaStartOp,
    mlir::LLVM::VaEndOp,
    mlir::LLVM::BrOp,
    mlir::LLVM::CondBrOp,
    mlir::LLVM::AllocaOp,
    mlir::LLVM::GlobalOp,
    mlir::LLVM::UndefOp,
    mlir::LLVM::ConstantOp,
    mlir::LLVM::AddressOfOp>;
using CustomPatternGeneration = CustomPatterns<
    RegionTerminator,
    MoveConstantToStack,
    MoveConstantToRegister,
    MoveBetweenRegisters,
    MoveStackToRegister,
    CopyOnStack,
    MoveRegisterToStack,
    ExternalThunk,
    Br,
    Malloc>;

using CustomCodegen = CustomPatterns<
    mlir::arith::ConstantOp,
    mlir::LLVM::InsertValueOp,
    mlir::LLVM::BrOp,
    mlir::LLVM::CondBrOp,
    mlir::LLVM::SwitchOp,
    mlir::LLVM::ConstantOp,
    mlir::LLVM::UndefOp,
    mlir::LLVM::AllocaOp,
    mlir::LLVM::GlobalOp,
#ifdef LESS_TEMPLATES
    mlir::LLVM::CallOp,
    mlir::func::CallOp,
#endif
    mlir::LLVM::AddressOfOp>;
using SkipCodegen = CustomPatterns<
    mlir::ModuleOp,
    mlir::linalg::YieldOp,
    mlir::scf::YieldOp,
#ifndef LINGODB_LLVM
    mlir::LLVM::NoAliasScopeDeclOp,
#endif
    mlir::LLVM::LifetimeStartOp,
    mlir::LLVM::LifetimeEndOp,
    mlir::scf::ConditionOp>;
