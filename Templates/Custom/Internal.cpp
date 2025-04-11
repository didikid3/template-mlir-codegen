#include "Execution/JIT.hpp"
#include "Templates/Custom/Utils.hpp"
#include "Templates/PredefinedPatterns.hpp"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "llvm/Support/FormatVariadic.h"

uint32_t MoveStackToRegister::id(uint32_t target)
{
    return target - 1;
}
void CustomPattern<MoveStackToRegister>::GenerateTemplate(TemplateBuilder& builder, [[maybe_unused]] uint32_t expectedId, uint32_t nr)
{
    mlir::OpBuilder opBuilder(&builder.getContext());
    const auto ptr = mlir::LLVM::LLVMPointerType::get(opBuilder.getI8Type());
    std::vector<mlir::Type> argTypes{ptr};
    for (int i = 0; i < OPERAND_REGISTERS; i++) {
        argTypes.push_back(opBuilder.getI64Type());
    }

    const auto target = nr + 1;
    const auto id = builder.reserveInternal();
    ASSURE(id, ==, expectedId);

    mlir::OwningOpRef<mlir::ModuleOp> mod = opBuilder.create<mlir::ModuleOp>(opBuilder.getUnknownLoc());
    opBuilder.setInsertionPointToStart(mod->getBody());
    auto indx = opBuilder.create<mlir::LLVM::GlobalOp>(opBuilder.getUnknownLoc(),
        opBuilder.getI8Type(),
        true,
        mlir::LLVM::Linkage::External,
        formatIndexPlaceholder(id, kind, 0),
        opBuilder.getZeroAttr(opBuilder.getI8Type()),
        1);
    auto continuation = opBuilder.create<mlir::LLVM::LLVMFuncOp>(opBuilder.getUnknownLoc(),
        formatNextCall(id, static_cast<uint8_t>(kind)),
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(opBuilder.getContext()), argTypes),
        mlir::LLVM::Linkage::External,
        true,
        mlir::LLVM::CConv::CALLING_CONVENTION);
    auto func = opBuilder.create<mlir::LLVM::LLVMFuncOp>(opBuilder.getUnknownLoc(),
        formatFunction(id, static_cast<uint8_t>(kind)),
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(opBuilder.getContext()), argTypes),
        mlir::LLVM::Linkage::External,
        true,
        mlir::LLVM::CConv::CALLING_CONVENTION);
    auto* bodyBlock = func.addEntryBlock();
    opBuilder.setInsertionPointToStart(bodyBlock);
    auto addr = opBuilder.create<mlir::LLVM::AddressOfOp>(opBuilder.getUnknownLoc(), indx);
    auto integer = opBuilder.create<mlir::LLVM::PtrToIntOp>(opBuilder.getUnknownLoc(), opBuilder.getI64Type(), addr);
    auto gepOp = opBuilder.create<mlir::LLVM::GEPOp>(opBuilder.getUnknownLoc(), ptr, *bodyBlock->getArguments().begin(), integer.getResult());
    auto val = opBuilder.create<mlir::LLVM::LoadOp>(opBuilder.getUnknownLoc(), opBuilder.getI64Type(), gepOp);
    std::vector<mlir::Value> forwardedVals(bodyBlock->getArguments().begin(), bodyBlock->getArguments().end());
    forwardedVals[target] = val;
    opBuilder.create<mlir::LLVM::CallOp>(opBuilder.getUnknownLoc(), continuation, forwardedVals);
    opBuilder.create<mlir::LLVM::ReturnOp>(opBuilder.getUnknownLoc(), std::vector<mlir::Value>{});

    builder.compileInternal(id, *mod.operator->());
}

uint32_t MoveBetweenRegisters::id(uint32_t source, uint32_t target)
{
    return (target - 1) * CACHE_REGISTERS + (source - 1);
}
void CustomPattern<MoveBetweenRegisters>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t nr)
{
    static constexpr const char* str = R"""(
        llvm.func @__placeholder__next[NR]_[KIND]()
        llvm.func @stencil[NR]_[KIND]() attributes {passthrough = ["naked", "NO_ADJUST"]} {
            llvm.inline_asm [ASM] : () -> ()
            llvm.call @__placeholder__next[NR]_[KIND]() : () -> ()
            llvm.return
        }
    )""";
    // ASM corresponds to e.g.: "%rax, %rbx" or "x10, x12"
    const auto target = (nr / CACHE_REGISTERS) + 1;
    const auto source = (nr % CACHE_REGISTERS) + 1;
    ASSURE(MoveBetweenRegisters::id(source, target), ==, nr);
#if defined(__x86_64__)
    constexpr const char* Fmt = "\"mov %{0}, %{1}\", \"~{{{2}}\"";
#elif defined(__aarch64__)
    constexpr const char* Fmt = "\"mov {1}, {0}\", \"~{{{2}}\"";
#endif
    std::string lowercaseClobberRegister(REGISTERS[target - 1]);
    std::transform(lowercaseClobberRegister.begin(), lowercaseClobberRegister.end(), lowercaseClobberRegister.begin(),
        [](unsigned char c) { return std::tolower(c); });
    std::string _asm = llvm::formatv(Fmt, REGISTERS[source - 1], REGISTERS[target - 1], lowercaseClobberRegister);
    parseAndRegisterTemplate(ReplaceAll(str, "[ASM]", _asm), builder, expectedId);
}

uint32_t MoveConstantToStack::id()
{
    return 0;
}
void CustomPattern<MoveConstantToStack>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
    static constexpr const char* mlir = R"""(
        llvm.mlir.global external @__placeholder__idx[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.mlir.global external @__placeholder[NR]__1() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.func @__placeholder__next[NR]_[KIND](!llvm.ptr)
        llvm.func @stencil[NR]_[KIND](%arg0: !llvm.ptr) {
            %addr1 = llvm.mlir.addressof @__placeholder__idx[NR]__0 : !llvm.ptr
            %addr2 = llvm.mlir.addressof @__placeholder[NR]__1 : !llvm.ptr
            %c1 = llvm.ptrtoint %addr1 : !llvm.ptr to i64
            %c2 = llvm.ptrtoint %addr2 : !llvm.ptr to i64
            %loc = llvm.getelementptr %arg0[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            llvm.store %c2, %loc : i64, !llvm.ptr
            llvm.call @__placeholder__next[NR]_[KIND](%arg0) : (!llvm.ptr) -> ()
            llvm.return
        }
    )""";
    parseAndRegisterTemplate(mlir, builder, expectedId);
}

uint32_t MoveConstantToRegister::id(uint32_t target)
{
    return target - 1;
}
void CustomPattern<MoveConstantToRegister>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t nr)
{
#if defined(__x86_64__)
    static constexpr const char* mlir = R"""(
        llvm.func @__placeholder__next[NR]_[KIND](!llvm.ptr, i64, i64, i64, i64, i64)
        llvm.mlir.global external @__placeholder[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.func @stencil[NR]_[KIND](
            %arg0: !llvm.ptr,
            %arg1: i64,
            %arg2: i64,
            %arg3: i64,
            %arg4: i64,
            %arg5: i64
        ) {
            %addr = llvm.mlir.addressof @__placeholder[NR]__0 : !llvm.ptr
            %c = llvm.ptrtoint %addr : !llvm.ptr to i64
            llvm.call @__placeholder__next[NR]_[KIND](
                [TARGET_CONFIG]
            ) : (!llvm.ptr, i64, i64, i64, i64, i64) -> ()
            llvm.return
        }
    )""";
    const auto target = nr + 1;
    std::string arguments = "%arg0,%arg1,%arg2,%arg3,%arg4,%arg5";
#else
    static constexpr const char* mlir = R"""(
        llvm.func @__placeholder__next[NR]_[KIND](!llvm.ptr, i64, i64, i64, i64, i64, i64, i64)
        llvm.mlir.global external @__placeholder[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.func @stencil[NR]_[KIND](
            %arg0: !llvm.ptr,
            %arg1: i64,
            %arg2: i64,
            %arg3: i64,
            %arg4: i64,
            %arg5: i64,
            %arg6: i64,
            %arg7: i64
        ) {
            %addr = llvm.mlir.addressof @__placeholder[NR]__0 : !llvm.ptr
            %c = llvm.ptrtoint %addr : !llvm.ptr to i64
            llvm.call @__placeholder__next[NR]_[KIND](
                [TARGET_CONFIG]
            ) : (!llvm.ptr, i64, i64, i64, i64, i64, i64, i64) -> ()
            llvm.return
        }
    )""";
    const auto target = nr + 1;
    std::string arguments = "%arg0,%arg1,%arg2,%arg3,%arg4,%arg5,%arg6,%arg7";
#endif
    arguments.replace((target * 6), 5ul, "%c");

    parseAndRegisterTemplate(ReplaceAll(mlir, "[TARGET_CONFIG]", arguments), builder, expectedId);
}

uint32_t CopyOnStack::id()
{
    return 0;
}
void CustomPattern<CopyOnStack>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
    static constexpr const char* mlir = R"""(
        llvm.mlir.global external @__placeholder__idx[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.mlir.global external @__placeholder__idx[NR]__1() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.func @__placeholder__next[NR]_[KIND](!llvm.ptr)
        llvm.func @stencil[NR]_[KIND](%arg0: !llvm.ptr) {
            %addr1 = llvm.mlir.addressof @__placeholder__idx[NR]__0 : !llvm.ptr
            %addr2 = llvm.mlir.addressof @__placeholder__idx[NR]__1 : !llvm.ptr
            %c1 = llvm.ptrtoint %addr1 : !llvm.ptr to i64
            %c2 = llvm.ptrtoint %addr2 : !llvm.ptr to i64
            %loc1 = llvm.getelementptr %arg0[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %loc2 = llvm.getelementptr %arg0[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %val = llvm.load %loc1 : !llvm.ptr -> i64
            llvm.store %val, %loc2 : i64, !llvm.ptr
            llvm.call @__placeholder__next[NR]_[KIND](%arg0) : (!llvm.ptr) -> ()
            llvm.return
        }
    )""";
    parseAndRegisterTemplate(mlir, builder, expectedId);
}

uint32_t MoveRegisterToStack::id(uint32_t source)
{
    return source - 1;
}
void CustomPattern<MoveRegisterToStack>::GenerateTemplate(TemplateBuilder& builder, [[maybe_unused]] uint32_t expectedId, uint32_t nr)
{
    mlir::OpBuilder opBuilder(&builder.getContext());
    const auto ptr = mlir::LLVM::LLVMPointerType::get(opBuilder.getI8Type());
    std::vector<mlir::Type> argTypes{ptr};
    for (int i = 0; i < OPERAND_REGISTERS; i++) {
        argTypes.push_back(opBuilder.getI64Type());
    }
    const auto source = nr + 1;
    const auto id = builder.reserveInternal();
    ASSURE(id, ==, expectedId);

    mlir::OwningOpRef<mlir::ModuleOp> mod = opBuilder.create<mlir::ModuleOp>(opBuilder.getUnknownLoc());
    opBuilder.setInsertionPointToStart(mod->getBody());
    auto indx = opBuilder.create<mlir::LLVM::GlobalOp>(opBuilder.getUnknownLoc(),
        opBuilder.getI8Type(),
        true,
        mlir::LLVM::Linkage::External,
        formatIndexPlaceholder(id, kind, 0),
        opBuilder.getZeroAttr(opBuilder.getI8Type()),
        1);
    auto tailCall = opBuilder.create<mlir::LLVM::LLVMFuncOp>(opBuilder.getUnknownLoc(),
        formatNextCall(id, static_cast<uint8_t>(kind)),
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(opBuilder.getContext()), argTypes),
        mlir::LLVM::Linkage::External,
        true,
        mlir::LLVM::CConv::CALLING_CONVENTION);
    auto func = opBuilder.create<mlir::LLVM::LLVMFuncOp>(opBuilder.getUnknownLoc(),
        formatFunction(id, static_cast<uint8_t>(kind)),
        mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMVoidType::get(opBuilder.getContext()), argTypes),
        mlir::LLVM::Linkage::External,
        true,
        mlir::LLVM::CConv::CALLING_CONVENTION);
    auto* bodyBlock = func.addEntryBlock();
    opBuilder.setInsertionPointToStart(bodyBlock);
    auto addr = opBuilder.create<mlir::LLVM::AddressOfOp>(opBuilder.getUnknownLoc(), indx);
    auto integer = opBuilder.create<mlir::LLVM::PtrToIntOp>(opBuilder.getUnknownLoc(), opBuilder.getI64Type(), addr);
    auto gepOp = opBuilder.create<mlir::LLVM::GEPOp>(opBuilder.getUnknownLoc(), ptr, *bodyBlock->getArguments().begin(), integer.getResult());
    opBuilder.create<mlir::LLVM::StoreOp>(opBuilder.getUnknownLoc(), *(bodyBlock->getArguments().begin() + source), gepOp);
    opBuilder.create<mlir::LLVM::CallOp>(opBuilder.getUnknownLoc(), tailCall, bodyBlock->getArguments());
    opBuilder.create<mlir::LLVM::ReturnOp>(opBuilder.getUnknownLoc(), std::vector<mlir::Value>{});

    builder.compileInternal(id, *mod.operator->());
}

uint32_t RegionTerminator::id()
{
    return 0;
}

void CustomPattern<RegionTerminator>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
    static constexpr const char* mlir = R"""(
        module {
            llvm.func @stencil[NR]_[KIND]() {
                llvm.return
            }
        }
        )""";
    parseAndRegisterTemplate(mlir, builder, expectedId);
}

uint32_t ExternalThunk::id()
{
    return 0;
}
void CustomPattern<ExternalThunk>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
#if defined(__x86_64__)
    static constexpr const char* mlir = R"""(
        module attributes {"custom.cmodel-large"} {
            llvm.mlir.global external @__placeholder[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
            llvm.func @stencil[NR]_[KIND]() {
                %addr = llvm.mlir.addressof @__placeholder[NR]__0 : !llvm.ptr
                llvm.call %addr() : !llvm.ptr, () -> ()
                llvm.return
            }
        }
        )""";
#elif defined(__aarch64__)
    // even with large code model, arm does not emit code for a "far" jump but allows the linker to add a veneer.
    // Therefore, we manually handcraft the veneer in inline-asm here filling the address into x17 (one of the scratch registers)
    static constexpr const char* mlir = R"""(
        module attributes {"custom.cmodel-large"} {
            llvm.func @stencil[NR]_[KIND]() attributes {passthrough = ["naked", ["alignstack", "16"]]} {
                llvm.inline_asm has_side_effects "movz x17, #:abs_g0_nc:__placeholder[NR]__0\0A\09movk x17, #:abs_g1_nc:__placeholder[NR]__0\0A\09movk x17, #:abs_g2_nc:__placeholder[NR]__0\0A\09movk x17, #:abs_g3:__placeholder[NR]__0\0A\09br x17", "" : () -> ()
                llvm.return
            }
        }
    )""";
#endif
    parseAndRegisterTemplate(mlir, builder, expectedId);
}

uint32_t Br::id()
{
    return 0;
}

void CustomPattern<Br>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
    static constexpr const char* mlir = R"""(
        llvm.mlir.global external @__placeholder__idx[NR]__0() {addr_space = 0 : i32, alignment = 1 : i64, dso_local} : i8
        llvm.func @stencil[NR]_[KIND]() {
            %addr = llvm.mlir.addressof @__placeholder__idx[NR]__0 : !llvm.ptr
            llvm.call %addr() : !llvm.ptr, () -> ()
            llvm.return
        }
        )""";
    parseAndRegisterTemplate(mlir, builder, expectedId);
}

uint32_t Malloc::id()
{
    return 0;
}

void CustomPattern<Malloc>::GenerateTemplate(TemplateBuilder& builder, uint32_t expectedId, uint32_t)
{
    static constexpr const char* mlir = R"""(
        llvm.func @__placeholder__next[NR]_[KIND](!llvm.ptr, i64)
        llvm.func @malloc(i64) -> !llvm.ptr
        llvm.func @stencil[NR]_[KIND](%arg0: !llvm.ptr, %arg1: i64) {
            %0 = llvm.call @malloc(%arg1) : (i64) -> !llvm.ptr
            %1 = llvm.ptrtoint %0 : !llvm.ptr to i64
            llvm.call @__placeholder__next[NR]_[KIND](%arg0, %1) : (!llvm.ptr, i64) -> ()
            llvm.return
        }
        )""";
    parseAndRegisterTemplate(mlir, builder, expectedId);
}
