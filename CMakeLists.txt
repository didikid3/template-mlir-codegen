cmake_minimum_required(VERSION 3.0.0)
project(mlir-codegen VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

set(EXECUTORS
    Execution/MemInterface.cpp
    Execution/LLI.cpp
    Execution/JIT.cpp
)
set(CUSTOM_TEMPLATES
    Templates/Custom/Utils.cpp
    Templates/Custom/LLVM.cpp
    Templates/Custom/Internal.cpp
    Templates/Custom/Builtins.cpp
)
set(PREPARATION
    Preparation/Abstraction.cpp
    Preparation/SectionProvider.cpp
    Preparation/BinaryParser.cpp
    Preparation/CustomPass.cpp
    Preparation/Utils.cpp
)
set(SRC
    ${EXECUTORS}
    ${CUSTOM_TEMPLATES}
    ${PREPARATION}
    TemplateCompiler.cpp
    Templates/Template.cpp
    Templates/Signature.cpp
    TemplateLibrary.cpp
    TemplateBuilder.cpp
    TypeSize.cpp
)

option(FOR_AARCH64 "aarch64" OFF)
option(FORCE_STACK "only use stack-only templates" OFF)
option(FIXED_STACKSIZE "use a fixed siize stack for each call frame" OFF)
option(VTUNE "build with VTune JIT profiling support (path to vtune)" OFF)
option(VTUNE_COMPILE "build with VTune JIT profiling support focused on compile" OFF)
option(OPTIMIZED_VARIABLE_PASSING "eliminate stores to stack that are loaded immediately after" ON)
option(FORTRAN "add fortran runtime libraries" OFF)
option(EVICTION_STRATEGY "store the value temporarily in a register in hope to using it again and avoid the stack load (eviction strategy 1: no-evict, 2: round-robin, OFF: no caching)" OFF)

if(FIXED_STACKSIZE)
    add_compile_definitions(FIXED_STACKSIZE=${FIXED_STACKSIZE})
endif()

if(EVICTION_STRATEGY)
    add_compile_definitions(EVICTION_STRATEGY=${EVICTION_STRATEGY})
endif()

if(FORCE_STACK)
    add_compile_definitions(FORCE_STACK_TEMPLATE)
    list(APPEND SRC Execution/CallingConventions/Stack.cpp)
else()
    list(APPEND SRC Execution/CallingConventions/Register.cpp)
endif()

if(APPLE)
    add_compile_definitions(NO_MAP_WX)
endif()

if(OPTIMIZED_VARIABLE_PASSING)
    add_compile_definitions(OPTIMIZED_VARIABLE_PASSING)
endif()

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
add_compile_definitions(CUSTOM_INSERT_VALUE)

if(FOR_AARCH64)
    set(TARGET_DEPS
        LLVMAArch64CodeGen
        LLVMAArch64Info
        LLVMAArch64Desc
        LLVMAArch64AsmParser
        LLVMAArch64Disassembler
    )
    set(TARGET_COMPONENT AARCH64)
    list(APPEND SRC Preparation/BinaryParserAArch64.cpp)
else()
    set(TARGET_DEPS
        LLVMX86CodeGen
        LLVMX86Info
        LLVMX86Desc
        LLVMX86AsmParser
        LLVMX86Disassembler
    )
    set(TARGET_COMPONENT X86)
    list(APPEND SRC Preparation/BinaryParserX86.cpp)
endif()

set(LLVM_LINK_COMPONENTS
    Support
    Core
    nativecodegen
    OrcJIT
    ${TARGET_COMPONENT}
)

find_library(_MLIRExecEngine MLIRExecutionEngine PATHS ${LLVM_BUILD_LIBRARY_DIR})

set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs}
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRLLVMToLLVMIRTranslation
    MLIRTargetLLVMIRExport
    MLIRSupport
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRSideEffectInterfaces
    MLIRTransforms
    MLIRLLVMCommonConversion

    ${TARGET_DEPS}

    LLVMExecutionEngine
    LLVMInterpreter
)

if(NOT _MLIRExecEngine)
    message("no execution engine library found")
else()
    add_compile_definitions(__USE_MLIRExecutionEngine__)
    list(APPEND SRC Execution/MLIRExecutionEngine.cpp)
    list(APPEND LIBS MLIRExecutionEngine)
endif()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})

if(VTUNE)
    include_directories(${VTUNE}/include)
    list(APPEND LIBS ${VTUNE}/lib64/libjitprofiling.a)
    list(APPEND LIBS ${VTUNE}/lib64/libittnotify.a)
    add_compile_definitions(VTUNE_SUPPORT)

    if(VTUNE_COMPILE)
        add_compile_definitions(VTUNE_COMPILE)
    endif()
endif()

add_library(mlir-codegen ${SRC})
target_link_libraries(mlir-codegen PRIVATE ${LIBS})

if(FORTRAN)
    message("link fortran runtime")
    target_compile_definitions(mlir-codegen PUBLIC "FORTRAN")

    # dump all fortran symbols into the binary
    if(APPLE)
        target_link_options(mlir-codegen PUBLIC "-Wl,-all_load,-lFortranRuntime,-lFortranDecimal,-noall_load")
    else()
        target_link_options(mlir-codegen PUBLIC "-Wl,--whole-archive,-lFortranRuntime,-lFortranDecimal,--no-whole-archive")
    endif()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)

if(${UPPERCASE_CMAKE_BUILD_TYPE} MATCHES "REL.*")
    message("release build - omit most of the checks and outputs")
    target_compile_definitions(mlir-codegen PUBLIC "CODEGEN_RELEASE_BUILD")
endif()


target_compile_options(mlir-codegen PUBLIC "-fvisibility=hidden")

if(APPLE)
    target_link_options(mlir-codegen PUBLIC -Wl,-export_dynamic)
else()
    target_link_options(mlir-codegen PUBLIC -Wl,--export-dynamic)
endif()

add_executable(mlir-codegen-cli MLIRCodegen.cpp)
target_link_libraries(mlir-codegen-cli PUBLIC mlir-codegen LLVMSupport)
set_target_properties(mlir-codegen-cli PROPERTIES OUTPUT_NAME "mlir-codegen")

if(NOT FOR_AARCH64)
    set(FLAGS -Wall -Wextra -Wpedantic -Werror)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND FLAGS -Wno-subobject-linkage)
endif()

llvm_update_compile_flags(mlir-codegen-cli)
llvm_update_compile_flags(mlir-codegen)
target_compile_options(mlir-codegen PRIVATE ${FLAGS})
target_compile_options(mlir-codegen-cli PRIVATE ${FLAGS})
