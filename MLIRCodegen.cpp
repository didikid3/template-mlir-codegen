#include "MLIRCodegen.hpp"

#include "Execution/Baseline.hpp"
#include "Execution/JIT.hpp"
#include "Execution/LLI.hpp"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/InitAllDialects.h"

#include <fstream>

#ifdef __USE_MLIRExecutionEngine__
#include "Execution/MLIRExecutionEngine.hpp"
#endif

extern uint64_t clobberedTotal;

using namespace llvm;

#ifdef INSTRUMENT
uint64_t ScopeTimer::templateLookup;
uint64_t ScopeTimer::globalPreparation;
uint64_t ScopeTimer::copyAndPatching;
uint64_t ScopeTimer::valueStorage;
uint64_t ScopeTimer::templateGeneration;
#endif

namespace impl {
enum ExecutionMode
{
    Interpreter,
    JIT,
    O0,
    O1,
    O2,
    O3,
    Baseline
};
}
std::string INPUT;
cl::opt<std::string, true> Input(cl::Positional, cl::desc("<input>"), cl::Required, cl::location(INPUT));
cl::list<std::string> Args(cl::Positional, cl::desc("<args>"));
cl::opt<std::string> Library("lib", cl::desc("Specify folder for template library"), cl::value_desc("library"), cl::Required);
cl::opt<impl::ExecutionMode> ExecMode(cl::desc("Choose execution mode:"),
    cl::values(
        clEnumValN(impl::ExecutionMode::JIT, "fast", "use jit"),
        clEnumValN(impl::ExecutionMode::Baseline, "base", "baseline for traversing/lowering the mlir"),
        clEnumValN(impl::ExecutionMode::O0, "O0", "use llvm execution engine without optimizations"),
        clEnumValN(impl::ExecutionMode::O2, "O2", "use llvm execution engine with optimizations"),
        clEnumValN(impl::ExecutionMode::O1, "O1", "use llvm execution engine with optimizations"),
        clEnumValN(impl::ExecutionMode::O3, "O3", "use llvm execution engine with optimizations"),
        clEnumValN(impl::ExecutionMode::Interpreter, "interpreter", "use LLVM interpreter")),
    cl::init(impl::ExecutionMode::JIT));
bool TIME_FLAG;
cl::opt<bool, true> Time("time", cl::desc("Print timing of lowering, compilation and execution time"), cl::location(TIME_FLAG));
bool MAIN;
cl::opt<bool, true> EmulateMain("main", cl::desc("Call the main function with argv/argc arguments"), cl::location(MAIN));
cl::opt<bool> AllowUnregistered("allow-unregistered", cl::desc("Allow unregistered ops/attributes etc."));
int RUNS;
cl::opt<int, true> Runs("runs", cl::desc("repeat running"), cl::location(RUNS), cl::init(1));
bool DUMP_CODE;
cl::opt<bool, true> Dump("dump", cl::desc("dump generated code"), cl::location(DUMP_CODE));
bool NO_TEMPLATE_OUTLINES;
cl::opt<bool, true> NoOutlines("nooutline", cl::desc("do not print the template boundaries into the assembly when dumping"), cl::location(NO_TEMPLATE_OUTLINES));
int TRACING_LEVEL;
cl::opt<int, true> Trace("verbose", cl::desc("tracing level"), cl::location(TRACING_LEVEL), cl::init(1));
bool COMPILE_ONLY;
cl::opt<bool, true> CompileOnlyFlag("compileonly", cl::desc("only generate code without executing it"), cl::location(COMPILE_ONLY), cl::Hidden);
#ifdef NO_MAP_WX
bool TEMPLATE_CONST_EVAL = false;
#else
bool TEMPLATE_CONST_EVAL = true;
cl::opt<bool, true> ConstEval("consteval", cl::desc("evaluate constant templates during compilation"), cl::location(TEMPLATE_CONST_EVAL), cl::init(true));
#endif
cl::list<uint64_t> CodeSizes("code", cl::desc("size for code"), cl::ZeroOrMore);
int LLVM_CMODEL = 0;
cl::opt<int, true> CModel("llvmcmodel", cl::desc("code model for llvm execution engines (0: small, 1:medium, 2:large)"), cl::location(LLVM_CMODEL), cl::init(0));
bool LLVM_OPT;
cl::opt<bool, true> Opt("optimize", cl::desc("use the LLVM (middle-end) optimizations according to the opt level"), cl::location(LLVM_OPT), cl::init(true));

mlir::OwningOpRef<mlir::ModuleOp> loadModule(mlir::MLIRContext& context)
{
    mlir::OwningOpRef<mlir::ModuleOp> mod;
    llvm::SourceMgr sourceMgr;
    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

    mod = mlir::parseSourceFile<mlir::ModuleOp>(INPUT, sourceMgr, mlir::ParserConfig{&context});
    REL_ASSURE(static_cast<bool>(mod), ==, true);
    setDataLayout(*mod);
    return mod;
}

int main(int argc, char** argv)
{
    StringMap<cl::Option*> optionMap = cl::getRegisteredOptions();
    // -tail-dup-placement=0 required for aarch64
    reinterpret_cast<cl::opt<bool>*>(optionMap["tail-dup-placement"])->setValue(false);
    cl::ParseCommandLineOptions(argc, argv, "MLIR codegen jit-compiler cli");

    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::MLIRContext context(mlir::MLIRContext::Threading::DISABLED);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    context.allowUnregisteredDialects(AllowUnregistered);

#ifdef FORTRAN
    std::atexit(atexit_print_handler);
#endif

    auto mod = loadModule(context);

    std::vector<uint64_t> argumentVec;
    if (MAIN) {
        argumentVec.push_back(Args.size() + 1);
        argumentVec.push_back(reinterpret_cast<uint64_t>(new char*[Args.size() + 1]));
        reinterpret_cast<char**>(argumentVec.back())[0] = const_cast<char*>("a.out");
        for (unsigned i = 0; i < Args.size(); i++) {
            reinterpret_cast<char**>(argumentVec.back())[i + 1] = const_cast<char*>(Args[i].c_str());
        }
    } else {
        for (const auto& arg : Args) {
            argumentVec.push_back(std::stoull(arg));
        }
    }
    TimedResult result = [&] {
        switch (ExecMode) {
            case impl::ExecutionMode::Baseline:
                return run<Baseline>(*mod, argumentVec);
            case impl::ExecutionMode::Interpreter:
                return run<LLI>(*mod, argumentVec);
            case impl::ExecutionMode::JIT:
                return run<JIT>(*mod,
                    Library,
                    Abstraction::defaultLowering,
                    std::vector<uint64_t>(CodeSizes.begin(), CodeSizes.end()),
                    argumentVec);
#ifdef __USE_MLIRExecutionEngine__
            case impl::ExecutionMode::O0:
                return run<MLIRExecEngineO0>(*mod, argumentVec);
            case impl::ExecutionMode::O1:
                return run<MLIRExecEngineO1>(*mod, argumentVec);
            case impl::ExecutionMode::O2:
                return run<MLIRExecEngineO2>(*mod, argumentVec);
            case impl::ExecutionMode::O3:
                return run<MLIRExecEngineO3>(*mod, argumentVec);
#endif
            default:
                UNREACHABLE();
        }
    }();

    if (TIME_FLAG) {
        auto file = std::ofstream(result.name + "-time.txt");
        file << result;
        file.close();
    }
    PRINT_TIMERS();
    TIME_FLAG = false;

    llvm::outs() << "Output: " << result.result << "\n";
    return 0;
}
