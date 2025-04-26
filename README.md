# CSE 583 Notes

This repository is only a section of the total software artifact needed to reproduce our results. To do so the original artifact from the Fast Template Based Code Generation for MLIR paper must be downloaded. Note that just installing and compiling the base version requires around 50 GB. Compiling and running benchmark times vary depending on the machine used. Across different x86 and arm platforms, we got around ~ 1-3 hours to compile and benchmark times vary drastically depending on which one is run. Coremark may take around 20 minutes - 1 hour. Polybench ran for more than 6 hours.

Follow the steps in the artifact README to understand how to compile and run the base benchmarks. Once that is understood, replace thier mlir-codegen directory with our repository which modifies some utility files to test different eviction strategies. To get access and eviction logs, a debug build (NOT RELEAE) must be compiled and will utilize the following command. 

```bash
make benchmark-debug
```

All other results and logs can still utilize the same workflow as the paper used.

Link to the paper: https://dl.acm.org/doi/10.1145/3640537.3641567  
Link to Artifact: https://dl.acm.org/do/10.5281/zenodo.10571103/full/



# Template-based MLIR Compiler

The repository contains the sources for building the template-based MLIR compiler and the dependent LLVM sources (commit `5d4927` with some modifications). It compiles and executes MLIR programs consisting of supported operations (multiple sample programs are included; similar to `mlir-cpu-runner`); on first execution, it generates required templates and persists them.
Furthermore, the artifact contains the modified sources for LingoDB with integrated template-based code-generation backend and Polygeist (commit `fd4194b`) for conversion of C files to MLIR upstream dialect operations. Sample MLIR programs and scripts for [preparing/running](#reproducible-artifact) the benchmarks from Figures 2-5 are attached.

## Benchmarks

### Reproducible Artifact
refer [instructions for reproduction](Reproduction.md)

### Evaluation and expected results
In general, the x86-64 benchmarks in the paper were run on an older server CPU, so latency of individual instructions as well as memory access costs might vary on modern desktop CPUs and thus end up with slightly different results. The following applies for both architectures - x86-64 and aarch64; the experiments reproduce compilation and execution times of the respective benchmarks visualized as diagrams similar to the ones presented in the paper. In general, there should be a one to two orders of magnitude difference in the compilation time of our approach compared to the LLVM backends and a slowdown below 3x (at least in the geomean) for the execution time.

#### Microbenchmarks
The experiment additionally reproduces the effect of the individual applied optimizations.
The individual optimizations might vary heavily based on the exact CPU. The faster the execution of the benchmarks the less difference between the individual stages. Depending on the memory access costs, the difference effectiveness of the register caching might be reduced (barely any improvement to template calling convention).

#### LingoDB
Faster systems might end up with a different speedup factor for total query runtime as execution time is a larger factor to our approach than it is to the others.

#### PolyBenchC, Spec and Coremark
Expect an order of magnitude between the three approaches on compilation time; similar results in the geomean for execution time. As the results are normalized to the execution of the optimized code generation backend of LLVM results might shift quite a bit for individual benchmarks as faster execution times for the baseline results in comparably high slowdowns for the other approaches.

## Building

Use `make build/mlir-codegen` to build the release binary of the template compiler with its default settings.

> `make build-debug/mlir-codegen` can be used for a equivalent debug build with many more assertions and infos

For further customization, useful build flags are:
- `MLIR_DIR`: path to find MLIR (e.g.: _llvm-project/build/lib/cmake/mlir_)
- `EVICTION_STRATEGY`: pick an eviction strategy for saving redundant-stack-loads
- `OPTIMIZED_VARIABLE_PASSING` : enable optimized variable passing (dead-store elimination)
- `FIXED_STACKSIZE`: if specified sets the used stackframe size for each function (in bytes; must be aligned)
- `FORCE_STACK`: forces usage of variable storage templates (no register passing). This additionally requires a fixed stacksize, as we otherwise end up with an LLVM Bug (referencing a clobbered register after a called that is marked as `no_callee_saved_registers`).

## Usage
For examples on how to use the CLI, take a look at `utils/spec.py`, `utils/benchmark.py` and `utils/tests.py` or start with something like: `build/mlir-codegen -lib dummy Samples/sample1.mlir`
> sometimes manually removing the generated template library can help (`rm -r dummy/`) in certain situations (e.g. if you end up in an inconsistent state due to an exception in an earlier execution).

### CLI Options
- input mlir file
- execution mode (one of):
    - `-fast`: template-based compilation
    - `-O0`: LLVM O0 backend
    - `-O1`: LLVM O1 backend
    - `-O2`: LLVM O2 backend
    - `-interpreter`: LLI
> Using LLI with external symbols requires LLVM to be build with FFI support
- `-lib`: specifies template library directory
- `-time`: writes files with timings of the run (compile and execution time)
- `-main`: treat the entry point as main-function and pass the argument(s) as `argc` and `argv`
- `-code`: set the memory size for generated binary code
- `-llvmcmodel`: set the code-model for LLVM backends (0-3: small-large)
- `-verbose`: controls verbosity of debug output/assertions (only applicable in debug builds)
- positional arguments to main function

## General

- `rdi` (`x0`) is the pinned value storage pointer (also referred to as stack pointer)
- `rsi`, `rdx`, `rcx`, `r8`, `r9` (`x1` - `x7`) are other argument passing registers (in that order)
- `r10`, `r11`, `r12`, `rax`, `r13`, `r14`, `r15`, `rbx` (`x8`- `x15`) are caching registers to cache values for redundant-store elimination, but can not be used to pass arguments to a template directly (due to the restrictions of the calling convention)

Each MLIR value is assigned a storage location consisting of one or more __slots__. A __slot__ is a 8-byte aligned 64-bit storage location but can also be a register.
> Although we could address the slots on our stack with their slot index all patched stack offsets are in bytes (and therefore a multiple of 8). This is done, so that LLVM can fuse the offset directly into a `mov` instruction, which wouldn't be possible if the multiplication with 8 would be still outstanding.

For values, that are already known during compile time (e.g. constant evaluated templates) we also allow out-of-line storage as constants and directly materialize the value into the generated code using const2reg/const2stack code snippets.


## Debugging/Code Visualization

### Template Inspection
Templates are persistently stored across runs in `dummy/stencilXXX.o`, where `XXX` is the template id (view with `objdump -rS`). Each binary contains a section per kind of template of the form `.text.stencilXXX_KIND`, where `KIND` indicates, whether the function retrieves their parameters from arguments (__0__, used for constant evaluation), gets them patched into the binary (__1__, used for stack-cc) or in registers (__2__, used for register-cc).

### Code Inspection (requires debug build)
Specifying the `dump` option writes the generated code (as textual assembler) to the output and additionally indicates where the corresponding templates begin and end together with their name and template id. Getting plain assembler is also possible, by specifying `nooutline` to skip visualization of template boundaries.

> For a quick comparison, the `dump` option also works with `O0` and `O2` execution, however the output format slightly differs.