# Reproduction
- as all dependent projects (lingodb, polygeist and our approach) require their own llvm version, reproducing the results requires building the llvm project three times.
- experiments from the paper:
	- Microbenchmarks (x86 only)
	- PolyBenchC (x86 only)
	- LingoDB (x86 only)
	- Coremark (x86 and aarch64)
	- SPEC (x86 and aarch64)

## Requirements
 - linux operating system on x86 and aarch64
 - `podman` container runtime
 - disk space: 40GB (x86); 20GB (aarch64)
 - DRAM: 32GB(x86); 16GB (aarch64)

## Setup
Folder structure like:
```
src
 |- mlir-codegen <- run scripts inside here
 | |- spec-data (spec only; spec benchmark input data)
 | |- spec-programs (spec only; spec benchmark mlir programs)
 | \- results <- results will appear here
 |- llvm-project
 |- coremark
 |- lingo-db
 |- mlir-codegen-lingodb
 |- Polygeist
 \- PolybenchC-4.2.1
```

1. build container for build and runtime environment
`podman build --squash . -t mlir-codegen-build`
2. run the build container and mount the above folder structure
`podman run -it --rm --mount type=bind,source=${PWD}/..,target=/src mlir-codegen-build bash`
3. build the dependent projects (you might want to adjust the `CMAKE_BUILD_PARALLEL_LEVEL` environment variable to control the number of compiling jobs --- default is set to `2`)
`make prepare-x86`/`make prepare-aarch64`
> On some hosts, we encountered a sporadic internal compiler error of gcc while building LLVM; in these cases rerun the target until it finishes successfully
4. (spec only; run on target machine) prepare spec programs and data. As we can not distribute the SPEC benchmarks and data, there is some manual effort required. Export `SPEC_BENCHSPEC_FOLDER` environment variable to point to the `benchspec` folder of the unpacked spec benchmark data. Run `make spec` to create and fill the `spec-data` and `spec-program` folder (There might be some changes/removals to the `./simple-build-ldecod_r-525.sh` necessary, as some versions of `gcc` disallow the `-m64` option on aarch64).

## Execution
5. run the benchmarks (except SPEC) using the architecture specific benchmark make targets (`make benchmark-x86`/`make benchmark-aarch64`). The spec benchmarks can be run on both architectures using `make benchmark-spec`. The benchmarks produce output log files in the result directory.

## Visualization
6. visualize the results in diagrams similar to the ones presented in the paper by using the `make viz` target. It produces output diagrams as `pdf` files in the `results` folder for whatever benchmark results files are present.

## Summary
Reproduce all result diagrams in `results` folder (SPEC commands can be left out to skip reproduction of the SPEC results):

### X86
```
podman build . -t mlir-codegen-build
podman run -it --rm --mount type=bind,source=${PWD}/..,target=/src mlir-codegen-build bash
make prepare-x86
SPEC_BENCHSPEC_FOLDER=[...]/benchspec make spec
make benchmark-x86
make benchmark-spec
make viz
```

### AArch64
```
podman build . -t mlir-codegen-build
podman run -it --rm --mount type=bind,source=${PWD}/..,target=/src mlir-codegen-build bash
make prepare-aarch64
SPEC_BENCHSPEC_FOLDER=[...]/benchspec make spec
make benchmark-aarch64
make benchmark-spec
make viz
```