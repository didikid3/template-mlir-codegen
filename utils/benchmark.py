from common import *

ARM_CONFIGS = " -DFOR_AARCH64=ON -DFIXED_STACKSIZE=2048"

CONFIGURATIONS = [
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=OFF -DEVICTION_STRATEGY=OFF -DFORCE_STACK=ON -DFIXED_STACKSIZE=384"
    },
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=1"
    },
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=OFF -DEVICTION_STRATEGY=OFF"
    },
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=2"
    },
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=3"
    },
    {
        "cmake": "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=4"
    }
]

for j, config in enumerate(CONFIGURATIONS):
    assert (not AARCH64 or os.environ.get("COREMARK"))
    if (j == 0 or j == 3) and os.environ.get("COREMARK"):
        continue
    recompile(config["cmake"] + (ARM_CONFIGS if AARCH64 else ""))
    for i in range(RUNS):
        print("\n========================INPUT========================", file=OUT_FILE)
        print("input: ", "02", config["cmake"], file=OUT_FILE)
        print("=====================================================\n", file=OUT_FILE)
        base = ALL if i < 3 else NO_INTERPRETER
        modes = base if j == 0 else FAST_ONLY
        if not os.environ.get("COREMARK"):
            benchmark("bm1.mlir", ["10000"], execution_modes=modes)
            benchmark("bm2.mlir", ["10000"], execution_modes=modes)
            benchmark("bm3.mlir", ["50000000"], execution_modes=modes)
            benchmark("bm4.mlir", ["50000000"], execution_modes=modes)
            benchmark("bm5.mlir", ["40"], execution_modes=modes)
            benchmark("bm6.mlir", ["40"], execution_modes=modes)
        else:
            coremark = os.environ.get("COREMARK_DIR")
            assert (coremark is not None)
            benchmark(
                f"{coremark}/coremark.mlir", ["-main", "-llvmcmodel=2" if AARCH64 else "-llvmcmodel=1", "-code=32000", "-code=84000"],
                execution_modes=NO_INTERPRETER if j == 1 else FAST_ONLY)
