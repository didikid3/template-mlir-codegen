from common import *
import os
from subprocess import PIPE, Popen, TimeoutExpired
import traceback

ARM_CONFIGS = " -DFOR_AARCH64=ON -DFIXED_STACKSIZE=2048"

configurations = [
    # "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=OFF",
    # "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=1",
    "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=2",
    "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=3",
    "-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=4"
]

coremark = os.environ.get("COREMARK_DIR")

command = EXECUTABLE
command += " -" + "fast"
command += f" -lib {LIBRARY_PATH}"
command += " -time"
command += " -verbose=2"
command += " -allow-unregistered "
command += f'{coremark}/coremark.mlir'
command += " "
command += " ".join(["-main", "-llvmcmodel=2" if AARCH64 else "-llvmcmodel=1", "-code=32000", "-code=84000"])

RUNS = 5

for i, config in enumerate(configurations):
    print(f"---DEBUG CONFIGURATION {i}: {config}---")

    print(("Recompiling ..."))
    recompile(config + (ARM_CONFIGS if AARCH64 else ""))
    print("Recompile done")

    if os.path.exists(LIBRARY_PATH):
        os.system(f"rm -r {LIBRARY_PATH}/")
    if not os.path.exists(coremark):
        print("Coremark directory does not exist")
        break
    

    print("Executing benchmark...")
    for j in range(RUNS):
        print(f"###Staring Run {j+1}/{RUNS}###")
        try:
            subp = Popen(command, stdin=PIPE, shell=True, stdout=PIPE, stderr=PIPE)
            # maximum 4h
            out, err = subp.communicate(timeout=60 * 60 * 4)
            print(out.decode())
            print(err.decode())

            if os.path.exists("register_cache.log"):
                log_dir = 'benchmark_logs'
                os.makedirs(log_dir, exist_ok=True)

                newFile = f'{log_dir}/register_cache{config}_{j+1}.log'
                os.rename("register_cache.log", newFile)
                print(f"register_cache.log moved to {newFile}")
        except TimeoutExpired:
            print("!!!!!!!!!!!!!!!! timeout !!!!!!!!!!!!!!!!")
        except Exception as ex:
            print("!!!!!!!!!!!!!!!! exception !!!!!!!!!!!!!!!!")
            traceback.print_exc()
