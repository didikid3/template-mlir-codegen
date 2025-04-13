import signal
import os
import sys
from subprocess import PIPE, Popen, TimeoutExpired
import platform
import traceback

def signal_handler(sig, frame):
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

assert(("AARCH64" in platform.machine().upper()) or ("X86_64" in platform.machine().upper()))
AARCH64 = "AARCH64" in platform.machine().upper()
RUNS = int(os.environ.get("RUNS") or 1)
ABS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.environ.get("BUILD_DIR") or f"{ABS_DIR}/../build"
EXECUTABLE = os.path.join(BUILD_DIR, "mlir-codegen")
MLIR_DIR = os.environ.get("MLIR_DIR")
ALL = ["fast", "O0", "O2", "interpreter"]
NO_INTERPRETER = ["fast", "O0", "O2"]
FAST_ONLY = ["fast"]
NAMES = {
    "fast": "JIT",
    "interpreter": "LLI",
    "O0": "MLIRExecEngineOpt0",
    "O1": "MLIRExecEngineOpt1",
    "O2": "MLIRExecEngineOpt2",
    "O3": "MLIRExecEngineOpt3"
}
LIBRARY_PATH = "dummy"

OUT_FILE = sys.stderr
if len(sys.argv) >= 2:
    OUT_FILE = open(sys.argv[1], 'w')

last_options = None
def recompile(options):
    buildType = os.environ.get("BUILD_TYPE") or "Release"
    global last_options
    if last_options == options:
        return
    last_options = options
    if os.path.exists(BUILD_DIR):
        os.system(f"rm -rf {BUILD_DIR}")
    assert (MLIR_DIR is not None)
    Popen(
        f"cmake -DCMAKE_BUILD_TYPE={buildType} -DCMAKE_CXX_FLAGS=\"-fPIE -fPIC\" -DMLIR_DIR={MLIR_DIR} {options} -S{ABS_DIR}/.. -B{BUILD_DIR}",
        shell=True, stdout=PIPE, stderr=PIPE).communicate()
    Popen(f"cmake --build {BUILD_DIR} --target mlir-codegen-cli -j4",
          shell=True, stdout=PIPE, stderr=PIPE).communicate()


def benchmark(file, args, execution_modes=ALL, stdin=None, noclear=False):
    for mode in execution_modes:
        if os.path.exists(LIBRARY_PATH) and not noclear:
            os.system(f"rm -r {LIBRARY_PATH}/")
        if not os.path.exists(file):
            print("file does not exist", file=OUT_FILE)
            return
        command = EXECUTABLE
        command += " -" + mode
        command += f" -lib {LIBRARY_PATH}"
        command += " -time"
        command += " -verbose=0"
        command += " -allow-unregistered "
        command += file
        command += " "
        command += " ".join(args)

        assert (NAMES[mode] is not None)
        print("EXECUTION:" + NAMES[mode] + ":" + command, file=OUT_FILE)
        try:
            subp = Popen(command, stdin=PIPE, shell=True, stdout=PIPE, stderr=PIPE)
            # maximum 4h
            out, err = subp.communicate(input=stdin, timeout=60 * 60 * 4)
            print(out.decode(), file=OUT_FILE)
            print(err.decode(), file=OUT_FILE)

            filename = NAMES[mode] + "-time.txt"
            print(open(filename, 'r').read(), file=OUT_FILE)
            os.remove(filename)
        except TimeoutExpired:
            print("!!!!!!!!!!!!!!!! timeout !!!!!!!!!!!!!!!!", file=OUT_FILE)
        except Exception as ex:
            print("!!!!!!!!!!!!!!!! exception !!!!!!!!!!!!!!!!", file=OUT_FILE)
            traceback.print_exc(file=OUT_FILE)  # full traceback

if __name__ == "__main__":
    print("Compiler mlir-codege with default")
    recompile("-DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=1")
