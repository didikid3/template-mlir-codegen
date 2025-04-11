import os
import sys
import platform

EXECUTABLE = "build/mlir-codegen"
if len(sys.argv) >= 2:
    EXECUTABLE = sys.argv[1]
ALL_MODES = ["fast", "interpreter", "O2", "O0"]
NO_INTERPRETER = ["fast", "O2", "O0"]
FAST_ONLY = ["fast"]
assert(("AARCH64" in platform.machine().upper()) or ("ARM64" in platform.machine().upper()) or ("X86_64" in platform.machine().upper()))
AARCH64 = not "X86_64" in platform.machine().upper()


def run_and_verify(nr, args, expected_result, execution_modes=ALL_MODES):
    for mode in execution_modes:
        if os.path.exists("dummy"):
            os.system("rm -r dummy/")
        command = EXECUTABLE
        command += " -" + mode
        command += " -lib dummy"
        command += " Samples/sample" + str(nr) + ".mlir "
        if args is not None:
            for arg in args:
                command += str(arg)

        stream = os.popen(command)
        output = stream.read()
        expected = "Output: " + str(expected_result)
        if (not expected in output):
            print(command)
            print(output)
            print("Expected: " + expected)
            assert(False)
        else:
            print("Verified: " + str(nr) + " (" + mode + ")")


run_and_verify(0, None, 2)
run_and_verify(1, None, 2, FAST_ONLY)
run_and_verify(2, None, 0, FAST_ONLY)
run_and_verify(3, None, 2)
run_and_verify(4, None, 2)
run_and_verify(5, None, 6)
run_and_verify(6, [1], 1)
run_and_verify(7, None, 2)
run_and_verify(8, None, 2)
run_and_verify(9, None, 6)
run_and_verify(10, None, 4, NO_INTERPRETER)
run_and_verify(11, None, 0)
run_and_verify(12, None, 128)
run_and_verify(13, None, 9)
run_and_verify(14, None, 0, FAST_ONLY)
run_and_verify(15, None, 11)
run_and_verify(16, [5], 2, NO_INTERPRETER)
run_and_verify(17, [5], 5, NO_INTERPRETER)
run_and_verify(18, [5], 2, FAST_ONLY)
run_and_verify(19, None, 8, FAST_ONLY)
run_and_verify(20, None, 13)
run_and_verify(21, [10], 2, FAST_ONLY)
run_and_verify(22, None, 1)
run_and_verify(23, None, 4294967293)  # -3
run_and_verify(24, [10], 144)
run_and_verify(25, None, 0, FAST_ONLY)
run_and_verify(26, [10], 144)
run_and_verify(27, None, 0)
run_and_verify(28, None, 125)
run_and_verify(29, None, 0)
run_and_verify(30, None, 13)
run_and_verify(31, None, 3)
if AARCH64 or "Darwin" in platform.system():
    run_and_verify(32, ["42", " -llvmcmodel=2"], 7)
else:
    run_and_verify(32, ["42", " -llvmcmodel=1"], 7)

if not (AARCH64 or "Darwin" in platform.system()):
    run_and_verify(33, None, 100, FAST_ONLY) # broken on aarch64...
run_and_verify(34, None, 0)
run_and_verify(35, None, 1)
run_and_verify(36, None, 0)
run_and_verify(37, None, 160, NO_INTERPRETER) # interpreter bug ...
run_and_verify(38, None, 1)
run_and_verify(39, None, 123456, NO_INTERPRETER)
run_and_verify(40, None, 1)
run_and_verify(41, None, 6)
run_and_verify(42, None, 7)
run_and_verify(43, None, 16)
run_and_verify(44, None, 4294967295)
run_and_verify(45, None, 6)
run_and_verify(46, ["-main", " --", " 1", " 2", " 3", " 4"], 6)
