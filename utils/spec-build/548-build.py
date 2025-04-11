import sys
from subprocess import PIPE, Popen
import os

FLANG_BIN = os.environ.get("COMPILER_BIN_DIR") + "/flang-new"
command = f"{FLANG_BIN} -fc1 exchange2.F90 -mmlir --main-entry-name=main -mmlir -mlir-print-ir-after-all -emit-llvm"
print(command)
p = Popen(command, shell=True, stderr=PIPE)
_, stream = p.communicate()

dump = stream.decode()
splits = dump.split("// -----// IR Dump After FIRToLLVMLowering (fir-to-llvm-ir) //----- //")
assert (len(splits) == 2)

output = open(f"spec548.mlir", "w")
output.write(splits[1])
output.close()
