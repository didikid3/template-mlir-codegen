from common import *

ARM_CONFIGS = " -DFOR_AARCH64=ON -DFIXED_STACKSIZE=52384"

CONFIGURATIONS = [
    {
        "subfolder": "O0",
        "cmake": "-DFORTRAN=ON -DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=1"
    }
#    ,
#    {
#        "subfolder": "O0",
#        "cmake": "-DFORTRAN=ON -DOPTIMIZED_VARIABLE_PASSING=OFF -DEVICTION_STRATEGY=OFF"
#    }
#    ,
#    {
#        "subfolder": "O0",
#        "cmake": "-DFORTRAN=ON -DOPTIMIZED_VARIABLE_PASSING=ON -DEVICTION_STRATEGY=2"
#    }
]

for _ in range(RUNS):
    for i, config in enumerate(CONFIGURATIONS):
        print("\n========================INPUT========================", file=OUT_FILE)
        print("input: ", config["subfolder"], config["cmake"], file=OUT_FILE)
        print("=====================================================\n", file=OUT_FILE)
        if AARCH64:
            recompile(config["cmake"] + " " + ARM_CONFIGS)
            cmodel = 2
        else:
            recompile(config["cmake"])
            cmodel = 1

        folder = os.environ.get("SPEC_PROGRAMS")
        assert(folder is not None)
        modes = NO_INTERPRETER if i == 0 else FAST_ONLY

        # test data
        # benchmark(
        #     folder + "/spec505.mlir",
        #     ["-code 90000", "-code 200000", "-main", "-llvmcmodel 1",
        #      "-- inp.in"], execution_modes=modes)
        # benchmark(folder + "/spec548.mlir", ["-code 80000", "-code 1000000",
        #                                      "-code 5000", "-llvmcmodel 1"], stdin="0\n".encode(), execution_modes=modes)
        # benchmark(
        #     folder + "/spec557.mlir",
        #     ["-code 90000", "-code 2000000", "-main", "-llvmcmodel 1",
        #      "-- cpu2006docs.tar.xz 4 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1548636 1555348 0"],
        #     execution_modes=modes)
        # benchmark(
        #     folder + "/spec525.mlir",
        #     ["-code 200000", "-code 40000000", "-main", "-llvmcmodel 1",
        #      "-- -o BuckBunny_New.264 BuckBunny.yuv 1280x720"], execution_modes=modes)
        #####################################################################################
        # reference data (spec speed)
        benchmark(
            folder + "/spec505.mlir",
            ["-code 90000", "-code 200000", "-main", f"-llvmcmodel {cmodel}", "-- inp.in"],
            execution_modes=modes)
        benchmark(
            folder + "/spec548.mlir",
            ["-code 80000", "-code 1700000", "-code 10000", f"-llvmcmodel {cmodel}"],
            stdin="6\n".encode(),
            execution_modes=modes)
        # on ARM: Killed by OOM for fast and O0 (O2 crashes elsewhere...), test-data however work just fine
        if not AARCH64:
            benchmark(
                folder + "/spec557.mlir",
                ["-code 90000", "-code 2000000", "-main", f"-llvmcmodel {cmodel}",
                "-- cpu2006docs.tar.xz 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1036078272 1111795472 4"],
                execution_modes=modes)
        benchmark(
            folder + "/spec557.mlir",
            ["-code 90000", "-code 2000000", "-main", f"-llvmcmodel {cmodel}",
            "-- cld.tar.xz 1400 19cf30ae51eddcbefda78dd06014b4b96281456e078ca7c13e1c0c9e6aaea8dff3efb4ad6b0456697718cede6bd5454852652806a657bb56e07d61128434b474 536995164 539938872 8"],
            execution_modes=modes, noclear=True)
        benchmark(
            folder + "/spec525.mlir",
            ["-code 800000", "-code 40000000", "-main", f"-llvmcmodel {cmodel}",
             "-- --pass 1 --stats x264_stats.log --bitrate 1000 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720"],
            execution_modes=modes)
        benchmark(
            folder + "/spec525.mlir",
            ["-code 800000", "-code 40000000", "-main", f"-llvmcmodel {cmodel}",
             "-- --pass 2 --stats x264_stats.log --bitrate 1000 --dumpyuv 200 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720"],
            execution_modes=modes, noclear=True)
        benchmark(
            folder + "/spec525.mlir",
            ["-code 800000", "-code 40000000", "-main", f"-llvmcmodel {cmodel}",
             "-- --seek 500 --dumpyuv 200 --frames 1250 -o BuckBunny_New.264 BuckBunny.yuv 1280x720"],
            execution_modes=modes, noclear=True)
