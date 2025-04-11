import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib
import os
import numpy as np
import pandas as pd

cc = colors.ColorConverter()
colors = [cc.to_rgba_array(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
c_O0 = colors[0][0, :3]
c_O2 = colors[1][0, :3]
c_llvm = colors[3][0, :3]
c_builtin = colors[2][0, :3]
c_lli = colors[4][0, :3]

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 8
})


def parse(file: str, cmake_configuration=None, optimization=None):
    f = open(file, 'r')
    lines = f.readlines()
    assert (all(['file does not exist' not in l for l in lines]))
    executions = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if '========================INPUT========================' in line:
            next_line = lines[i + 1]
            assert (next_line.startswith("input:  "))
            next_line = next_line[8:]
            optimization = next_line[:2]
            cmake_configuration = next_line[3:]
            i += 2
        elif 'EXECUTION' in line:
            _exec, mode, command = line.split(":")
            assert (_exec == "EXECUTION")
            executions.append(
                {
                    "mode": mode,
                    "command": command[:-1],
                    "cmake": cmake_configuration[:-1],
                    "optimization": optimization
                }
            )
        elif '=====================|' in line:
            assert (line.split("|")[1] == executions[-1].get("mode"))
            executions[-1]["timeout"] = False
            executions[-1]["exception"] = False
            if 'Lowering' in lines[i + 1]:
                for time in lines[i + 1: i + 4]:
                    executions[-1][time.split(" ")[0]] = float(time.split(":")[-1].replace(" μs", ""))
                i += 4
            else:
                for time in lines[i + 1: i + 3]:
                    executions[-1][time.split(" ")[0]] = float(time.split(":")[-1].replace(" μs", ""))
                executions[-1]["Lowering"] = 0
                i += 3
        elif '!!!!!!!!!!!!!!!! timeout !!!!!!!!!!!!!!!!' in line:
            executions[-1]["timeout"] = True
            executions[-1]["Lowering"] = -1
            executions[-1]["Preparation"] = -1
            executions[-1]["Execution"] = -1
        elif '!!!!!!!!!!!!!!!! exception !!!!!!!!!!!!!!!!' in line:
            executions[-1]["exception"] = True
            executions[-1]["Lowering"] = -1
            executions[-1]["Preparation"] = -1
            executions[-1]["Execution"] = -1
        elif len(executions) >= 1:
            assert (len(executions) >= 1)
            if "Elapsed time" not in line:
                executions[-1]["output"] = executions[-1].get("output", "") + line
        i += 1
    df = pd.DataFrame(executions)
    df["valid"] = df.apply(lambda x: not x['timeout'] and not x['exception'], axis=1)
    valid_samples = list(df.groupby("mode")["valid"].sum())
    return df.where(df["valid"])


def viz_polybench():
    df_polybench = parse("results/polybench-llvm.log", cmake_configuration=["DEFAULT "], optimization="O0")
    df_polybench2 = parse("results/polybench-polygeist.log", cmake_configuration=["DEFAULT "], optimization="O0")
    bms = list(map(lambda x: x.strip(), list(df_polybench.where(df_polybench["mode"] == "MLIRExecEngineOpt2").dropna().groupby(
        ["command"], as_index=False)[["command", "Execution", "Preparation"]].median(numeric_only=True).sort_values(by=["command"])["command"])))
    exec_baseline = np.array(df_polybench.where(df_polybench["mode"] == "MLIRExecEngineOpt2").dropna().groupby(["command"], as_index=False)[
                             ["command", "Execution", "Preparation"]].median(numeric_only=True).sort_values(by=["command"])["Execution"])
    prep_baseline = np.array(df_polybench.where(df_polybench["mode"] == "MLIRExecEngineOpt2").dropna().groupby(["command"], as_index=False)[
                             ["command", "Execution", "Preparation"]].median(numeric_only=True).sort_values(by=["command"])["Preparation"])
    assert (len(prep_baseline) == len(exec_baseline))
    x = list(range(0, len(prep_baseline)))

    distance = 0.2
    width = 0.2
    ranges = [
        list(np.arange(len(prep_baseline)) - 1.5 * distance) + [len(prep_baseline) + 1 - 1.5 * distance],
        list(np.arange(len(prep_baseline)) - 0.5 * distance) + [len(prep_baseline) + 1 - 0.5 * distance],
        list(np.arange(len(prep_baseline)) + 0.5 * distance) + [len(prep_baseline) + 1 + 0.5 * distance],
        list(np.arange(len(prep_baseline)) + 1.5 * distance) + [len(prep_baseline) + 1 + 1.5 * distance]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 2.3), sharex=True)

    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.grid(axis='y', zorder=-1000000)
    ax2.set_yticks([0, 2, 4, 6, 8], [0, 2, 4, 6, 8])
    ax2.grid(axis='y')

    i = 0
    def plot(df, index):
        nonlocal i
        labels = ["LLVM O2", "LLVM O0", "Ours (built-in dialect)", "Ours (llvm-dialect)"]
        modes = ["MLIRExecEngineOpt2", "MLIRExecEngineOpt0", "JIT", "JIT"]
        cs = [c_O2, c_O0, c_builtin, c_llvm]
        label = labels[index]
        mode = modes[index]
        execution = np.array(df.where(df["mode"] == mode).dropna().groupby(["command"], as_index=False)[["command", "Execution", "Preparation"]].median(numeric_only=True).sort_values(by=["command"])["Execution"]) / exec_baseline
        prep = np.array(df.where(df["mode"] == mode).dropna().groupby(["command"], as_index=False)[["command", "Execution", "Preparation"]].median(numeric_only=True).sort_values(by=["command"])["Preparation"]) / prep_baseline
        execution = np.append(execution, [execution.prod() ** (1.0 / len(execution))])
        prep = np.append(prep, [prep.prod() ** (1.0 / len(prep))])
        ax2.bar(ranges[i], execution, color=cs[index], label=label, width=width)
        ax1.bar(ranges[i], prep, color=cs[index], label=label, width=width)
        i += 1

    plot(df_polybench, 0)
    plot(df_polybench, 1)
    plot(df_polybench, 3)
    plot(df_polybench2, 2)
    ax2.set_xticks(x + [31], bms + ["Geomean"], rotation=45, ha="right")
    ax2.get_xticklabels()[-1].set_rotation(0)
    ax2.get_xticklabels()[-1].set_ha("center")
    ax1.set_ylabel("normalized\ncompilation time")
    ax1.set_yscale("log")
    ax2.set_ylabel("normalized\nexecution time")
    ax1.axvline(x=30, color='black', linestyle=':')
    ax2.axvline(x=30, color='black', linestyle=':')
    ax1.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.), edgecolor="black", fancybox=False)
    ax1.set_xlim(-1, 33.5)
    fig.savefig("results/polybench.pdf", bbox_inches="tight")


def viz_microbm():
    df = parse("results/benchmark.log")

    def plot_per_file(nr, ax, notLower=False, addLowering=False, i=0):
        file = f"bm{nr + i}.mlir"

        def select(s):
            a, b, _ = df.where(
                df["command"].str.contains(file)).where(
                df["cmake"].str.contains(s)).where(
                df["mode"] == mode).groupby(["mode"])[["Preparation", "Execution", "Lowering"]].median().iloc[0]

            return a / 1000000, b / 1000000
        ax.set_xscale('log')
        file = f"bm{nr}.mlir"
        mode = "LLI"
        lli_data = select(".*")
        ax.scatter(*lli_data, label="LLI", marker="+", color=c_lli)
        if notLower:
            return lli_data
        mode = "MLIRExecEngineOpt0"
        data_O0 = select(".*")
        ax.scatter(*data_O0, label="O0", marker="+", color=c_O0)
        mode = "MLIRExecEngineOpt2"
        ax.scatter(*select(".*"), label="O2", marker="+", color=c_O2)

        mode = "JIT"
        file = f"bm{nr + i}.mlir"
        if i == 0:
            c = c_builtin
        else:
            c = c_llvm
        ax.scatter(*select("FORCE_STACK=ON"), label="Ours (basline)", color=[*c, 0.3])
        ax.scatter(*select("EVICTION_STRATEGY=OFF$"), label="+ calling convention", color=[*c, 0.6])
        ax.scatter(*select("EVICTION_STRATEGY=(1|2)"), label="+ register caching", color=c)
        return None

    fig = plt.figure(figsize=(7.8, 2.))
    subfig1, subfig2 = fig.subfigures(2, 1)

    def plot_for_figure(fig, i, upperRow):
        top, bot = fig.subplots(2, 3, sharex=True, gridspec_kw={'height_ratios': [0.25, 0.75]})
        (ax1_, ax2_, ax3_) = top
        (ax1, ax2, ax3) = bot
        d = .25
        kwargs = dict(marker=[(-1, -d), (1, d)], markersize=10,
                      linestyle="none", color='k', mec='k', clip_on=False)
        for a in top:
            a.spines['bottom'].set_visible(False)
            a.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            kwargs.update(transform=a.transAxes)
            a.plot([0, 1], [0, 0], **kwargs)
        for a in bot:
            kwargs.update(transform=a.transAxes)
            a.plot([0, 1], [1, 1], **kwargs)
            a.spines['top'].set_visible(False)
        plot_per_file(1, ax2, addLowering=upperRow, i=i)
        lli2 = plot_per_file(1, ax2_, True, addLowering=upperRow)
        plot_per_file(3, ax1, addLowering=upperRow, i=i)
        lli1 = plot_per_file(3, ax1_, True, addLowering=upperRow, i=i)
        plot_per_file(5, ax3, addLowering=upperRow, i=i)
        lli3 = plot_per_file(5, ax3_, True, addLowering=upperRow, i=i)
        plt.subplots_adjust(hspace=0.06, wspace=0.30)
        for (lli, ax) in [(lli2, ax2_), (lli1, ax1_), (lli3, ax3_)]:
            y = int(lli[1])
            ax.set_ylim(0.9 * y, 1.1 * y)
            ax.set_yticks([y], [y])
        if upperRow:
            ax1.set_ylim(0, 2.2)
            ax1.set_yticks([0, 1, 2], [0, 1, 2])
            ax1.set_xticklabels([])
            ax2.set_ylim(0, 440)
            ax2.set_yticks([0, 0.2, 0.4], [0, 0.2, 0.4])
            ax3.set_ylim(0, 2210)
            ax3.set_yticks([0, 1, 2], [0, 1, 2])
            ax1_.set_xlabel("Sieve of Eratosthenes")
            ax2_.set_xlabel("Quicksort")
            ax3_.set_xlabel("Fibonacci")
            ax3.set_ylabel("Built-in Dialect")
            ax3.yaxis.set_label_coords(1.2, 0.7)
        else:
            ax1.set_ylim(0, 3.5)
            ax1.set_yticks([0, 1, 2, 3], [0, 1, 2, 3])
            ax2.set_ylim(0, 1.09)
            ax2.set_yticks([0, 0.5, 1], [0, 0.5, 1])
            ax3.set_ylim(0, 3.29)
            ax3.set_yticks([0, 1, 2, 3], [0, 1, 2, 3])
            ax3.set_ylabel("LLVM IR Dialect")
            ax3.yaxis.set_label_coords(1.2, 0.7)

        ax1_.xaxis.set_label_position('top')
        ax2_.xaxis.set_label_position('top')
        ax3_.xaxis.set_label_position('top')

        return bot
    a, b, c = plot_for_figure(subfig1, 0, True)
    d, e, f = plot_for_figure(subfig2, 1, False)

    legend_elements = [
        Line2D([0],
               [0],
               linestyle="none", marker='+', color=c_O2),
        Line2D([0],
               [0],
               linestyle="none", marker='+', color=c_O0),
        Line2D([0],
               [0],
               linestyle="none", marker='+', color=c_lli),
        Line2D(
            [0],
            [0],
            fillstyle="left", linestyle="none", marker="o", color=[*c_builtin, 0.3],
            mec=[0, 0, 0, 0],
            markerfacecoloralt=[*c_llvm, 0.3]),
        Line2D(
            [0],
            [0],
            fillstyle="left", linestyle="none", marker="o", color=[*c_builtin, 0.6],
            mec=[0, 0, 0, 0],
            markerfacecoloralt=[*c_llvm, 0.6]),
        Line2D(
            [0],
            [0],
            fillstyle="left", linestyle="none", marker="o", color=[*c_builtin, 1.0],
            mec=[0, 0, 0, 0],
            markerfacecoloralt=[*c_llvm, 1.0])]
    b.legend(
        handles=legend_elements,
        labels=["O2", "O0", "LLI", "Ours (baseline)", "+ calling convention", "+ register caching"],
        loc="lower center", numpoints=1, ncol=6, bbox_to_anchor=(0.5, 1.6),
        columnspacing=0.08, borderpad=0.3, handletextpad=0.2, edgecolor="black", fancybox=False)
    a.sharey(d)
    b.sharey(e)
    c.sharey(f)

    subfig2.supylabel("execution time [$s$]", y=1, x=0.05)
    subfig2.supxlabel("startup time [$s$]", y=-0.25)
    fig.savefig("results/microbm.pdf", bbox_inches="tight")


def viz_lingodb():
    headers = ["name", "QOpt", "lowerRelAlg", "lowerSubOp", "lowerDB", "lowerDSA",
               "lowerToLLVM", "toLLVMIR", "llvmOptimize", "codeGen", "executionTime", "total"]
    headers_mlir = ["name", "QOpt", "lowerRelAlg", "lowerSubOp",
                    "lowerDB", "lowerDSA", "codeGen", "executionTime", "total"]
    skipLines = 7
    folder = "results/lingodb/"
    mlir0 = pd.DataFrame(
        [map(lambda x: x if ".sql" in x else float(x), r.split())
         for r in open(folder + "time-mlir0.log").readlines()[skipLines:]],
        columns=headers_mlir)
    mlir = pd.DataFrame(
        [map(lambda x: x if ".sql" in x else float(x), r.split())
         for r in open(folder + "time-mlir.log").readlines()[skipLines:]],
        columns=headers_mlir)
    O0 = pd.DataFrame([map(lambda x: x if ".sql" in x else float(x), r.split())
                      for r in open(folder + "time-cheap.log").readlines()[skipLines:]], columns=headers)
    O2 = pd.DataFrame([map(lambda x: x if ".sql" in x else float(x), r.split())
                      for r in open(folder + "time-speed.log").readlines()[skipLines:]], columns=headers)
    O2 = O2.groupby("name").mean()
    O0 = O0.groupby("name").mean()
    mlir = mlir.groupby("name").mean()
    mlir0 = mlir0.groupby("name").mean()
    for df in [mlir, mlir0, O0, O2]:
        df.loc['Geomean'] = df.apply(lambda x: np.array(x).prod() ** (1 / len(x)), axis=0)
    distance = 0.2
    width = 0.2
    ranges = [
        list(np.arange(22) - 1.5 * distance) + [23 - 1.5 * distance],
        list(np.arange(22) - 0.5 * distance) + [23 - 0.5 * distance],
        list(np.arange(22) + 0.5 * distance) + [23 + 0.5 * distance],
        list(np.arange(22) + 1.5 * distance) + [23 + 1.5 * distance]]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.8, 2.6), sharex=True)
    opacity = 0.5
    ax1.bar(ranges[1], O0["codeGen"], width=width, color=[*c_O0, opacity])
    ax1.bar(ranges[0], O2["codeGen"] + O2["llvmOptimize"], width=width, color=[*c_O2, opacity])
    ax1.bar(ranges[3], mlir["codeGen"], width=width, color=[*c_builtin, opacity])
    ax1.bar(ranges[2], mlir0["codeGen"], width=width, color=[*c_llvm, opacity])
    ax1.set_ylabel("compile time\n[$ms$]")
    ax2.bar(ranges[0], O2["executionTime"], label="SPEED (O2)", width=width, color=c_O2)
    ax2.bar(ranges[1], O0["executionTime"], label="CHEAP (O0)", width=width, color=c_O0)
    ax2.bar(ranges[2], mlir0["executionTime"], label="Ours (LLVM dialect)", width=width, color=c_llvm)
    ax2.bar(ranges[3], mlir["executionTime"], label="Ours (built-in dialect)", width=width, color=c_builtin)
    ax2.set_ylabel("execution time\n[$ms$]")
    ax3.bar(ranges[1], O0["codeGen"], width=width, color=[*c_O0, opacity])
    ax3.bar(ranges[0], O2["codeGen"] + O2["llvmOptimize"], width=width, color=[*c_O2, opacity])
    ax3.bar(ranges[3], mlir["codeGen"], width=width, color=[*c_builtin, opacity])
    ax3.bar(ranges[2], mlir0["codeGen"], width=width, color=[*c_llvm, opacity])
    opacity = 1
    ax3.bar(ranges[1], O0["executionTime"],   bottom=O0["codeGen"], width=width, color=[*c_O0, opacity])
    ax3.bar(
        ranges[0],
        O2["executionTime"],
        bottom=O2["codeGen"] + O2["llvmOptimize"],
        width=width, color=[*c_O2, opacity])
    ax3.bar(ranges[3], mlir["executionTime"], bottom=mlir["codeGen"], width=width, color=[*c_builtin, opacity])
    ax3.bar(ranges[2], mlir0["executionTime"], bottom=mlir0["codeGen"], width=width, color=[*c_llvm, opacity])
    ax3.set_ylabel("query time\n[$ms$]\n")
    ax1.set_yscale("log")
    ax3.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.set_axisbelow(True)
    ax3.grid(axis='y')
    ax2.grid(axis='y')
    ax1.grid(axis='y')
    ax1.set_xlim(-1, 25)
    ax3.set_yticks([0, 20, 40, 60], [0, 20, 40, 60])
    ax1.axvline(x=22, color='black', linestyle=':')
    ax2.axvline(x=22, color='black', linestyle=':')
    ax3.axvline(x=22, color='black', linestyle=':')

    labels = plt.xticks(np.array(ranges[1]) + (0.5 * distance),
                        list(map(lambda x: "Q"+str(x), range(1, 23))) + ["Geomean"])
    labels[-1][-1].set_rotation(0)
    fig.legend(loc="lower center", ncol=5, bbox_to_anchor=(0.513, 0.88), edgecolor="black", fancybox=False)
    fig.savefig("results/lingodb.pdf", bbox_inches="tight")


def viz_coremark_and_spec():
    df_coremark = parse("results/coremark.log")
    df_coremark = df_coremark.groupby(
        ["mode", "command", "cmake"],
        as_index=False)[
        ["mode", "command", "cmake", "Preparation", "Execution", "Lowering"]].median(
        numeric_only=True)
    coremark_only = not os.path.exists("results/spec.log")
    if not coremark_only:
        dfO0_x86 = parse("results/spec.log")
        dfO0_x86 = dfO0_x86.where(dfO0_x86["cmake"].str.contains("EVICTION_STRATEGY=(1|2)"))
        ##########################
        tmp = dfO0_x86.groupby(
            ["mode", "command", "cmake"],
            as_index=False)[
            ["mode", "command", "cmake", "Preparation", "Execution", "Lowering"]].median(
            numeric_only=True)
        dfO0_x86 = tmp.groupby(["mode", "command", "cmake"], as_index=False)[
            ["cmake", "Preparation", "Execution", "Lowering"]].sum()
    else:
        dfO0_x86 = None

    def select(df, bm, mode, nobaseline=False, eviction_strat="(1|2)", baseline_df=None):
        if df is None:
            return 0, 0, 0
        pattern = bm if isinstance(bm, str) else f"spec{bm}.mlir"
        baseline_df = df if baseline_df is None else baseline_df
        prep, ex, lowering = df.where(
            df["mode"].str.contains(mode)).where(
            df["cmake"].str.contains("EVICTION_STRATEGY=" + eviction_strat)).where(
            df["command"].str.contains(pattern)).groupby(
                ["mode"])[["Preparation", "Execution", "Lowering"]].median().iloc[0]
        if nobaseline:
            return prep, ex, lowering
        else:
            assert (baseline_df is not None)
            baseline = select(baseline_df, bm, "MLIRExecEngineOpt2", True)
            return prep / baseline[0], ex / baseline[1], lowering / baseline[0]

    def diagram(prep_ax, run_ax, df, df_coremark=None):
        distance = 0.2
        width = 0.2
        bms = 5
        ranges = [
            list(np.arange(bms) - 1 * distance) + [bms - distance + 0.5],
            list(np.arange(bms)) + [bms + 0.5],
            list(np.arange(bms) + 1 * distance) + [bms + distance + 0.5]]
        data = np.asarray([
            select(df, 505, "MLIRExecEngineOpt0", baseline_df=df),
            select(df, 525, "MLIRExecEngineOpt0", baseline_df=df),
            select(df, 548, "MLIRExecEngineOpt0", baseline_df=df),
            select(df, 557, "MLIRExecEngineOpt0", baseline_df=df),
            select(df_coremark, ".*", "MLIRExecEngineOpt0"),
            select(df, 505, "MLIRExecEngineOpt2", baseline_df=df),
            select(df, 525, "MLIRExecEngineOpt2", baseline_df=df),
            select(df, 548, "MLIRExecEngineOpt2", baseline_df=df),
            select(df, 557, "MLIRExecEngineOpt2", baseline_df=df),
            select(df_coremark, ".*", "MLIRExecEngineOpt2"),
            select(df, 505, "JIT", baseline_df=df),
            select(df, 525, "JIT", baseline_df=df),
            select(df, 548, "JIT", baseline_df=df),
            select(df, 557, "JIT", baseline_df=df),
            select(df_coremark, ".*", "JIT")
        ], dtype=np.float64)

        def add_geomean(arr):
            geomean = arr.prod() ** (1. / len(arr))
            return np.append(arr, geomean)
        prep_ax.bar(ranges[0], add_geomean(data[bms:2*bms, 0]), width=width, label="O2", color=c_O2)
        prep_ax.bar(ranges[1], add_geomean(data[:bms, 0]), width=width, label="O0", color=c_O0)
        prep_ax.bar(ranges[2], add_geomean(data[2*bms:3*bms, 0]), width=width, label="Ours", color=c_llvm)
        prep_ax.set_yscale("log")
        run_ax.bar(ranges[0], add_geomean(data[bms:2*bms, 1]), width=width, label="O2", color=c_O2)
        run_ax.bar(ranges[1], add_geomean(data[:bms, 1]), width=width, label="O0", color=c_O0)
        run_ax.bar(ranges[2], add_geomean(data[2*bms:3*bms, 1]), width=width, label="Ours", color=c_llvm)
        run_ax.set_xticks(ranges[1], ["SPEC 505\nmcf", "SPEC 525\nx264", "SPEC 548\nexchange2",
                          "SPEC 557\nxz", "CoreMark", "Geomean"], rotation=45)
        run_ax.get_xticklabels()[-1].set_rotation(0)

    fig, (compile_axes, run_axes) = plt.subplots(2, 1, figsize=(7.5, 2.), sharex=True, sharey='row')
    compile_axes.set_axisbelow(True)
    run_axes.set_yticks([2, 4, 6, 8, 10, 12, 14], [2, 4, 6, 8, 10, 12, 14])
    run_axes.set_axisbelow(True)
    run_axes.grid(axis='y')
    compile_axes.grid(axis='y')
    diagram(compile_axes, run_axes, dfO0_x86, df_coremark)
    handles, labels = run_axes.get_legend_handles_labels()
    run_axes.legend(
        [handles[1],
         handles[2],
         handles[0]],
        [labels[1],
         labels[2],
         labels[0]],
        loc="lower center", ncol=5, bbox_to_anchor=(0.5, 2.35),
        edgecolor="black", fancybox=False)
    compile_axes.set_ylabel("normalized\ncompile time")
    run_axes.set_ylabel("normalized\nexecution time")
    compile_axes.xaxis.set_label_position('top')
    run_axes.set_ylim(0, 15)
    run_axes.axvline(5 - 0.25, color="black", linestyle=":")
    compile_axes.axvline(5 - 0.25, color="black", linestyle=":")
    fig.savefig("results/coremark.pdf" if coremark_only else "results/spec.pdf", bbox_inches="tight")


files = os.listdir("results/")
for result in files:
    if 'lingodb' in result and os.path.isdir("results/lingodb"):
        viz_lingodb()
    if not result.endswith(".log"):
        continue
    if 'benchmark' in result:
        viz_microbm()
    elif 'polybench' in result:
        viz_polybench()
    elif 'coremark' in result:
        viz_coremark_and_spec()
