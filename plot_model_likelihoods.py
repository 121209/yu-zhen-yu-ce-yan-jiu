"""在仓库根目录运行 python plot_model_likelihoods.py，生成两张 PNG 柱状图。

图 1：各数据集的测试集联合总对数似然，每个子图独立标尺。
图 2：各数据集内平均每事件的区域时间项、条件空间项，单位 nats/event。
输入为两次评估的 summary.csv，无需 PyTorch 或逐事件 CSV。
"""
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASETS = ("SCEDC_30", "SCEDC_25", "SCEDC_20", "SaltonSea", "SanJac", "WHITE", "ComCat")
# 顺序和颜色在两张图中共用。
MODELS = {
    "etas": ("ETAS", "#939BA3"),
    "fcnn": ("ST-NKF (FCNN)", "#35445C"),
    "scheme1": ("Scheme 1", "#337FA5"),
    "scheme2": ("Scheme 2", "#479D8E"),
    "scheme3": ("Scheme 3", "#9A83BA"),
    "scheme4": ("Scheme 4", "#D49155"),
    "scheme5": ("Scheme 5", "#B66978"),
}
TOTAL = "LL_ts_total"
TIME = "LL_t_region_mean"
SPACE = "LL_s_given_t_region_mean"


def read_scores():
    scores = {}
    for folder in ("decomposition", "etas_decomposition"):
        path = ROOT / "results" / folder / "summary.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["dataset"], row["model"]
                if key[0] not in DATASETS or key[1] not in MODELS or int(row["seed"]) != 1:
                    continue
                if key in scores:
                    raise ValueError(f"重复的结果行：{key}")
                values = {field: float(row[field]) for field in (TOTAL, TIME, SPACE)}
                n = int(row["n_test"])
                if n <= 0 or not all(math.isfinite(x) for x in values.values()):
                    raise ValueError(f"无效的事件数或得分：{key}")
                if not math.isclose(values[TOTAL] / n, values[TIME] + values[SPACE],
                                    rel_tol=0, abs_tol=1e-5):
                    raise ValueError(f"区域分解与联合得分不一致：{key}")
                scores[key] = dict(values, n=n)
    for dataset in DATASETS:
        missing = [m for m in MODELS if (dataset, m) not in scores]
        if missing:
            raise ValueError(f"{dataset} 缺少 {', '.join(missing)}")
        if len({scores[dataset, m]["n"] for m in MODELS}) != 1:
            raise ValueError(f"{dataset} 的测试事件数不一致")
    return scores


def style_axis(ax, grid_axis):
    ax.set_axisbelow(True)
    ax.grid(axis=grid_axis, color="#E1E5E9", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=4)


def plot_totals(plt, scores, output):
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    fig, axes = plt.subplots(4, 2, figsize=(13, 13), layout="constrained")
    fig.suptitle("Joint test log-likelihood by dataset", fontsize=19, fontweight="bold")
    for dataset, ax in zip(DATASETS, axes.flat):
        values = [scores[dataset, m][TOTAL] for m in MODELS]
        n = scores[dataset, "etas"]["n"]
        bars = ax.barh(range(len(MODELS)), values, color=[v[1] for v in MODELS.values()])
        ax.set_yticks(range(len(MODELS)), [v[0] for v in MODELS.values()])
        ax.invert_yaxis()
        ax.set_title(f"{dataset}   |   n = {n:,}", loc="left", fontsize=12)
        ax.set_xlabel("Total joint log-likelihood (nats)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        # 单独留出右侧数值栏，避免接近的总分难以读出。
        for bar, score in zip(bars, values):
            ax.text(1.02, bar.get_y() + bar.get_height()/2, f"{score:,.1f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=9)
        ax.axvline(0, color="#606B74", linewidth=0.8)
        style_axis(ax, "x")
    note = axes.flat[-1]
    note.axis("off")
    note.text(0.04, 0.6, "Higher scores are better.\nCompare models within each dataset.\n"
              "Panels use independent scales.\nH = 128   /   seed = 1",
              transform=note.transAxes, fontsize=12, linespacing=1.6, va="center")
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def plot_components(plt, scores, output):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.3), sharex=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.82, bottom=0.12, hspace=0.32)
    fig.suptitle("Regional decomposition of joint test log-likelihood",
                 fontsize=19, fontweight="bold", y=0.97)
    width = 0.115
    for ax, field, title in zip(axes, (TIME, SPACE),
                                ("Temporal log-likelihood", "Conditional spatial log-likelihood")):
        for j, (model, (label, color)) in enumerate(MODELS.items()):
            x = [i + (j - (len(MODELS)-1)/2) * width for i in range(len(DATASETS))]
            ax.bar(x, [scores[d, model][field] for d in DATASETS],
                   width=width * 0.93, label=label, color=color)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.set_ylabel("nats / event")
        ax.set_xlim(-0.52, len(DATASETS)-0.48)
        ax.axhline(0, color="#606B74", linewidth=0.8)
        style_axis(ax, "y")
    axes[-1].set_xticks(range(len(DATASETS)),
                        [f"{d}\nn = {scores[d, 'etas']['n']:,}" for d in DATASETS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="upper center",
               bbox_to_anchor=(0.52, 0.92), frameon=False)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def main():
    scores = read_scores()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = ROOT / "results" / "likelihood_figures"
    output.mkdir(parents=True, exist_ok=True)
    for draw, name in ((plot_totals, "joint_total.png"),
                       (plot_components, "regional_decomposition_per_event.png")):
        draw(plt, scores, output / name)
        print(output / name)


if __name__ == "__main__":
    main()
