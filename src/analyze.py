"""Aggregate raw per-seed CSVs, apply the pre-registered decision rule, draw the figures.

    python src/analyze.py --config configs/default.yaml

Writes to results/: results.csv (ablation), probe.csv, erasure.csv, summary.csv, gap_by_layer.csv,
verdict.md, figure1.png (primary result), figure2_probe_layers.png (decodability by layer).
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import load_config, resolve  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, validated all-pairs
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
LANG_NAME = {"bn": "Bengali", "hi": "Hindi", "en": "English"}
READOUT = "acc"  # accuracy column the decision rule uses; set from cfg["readout"] in run()


def collect(results_dir, name: str) -> pd.DataFrame:
    files = sorted((results_dir / "raw").glob(f"*/seed*/{name}.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def run_meta(results_dir) -> dict:
    meta = {}
    for f in sorted((results_dir / "raw").glob("*/seed*/run_meta.json")):
        with open(f, encoding="utf-8") as fh:
            m = json.load(fh)
        meta[m["model"]] = m
    return meta


def drops(abl: pd.DataFrame, model: str, concept: str, layer: int, rank: int) -> pd.DataFrame:
    """Per-seed accuracy drop, columns (direction, eval_lang), informative context."""
    df = abl[(abl.model == model) & (abl.concept == concept) & (abl.context == "informative")]
    clean = df[df.direction == "none"][["seed", "eval_lang", READOUT]].rename(columns={READOUT: "acc_clean"})
    cell = df[(df.layer == layer) & (df["rank"] == rank)].merge(clean, on=["seed", "eval_lang"])
    cell = cell.assign(drop=cell.acc_clean - cell[READOUT])
    return cell.pivot_table(index="seed", columns=["direction", "eval_lang"], values="drop")


def metrics_for(abl: pd.DataFrame, model: str, cfg: dict, layer: int, rank: int) -> dict[str, pd.Series]:
    s, tg, idl = cfg["source_lang"], cfg["target_lang"], cfg["identity_lang"]
    clean = abl[(abl.model == model) & (abl.concept == "side") & (abl.context == "informative")
                & (abl.direction == "none")].pivot_table(index="seed", columns="eval_lang", values=READOUT)
    side = drops(abl, model, "side", layer, rank)
    m = {
        "clean_acc_src": clean[s],
        "clean_acc_tgt": clean[tg],
        **({"clean_acc_identity": clean[idl]} if idl in clean.columns else {}),
        "drop_src": side[(s, s)],
        "drop_tgt": side[(s, tg)],
        "drop_random_src": side[("random", s)],
        "gap": side[(s, s)] - side[(s, tg)],
        "specificity": side[(s, s)] - side[("random", s)],
    }
    if (s, idl) in side.columns:
        m["drop_identity"] = side[(s, idl)]
    if "gender" in set(abl.concept):
        gen = drops(abl, model, "gender", layer, rank)
        m["gap_matched"] = gen[(s, s)] - gen[(s, tg)]
    return m


def decide(mean: dict, t: dict) -> tuple[str, list[str]]:
    pts = lambda x: f"{100 * x:.1f} pts"  # noqa: E731
    if mean["clean_acc_src"] < t["gate_clean_acc"] or mean["clean_acc_tgt"] < t["gate_clean_acc"]:
        return "INCONCLUSIVE", [f"task gate failed: clean accuracy src {mean['clean_acc_src']:.2f}, "
                                f"tgt {mean['clean_acc_tgt']:.2f} (need >= {t['gate_clean_acc']})"]
    falsified = []
    if mean["specificity"] < t["specificity_min"]:
        falsified.append(f"F1 concept direction is no worse than random (specificity {pts(mean['specificity'])})")
    if mean["gap"] < t["falsify_gap"]:
        falsified.append(f"F2 cross-language gap {pts(mean['gap'])} < {pts(t['falsify_gap'])}: shared encoding")
    if "gap_matched" in mean and mean["gap_matched"] >= mean["gap"] - t["matched_margin"]:
        falsified.append(f"F3 matched concept shows the same gap ({pts(mean['gap_matched'])}): "
                         f"language-channel artifact")
    if falsified:
        return "FALSIFIED", falsified
    if mean["gap"] >= t["support_gap"] and mean["drop_random_src"] <= t["random_max_drop"]:
        return "SUPPORTED", [f"gap {pts(mean['gap'])} >= {pts(t['support_gap'])}, "
                             f"random drop {pts(mean['drop_random_src'])} <= {pts(t['random_max_drop'])}"]
    return "INCONCLUSIVE", ["between the support and falsification thresholds"]


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figure1(abl, cfg, meta, verdicts, path) -> None:
    s, rank = cfg["source_lang"], cfg["ablation"]["primary_rank"]
    langs = [l for l in (s, cfg["identity_lang"], cfg["target_lang"]) if l in set(abl.eval_lang)]
    models = list(verdicts)
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 4.4), sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, model in zip(axes[0], models):
        layer = meta[model]["primary_layer"]
        side = drops(abl, model, "side", layer, rank)
        bars = [(f"{LANG_NAME.get(s, s)} side direction", side, s),
                ("random direction", side, "random")]
        if "gender" in set(abl.concept):
            bars.append((f"{LANG_NAME.get(s, s)} gender direction (matched control)",
                         drops(abl, model, "gender", layer, rank), s))
        x = np.arange(len(langs))
        w = 0.8 / len(bars)
        for k, (label, table, direction) in enumerate(bars):
            vals = np.array([100 * table[(direction, l)].mean() for l in langs])
            errs = np.array([100 * table[(direction, l)].std(ddof=0) for l in langs])
            ax.bar(x + (k - (len(bars) - 1) / 2) * w, vals, width=w - 0.03, color=SERIES[k], label=label,
                   yerr=errs, error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
        style(ax)
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xticks(x, [LANG_NAME.get(l, l) for l in langs], color=INK)
        status, gap = verdicts[model]
        ax.set_title(f"{model} (block {layer})\n{status}: gap {100 * gap[0]:.1f} ± {100 * gap[1]:.1f} pts",
                     fontsize=10, color=INK, loc="left")
    axes[0][0].set_ylabel("Accuracy drop after ablation (points)", color=INK2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def figure2(probe, cfg, path) -> None:
    same = probe[(probe.train_lang == probe.test_lang) & (probe.train_context == probe.test_context)]
    models, concepts = list(dict.fromkeys(same.model)), list(dict.fromkeys(same.concept))
    fig, axes = plt.subplots(len(models), len(concepts), figsize=(4.6 * len(concepts), 3.2 * len(models)),
                             sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, model in enumerate(models):
        for c, concept in enumerate(concepts):
            ax = axes[r][c]
            df = same[(same.model == model) & (same.concept == concept)]
            for k, lang in enumerate(cfg["langs"]):
                for ctx, dash in (("informative", "-"), ("neutral", "--")):
                    sub = df[(df.test_lang == lang) & (df.test_context == ctx) & (df.control == "none")]
                    curve = sub.groupby("layer").acc.mean()
                    # widths shrink with k so a line drawn later never fully hides one at the same value
                    ax.plot(curve.index, curve.values, dash, color=SERIES[k], linewidth=4.5 - 1.5 * k,
                            label=f"{LANG_NAME.get(lang, lang)}, {ctx} context")
            shuf = df[df.control == "shuffled"].groupby("layer").acc.mean()
            ax.plot(shuf.index, shuf.values, ":", color=MUTED, linewidth=2, label="shuffled-label control")
            style(ax)
            ax.set_ylim(0.3, 1.02)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{model} / {concept}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("decoder block", color=INK2, fontsize=9)
        axes[r][0].set_ylabel("probe accuracy (test names)", color=INK2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def run(cfg: dict) -> None:
    global READOUT
    READOUT = cfg.get("readout", "acc")
    out = resolve(cfg["results_dir"])
    abl, probe, erasure = collect(out, "ablation"), collect(out, "probe"), collect(out, "erasure")
    if abl.empty:
        print(f"no raw results under {out / 'raw'}")
        return
    meta = run_meta(out)
    abl.to_csv(out / "results.csv", index=False)
    probe.to_csv(out / "probe.csv", index=False)
    erasure.to_csv(out / "erasure.csv", index=False)

    t, rank = cfg["thresholds"], cfg["ablation"]["primary_rank"]
    summary, lines, verdicts, gap_rows = [], [], {}, []
    lines += ["# Verdict", "",
              f"Primary cell: concept `side`, informative context, primary block, rank {rank}, "
              f"direction fitted on `{cfg['source_lang']}`. Readout `{READOUT}`. Values are mean ± std over seeds, in accuracy points.", ""]
    for model in dict.fromkeys(abl.model):
        layer = meta[model]["primary_layer"]
        m = metrics_for(abl, model, cfg, layer, rank)
        mean = {k: float(v.mean()) for k, v in m.items()}
        for k, v in m.items():
            summary.append(dict(model=model, layer=layer, rank=rank, metric=k, mean=v.mean(),
                                std=v.std(ddof=0), n_seeds=int(v.count())))
        status, reasons = decide(mean, t)
        verdicts[model] = (status, (mean["gap"], float(m["gap"].std(ddof=0))))

        lines += [f"## {model} (block {layer})", "", f"**{status}**", ""] + [f"- {r}" for r in reasons] + [""]
        lines += ["| metric | mean | std |", "|---|---|---|"]
        for k, v in m.items():
            scale = 1 if k.startswith("clean") else 100
            lines.append(f"| {k} | {scale * v.mean():.2f} | {scale * v.std(ddof=0):.2f} |")
        if "drop_identity" in mean and mean.get("clean_acc_identity", 1.0) < t["gate_clean_acc"]:
            lines += ["", f"Language-identity control: not interpretable ({cfg['identity_lang']} clean accuracy "
                          f"{mean['clean_acc_identity']:.2f} < gate {t['gate_clean_acc']})."]
        elif "drop_identity" in mean:
            tracks = abs(mean["drop_identity"] - mean["drop_src"]) < abs(mean["drop_identity"] - mean["drop_tgt"])
            lines += ["", f"Language-identity control: the {cfg['identity_lang']} drop is closer to the "
                          f"{'source' if tracks else 'target'} language -> effect "
                          f"{'tracks the lexical distinction' if tracks else 'looks specific to one language'}."]
        for extra in cfg["ablation"]["extra_ranks"]:
            e = metrics_for(abl, model, cfg, layer, extra)
            lines.append(f"Sensitivity, rank {extra}: gap {100 * e['gap'].mean():.1f} ± "
                         f"{100 * e['gap'].std(ddof=0):.1f}, random drop {100 * e['drop_random_src'].mean():.1f}")
        lines.append("")

        for concept in dict.fromkeys(abl.concept):
            for lyr in sorted(l for l in abl[abl.model == model].layer.unique() if l >= 0):
                d = drops(abl, model, concept, lyr, rank)
                s, tg = cfg["source_lang"], cfg["target_lang"]
                if (s, s) in d.columns and (s, tg) in d.columns:
                    g = d[(s, s)] - d[(s, tg)]
                    gap_rows.append(dict(model=model, concept=concept, layer=lyr, rank=rank,
                                         gap_mean=g.mean(), gap_std=g.std(ddof=0),
                                         random_drop_src=d[("random", s)].mean()))

    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(out / "gap_by_layer.csv", index=False)
    (out / "verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    figure1(abl, cfg, meta, verdicts, out / "figure1.png")
    figure2(probe, cfg, out / "figure2_probe_layers.png")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
