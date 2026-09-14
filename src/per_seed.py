"""Per-seed primary-cell numbers for the tested concept and the matched control (gender).

    python src/per_seed.py --config configs/backup.yaml
    python src/per_seed.py --config configs/amended.yaml

Reads results.csv written by analyze.py. Use it to see how many seeds clear the threshold and whether
the matched control is at ceiling (clean accuracy 1.0 and drops of exactly 0).
"""
from __future__ import annotations

import argparse

import pandas as pd

import analyze
from utils import load_config, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/amended.yaml")
    cfg = load_config(ap.parse_args().config)
    analyze.READOUT = cfg.get("readout", "acc")
    analyze.PRIMARY = cfg.get("primary_concept", "side")

    out = resolve(cfg["results_dir"])
    abl = pd.read_csv(out / "results.csv")
    meta = analyze.run_meta(out)
    s, tg, rank = cfg["source_lang"], cfg["target_lang"], cfg["ablation"]["primary_rank"]
    threshold = 100 * cfg["thresholds"]["support_gap"]

    for model in dict.fromkeys(abl.model):
        layer = meta[model]["primary_layer"]
        for concept in dict.fromkeys([analyze.PRIMARY, "gender"]):
            d = analyze.drops(abl, model, concept, layer, rank)
            t = pd.DataFrame({"drop_src": d[(s, s)], "drop_tgt": d[(s, tg)], "drop_random_src": d[("random", s)]}) * 100
            t["gap"] = t.drop_src - t.drop_tgt
            clean = abl[(abl.model == model) & (abl.concept == concept) & (abl.context == "informative")
                        & (abl.direction == "none")].pivot_table(index="seed", columns="eval_lang",
                                                                 values=analyze.READOUT)
            t["clean_src"], t["clean_tgt"] = clean[s], clean[tg]
            print(f"\n{model} / {concept} / block {layer} / {analyze.READOUT}")
            print(t.round(2).to_string())
            if concept == analyze.PRIMARY:
                print(f"seeds with gap >= {threshold:.0f} pts: {int((t.gap >= threshold).sum())} of {len(t)}")


if __name__ == "__main__":
    main()
