"""Frozen follow-up to the aunt-side lead: a second model family and a non-ceiling matched control, one command.

    python src/followup.py                                         # configs/followup.yaml (GPU)
    python src/followup.py --config configs/followup_smoke.yaml    # offline plumbing check, tiny random model (CPU)

Stage 1, no ablation. Cache un-ablated scores and apply two fixed rules, in config order:
  model    first second-family candidate whose all-names pair accuracy on the primary concept is >= the gate
           in the source and target language
  control  first control candidate whose all-names pair accuracy is inside [gate, control_ceiling] in the source
           and target language, in every follow-up model (anchor + selected)
Stage 2: the unchanged pipeline (run_all.run_model, analyze.run) on anchor + selected model with the selected
control as matched concept, then the follow-up outcome rule (outcome()). Nothing is picked by hand.
Everything lands in results_followup/: screen.json, followup_verdict.md, and the usual analyze.py outputs.
"""
from __future__ import annotations

import sentencepiece  # noqa: F401  must load before torch/sklearn (see run_all.py)

import argparse
import json
import subprocess
import sys

import numpy as np
import torch

import analyze
from build_stimuli import build_all
from models import pick_device, pick_dtype
from run_all import pair_accuracy, run_model
from utils import load_config, read_jsonl, resolve, slug, write_jsonl


def load_items(cfg: dict) -> list[dict]:
    concepts = [cfg["primary_concept"], *cfg["control_candidates"]]
    path = resolve(cfg["stimuli"])
    if not path.exists():
        write_jsonl(build_all(concepts, cfg["langs"]), path)
    return [it for it in read_jsonl(path) if it["concept"] in concepts and it["lang"] in cfg["langs"]]


def clean_pair_acc(cfg: dict, name: str, items: list[dict]) -> dict[str, float]:
    """All-names, no-ablation pair accuracy per 'concept/lang' in the screening context, from the cached scores."""
    blob = torch.load(resolve(cfg["cache_dir"]) / slug(name) / "baseline_logp.pt")
    if blob["ids"] != [it["id"] for it in items]:
        raise ValueError(f"{name}: cached baseline was built from different items")
    logp = blob["logp"].numpy()
    out = {}
    for concept in [cfg["primary_concept"], *cfg["control_candidates"]]:
        for lang in cfg["langs"]:
            idx = np.array([k for k, it in enumerate(items) if it["concept"] == concept and it["lang"] == lang
                            and it["context"] == cfg["screen"]["context"]])
            out[f"{concept}/{lang}"] = pair_accuracy(logp[idx, 1] - logp[idx, 0],
                                                     np.array([items[k]["label"] for k in idx]),
                                                     np.array([items[k]["pair_id"] for k in idx]))
    return out


def passes_gate(acc: dict, concept: str, cfg: dict) -> bool:
    gate = cfg["thresholds"]["gate_clean_acc"]
    return all(acc[f"{concept}/{lang}"] >= gate for lang in (cfg["source_lang"], cfg["target_lang"]))


def off_ceiling(acc: dict, concept: str, cfg: dict) -> bool:
    gate, ceiling = cfg["thresholds"]["gate_clean_acc"], cfg["screen"]["control_ceiling"]
    return all(gate <= acc[f"{concept}/{lang}"] <= ceiling for lang in (cfg["source_lang"], cfg["target_lang"]))


def select(accs: dict[str, dict], cfg: dict) -> tuple[str | None, str | None]:
    """The two stage-1 rules. accs holds every model screened so far; candidates are tried in config order."""
    model = next((m for m in cfg["second_family_candidates"]
                  if m in accs and passes_gate(accs[m], cfg["primary_concept"], cfg)), None)
    if model is None:
        return None, None
    models = list(dict.fromkeys([cfg["anchor_model"], model]))
    control = next((c for c in cfg["control_candidates"] if all(off_ceiling(accs[m], c, cfg) for m in models)), None)
    return model, control


def outcome(per_model: dict[str, dict]) -> str:
    """SURVIVED: every model SUPPORTED with an informative control. NULL: every model FALSIFIED. Else MIXED."""
    if all(m["survives"] for m in per_model.values()):
        return "SURVIVED"
    if all(m["verdict"] == "FALSIFIED" for m in per_model.values()):
        return "NULL"
    return "MIXED"


def check_pins(cfg: dict) -> None:
    """Every model the screen may select must be pinned to a full Hugging Face commit SHA."""
    revs = cfg.get("revisions", {})
    bad = [m for m in [cfg["anchor_model"], *cfg["second_family_candidates"]]
           if m != "__tiny_random__" and not (isinstance(revs.get(m), str) and len(revs[m]) == 40)]
    if bad:
        raise SystemExit(f"unpinned models (add a 40-char commit to revisions:): {bad}")


def check_env(cfg: dict, out) -> None:
    """Record the installed packages and stop if they differ from the committed lock."""
    lock = cfg.get("env_lock")
    if lock is None:
        return
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (out / "env_freeze.txt").write_text(frozen, encoding="utf-8")
    want = {l.strip() for l in resolve(lock).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    have = {l.strip() for l in frozen.splitlines() if l.strip()}
    if want != have:
        raise SystemExit(f"environment differs from {lock}: missing {sorted(want - have)}, extra {sorted(have - want)}")


def write_verdict(out, lines: list[str]) -> None:
    (out / "followup_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/followup.yaml")
    ap.add_argument("--batch-size", type=int, help="only if out of memory; default: config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(cfg["device"])
    items = load_items(cfg)
    out = resolve(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    check_pins(cfg)
    check_env(cfg, out)
    bs = args.batch_size or cfg["batch_size"]
    primary, anchor = cfg["primary_concept"], cfg["anchor_model"]
    screen_cfg = {**cfg, "concepts": [primary, *cfg["control_candidates"]]}

    def dtype(name):
        return pick_dtype(cfg.get("model_dtype", {}).get(name, cfg["dtype"]), device)

    # ---- stage 1: no-ablation screen
    accs: dict[str, dict] = {}
    for name in [anchor, *cfg["second_family_candidates"]]:
        if name not in accs:
            run_model(name, screen_cfg, items, [], device, dtype(name), bs, baseline_only=True)
            accs[name] = clean_pair_acc(cfg, name, items)
        if name != anchor and passes_gate(accs[name], primary, cfg):
            break
    selected, control = select(accs, cfg)
    models = list(dict.fromkeys([anchor, selected])) if selected else [anchor]
    with open(out / "screen.json", "w", encoding="utf-8") as f:
        json.dump({"anchor_model": anchor, "selected_model": selected, "control": control,
                   "screened_models": list(accs), "clean_pair_acc_all_names": accs,
                   "context": cfg["screen"]["context"], "gate": cfg["thresholds"]["gate_clean_acc"],
                   "control_ceiling": cfg["screen"]["control_ceiling"]}, f, indent=2)

    head = ["# Follow-up verdict", "", f"Config: `{args.config}`. Screen: `screen.json`.", "",
            "| model | " + " | ".join(f"{c}/{l}" for c in screen_cfg["concepts"] for l in cfg["langs"]) + " |",
            "|---|" + "---|" * len(screen_cfg["concepts"]) * len(cfg["langs"])]
    head += [f"| {m} | " + " | ".join(f"{a[f'{c}/{l}']:.2f}" for c in screen_cfg["concepts"] for l in cfg["langs"])
             + " |" for m, a in accs.items()]
    head += ["", f"No-ablation pair accuracy, all names, {cfg['screen']['context']} context.",
             f"Selected second-family model: `{selected}`. Selected matched control: `{control}`.", ""]
    if selected is None or control is None:
        why = "no second-family candidate passed the task gate" if selected is None else \
            "no control candidate was off ceiling in every follow-up model"
        write_verdict(out, head + ["## Outcome: NOT TESTABLE", "", f"- {why}; no ablation was run."])
        return

    # ---- stage 2: unchanged pipeline
    run_cfg = {**cfg, "models": models, "concepts": [primary, control], "matched_concept": control}
    for name in models:
        run_model(name, run_cfg, items, cfg["seeds"], device, dtype(name), bs)
    analyze.run(run_cfg)

    abl, meta = analyze.collect(out, "ablation"), analyze.run_meta(out)
    s, t, rank = cfg["source_lang"], cfg["thresholds"], cfg["ablation"]["primary_rank"]
    per_model, rows = {}, []
    for name in models:
        layer = meta[name]["primary_layer"]
        mean = {k: float(v.mean()) for k, v in analyze.metrics_for(abl, name, run_cfg, layer, rank).items()}
        verdict, _ = analyze.decide(mean, t)
        ctrl = analyze.drops(abl, name, control, layer, rank)
        ctrl_spec = float((ctrl[(s, s)] - ctrl[("random", s)]).mean())
        informative = ctrl_spec >= t["specificity_min"]
        per_model[name] = dict(verdict=verdict, survives=verdict == "SUPPORTED" and informative)
        rows.append(f"| {name} | {layer} | {100 * mean['gap']:.1f} | {100 * mean['drop_src']:.1f} | "
                    f"{100 * mean['drop_tgt']:.1f} | {100 * mean['drop_random_src']:.1f} | "
                    f"{100 * mean['gap_matched']:.1f} | {100 * ctrl_spec:.1f} ({'informative' if informative else 'NOT informative'}) | "
                    f"{verdict} | {'yes' if per_model[name]['survives'] else 'no'} |")
    write_verdict(out, head + [
        f"## Outcome: {outcome(per_model)}", "",
        f"Primary cell: `{primary}`, informative context, block n_layers // 2, rank {rank}, direction fitted on `{s}`, "
        f"readout `{cfg['readout']}`, seeds {cfg['seeds']}. Points, mean over seeds; per-model reasons in `verdict.md`.", "",
        "| model | block | G | drop bn | drop en | random drop bn | control gap | control specificity | verdict | survives |",
        "|---|---|---|---|---|---|---|---|---|---|", *rows, "",
        "Rule (frozen): a model survives if `analyze.decide` returns SUPPORTED and the control's own direction beats a "
        f"random direction in `{s}` by >= {100 * t['specificity_min']:.0f} points (otherwise F3 cannot fire and is vacuous). "
        "SURVIVED = every model survives; NULL = every model FALSIFIED; MIXED = anything else."])


if __name__ == "__main__":
    main()
