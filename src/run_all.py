"""Single entry point: stimuli -> activations -> probes -> ablations + controls -> raw CSVs -> analysis.

    python src/run_all.py --seed 0                       # one seed, every model in the config
    python src/run_all.py                                # every seed in the config (0-4)
    python src/run_all.py --config configs/smoke.yaml    # offline pipeline check on a tiny random model (CPU)

Seed-independent work (activation extraction, un-ablated scoring) is cached under cache/<model>/.
Per-seed raw numbers go to results/raw/<model>/seed<k>/, then analyze.py aggregates everything.
"""
from __future__ import annotations

import sentencepiece  # noqa: F401  must load before torch/sklearn: importing it after them crashes on Windows

import argparse
import json
import platform
import time

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

import analyze
from ablate import ScoringSet, SubspaceAblation, score
from build_stimuli import build_all
from extract import extract
from lexicon import CONTEXTS, NAMES
from models import get_blocks, load_model, pick_device, pick_dtype
from probe import Probe, erase, inlp_basis, random_basis
from utils import load_config, name_split, read_jsonl, resolve, set_seed, slug, write_jsonl


def load_items(cfg: dict) -> list[dict]:
    path = resolve(cfg["stimuli"])
    if not path.exists():
        write_jsonl(build_all(), path)
    return [it for it in read_jsonl(path) if it["concept"] in cfg["concepts"] and it["lang"] in cfg["langs"]]


def cached(path, ids, compute, tag: str):
    """Reuse a cache only if it was built from the same items with the same dtype."""
    if path.exists():
        blob = torch.load(path)
        if blob["ids"] == ids and blob.get("tag") == tag:
            return blob
    blob = compute()
    blob["tag"] = tag
    torch.save(blob, path)
    return blob


def pair_accuracy(prefer1: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> float:
    """Share of minimal pairs where the label-1 item prefers answer 1 more than its label-0 partner.
    A constant answer preference cancels; identical prompts tie and count 0.5."""
    by_pair: dict = {}
    for p, y, v in zip(pair_ids, labels, prefer1):
        by_pair.setdefault(p, {})[int(y)] = v
    d = np.array([v[1] - v[0] for v in by_pair.values() if len(v) == 2])
    return float(np.mean(np.where(np.abs(d) < 1e-6, 0.5, d > 0))) if len(d) else float("nan")


def group_rows(logp: np.ndarray, idx: np.ndarray, meta: dict, row: dict) -> list[dict]:
    y = meta["label"][idx]
    r = np.arange(len(idx))
    margin = logp[idx][r, y] - logp[idx][r, 1 - y]
    prefer1 = logp[idx, 1] - logp[idx, 0]
    out = []
    for ctx in CONTEXTS:
        for lang in np.unique(meta["lang"][idx]):
            m = (meta["context"][idx] == ctx) & (meta["lang"][idx] == lang)
            if m.any():
                out.append({**row, "context": ctx, "eval_lang": lang, "acc": float((margin[m] > 0).mean()),
                            "pair_acc": pair_accuracy(prefer1[m], y[m], meta["pair_id"][idx][m]),
                            "logit_diff": float(margin[m].mean()), "n": int(m.sum())})
    return out


def cell(meta: dict, concept: str, context: str, lang: str) -> np.ndarray:
    return (meta["concept"] == concept) & (meta["context"] == context) & (meta["lang"] == lang)


def run_probes(X, meta, train, test, cfg, model_name, seed, n_layers, abl_layers, primary, ranks):
    """Probe accuracy for every (concept, train cell, layer) on every test cell, plus shuffled-label control.
    Returns rows and the concept subspaces fitted on informative-context items of each source language."""
    C, iters = cfg["probe"]["C"], cfg["probe"]["max_iter"]
    rng = np.random.default_rng(seed)
    rows, dirs = [], {}
    for concept in cfg["concepts"]:
        for tr_ctx in CONTEXTS:
            for tr_lang in cfg["langs"]:
                tr = np.where(cell(meta, concept, tr_ctx, tr_lang) & train)[0]
                y_tr = meta["label"][tr]
                y_shuf = rng.permutation(y_tr)
                for layer in tqdm(range(n_layers), desc=f"probe {concept}/{tr_ctx}/{tr_lang}", leave=False):
                    probe = Probe(X[tr, layer], y_tr, C, iters)
                    base = dict(model=model_name, seed=seed, concept=concept, layer=layer,
                                train_context=tr_ctx, train_lang=tr_lang)
                    for te_ctx in CONTEXTS:
                        for te_lang in cfg["langs"]:
                            te = np.where(cell(meta, concept, te_ctx, te_lang) & test)[0]
                            rows.append({**base, "test_context": te_ctx, "test_lang": te_lang, "control": "none",
                                         "acc": probe.score(X[te, layer], meta["label"][te])})
                    te = np.where(cell(meta, concept, tr_ctx, tr_lang) & test)[0]
                    rows.append({**base, "test_context": tr_ctx, "test_lang": tr_lang, "control": "shuffled",
                                 "acc": Probe(X[tr, layer], y_shuf, C, iters).score(X[te, layer], meta["label"][te])})
                    if tr_ctx == "informative" and layer in abl_layers:
                        for rank in ranks(layer):
                            dirs[(concept, tr_lang, layer, rank)] = (
                                probe.direction()[:, None].astype(np.float32) if rank == 1
                                else inlp_basis(X[tr, layer], y_tr, rank, C, iters))
    return rows, dirs


def run_model(name: str, cfg: dict, items: list[dict], seeds: list[int], device, dtype, batch_size: int,
              baseline_only: bool = False) -> None:
    t_model = time.time()
    revision = cfg.get("revisions", {}).get(name)
    model, tok = load_model(name, device, dtype, texts=[it["text"] + a for it in items for a in it["answers"]],
                            revision=revision)
    n_layers = len(get_blocks(model))
    d_model = model.config.hidden_size
    acfg = cfg["ablation"]
    primary = n_layers // 2 if acfg["primary_layer"] == "mid" else int(acfg["primary_layer"])
    abl_layers = sorted(set(range(0, n_layers, acfg["layer_stride"])) | {primary})

    def ranks(layer):
        return [acfg["primary_rank"]] + (acfg["extra_ranks"] if layer == primary else [])

    cache = resolve(cfg["cache_dir"]) / slug(name)
    cache.mkdir(parents=True, exist_ok=True)
    ids = [it["id"] for it in items]
    tag = str(dtype) if revision is None else f"{dtype}@{revision}"  # pilot caches keep their dtype-only tag
    acts = cached(cache / "activations.pt", ids, lambda: extract(model, tok, items, device, batch_size), tag)
    X = acts["kin"].numpy()
    lang_mean_np = {k: v.numpy().astype(np.float64) for k, v in acts["lang_mean"].items()}
    sset = ScoringSet(tok, items)
    base_logp = cached(cache / "baseline_logp.pt", ids,
                       lambda: {"ids": ids, "logp": torch.from_numpy(score(model, sset, device, batch_size))},
                       tag)
    base_logp = base_logp["logp"].numpy()
    if baseline_only:
        print(f"{name}: baseline cached -> {cache}  (inspect with: python src/diagnose_baseline.py)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return
    meta = {k: np.array([it[k] for it in items]) for k in ("concept", "context", "lang", "label", "name_id", "pair_id")}
    C, iters = cfg["probe"]["C"], cfg["probe"]["max_iter"]
    print(f"{name}: {n_layers} blocks, d={d_model}, primary layer {primary}, ablation layers {abl_layers}")

    for seed in seeds:
        t_seed = time.time()
        set_seed(seed)
        _, test_names = name_split(len(NAMES), cfg["n_test_names"], seed)
        test = np.isin(meta["name_id"], sorted(test_names))
        train = ~test

        probe_rows, dirs = run_probes(X, meta, train, test, cfg, name, seed, n_layers, abl_layers, primary, ranks)
        for layer in abl_layers:
            for rank in ranks(layer):
                V = random_basis(d_model, rank, np.random.default_rng([seed, layer, rank]))
                for concept in cfg["concepts"]:
                    dirs[(concept, "random", layer, rank)] = V

        abl_rows, erasure_rows = [], []
        for concept in cfg["concepts"]:
            idx = np.where((meta["concept"] == concept) & test)[0]
            abl_rows += group_rows(base_logp, idx, meta, dict(model=name, seed=seed, concept=concept,
                                                              layer=-1, rank=0, direction="none"))
        for (concept, src, layer, rank), V in tqdm(dirs.items(), desc=f"ablate seed {seed}"):
            idx = np.where((meta["concept"] == concept) & test)[0]
            with SubspaceAblation(model, layer, V, acts["lang_mean"], acfg["mode"]) as ablation:
                logp = score(model, sset, device, batch_size, idx, ablation)
            abl_rows += group_rows(logp, idx, meta, dict(model=name, seed=seed, concept=concept,
                                                         layer=layer, rank=rank, direction=src))

            lang = cfg["source_lang"] if src == "random" else src
            tr = np.where(cell(meta, concept, "informative", lang) & train)[0]
            te = np.where(cell(meta, concept, "informative", lang) & test)[0]
            mu = lang_mean_np[lang][layer] if acfg["mode"] == "mean" else 0.0
            erasure_rows.append(dict(
                model=name, seed=seed, concept=concept, layer=layer, rank=rank, direction=src, lang=lang,
                acc_before=Probe(X[tr, layer], meta["label"][tr], C, iters).score(X[te, layer], meta["label"][te]),
                acc_after=Probe(erase(X[tr, layer], V, mu), meta["label"][tr], C, iters)
                .score(erase(X[te, layer], V, mu), meta["label"][te])))

        out = resolve(cfg["results_dir"]) / "raw" / slug(name) / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(abl_rows).to_csv(out / "ablation.csv", index=False)
        pd.DataFrame(probe_rows).to_csv(out / "probe.csv", index=False)
        pd.DataFrame(erasure_rows).to_csv(out / "erasure.csv", index=False)
        with open(out / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "model": name, "revision": revision, "seed": seed, "n_layers": n_layers, "d_model": d_model, "primary_layer": primary,
                "ablation_layers": abl_layers, "device": str(device), "dtype": str(dtype),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(),
                "seed_seconds": round(time.time() - t_seed, 1), "model_seconds_so_far": round(time.time() - t_model, 1),
            }, f, indent=2)
        print(f"{name} seed {seed}: {time.time() - t_seed:.0f}s -> {out}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, nargs="+", help="seed(s) to run; default: all seeds in the config")
    ap.add_argument("--models", nargs="+", help="override the model list")
    ap.add_argument("--device", help="cuda | cuda:1 | cpu | mps; default: config (auto)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], help="default: config")
    ap.add_argument("--skip-analyze", action="store_true")
    ap.add_argument("--baseline-only", action="store_true",
                    help="cache activations + un-ablated scores, then stop (for diagnose_baseline.py)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = pick_device(args.device or cfg["device"])
    items = load_items(cfg)
    print(f"device={device} items={len(items)}")
    for name in args.models or cfg["models"]:
        # precedence: --dtype > model_dtype[name] > dtype
        dtype = pick_dtype(args.dtype or cfg.get("model_dtype", {}).get(name, cfg["dtype"]), device)
        print(f"{name}: dtype={dtype}")
        run_model(name, cfg, items, args.seed or cfg["seeds"], device, dtype, args.batch_size or cfg["batch_size"],
                  args.baseline_only)
    if not (args.skip_analyze or args.baseline_only):
        analyze.run(cfg)


if __name__ == "__main__":
    main()
