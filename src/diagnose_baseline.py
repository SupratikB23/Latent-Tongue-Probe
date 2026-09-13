"""Diagnose the un-ablated readout from the cached log-probs (no GPU, no model load).

    python src/diagnose_baseline.py                      # every model under cache/
    python src/diagnose_baseline.py --cache cache_smoke

For every model x concept x context x language (all 20 names, 200 minimal pairs) it reports:
  acc        item accuracy: log p(correct) > log p(other)            (the current readout)
  pick0      share of items where answer 0 wins ("father"/"brother"); 1.0 or 0.0 = constant answer
  pair_acc   minimal-pair accuracy: the label-1 item prefers answer 1 MORE than its label-0 partner does.
             A constant preference for one answer cancels out, so this isolates what the input changes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from utils import read_jsonl, resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--stimuli", default="data/stimuli.jsonl")
    args = ap.parse_args()

    by_id = {it["id"]: it for it in read_jsonl(args.stimuli)}
    for path in sorted(resolve(args.cache).glob("*/baseline_logp.pt")):
        blob = torch.load(path)
        logp = blob["logp"].numpy()
        items = [by_id[i] for i in blob["ids"]]
        prefer1 = logp[:, 1] - logp[:, 0]

        cells, pairs = defaultdict(list), defaultdict(dict)
        for k, it in enumerate(items):
            cells[(it["concept"], it["context"], it["lang"])].append(k)
            pairs[it["pair_id"]][it["label"]] = k

        print(f"\n{path.parent.name}")
        print(f"{'concept':<8}{'context':<13}{'lang':<6}{'acc':>6}{'pick0':>7}{'pair_acc':>10}")
        for (concept, context, lang), idx in sorted(cells.items()):
            y = np.array([items[k]["label"] for k in idx])
            acc = float(((prefer1[idx] > 0).astype(int) == y).mean())
            pick0 = float((prefer1[idx] <= 0).mean())
            pids = {items[k]["pair_id"] for k in idx}
            d = np.array([prefer1[pairs[p][1]] - prefer1[pairs[p][0]] for p in pids])
            pair_acc = float(np.mean(np.where(np.abs(d) < 1e-6, 0.5, d > 0)))  # identical prompts tie -> 0.5
            print(f"{concept:<8}{context:<13}{lang:<6}{acc:>6.2f}{pick0:>7.2f}{pair_acc:>10.2f}")


if __name__ == "__main__":
    main()
