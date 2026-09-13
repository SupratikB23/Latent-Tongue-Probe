"""Cache residual-stream activations at the kin-term token, for every item and every decoder block.

Also accumulates a per-language mean activation (over all non-initial tokens) per block, which is
the reference point for mean-ablation in ablate.py. The mean is label-free.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from models import block_hidden, encode, get_base, get_blocks, pad_batch, special_prefix


def kin_token_index(offsets: list[tuple[int, int]], n_prefix: int, start: int, end: int) -> int:
    """Last token whose character span overlaps the kin term."""
    hits = [i for i, (s, e) in enumerate(offsets) if i >= n_prefix and e > s and s < end and e > start]
    if not hits:
        raise ValueError(f"no token overlaps characters [{start}, {end})")
    return hits[-1]


@torch.no_grad()
def extract(model, tok, items: list[dict], device, batch_size: int) -> dict:
    blocks, base = get_blocks(model), get_base(model)
    n_layers, d_model = len(blocks), model.config.hidden_size
    prefix = special_prefix(tok)
    encoded = [encode(tok, it["text"], prefix) for it in items]
    kin_pos = [kin_token_index(off, len(prefix), it["kin_char_start"], it["kin_char_end"])
               for (_, off), it in zip(encoded, items)]

    langs = sorted({it["lang"] for it in items})
    kin = torch.zeros(len(items), n_layers, d_model, dtype=torch.float32)
    sums = {lang: torch.zeros(n_layers, d_model, dtype=torch.float64) for lang in langs}
    counts = {lang: 0 for lang in langs}
    state: dict = {}

    def make_hook(layer):
        def hook(module, inputs, output):
            h = block_hidden(output).float()
            state["kin"][:, layer] = h[state["rows"], state["pos"]].cpu()
            state["sum"][:, layer] = (h * state["content"][..., None]).sum(1).double().cpu()
        return hook

    handles = [b.register_forward_hook(make_hook(l)) for l, b in enumerate(blocks)]
    order = np.argsort([len(ids) for ids, _ in encoded], kind="stable")
    try:
        for start in tqdm(range(0, len(order), batch_size), desc="extract"):
            idx = order[start:start + batch_size]
            ids, mask = pad_batch([encoded[i][0] for i in idx], tok.pad_token_id, device)
            content = mask.float()
            content[:, 0] = 0  # position 0 is the attention-sink token; keep it out of the mean
            state.update(
                rows=torch.arange(len(idx), device=device),
                pos=torch.tensor([kin_pos[i] for i in idx], device=device),
                content=content,
                kin=torch.zeros(len(idx), n_layers, d_model),
                sum=torch.zeros(len(idx), n_layers, d_model, dtype=torch.float64),
            )
            base(input_ids=ids, attention_mask=mask, use_cache=False)
            kin[torch.as_tensor(idx)] = state["kin"]
            n_content = content.sum(1).long().cpu().tolist()
            for j, i in enumerate(idx):
                lang = items[i]["lang"]
                sums[lang] += state["sum"][j]
                counts[lang] += n_content[j]
    finally:
        for h in handles:
            h.remove()

    if not torch.isfinite(kin).all():
        raise FloatingPointError("non-finite activations; rerun with dtype: float32")
    return {
        "ids": [it["id"] for it in items],
        "kin": kin,
        "kin_pos": kin_pos,
        "lang_mean": {lang: (sums[lang] / counts[lang]).float() for lang in langs},
    }
