"""Continuation scoring, optionally under mean-ablation of a subspace at the output of one decoder block.

Readout: for each item, log p(answer_0 | prompt) and log p(answer_1 | prompt), summed over answer tokens.
An item is correct when the labelled answer scores strictly higher.

Intervention: h <- h - ((h - mu_lang) V) V^T at every position except 0, where V is an orthonormal
[d, k] basis and mu_lang is the label-free mean activation of the item's language at that block.
mode: zero uses mu = 0 (plain projection).
"""
from __future__ import annotations

import numpy as np
import torch

from models import block_hidden, get_base, get_blocks, pad_batch, special_prefix


class ScoringSet:
    """Pre-tokenized (prompt + answer) sequences, two per item."""

    def __init__(self, tok, items: list[dict]):
        prefix = special_prefix(tok)
        self.items = items
        self.langs = [it["lang"] for it in items]
        self.pad_id = tok.pad_token_id
        self.seqs: list[tuple[int, int, list[int], int, int]] = []  # item, answer, ids, prompt_len, answer_len
        self.item_seqs: list[list[int]] = []
        for i, it in enumerate(items):
            p_ids = tok(it["text"], add_special_tokens=False)["input_ids"]
            own = []
            for a, ans in enumerate(it["answers"]):
                full = tok(it["text"] + ans, add_special_tokens=False)["input_ids"]
                if len(full) > len(p_ids) and full[:len(p_ids)] == p_ids:
                    c_ids = full[len(p_ids):]
                else:  # a merge across the boundary; fall back to tokenizing the answer on its own
                    c_ids = tok(ans, add_special_tokens=False)["input_ids"]
                own.append(len(self.seqs))
                self.seqs.append((i, a, prefix + p_ids + c_ids, len(prefix) + len(p_ids), len(c_ids)))
            self.item_seqs.append(own)


class SubspaceAblation:
    def __init__(self, model, layer: int, V: np.ndarray, lang_mean: dict, mode: str = "mean"):
        p = next(model.parameters())
        self.block = get_blocks(model)[layer]
        self.V = torch.as_tensor(V, device=p.device, dtype=p.dtype)
        self.mu = {lang: (m[layer] if mode == "mean" else torch.zeros_like(m[layer])).to(p.device, p.dtype)
                   for lang, m in lang_mean.items()}
        self.batch_mu = None
        self.handle = None

    def set_batch(self, langs: list[str]) -> None:
        self.batch_mu = torch.stack([self.mu[lang] for lang in langs])[:, None, :]

    def _hook(self, module, inputs, output):
        h = block_hidden(output)
        delta = ((h - self.batch_mu) @ self.V) @ self.V.T
        delta[:, 0] = 0
        h = h - delta
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        self.handle.remove()


@torch.no_grad()
def score(model, sset: ScoringSet, device, batch_size: int, subset=None,
          ablation: SubspaceAblation | None = None) -> np.ndarray:
    """Returns [n_items, 2] summed answer log-probs; rows outside `subset` are NaN."""
    base, head = get_base(model), model.get_output_embeddings()
    items = range(len(sset.items)) if subset is None else subset
    seq_ids = sorted((s for i in items for s in sset.item_seqs[i]), key=lambda s: len(sset.seqs[s][2]))
    out = np.full((len(sset.items), 2), np.nan)

    for start in range(0, len(seq_ids), batch_size):
        batch = [sset.seqs[s] for s in seq_ids[start:start + batch_size]]
        ids, mask = pad_batch([b[2] for b in batch], sset.pad_id, device)
        if ablation is not None:
            ablation.set_batch([sset.langs[b[0]] for b in batch])
        h = base(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state

        rows, pos, tgt = [], [], []
        for j, (_, _, seq, p_len, c_len) in enumerate(batch):
            for k in range(c_len):
                rows.append(j)
                pos.append(p_len - 1 + k)
                tgt.append(seq[p_len + k])
        rows_t = torch.tensor(rows, device=device)
        tgt_t = torch.tensor(tgt, device=device)
        logits = head(h[rows_t, torch.tensor(pos, device=device)]).float()
        tok_logp = torch.log_softmax(logits, dim=-1).gather(1, tgt_t[:, None]).squeeze(1)
        seq_logp = torch.zeros(len(batch), device=device).index_add_(0, rows_t, tok_logp).cpu().numpy()
        for j, (i, a, *_) in enumerate(batch):
            out[i, a] = seq_logp[j]
    return out
