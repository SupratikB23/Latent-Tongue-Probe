"""Offline checks of the mechanics on a tiny random GPT-2 (no downloads)."""
import numpy as np
import pytest
import torch

from ablate import ScoringSet, SubspaceAblation, score
from build_stimuli import build_all
from extract import extract
from models import TINY, block_hidden, encode, get_blocks, load_model, special_prefix
from probe import Probe, erase, inlp_basis, random_basis

ITEMS = [it for it in build_all() if it["name_id"] < 2 and it["pred_id"] < 2 and it["concept"] != "side_aunt"]
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tiny():
    return load_model(TINY, CPU, torch.float32, texts=[it["text"] + a for it in ITEMS for a in it["answers"]])


@pytest.fixture(scope="module")
def acts(tiny):
    model, tok = tiny
    return extract(model, tok, ITEMS, CPU, batch_size=8)


def test_kin_token_overlaps_kin_term(tiny, acts):
    _, tok = tiny
    prefix = special_prefix(tok)
    for it, pos in zip(ITEMS, acts["kin_pos"]):
        _, offsets = encode(tok, it["text"], prefix)
        s, e = offsets[pos]
        assert s < it["kin_char_end"] and e > it["kin_char_start"]


def test_extract_shapes(tiny, acts):
    model, _ = tiny
    assert acts["kin"].shape == (len(ITEMS), len(get_blocks(model)), model.config.hidden_size)
    assert set(acts["lang_mean"]) == {"bn", "hi", "en"}


def test_score_matches_full_logits(tiny):
    model, tok = tiny
    sset = ScoringSet(tok, ITEMS[:4])
    got = score(model, sset, CPU, batch_size=3)
    for i, a, ids, p_len, c_len in sset.seqs:
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.tensor([ids])).logits[0].float(), -1)
        want = sum(logp[p_len - 1 + k, ids[p_len + k]].item() for k in range(c_len))
        assert got[i, a] == pytest.approx(want, abs=1e-4)


def test_ablation_removes_component_except_position_zero(tiny, acts):
    model, tok = tiny
    layer, d = 1, model.config.hidden_size
    V = random_basis(d, 3, np.random.default_rng(0))
    sset = ScoringSet(tok, ITEMS[:2])
    captured = {}

    def grab(key):
        return lambda m, i, o: captured.__setitem__(key, block_hidden(o).detach().clone())

    block = get_blocks(model)[layer]
    h = block.register_forward_hook(grab("clean"))
    score(model, sset, CPU, batch_size=4)
    h.remove()
    with SubspaceAblation(model, layer, V, acts["lang_mean"], "mean") as ab:
        h = block.register_forward_hook(grab("ablated"))  # registered after, so it sees the edited output
        score(model, sset, CPU, batch_size=4, ablation=ab)
        h.remove()
        mu = ab.batch_mu
    Vt = torch.as_tensor(V)
    resid = ((captured["ablated"] - mu) @ Vt)[:, 1:]
    assert resid.abs().max().item() < 1e-4
    assert torch.allclose(captured["ablated"][:, 0], captured["clean"][:, 0])


def test_inlp_basis_orthonormal_and_erasure():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 16))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    V = inlp_basis(X, y, rank=3)
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-5)
    mu = X.mean(0)
    Xe = erase(X, V, mu)
    assert np.allclose((Xe - mu) @ V, 0, atol=1e-6)
    assert Probe(X, y).score(X, y) > 0.95
