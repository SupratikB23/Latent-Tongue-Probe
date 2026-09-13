"""Model loading, device selection, tokenization, and architecture-agnostic access to decoder blocks."""
from __future__ import annotations

import torch

TINY = "__tiny_random__"  # offline 4-layer GPT-2 with a BPE tokenizer trained on the stimuli; for smoke tests only

_BLOCK_PATHS = ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers")


def pick_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]
    return torch.float32 if device.type == "cpu" else dtype


def load_model(name: str, device: torch.device, dtype: torch.dtype, texts: list[str] | None = None):
    if name == TINY:
        model, tok = build_tiny_random(texts)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    if not tok.is_fast:
        raise ValueError(f"{name}: a fast tokenizer is required (offset mapping locates the kin-term token)")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.to(device=device, dtype=dtype).eval()
    return model, tok


def build_tiny_random(texts: list[str]):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    bpe = Tokenizer(models.BPE())
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=2000, special_tokens=["<pad>", "<s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    bpe.train_from_iterator(texts, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, pad_token="<pad>", bos_token="<s>", eos_token="<s>")
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=len(tok), n_positions=512, n_embd=64, n_layer=4, n_head=4)
    return GPT2LMHeadModel(cfg), tok


def get_blocks(model) -> torch.nn.ModuleList:
    for path in _BLOCK_PATHS:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise ValueError(f"cannot locate decoder blocks in {type(model).__name__}")


def get_base(model):
    """The transformer without the LM head, so we never materialize [batch, seq, 250k-vocab] logits."""
    return getattr(model, model.base_model_prefix)


def special_prefix(tok) -> list[int]:
    """Special tokens the tokenizer prepends by default (XGLM adds </s>, BLOOM adds nothing)."""
    with_sp = tok("a")["input_ids"]
    without = tok("a", add_special_tokens=False)["input_ids"]
    for i in range(len(with_sp) - len(without) + 1):
        if with_sp[i:i + len(without)] == without:
            return with_sp[:i]
    return []


def encode(tok, text: str, prefix: list[int]) -> tuple[list[int], list[tuple[int, int]]]:
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return prefix + enc["input_ids"], [(0, 0)] * len(prefix) + [tuple(o) for o in enc["offset_mapping"]]


def pad_batch(seqs: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def block_hidden(output):
    return output[0] if isinstance(output, tuple) else output
