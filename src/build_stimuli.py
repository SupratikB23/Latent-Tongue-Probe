"""Generate data/stimuli.jsonl from lexicon.py.

    python src/build_stimuli.py              # write data/stimuli.jsonl
    python src/build_stimuli.py --preview    # print one minimal pair per cell for native-speaker review

Each item is   <context sentence> <target sentence containing the kin term> <question>
and is scored by comparing log p(answer_0) vs log p(answer_1) as the continuation.
Items come in minimal pairs: same concept, context type, language, name and predicate; only the label differs.
"""
from __future__ import annotations

import argparse
import unicodedata
from collections import Counter

from lexicon import (ANSWERS, CONCEPTS, CONTEXTS, KIN, KIN_HI_FEMININE, LABEL_NAMES, LANGS, NAMES,
                     PREDICATES, TEMPLATES)
from utils import write_jsonl


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def render(concept: str, context: str, lang: str, label: int, name_id: int, pred_id: int) -> dict:
    en, bn, bn_gen, hi = NAMES[name_id]
    pred = PREDICATES[pred_id]
    tpl = TEMPLATES[lang]
    kin = nfc(KIN[concept][lang][label])

    fill = {"en": {"N": en, "P": pred["en"]},
            "bn": {"N": bn_gen, "P": pred["bn"]},
            "hi": {"N": hi}}[lang]
    if lang == "hi":
        fem = KIN_HI_FEMININE[concept][label]
        fill["G"] = "की" if fem else "के"
        fill["P"] = pred["hi_f"] if fem else pred["hi_m"]

    ctx_tpl = tpl["context"][concept][label] if context == "informative" else tpl["neutral"]
    ctx = nfc(ctx_tpl.format(**fill))
    pre, post = tpl["target"].split("{K}")
    pre, post = nfc(pre.format(**fill)), nfc(post.format(**fill))
    query = nfc(tpl["query"][concept].format(K=kin, **fill))

    kin_start = len(ctx) + 1 + len(pre)
    text = f"{ctx} {pre}{kin}{post} {query}"
    assert text[kin_start:kin_start + len(kin)] == kin

    return {
        "id": f"{concept}|{context}|{lang}|n{name_id:02d}|p{pred_id:02d}|y{label}",
        "pair_id": f"{concept}|{context}|{lang}|n{name_id:02d}|p{pred_id:02d}",
        "concept": concept,
        "context": context,
        "lang": lang,
        "label": label,
        "label_name": LABEL_NAMES[concept][label],
        "name_id": name_id,
        "pred_id": pred_id,
        "text": text,
        "kin_term": kin,
        "kin_char_start": kin_start,
        "kin_char_end": kin_start + len(kin),
        "answers": [nfc(a) for a in ANSWERS[concept][lang]],
    }


def build_all() -> list[dict]:
    return [
        render(c, ctx, lang, y, n, p)
        for c in CONCEPTS for ctx in CONTEXTS for lang in LANGS
        for n in range(len(NAMES)) for p in range(len(PREDICATES)) for y in (0, 1)
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/stimuli.jsonl")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    items = build_all()
    if args.preview:
        for c in CONCEPTS:
            for ctx in CONTEXTS:
                for lang in LANGS:
                    for y in (0, 1):
                        it = render(c, ctx, lang, y, 0, 0)
                        print(f"[{c}/{ctx}/{lang}/{it['label_name']}] {it['text']}  ->{it['answers'][y]}")
                print()
        return

    write_jsonl(items, args.out)
    counts = Counter((it["concept"], it["context"], it["lang"]) for it in items)
    print(f"wrote {len(items)} items to {args.out}")
    for key, n in sorted(counts.items()):
        print("  ", "/".join(key), n)


if __name__ == "__main__":
    main()
