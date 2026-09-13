import re
import unicodedata
from collections import defaultdict

from build_stimuli import build_all
from lexicon import CONCEPTS, CONTEXTS, LANGS, NAMES, PREDICATES

ITEMS = build_all()


def pairs():
    grouped = defaultdict(list)
    for it in ITEMS:
        grouped[it["pair_id"]].append(it)
    return grouped


def test_counts_and_unique_ids():
    assert len(ITEMS) == len(CONCEPTS) * len(CONTEXTS) * len(LANGS) * len(NAMES) * len(PREDICATES) * 2
    assert len({it["id"] for it in ITEMS}) == len(ITEMS)


def test_kin_span_nfc_and_answers():
    for it in ITEMS:
        assert it["text"][it["kin_char_start"]:it["kin_char_end"]] == it["kin_term"]
        assert unicodedata.is_normalized("NFC", it["text"])
        assert it["answers"][0] != it["answers"][1]


def test_minimal_pairs_differ_only_in_label():
    for members in pairs().values():
        assert sorted(m["label"] for m in members) == [0, 1]
        assert members[0]["answers"] == members[1]["answers"]


def test_lexicalization_pattern():
    def terms(concept, lang):
        return {it["kin_term"] for it in ITEMS if it["concept"] == concept and it["lang"] == lang}

    assert len(terms("side", "en")) == 1  # English "uncle" does not mark side
    assert len(terms("side", "bn")) == 2 and len(terms("side", "hi")) == 2
    assert all(len(terms("gender", lang)) == 2 for lang in LANGS)  # gender is lexical everywhere


def test_neutral_context_sentence_is_label_free():
    for members in pairs().values():
        if members[0]["context"] == "neutral":
            first = [re.split(r"[.।]", m["text"], maxsplit=1)[0] for m in members]
            assert first[0] == first[1]
