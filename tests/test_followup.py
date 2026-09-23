"""Follow-up freeze: pilot stimuli untouched, follow-up stimuli invariants, and the stage-1 selection rules."""
from build_stimuli import build_all
from followup import off_ceiling, outcome, passes_gate, select
from lexicon import FOLLOWUP_CONCEPTS, FOLLOWUP_LANGS
from utils import load_config, read_jsonl

CFG = load_config("configs/followup.yaml")


def test_pilot_stimuli_file_unchanged():
    assert read_jsonl("data/stimuli.jsonl") == build_all()


def test_followup_stimuli_file_matches_lexicon():
    assert read_jsonl(CFG["stimuli"]) == build_all(FOLLOWUP_CONCEPTS, FOLLOWUP_LANGS)


def test_followup_items_match_pilot_side_aunt():
    pilot = {it["id"]: it for it in build_all() if it["concept"] == "side_aunt" and it["lang"] in FOLLOWUP_LANGS}
    items = build_all(FOLLOWUP_CONCEPTS, FOLLOWUP_LANGS)
    for it in items:
        if it["concept"] == "side_aunt":
            assert it == pilot[it["id"]]
        if it["concept"] == "side_aunt_named" and it["lang"] == "bn":  # the control differs only in English
            twin = pilot[it["id"].replace("side_aunt_named", "side_aunt")]
            assert (it["text"], it["answers"], it["label"]) == (twin["text"], twin["answers"], twin["label"])


def test_followup_controls_are_lexicalized_in_both_languages():
    items = build_all(FOLLOWUP_CONCEPTS, FOLLOWUP_LANGS)
    for concept in ("side_aunt_named", "gender_maternal"):
        for lang in FOLLOWUP_LANGS:
            assert len({it["kin_term"] for it in items if it["concept"] == concept and it["lang"] == lang}) == 2
    assert len({it["kin_term"] for it in items if it["concept"] == "side_aunt" and it["lang"] == "en"}) == 1


def acc(**cells):
    return {f"{c}/{l}": v for (c, l), v in ((tuple(k.split("__")), v) for k, v in cells.items())}


def test_gate_and_ceiling():
    a = acc(side_aunt__bn=0.93, side_aunt__en=0.64, side_aunt_named__bn=0.93, side_aunt_named__en=0.96)
    assert not passes_gate(a, "side_aunt", CFG)  # en below 0.65
    assert not off_ceiling(a, "side_aunt_named", CFG)  # en above 0.95
    assert off_ceiling(acc(x__bn=0.65, x__en=0.95), "x", CFG)  # both bounds inclusive


def test_select_takes_first_passing_model_and_control():
    ok = dict(side_aunt__bn=0.9, side_aunt__en=0.9, side_aunt_named__bn=0.9, side_aunt_named__en=0.9,
              gender_maternal__bn=0.9, gender_maternal__en=0.9)
    anchor, (m1, m2, _) = CFG["anchor_model"], CFG["second_family_candidates"]
    accs = {anchor: acc(**ok), m1: acc(**{**ok, "side_aunt__en": 0.5}), m2: acc(**ok)}
    assert select(accs, CFG) == (m2, "side_aunt_named")
    accs[m2] = acc(**{**ok, "side_aunt_named__bn": 1.0})  # first control at ceiling in one model -> second control
    assert select(accs, CFG) == (m2, "gender_maternal")
    assert select({anchor: acc(**ok), m1: acc(**{**ok, "side_aunt__bn": 0.1})}, CFG) == (None, None)


def test_outcome_rule():
    sup, fal, inc = (dict(verdict="SUPPORTED", survives=True), dict(verdict="FALSIFIED", survives=False),
                     dict(verdict="INCONCLUSIVE", survives=False))
    assert outcome({"a": sup, "b": sup}) == "SURVIVED"
    assert outcome({"a": fal, "b": fal}) == "NULL"
    assert outcome({"a": sup, "b": fal}) == "MIXED"
    assert outcome({"a": fal, "b": inc}) == "MIXED"
    assert outcome({"a": dict(verdict="SUPPORTED", survives=False), "b": sup}) == "MIXED"  # uninformative control


def test_every_candidate_is_pinned():
    revs = CFG["revisions"]
    for m in [CFG["anchor_model"], *CFG["second_family_candidates"]]:
        assert len(revs[m]) == 40 and all(c in "0123456789abcdef" for c in revs[m]), m
