"""Hand-built lexicon for the kinship stimuli.

Every string a native speaker needs to review lives in this file. build_stimuli.py
only combines these pieces; it never invents text.

Concepts
--------
side       paternal (0) vs maternal (1) uncle.  bn কাকা/মামা, hi चाचा/मामा, en "uncle" (not lexicalized)
gender     male (0) vs female (1) paternal sibling. bn কাকা/পিসি, hi चाचा/बुआ, en uncle/aunt (lexicalized everywhere)
side_aunt  paternal (0) vs maternal (1) aunt.   bn পিসি/মাসি, hi बुआ/मौसी, en "aunt" (backup concept)

Follow-up control candidates, bn and en only (configs/followup.yaml, results/followup_prereg.md)
side_aunt_named  side_aunt with English naming the side: en "paternal aunt"/"maternal aunt"; bn identical to side_aunt
gender_maternal  male (0) vs female (1) maternal sibling. bn মামা/মাসি, en uncle/aunt (lexicalized in both)

Placeholders
------------
{N}  name (en, hi) or name in genitive case (bn)
{K}  kin term
{P}  predicate
{G}  Hindi genitive particle agreeing with the kin term (के masc. honorific, की fem.)
"""

LANGS = ("bn", "hi", "en")
CONCEPTS = ("side", "gender", "side_aunt")  # pilot concepts; data/stimuli.jsonl is built from exactly these
FOLLOWUP_CONCEPTS = ("side_aunt", "side_aunt_named", "gender_maternal")  # data/stimuli_followup.jsonl
FOLLOWUP_LANGS = ("bn", "en")
CONTEXTS = ("informative", "neutral")

LABEL_NAMES = {
    "side": ("paternal", "maternal"),
    "gender": ("male", "female"),
    "side_aunt": ("paternal", "maternal"),
    "side_aunt_named": ("paternal", "maternal"),
    "gender_maternal": ("male", "female"),
}

# (en, bn, bn_genitive, hi). Genitive is stored, not derived: -এর after consonant-final, -র after vowel-final.
NAMES = [
    ("Rahul", "রাহুল", "রাহুলের", "राहुल"),
    ("Amit", "অমিত", "অমিতের", "अमित"),
    ("Sumon", "সুমন", "সুমনের", "सुमन"),
    ("Arnab", "অর্ণব", "অর্ণবের", "अर्णब"),
    ("Rohan", "রোহন", "রোহনের", "रोहन"),
    ("Kabir", "কবির", "কবিরের", "कबीर"),
    ("Arjun", "অর্জুন", "অর্জুনের", "अर्जुन"),
    ("Vikram", "বিক্রম", "বিক্রমের", "विक्रम"),
    ("Sourav", "সৌরভ", "সৌরভের", "सौरव"),
    ("Dev", "দেব", "দেবের", "देव"),
    ("Riya", "রিয়া", "রিয়ার", "रिया"),
    ("Priya", "প্রিয়া", "প্রিয়ার", "प्रिया"),
    ("Tania", "তানিয়া", "তানিয়ার", "तानिया"),
    ("Neha", "নেহা", "নেহার", "नेहा"),
    ("Ananya", "অনন্যা", "অনন্যার", "अनन्या"),
    ("Meera", "মীরা", "মীরার", "मीरा"),
    ("Puja", "পূজা", "পূজার", "पूजा"),
    ("Nisha", "নিশা", "নিশার", "निशा"),
    ("Soma", "সোমা", "সোমার", "सोमा"),
    ("Isha", "ঈশা", "ঈশার", "ईशा"),
]

# Predicates carry no kinship information. Bengali verbs use the honorific form (elder relative)
# and do not agree in gender; Hindi verbs agree, so both forms are stored.
PREDICATES = [
    {"en": "lives in Kolkata", "bn": "কলকাতায় থাকেন",
     "hi_m": "कोलकाता में रहते हैं", "hi_f": "कोलकाता में रहती हैं"},
    {"en": "works in a bank", "bn": "একটি ব্যাংকে কাজ করেন",
     "hi_m": "एक बैंक में काम करते हैं", "hi_f": "एक बैंक में काम करती हैं"},
    {"en": "cooks very well", "bn": "খুব ভালো রান্না করেন",
     "hi_m": "बहुत अच्छा खाना बनाते हैं", "hi_f": "बहुत अच्छा खाना बनाती हैं"},
    {"en": "reads the newspaper every morning", "bn": "প্রতিদিন সকালে খবরের কাগজ পড়েন",
     "hi_m": "हर सुबह अख़बार पढ़ते हैं", "hi_f": "हर सुबह अख़बार पढ़ती हैं"},
    {"en": "loves to sing", "bn": "গান গাইতে ভালোবাসেন",
     "hi_m": "गाना गाना पसंद करते हैं", "hi_f": "गाना गाना पसंद करती हैं"},
    {"en": "visits us every winter", "bn": "প্রতি শীতে আমাদের বাড়িতে আসেন",
     "hi_m": "हर सर्दी में हमारे घर आते हैं", "hi_f": "हर सर्दी में हमारे घर आती हैं"},
    {"en": "plays chess in the evening", "bn": "সন্ধ্যায় দাবা খেলেন",
     "hi_m": "शाम को शतरंज खेलते हैं", "hi_f": "शाम को शतरंज खेलती हैं"},
    {"en": "tells funny stories", "bn": "মজার গল্প বলেন",
     "hi_m": "मज़ेदार कहानियाँ सुनाते हैं", "hi_f": "मज़ेदार कहानियाँ सुनाती हैं"},
    {"en": "grows vegetables in the garden", "bn": "বাগানে সবজি চাষ করেন",
     "hi_m": "बगीचे में सब्ज़ियाँ उगाते हैं", "hi_f": "बगीचे में सब्ज़ियाँ उगाती हैं"},
    {"en": "drinks tea without sugar", "bn": "চিনি ছাড়া চা খান",
     "hi_m": "बिना चीनी की चाय पीते हैं", "hi_f": "बिना चीनी की चाय पीती हैं"},
]

# KIN[concept][lang] = (term for label 0, term for label 1)
KIN = {
    "side": {"en": ("uncle", "uncle"), "bn": ("কাকা", "মামা"), "hi": ("चाचा", "मामा")},
    "gender": {"en": ("uncle", "aunt"), "bn": ("কাকা", "পিসি"), "hi": ("चाचा", "बुआ")},
    "side_aunt": {"en": ("aunt", "aunt"), "bn": ("পিসি", "মাসি"), "hi": ("बुआ", "मौसी")},
    "side_aunt_named": {"en": ("paternal aunt", "maternal aunt"), "bn": ("পিসি", "মাসি")},
    "gender_maternal": {"en": ("uncle", "aunt"), "bn": ("মামা", "মাসি")},
}

# Hindi grammatical gender of the kin term per label (drives {G} and the predicate form).
KIN_HI_FEMININE = {
    "side": (False, False),
    "gender": (False, True),
    "side_aunt": (True, True),
}

# ANSWERS[concept][lang] = (continuation for label 0, continuation for label 1). Leading space is intentional.
ANSWERS = {
    "side": {"en": (" father", " mother"), "bn": (" বাবার", " মায়ের"), "hi": (" पिता", " माँ")},
    "gender": {"en": (" brother", " sister"), "bn": (" ভাই", " বোন"), "hi": (" भाई", " बहन")},
    "side_aunt": {"en": (" father", " mother"), "bn": (" বাবার", " মায়ের"), "hi": (" पिता", " माँ")},
    "side_aunt_named": {"en": (" father", " mother"), "bn": (" বাবার", " মায়ের")},
    "gender_maternal": {"en": (" brother", " sister"), "bn": (" ভাই", " বোন")},
}

TEMPLATES = {
    "en": {
        "target": "{N}'s {K} {P}.",
        "neutral": "{N} has a big family.",
        "context": {
            "side": ("{N}'s father has one brother.", "{N}'s mother has one brother."),
            "gender": ("{N}'s father has one brother.", "{N}'s father has one sister."),
            "side_aunt": ("{N}'s father has one sister.", "{N}'s mother has one sister."),
            "side_aunt_named": ("{N}'s father has one sister.", "{N}'s mother has one sister."),
            "gender_maternal": ("{N}'s mother has one brother.", "{N}'s mother has one sister."),
        },
        "query": {
            "side": "Question: Is {N}'s {K} the father's brother or the mother's brother? Answer:",
            "gender": "Question: Is {N}'s {K} the father's brother or the father's sister? Answer:",
            "side_aunt": "Question: Is {N}'s {K} the father's sister or the mother's sister? Answer:",
            "side_aunt_named": "Question: Is {N}'s {K} the father's sister or the mother's sister? Answer:",
            "gender_maternal": "Question: Is {N}'s {K} the mother's brother or the mother's sister? Answer:",
        },
    },
    "bn": {
        "target": "{N} {K} {P}।",
        "neutral": "{N} একটি বড় পরিবার আছে।",
        "context": {
            "side": ("{N} বাবার একজন ভাই আছেন।", "{N} মায়ের একজন ভাই আছেন।"),
            "gender": ("{N} বাবার একজন ভাই আছেন।", "{N} বাবার একজন বোন আছেন।"),
            "side_aunt": ("{N} বাবার একজন বোন আছেন।", "{N} মায়ের একজন বোন আছেন।"),
            "side_aunt_named": ("{N} বাবার একজন বোন আছেন।", "{N} মায়ের একজন বোন আছেন।"),
            "gender_maternal": ("{N} মায়ের একজন ভাই আছেন।", "{N} মায়ের একজন বোন আছেন।"),
        },
        "query": {
            "side": "প্রশ্ন: {N} {K} কি বাবার ভাই না মায়ের ভাই? উত্তর:",
            "gender": "প্রশ্ন: {N} {K} কি বাবার ভাই না বাবার বোন? উত্তর:",
            "side_aunt": "প্রশ্ন: {N} {K} কি বাবার বোন না মায়ের বোন? উত্তর:",
            "side_aunt_named": "প্রশ্ন: {N} {K} কি বাবার বোন না মায়ের বোন? উত্তর:",
            "gender_maternal": "প্রশ্ন: {N} {K} কি মায়ের ভাই না মায়ের বোন? উত্তর:",
        },
    },
    "hi": {
        "target": "{N} {G} {K} {P}।",
        "neutral": "{N} का एक बड़ा परिवार है।",
        "context": {
            "side": ("{N} के पिता के एक भाई हैं।", "{N} की माँ के एक भाई हैं।"),
            "gender": ("{N} के पिता के एक भाई हैं।", "{N} के पिता की एक बहन हैं।"),
            "side_aunt": ("{N} के पिता की एक बहन हैं।", "{N} की माँ की एक बहन हैं।"),
        },
        "query": {
            "side": "प्रश्न: क्या {N} {G} {K} पिता के भाई हैं या माँ के भाई? उत्तर:",
            "gender": "प्रश्न: क्या {N} {G} {K} पिता के भाई हैं या पिता की बहन? उत्तर:",
            "side_aunt": "प्रश्न: क्या {N} {G} {K} पिता की बहन हैं या माँ की बहन? उत्तर:",
        },
    },
}
