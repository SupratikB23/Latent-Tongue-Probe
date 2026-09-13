# Conversation Log

Date: 13 September 2026

Topics, in order:
1. ResearchDen project selection and outreach email
2. Linked list assignment (DOCX deliverable)
3. The Bu1ld outreach email to Ryan Gomez
4. Ryan's reply and what he is asking for
5. Brief on the CoT-Mediate paper (arXiv:2607.27304)
6. The seven fields of the research screen, explained
7. TLDR and concrete plan (Bengali kinship probe, Kaggle)

---

## 1. ResearchDen

Site: https://researchden.org/ (projects at /projects.html)

### The five listed projects

| # | Project | Area |
|---|---|---|
| 1 | Self-Supervised Learning for Medical Image Segmentation | CV |
| 2 | Benchmarking Multilingual Retrieval for Underrepresented Languages | NLP |
| 3 | Retrieval-Augmented Generation for Scientific Literature Synthesis | RAG |
| 4 | Efficient Fine-tuning of LLMs for Low-Resource Scientific Domains | LLM |
| 5 | Multimodal Understanding of Scientific Figures and Tables | CV + LLM |

No project on the page is a mechanistic interpretability project. The term does not appear.

### Recommendation given

- Best CV match: Multimodal Understanding of Scientific Figures and Tables. Needs document-level vision work, close to JEPA segmentation and B-JEPA histology experience.
- Closest to mech interp: Efficient Fine-tuning of LLMs for Low-Resource Scientific Domains. Works with LoRA/adapters, so it touches LLM internals. Not interpretability, but an interpretability angle could be proposed (for example: which layers or features shift most after fine-tuning).

### Caution flagged

Team names on the page are placeholders (A. Member, H. Scientist). Confirm scope with the team before assuming the projects are staffed and active.

### Decision

Chose projects 1 and 5 (both CV).

### Outreach email drafted

To: contact@researchden.org
Subject: Joining ResearchDen — Interest in CV/Medical Imaging and Multimodal Projects

```
Hello,

I am Sishir Gope (SS). I am a final-year B.Tech CSE (AI/ML) student at IEM Kolkata.
I want to join ResearchDen.

My core research areas are computer vision and self-supervised learning. I work on
JEPA-based methods for segmentation, including a boundary-aware JEPA model for
histology instance segmentation. I also work on 3D vision and ADAS-related perception
tasks. I have a separate paper on chain-of-thought reasoning in LLMs.

I want to contribute to two projects:

1. Self-Supervised Learning for Medical Image Segmentation. I can contribute methods
from my own histology segmentation work, including frozen-backbone contrastive
pretraining and boundary-aware objectives for scarce-label settings.

2. Multimodal Understanding of Scientific Figures and Tables. I can contribute
vision-language parsing methods from my segmentation and 3D vision work, and support
structured extraction pipelines for figures and tables.

I can commit regular time each week. I can write code, run experiments, and support
the manuscript process.

Please let me know the next step to join.

Thank you,
Sishir Gope (SS)
Final-year B.Tech CSE (AI/ML), IEM Kolkata
```

---

## 2. Linked list assignment

Deliverable produced: `Linked_List_Assignment.docx`

Plain formatting, no styling. For each of the 9 questions: the question text, the C code, then the tested output. All 9 programs were compiled with gcc and run; the outputs in the file are the actual program outputs.

| # | Problem | Approach | Output |
|---|---|---|---|
| 1 | Reverse a singly linked list | Three-pointer iterative reversal | `5 4 3 2 1` |
| 2 | Middle element | Slow/fast pointer | `Middle element: 3` |
| 3 | Cycle detection | Floyd slow/fast | `The linked list has a cycle.` |
| 4 | Remove Nth from end, one pass | Dummy node + gap of N between two pointers | `1 2 3 5` |
| 5 | Merge two sorted lists | Dummy node + tail splice | `1 2 3 4 5 6` |
| 6 | Palindrome check | Reversed copy, compare element-wise | `is a palindrome` |
| 7 | Rotate right by k | Close into ring, walk `len - k`, cut | `4 5 1 2 3` |
| 8 | Intersection node | Length difference, advance longer list, walk together | `intersect at 8` |
| 9 | Flatten multi-level list | Splice child list in, find its tail, reattach `next` | `1 2 3 6 7 9 10 8 4 5` |

---

## 3. The Bu1ld

Site: https://radiant-canvas.vertexed-25.workers.dev/
Contact: Ryan Gomez, ryan@thebu1ld.com (site also lists ryangomez.hs@gmail.com)

### Structure of the org

Six research threads:

| ID | Area | Thread |
|---|---|---|
| T-01 | world models | Counterfactual Defect Worlds |
| T-02 | interpretability | Sapir Whorf and the Latent Tongue |
| T-03 | representation | Adaptive Theory Geometry |
| T-04 | compression | Residual Event Tokenization |
| T-05 | scaling | Phase Transitions in Neural PDE Surrogates |
| T-06 | systems | On Device Agent Loops |

Three programs:

- P-01 Builder Cohort, 12 weeks, rolling intake
- P-02 Research Fellowship, 6 months, selective (this is the research track)
- P-03 Startup Incubation, ongoing

T-02 description: probes monolingual and multilingual checkpoints to see which concepts collapse, which split, and which exist only inside one language. Claimed joint working group with Stanford NLP.

T-05 description: scans width, depth, and data scale to find sharp transitions in representational behaviour, using those transitions as early warning that a PDE surrogate is about to degrade in deployment.

### Due diligence notes flagged

- The founder is a high school sophomore in Bangalore.
- The site's own numbers are inconsistent: 112 builders in the stat block, 200 in the signal log; 6 threads listed against a stat line claiming 12.
- Worth asking what the actual mentor structure is for T-02, given the Stanford NLP claim.

### Email sent

Subject: Research Fellowship (P-02) — interpretability (T-02) and PDE surrogates (T-05)

```
Hi Ryan,

I read through The Bu1ld site and I want in on the research side, specifically P-02.

I'm Supratik Bhowal, B.Tech CS (AI & ML) at IEM Kolkata, YGPA 9.11. Two threads line
up with work I'm already doing:

T-02, Sapir Whorf and the Latent Tongue. My most recent paper is "Position, Not
Provenance: Separating Reasoning Mediation from Sycophancy in Medical Vision-Language
Models" (arXiv:2607.27304). It separates where a model's answer actually gets decided
from where it only looks like it does. That is the same question your thread asks,
just with language as the intervention instead of position. If a concept only exists
inside one language, the mediation analysis should show it, and I would like to build
that probe.

T-05, Phase Transitions in Neural PDE Surrogates. Physics x ML was my first research
area and I've been in it for over a year, mostly PINNs and PINNsformer, plus JAX and
SciML tooling. Scanning width, depth, and data for the point where a surrogate flips
from modelling to memorizing is exactly the failure mode I keep hitting in practice.

I also saw your LinkedIn posts about a small Physics-Informed ML cohort, a JEPA
thread, and a Science Research cohort. All three are interesting, but I know that's
not how it works. Interpretability is where I want to spend the next several years,
and PhysML is where I started, so those two are my picks in that order.

[fallback line added here]

Resume attached. Happy to talk, or I can join the Discord first, whichever is faster
for you.

Best,
Supratik Bhowal
+91-6289552885 | supratikbhowal23@gmail.com
```

Fallback line added after the LinkedIn paragraph:

> If T-02 and T-05 are already staffed, treat this as an application to the Physics-Informed ML, JEPA, or Science Research cohorts instead. Any of the three works.

Alternates considered:
- "And if neither thread has room right now, put me down for whichever of those three cohorts opens first."
- "If neither thread has an open slot, I'd still like to be considered for the Physics-Informed ML or JEPA cohorts. I'd rather start somewhere adjacent than wait for the next intake."

---

## 4. Ryan's reply

```
Hi Supratik,

Thanks for reaching out. Based on what you described, T-02 is the strongest fit, with
T-05 as a very credible secondary lane. Your mediation-vs-sycophancy work is directly
relevant to treating language as an intervention rather than just another descriptive
feature.

I'd like to move you into a focused research screen rather than a generic call first.
Please send a compact one-page experiment spec for one T-02 probe: the exact
falsifiable hypothesis, intervention, model/dataset, control condition, primary metric,
a result that would falsify the claim, and the smallest reproducible artifact you could
ship in about a week.

No need to disclose anything beyond work you're comfortable making inspectable. The old
Discord link is stale, so email/GitHub is the canonical path for now. Once the scope is
frozen, we can decide whether a call would materially help.

Best,
Ryan
The Bu1LD
```

### What this means

- One document, one page, one experiment. Not a thread proposal.
- The spec is the interview. There is no call unless the spec is good.
- "Nothing beyond work you're comfortable making inspectable" means do not send unpublished results from the arXiv paper or anything under review. Design a fresh probe.
- "Once the scope is frozen" means small enough to defend.

---

## 5. Brief on the paper (arXiv:2607.27304)

Note: this brief is built from the abstract and title only. The full text fetch hit a rate limit. Verify section numbers against the PDF before quoting.

**Title:** Position, Not Provenance: Separating Reasoning Mediation from Sycophancy in Medical Vision-Language Models
**Authors:** Supratik Bhowal, Subhrajyoti Basu, Aritra Gir Mahanta, Anik Pal Chowdhury
**Submitted:** 29 July 2026, cs.LG and cs.CV

### The question

A medical VLM writes a chain-of-thought before answering. Does that reasoning cause the answer, or did the model already decide and the text is decoration? The term for this is **faithfulness**.

### The method: CoT-Mediate

A **behavioral** framework. Only inputs and outputs are touched; internal activations are never edited. (This matters, because T-02 is the opposite kind of work.)

Core move: let the model generate its own reasoning, edit exactly one clinically meaningful attribute inside it, feed it back, and check whether the answer follows the edit.

Example shape: the model writes "opacity in the left lower lobe." Change "left" to "right." If the final answer flips, the reasoning mediated it. If not, the reasoning was decoration.

**Mediation** is the causal-inference term. A mediator sits between cause and effect and carries the influence. The question is whether the CoT text is a real mediator between image and answer.

### The two arms

- **Re-prompting.** Hand the edited reasoning back as new context in a fresh prompt.
- **Prefix-forced continuation.** Force the edited reasoning to be the model's own already-written output, then make it continue from there.

**Finding:** prefix-forcing gave consistently higher measured faithfulness than re-prompting. So the injection mechanism changes the number you measure. Two labs running "the same" faithfulness experiment with different injection methods report different faithfulness for the same model. This is the methodological headline.

### The provenance control

**Sycophancy** is a model agreeing with whatever the user asserts, regardless of content. It confounds mediation: if injected reasoning moves the answer, maybe the reasoning mediated, or maybe the model just defers to authoritative-looking text.

Control: hold the reasoning text **identical**, vary only the stated source (model itself, user, expert radiologist). Movement under attribution change alone is sycophancy, not mediation.

**Result, and the title:** position, not provenance. Where the reasoning sits in context (own-output vs supplied-context) determines whether it gets used. Who it is claimed to come from matters much less.

### Setup and secondary findings

- Models: LLaVA-Med, MedGemma
- Data: VQA-RAD, 1000 samples each
- Removing the image increases reliance on injected reasoning.
- **Laterality** (left vs right) was the least faithfully tracked attribute. The model accepts a laterality edit in its reasoning and still answers with the original side. Real clinical safety finding, since left/right errors are a known harm category in radiology.

### Why Ryan flagged it

T-02 asks whether training language selects which concepts a model can represent. The paper is a template for the causal machinery: change exactly one thing, hold everything else fixed, add a control absorbing the obvious confound, measure whether output follows. His line about "language as an intervention" means: do to language what you did to laterality.

**Honest gap:** the paper is behavioral, working through prompts. T-02 is interpretability, meaning internals (activations, directions, subspaces). The methodology transfers. The tooling does not. Do not oversell fluency not yet built.

---

## 6. The seven fields, explained

### 6.1 Falsifiable hypothesis

One sentence, about a specific model, stated so a specific measurement could show it wrong.

Bad: "Language shapes model representations." Nothing could contradict it.

Good: "In mT5-base, the color-category boundary between blue and green is encoded in a linear subspace recoverable from Russian-language activations at layers 6 to 9 but not from English-language activations at the same layers."

Test: what result would make you say the hypothesis was false? If you cannot answer in one sentence, rewrite.

Candidate T-02 concepts (places languages genuinely differ):

- Russian has two basic color terms where English has "blue" (siniy, goluboy). Classic Winawer et al. finding in humans.
- Grammatical gender differs across languages for the same object.
- Evidentiality: Turkish grammatically marks witnessed vs hearsay. English does not.
- Kinship: Bengali distinguishes paternal and maternal uncle lexically. English collapses both to "uncle."

### 6.2 Intervention

The one thing you change, described mechanically enough to be implemented from your sentence.

#### Activation patching (causal tracing)

The standard causal tool in mech interp.

1. Run the model on a **clean** input. Cache all internal activations. (Cache = save intermediate tensors, one `forward_hook` in PyTorch.)
2. Run on a **corrupted** input differing in exactly the thing you care about. The output is now different.
3. Run the corrupted input again, but at one chosen layer and one chosen token position, **overwrite** the activation with the cached clean one.
4. Measure how much of the correct output returns.

If patching layer 8, token 4 restores the clean answer, layer 8 token 4 carries the information. Sweep all layers and positions for a heatmap of where the model stores it.

Layman version: the model is a factory line. Run a good batch and a bad batch, splice one station's output from the good run into the bad run, see if the final product comes out right. The station that fixes it is where the information lives.

Tooling: `nnsight`, `TransformerLens`, or plain PyTorch hooks. TransformerLens is easiest for decoder-only models and has patching helpers built in. Minutes on a 500M model.

#### Probing

Train a small linear classifier on internal activations to predict a property.

1. Collect activations at layer L for N inputs.
2. Label each input (contains concept X / does not).
3. Fit logistic regression on activations to predict the label.
4. Test on held-out data.

High accuracy means the information is linearly present at that layer. Standard caveat: a probe shows information is **decodable**, not that the model **uses** it. This is the same read-out vs use-it gap the CoT-Mediate paper is about, so saying it signals familiarity with the field.

#### Directional ablation (concept erasure)

Once a probe gives a direction, remove it and see what breaks.

1. Take the probe's weight vector as direction **v**.
2. At layer L, project every activation onto **v** and subtract that component. Activations now carry no information along **v**.
3. Rerun the task, measure the drop.

If ablating **v** kills performance on concept X, **v** carried X causally. Stronger than a probe alone.

**Composite intervention for T-02:** find the direction encoding concept X using language A data, ablate it, test whether performance drops in language A **and** in language B. Drop only in A means the encoding is language-specific. That is the Sapir-Whorf claim in mechanistic form, and it is small and runnable.

### 6.3 Model and dataset

Exact checkpoints and exact data. Ryan is checking whether compute has been thought about.

Models must be multilingual, open-weight, small enough for free-tier GPU:

- XGLM-564M or XGLM-1.7B (Meta, multilingual, decoder-only, TransformerLens-friendly)
- mT5-base (580M, encoder-decoder, wide coverage including Bengali)
- Bloom-560M or Bloom-1b7 (46 languages)
- Qwen2.5-0.5B (strong multilingual for the size)

Pick two. One cannot support a general claim. Three is too many for a week.

Datasets:

- XNLI: same inference task in 15 languages, parallel by construction
- FLORES-200: parallel sentences, 200 languages
- Hand-built: 200 to 400 minimal pairs targeting one concept. Slower, far more precise, and a contribution in itself.

For a one-week artifact, 200 hand-built items is right. State the size explicitly. "N=200 minimal pairs" reads as someone who has done this.

### 6.4 Control condition

The comparison proving the effect comes from the claimed cause and not from poking the model at all. Most commonly skipped section, so most likely to be graded hardest.

The paper's provenance control (identical text, only attribution moves) is exactly the right instinct. Say so in the spec.

Three controls T-02 needs:

- **Random direction control.** Ablate a random direction of the same rank. If random ablation degrades performance equally, the direction was not special, the model was just damaged. Non-negotiable. Report both numbers side by side.
- **Matched-concept control.** Run the identical intervention on a concept lexicalized the same way in both languages. If maternal/paternal uncle shows a language-specific effect but "table" does not, the effect tracks the linguistic difference. If "table" shows it too, something generic about the language channel is being measured.
- **Language-identity control.** Multilingual models encode "which language am I in" as its own direction, and the concept direction may partly pick that up. Use two languages sharing the distinction and one that does not. If the effect splits by distinction rather than language identity, it holds.

### 6.5 Primary metric

One number, chosen before running anything, with a stated threshold for a positive result.

Why "primary": collecting ten metrics and reporting the best one is p-hacking. Naming one in advance is a credibility signal.

Options:

- **Indirect effect.** Standard for patching. IE = (output with patch) minus (output without patch), measured in logit difference on the target token. Report as a fraction of total effect for cross-layer comparability.
- **Ablation delta.** Task accuracy before minus after. What matters is the **difference of deltas** across languages: (drop in A) minus (drop in B). Near zero means shared encoding.
- **Probe accuracy gap.** AUC in language A minus AUC in language B, same concept.

State the threshold, for example: "A cross-language ablation gap above 15 accuracy points, with the random-direction control under 3 points, counts as support."

Add error bars. Five seeds minimum, report mean and standard deviation. A single run with no variance estimate gets dismissed fastest.

### 6.6 Falsifying result

The outcome that would make you abandon the hypothesis, written down before running anything. This separates a research spec from a pitch. The Bu1ld manifesto has a line about being willing to be wrong in public; this is where that gets checked.

Three falsifiers, one per control:

- "If the random-direction control produces an ablation drop within 3 points of the concept-direction drop, the effect is not concept-specific and the hypothesis fails."
- "If the cross-language gap is under 5 points, the concept is encoded in a shared multilingual subspace and the Sapir-Whorf claim does not hold for this concept in this model."
- "If the matched-concept control shows the same gap, I am measuring a language-identity artifact, not conceptual structure."

Add one line noting that a negative result is publishable: "This concept is encoded in a shared subspace across languages" is a real finding about multilingual representation. It signals you will report what you find rather than what you hoped for.

### 6.7 Smallest reproducible artifact, one week

A repo he can clone and run. Not a paper, not a deck.

What a week buys on free-tier GPU: two 500M models, 200 stimulus items, one concept, three controls, five seeds.

```
latent-tongue-probe/
  README.md          # one command to reproduce, expected runtime, expected output
  requirements.txt   # pinned versions
  data/
    stimuli.jsonl    # 200 minimal pairs, documented construction
  src/
    probe.py         # fit direction
    ablate.py        # ablation + controls
    run_all.py       # single entry point, --seed flag
  results/
    results.csv      # raw numbers, all seeds
    figure1.png      # cross-language gap with error bars
```

Reproducibility requirements:

- One command runs everything: `python src/run_all.py --seed 0`
- Seeds fixed and exposed
- Versions pinned in requirements.txt
- README states expected runtime and expected output values so a reader knows if their run went wrong
- Raw per-seed numbers committed, not just the summary

**Scope discipline.** One concept, two models, three controls, one metric. Cut before adding. A "future work" section listing five more experiments should be one sentence or deleted.

**Page limit is real.** One page is roughly 500 to 600 words with headers, about 60 to 80 words per section. If it does not fit, the experiment is too big. That is diagnostic, not a formatting problem.

---

## 7. TLDR and plan

### The ask, in one table

| Field | Meaning | One-line answer |
|---|---|---|
| Hypothesis | A claim that can be proven wrong | "Concept X lives in a language-specific subspace in language A but not B" |
| Intervention | The one thing you change, mechanically | Find the concept direction with a probe, then delete it and see what breaks |
| Model + dataset | Exact checkpoints, exact data | Two ~500M multilingual models, 200 hand-built sentence pairs |
| Control | Proves the effect is your cause, not random damage | Ablate a random direction, and a concept identical in both languages |
| Primary metric | One number, threshold set in advance | Accuracy drop in language A minus accuracy drop in language B |
| Falsifier | What result kills the claim | If random ablation hurts just as much, the claim fails |
| Artifact | Clonable repo, one week | One script, fixed seeds, results.csv, one figure |

### Chosen concept: Bengali kinship

English says "uncle." Bengali splits it: কাকা (father's brother), মামা (mother's brother). Same referent, one language forces the distinction, the other does not.

Hypothesis: a multilingual model encodes maternal-vs-paternal side as a recoverable direction when processing Bengali, but that direction is weak or absent in English.

Why this over Russian blue/goluboy: the stimulus set can be built and verified natively, and almost nobody has probed Bengali. Russian has more prior work, which makes motivation easier but novelty lower. The Bu1ld says "ship beats publish" and screens for taste, so the one only you can build is the better pick.

### Compute: Kaggle, not Colab

Kaggle free tier gives 30 GPU-hours a week on a P100 or dual T4, 16GB, sessions up to 12 hours, no random disconnects. Colab free will kill a long sweep. Use Kaggle as primary, Colab for quick debugging.

What fits comfortably:

- XGLM-564M and mT5-base, or Qwen2.5-0.5B
- 200 stimulus pairs
- Probe fit, ablation, 3 controls, 5 seeds
- Full sweep in roughly 2 to 4 GPU-hours

30 hours a week available, so this is not tight. State in the spec: "single T4, Kaggle free tier, under 4 GPU-hours end to end." Naming the compute constraint and designing inside it is a positive signal.

### Order of work

1. Build 200 Bengali/English minimal pairs. Slowest part, about 2 days. Do it first.
2. Extract activations, fit the linear probe per layer, find which layers carry the distinction.
3. Ablate that direction. Measure the drop in both languages.
4. Run the two controls (random direction, matched concept).
5. Five seeds, write results.csv, one figure with error bars.
6. README with the single reproduce command.

### Open decisions before the spec gets drafted

- XGLM plus mT5, or a different model pair
- Kinship only, or kinship plus one backup concept in case kinship turns out flat
