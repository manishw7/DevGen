# Understanding Paper 2 — A Complete Guide

**Paper title (working):** *Glyph-Aware Latent Diffusion for Devanagari Handwritten Word Generation*

This document explains everything: the problem, every design decision, every file,
every metric, and the questions a supervisor or reviewer will ask. Read it top to
bottom once, then use it as reference.

---

## 1. What are we building, in one sentence?

A neural network that, given any Devanagari word as text (e.g. "नमस्ते"), draws a
realistic **handwritten image** of that word — as if a human wrote it with a pen.

Input: `"नमस्ते"` (a string) → Output: a 64×256 pixel image of handwritten नमस्ते.

## 2. Why does this matter? (the Introduction argument)

1. **Data scarcity.** Devanagari handwriting recognition (OCR/HTR) is behind
   English mostly because of data. English has IAM, CVL, RIMES...; Devanagari has
   basically one word-level dataset (IIIT-INDIC-HW-WORDS Hindi, ~95k words).
   A good generator = unlimited synthetic training data = better OCR for Nepali/Hindi.
2. **Nobody has done it.** We verified (multiple searches, latest 2026-07-03):
   diffusion-based handwriting generation exists for English, Chinese, Ukrainian,
   math notation — **not for Devanagari or any Indic script**. We are first.
3. **It is not a trivial port.** Devanagari has properties that break the methods
   used for English (Section 4). Showing *how* to handle them is the science.

## 3. Why is Devanagari harder than English?

Three script properties, know these cold:

- **Matras (dependent vowels).** A consonant + vowel doesn't sit side by side —
  the vowel sign wraps around the consonant: above (े), below (ु), left (ि),
  right (ा). The letter "कि" is written with the ि *before* the क visually,
  even though क comes first logically.
- **Conjuncts (ligatures).** When consonants cluster, they FUSE into a new shape.
  क + ् + ष = क्ष — which looks nothing like क followed by ष. There are hundreds
  of these, with a long tail of rare ones. A generator that just draws letters
  side by side produces text no Nepali reader accepts.
- **Shirorekha (headline).** The horizontal line connecting all letters in a word.
  It must be continuous — breaks look wrong instantly.

**The unit of writing is the akshara** — an orthographic syllable like क्षे or हिं —
not the individual Unicode character. This one insight drives the whole paper.

## 4. Why did the old approach (our ControlNet) fail?

Our first attempt (the DevGen app's generator) took Stable Diffusion 1.5 (a
photo generator) and bolted on a ControlNet that traced font-rendered images
of the word. Three architectural mistakes:

1. **CLIP cannot read Devanagari.** SD1.5 understands prompts through CLIP,
   whose tokenizer has almost no Devanagari coverage. We worked around it by
   transliterating ("नमस्ते" → "namastea") — a lossy hack; the model never truly
   knew what word it was drawing.
2. **Wrong prior.** SD1.5 knows photos: faces, dogs, sunsets. That 860M-parameter
   knowledge is dead weight for black-ink-on-white-paper, and fighting it wastes
   capacity.
3. **Wrong shape.** SD wants 512×512 (we used 256×256 squares). Words are wide
   strips — 64×256 is the natural shape (what WordStylist uses for English).

Lesson (goes in the paper as motivation): *for handwriting, a small model trained
from scratch on the right representation beats a huge model adapted with hacks.*

## 5. The new architecture — piece by piece

### 5.1 What is a diffusion model? (30-second version)

Training: take a real image, add random noise to it (a random amount, from
"slightly grainy" to "pure static"), and train a network to predict what noise
was added. That's it — the loss is mean-squared error on the noise.

Generation: start from pure random static and repeatedly ask the network
"what noise is in this?" — subtracting a bit each time. After ~50 steps the
static has been "denoised" into a brand-new image. Conditioning (the word we
want) steers what the image becomes.

### 5.2 Why *latent* diffusion (the L in LDM)?

Doing diffusion on 64×256×3 pixels is wasteful. Instead a **VAE**
(variational autoencoder) first compresses each image 8× per side:
64×256 image → **4×8×32 latent** (a tiny abstract summary). Diffusion runs in
that small space; afterwards the VAE decoder turns the final latent back into
pixels.

We use the public pre-trained `sd-vae-ft-mse` VAE and **freeze it** — it
already reconstructs handwriting well, no need to train it.

**Our efficiency trick:** because the VAE is frozen, every training image's
latent never changes. So we encode all 70k images ONCE, cache the latents
(~150MB), and training touches only the small UNet. This is why a
from-scratch model trains in ~2.5 hours on a free Kaggle T4 instead of days.

### 5.3 The UNet (the actual trainable model)

A ~65M-parameter UNet2DConditionModel — the standard diffusion backbone:
downsampling path → bottleneck → upsampling path with skip connections. At two
resolution levels it has **cross-attention** layers: places where the image
representation "looks at" the text condition to decide what to draw where.

Compare: SD1.5's UNet is 860M parameters. Ours is 13× smaller because
handwriting on white paper is a vastly simpler visual world than photographs.

### 5.4 Conditioning — where the paper's novelty lives

The text "नमस्ते" must become vectors the UNet can attend to:

1. **Tokenize** the word into units (see 5.5 — THE design choice).
2. Each unit gets a **learned embedding** (a 384-dim vector, trained from scratch).
3. Add positional embeddings (so the model knows unit order).
4. Pass through a small 2-layer Transformer encoder (lets units see context).
5. The result is the "memory" that the UNet's cross-attention reads.

No CLIP anywhere. The model learns Devanagari directly from the data.

### 5.5 Codepoint vs Akshara tokenization — THE experiment

Two ways to split "क्षेत्र" into units:

| Mode | Units | Count |
|---|---|---|
| **codepoint** | क ् ष े त ् र | 7 tokens |
| **akshara** | क्षे त्र | 2 tokens |

- **Codepoint model** must *learn* that क+्+ष fuses into the क्ष shape — the
  ligature rule is implicit in data statistics.
- **Akshara model** gets क्षे as ONE token with its own embedding — the ligature
  is explicit. But rare conjuncts have few training examples, and **unseen**
  conjuncts have no embedding at all — our tokenizer then falls back to
  decomposing them into codepoints (so nothing is unrepresentable).

**Research question: which representation generalizes better — especially to
conjuncts never seen in training?** Nobody has asked this for any abugida.
We train both models identically and compare. Either outcome is a finding:
- Akshara wins → orthographic-unit conditioning matters for abugidas.
- Codepoint wins → compositional learning beats holistic units (also citable!).

### 5.6 Classifier-free guidance (CFG)

During training, 10% of the time we hide the word (replace with a NULL token).
The model thus learns both "draw this word" and "draw generic handwriting".
At generation we compute both predictions and push the output *away* from
generic, *toward* the word: `eps = eps_uncond + 2.5 × (eps_cond − eps_uncond)`.
The 2.5 is the guidance scale — higher = more literal text, lower = more
natural variation. This measurably improves whether the right word appears.

### 5.7 EMA (exponential moving average)

Diffusion training is noisy; the weights jitter. We keep a slow-moving average
copy of the UNet weights (decay 0.9995) and use THAT for generation — standard
practice, visibly better samples.

## 6. Evaluation — how we prove it works

Three questions, three measurements:

### 6.1 Does it LOOK like real handwriting? → FID

Fréchet Inception Distance: run real images and generated images through a
pretrained vision network, compare the two feature distributions. Lower = more
realistic. FID is the standard generation metric (every related paper reports it).
We compare 1000 generated in-vocab images against 5000 real test images.

### 6.2 Does it write the RIGHT word? → recognizer-judge content fidelity

We take our best Devanagari OCR model (the TrOCR-LoRA from Paper 1), make it
read every generated image, and compare its output with the intended word:
- **CER** — character error rate
- **AER** — akshara error rate (our script-aware metric from Paper 1)
- **Word accuracy** — exact match

If the OCR reads "नमस्ते" off an image we generated for "नमस्ते", the model
genuinely wrote the word. This catches pretty-but-wrong generations that FID misses.

### 6.3 Does it GENERALIZE? → the conjunct-stratified OOV protocol (our invention)

Three test vocabularies, 200 words each (`experiments/vocab/`):

| Set | Contents | Tests |
|---|---|---|
| `iv` | real words from training vocab | basic quality |
| `oov_seen` | NEW pseudo-words built from seen aksharas | novel combinations |
| `oov_unseen_conjunct` | words containing conjuncts NEVER seen in training | true generalization — can it draw a ligature it never saw? |

Content fidelity is computed per set. The interesting row is the last one,
compared between the codepoint and akshara models. That comparison IS the paper.

## 7. The files — what does what

```
paper2/
├── ldm/
│   ├── tokenizer.py       codepoint + akshara tokenizers; akshara falls back to
│   │                      codepoint decomposition for unseen conjuncts
│   ├── train_ldm.py       precompute latents → train UNet (CFG dropout, EMA,
│   │                      fp16, checkpoints every 10 epochs, previews every 25)
│   └── sample_ldm.py      DDIM sampling with CFG; vocab file → images/ + labels.csv
├── experiments/
│   ├── build_eval_vocab.py  built the three 200-word test sets (already done)
│   ├── vocab/               the test sets themselves
│   ├── content_fidelity.py  OCR-judge scoring of generated images
│   └── compute_fid.py       FID vs real test images
├── paper/main.tex          the paper draft (results section pending)
├── Kaggle_Paper2_LDM_Scratch.ipynb   one-shot: trains both models + evaluates
└── README.md               short roadmap
```

## 8. What the Kaggle run produces, and how to read it

The summary cell prints a table like:

```
run                               CER     AER    WAcc
v2_akshara_iv                   0.08    0.11    0.81   <- in-vocab quality
v2_akshara_oov_seen             0.15    0.19    0.62
v2_akshara_oov_unseen_conjunct  0.31    0.38    0.35   <- generalization
v2_codepoint_iv                 0.09    0.12    0.79
v2_codepoint_oov_seen           0.16    0.21    0.60
v2_codepoint_oov_unseen_conjunct 0.24   0.30    0.44   <- compare with akshara row!
FID v2_akshara_iv: 28.4
FID v2_codepoint_iv: 30.1
```
(numbers above invented — yours will differ)

How to read: iv rows should be clearly best. The unseen-conjunct rows decide the
headline: whichever model reads better there generalizes better. Also check the
`preview_epochXXXX.png` grids — you should SEE handwriting getting cleaner
through training; by ~epoch 100 words should be legible.

Rule of thumb for success: iv word accuracy above ~0.5 and FID under ~60 means
the approach works; polish comes from more epochs.

## 9. Questions you will be asked (defense prep)

**"Why diffusion, not GAN?"** GANs (our earlier StyleGAN) are unstable to train,
mode-collapse on long-tail conjuncts, and give no likelihood-based control.
Diffusion training is a simple regression (predict noise), stable, and now
state of the art for handwriting in every recent paper (WordStylist, DiffusionPen).

**"Why train from scratch instead of fine-tuning Stable Diffusion?"** Section 4:
CLIP can't encode Devanagari, the photo prior is useless, and our from-scratch
UNet is 13× smaller yet task-matched. We also have the failed-ControlNet
experience as evidence.

**"Why 64×256?"** Words are wide strips. Standard in the field (WordStylist uses
the same for English). Height 64px preserves matra detail above/below the line.

**"Where do writer styles come in?"** They don't, yet — the public dataset
exposes no writer identities. We generate *content-faithful, style-averaged*
handwriting. Style conditioning is future work (needs writer metadata or an
unsupervised style encoder). Say this upfront; hiding limitations kills papers.

**"Is your judge biased?"** The OCR judge was trained on real data only, never
on generated images, so it has no reason to favor our generator. We also report
FID, which is judge-free. (If both models score identically on the judge but
differ on FID, we still learn something.)

**"Only one language?"** Devanagari serves Nepali, Hindi, Marathi, Sanskrit —
~600M users — and is where we can validate quality natively. The method has no
Devanagari-specific hardcoding (the tokenizer generalizes to any abugida), so
multi-script extension is the journal version.

**"How is this different from WordStylist?"** WordStylist = English, character
embeddings, writer styles, IAM. Us = first abugida, akshara-vs-codepoint
ablation (a question that doesn't exist in English — no conjuncts), conjunct-
stratified OOV protocol, recognizer-judge fidelity. Same family, new science.

## 10. Glossary

| Term | Meaning |
|---|---|
| LDM | Latent Diffusion Model — diffusion in VAE-compressed space |
| VAE | autoencoder that compresses images 8×/side; frozen in our system |
| UNet | the denoising network (the only thing we train, ~65M params) |
| Cross-attention | mechanism letting image features "read" the text condition |
| Akshara | orthographic syllable — the natural unit of Devanagari writing |
| Conjunct | fused consonant-cluster ligature (क्ष, त्र, स्थ...) |
| Matra | dependent vowel sign attached around a consonant |
| CFG | classifier-free guidance — amplifies the text condition at sampling |
| EMA | slow-averaged weights used for generation |
| DDPM/DDIM | noise schedulers: DDPM for training (1000 steps), DDIM for fast sampling (50) |
| FID | distribution distance between real and generated images (lower better) |
| CER/AER | character / akshara error rate of the OCR judge on generated images |
| OOV | out-of-vocabulary — words never seen in training |
| IV | in-vocabulary — words from the training set |
