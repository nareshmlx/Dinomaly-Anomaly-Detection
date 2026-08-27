# PCB Anomaly Detection Without Defect Images

Train a defect detector on photographs of **correct** circuit boards and nothing else. No defect
images, no masks, no annotation of any kind.

This is the code companion to the LearnOpenCV article **Anomaly Detection for Product-Line Parts:
Training a Detector Without a Single Defect** (link added when the article goes live).

| | |
|---|---|
| Method | [Dinomaly](https://arxiv.org/abs/2405.14325), CVPR 2025, via [anomalib](https://github.com/open-edge-platform/anomalib) 2.5.1 |
| Data | [VisA](https://github.com/amazon-science/spot-diff) `pcb1`–`pcb4`, a public benchmark with pixel-accurate masks |
| Trained on | 3,614 good boards. **Zero defective images, zero masks** |
| Held-out split | 802 boards, 402 good and 400 defective |

## What this repo does, and what it does not

Two things. It **scores** a board against that product's pass/fail limit, and it **points** at the regions
that earned the score. `pcb_anomaly.inspect(image, board="pcb3")` takes an image and a product name and
returns a dict holding the score, the verdict and up to three peak locations.

It does not recognise which product it is looking at, name the kind of defect it found, or serve anything
over HTTP. Those are separate problems. This is the detector and nothing else.

---

## Why train on good parts only

A new product goes into production on Monday and needs a visual inspection step by Friday. Ask what
the defects look like and nobody knows yet: the line has been running four days, and the faults that
will actually hurt you have not happened. You cannot train a defect detector, because you have no
defects.

What you do have, in abundance, is correct units. That asymmetry is not a ramp-up annoyance, it is
the permanent shape of the problem. Good parts are the output of the process; defective ones are the
exception it is designed to avoid.

So the model learns in fine detail what *correct* looks like and flags whatever it cannot reconcile
with that. It is never shown a fault. The failure mode it cannot have is "I was not trained on that
kind of defect".

---

## Quick start

```bash
git clone https://github.com/nareshmlx/Dinomaly-Anomaly-Detection
cd Dinomaly-Anomaly-Detection

pip install -r requirements.txt          # or: uv sync
jupyter lab notebooks/
```

On an RTX 50-series card install torch from the CUDA 12.8 index first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### The two notebooks

| Notebook | What it does | Runtime |
|---|---|---|
| **[`01_train.ipynb`](notebooks/01_train.ipynb)** | Download one product, verify the training split has no defects, look inside the model, train, score | ~10 min on a GPU |
| **[`02_inspect.ipynb`](notebooks/02_inspect.ipynb)** | Load the published checkpoint, score a board, draw the regions, sweep the accept/reject limit | ~2 min |

`02_inspect.ipynb` does not depend on `01_train.ipynb`. It downloads the trained checkpoint, so you
can see the detector work before deciding whether to train one.

### The trained checkpoint

234 MB, too large for git, published as a release asset:

```bash
python scripts/fetch_checkpoint.py
```

It holds only the weights we trained, the Dinomaly bottleneck and decoder. The frozen DINOv2 encoder
is not in the file because timm downloads it whenever the model is constructed.

---

## Results

Measured on the full held-out split, 400 defective and 402 good boards, with all four products
trained as a **single** model rather than four.

| Board | Image AUROC | Pixel AUROC | AUPIMO |
|---|---|---|---|
| pcb1 | 98.7 | 99.3 | 0.690 |
| pcb2 | 99.2 | 97.7 | 0.413 |
| pcb3 | 99.2 | 98.9 | 0.753 |
| pcb4 | 99.8 | 98.4 | 0.500 |
| **Mean** | **99.2** | **98.6** | **0.589** |

Turned into a decision, with the limit set per product at the 95th percentile of that product's
good-board scores and **no defect score consulted**:

| | |
|---|---|
| Defective boards rejected | 97.25% |
| Good boards falsely rejected | 4.95% |

Full protocols, confidence intervals and the refuted experiments are in
[`docs/results.md`](docs/results.md). What the detector is doing internally, every number read out of the
loaded model rather than from documentation, is in [`docs/how-it-works.md`](docs/how-it-works.md).

---

## Two things worth knowing before you copy this

**Pin the evaluation protocol before you tune the model.** Anomalib's `Folder` defaults to
`val_split_mode=FROM_TEST` with `val_split_ratio=0.5`, which quietly consumes half your test set as
validation, and `seed=0` is falsy so it silently means *unseeded*. Neither raises an error. Fixing
how the data was split moved our numbers more than any model change did. `01_train.ipynb` reads both
of these out of the installed library so you can see them for yourself.

**A fixed, well-lit station is part of the method.** The model learns what a good unit looks like
*including* how it is lit and framed, so variation you allow in lighting or camera position becomes
variation it must treat as normal. Stated the other way, on this public benchmark background
brightness alone separates good from defective at 0.68 to 0.77 AUROC, so some of what any method
reports here reflects capture conditions rather than defects.

---

## Layout

```
notebooks/       01_train.ipynb · 02_inspect.ipynb
pcb_anomaly/     the inference runtime the notebooks import
  runtime.py       load the detector once, then score, decide and locate per board
  board.py         find the board in the frame, so peaks land on the part
  peaks.py         top-k peaks with non-maximum suppression
  render.py        heatmaps and boxes
scripts/         get_visa.py · build_all4.py · fetch_checkpoint.py · measure_metrics.py
artifacts/       thresholds.json, and detector.pt once fetched
docs/            how-it-works.md · results.md
```

## Licence

[MIT](LICENSE). The dataset and the pretrained encoder are fetched at runtime rather than
redistributed here, and carry their own terms — VisA is CC BY 4.0, the DINOv2 weights are Apache 2.0.
Both are named in the LICENSE file.
