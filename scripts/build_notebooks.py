"""Build the two tutorial notebooks as valid .ipynb JSON.

Written as a builder rather than by hand so the JSON is always well formed and the prose stays
reviewable in one place. Cells are left unexecuted on purpose: training takes a quarter of an hour
and a repository should not ship stale outputs.

If you edit a notebook, edit it HERE and re-run this script, or the next run overwrites your change.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(exist_ok=True)


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}


def write(name, cells):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    (OUT / name).write_text(json.dumps(nb, indent=1) + "\n")
    print(f"  {name:22s} {len(cells)} cells")


# The cell that finds the repository from wherever the kernel happened to start. Jupyter usually
# puts the kernel in the notebook's own directory, but VS Code and Colab do not always, and a wrong
# root fails several cells later with a confusing FileNotFoundError instead of here.
FIND_REPO = """
HERE = Path.cwd()
REPO = next((p for p in [HERE, *HERE.parents] if (p / "pcb_anomaly").is_dir()), None)
if REPO is None:
    raise SystemExit(f"no repository root above {HERE}. Start Jupyter inside the clone.")
sys.path.insert(0, str(REPO))
"""

# ======================================================================================
# 01 — TRAIN
# ======================================================================================
train = [
md("""
# Training an anomaly detector without a single defect

This notebook fits a defect detector on photographs of **correct** circuit boards and nothing else.
No defect images, no masks, no annotation of any kind.

The idea is a bet about manufacturing: good units are the output of the process and defective ones
are the exception it is designed to avoid, so images of correct parts are plentiful and images of
faults are not. Instead of learning what is wrong, the model learns in fine detail what *correct*
looks like and flags whatever it cannot reconcile with that. The failure mode it cannot have is
"I was not trained on that kind of defect".

**What you will do here**

1. Download one VisA circuit-board product
2. Confirm for yourself that the training split contains no defects
3. Meet the anomalib evaluation defaults that will quietly ruin your numbers, and see the proof
4. Look inside the model, printing the real tensor shapes rather than trusting a diagram
5. Train on good boards only
6. Score the held-out split

**What this notebook is not.** It trains **one** product, `pcb3`, so it does not reproduce the
article's headline table. Those figures come from all four products trained as a single model over a
merged directory of symlinks, and this repository does not build that directory. What you get here
is the same method at the same per-image training budget, on one board. Expect image-AUROC near
0.99. AUPIMO is the volatile one: our runs of this model on this board have read anywhere between
0.69 and 0.76, so do not treat a single figure from it as precise.

**What you need.** A CUDA GPU: training and the test pass together peaked at 5.5 GB reserved on our
run, so 8 GB is comfortable. Also about 3.5 GB of disk, being 1.8 GB of Hugging Face parquet cache plus 1.4 GB of
decoded `pcb3` images.

**Runtime.** About 17 minutes on an RTX 5090, roughly 15 of them the 5,000 training steps and 90
seconds the final test pass. Other hardware was not timed. The four-product checkpoint used by
`02_inspect.ipynb` took 38 minutes on the same card at 20,000 steps.
If you only want to see the detector work, skip to `02_inspect.ipynb`, which downloads the
four-product checkpoint instead of training one.
"""),

code("""
import sys, subprocess
from pathlib import Path
""" + FIND_REPO + """
import torch
print("torch     ", torch.__version__)
print("cuda       ", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import anomalib
print("anomalib  ", anomalib.__version__)
print("repo      ", REPO)
"""),

md("""
## 1. The data

VisA is a public industrial inspection benchmark. We use one of its four printed circuit board
products. `pcb3` is a good starting point: an infrared sensor board, and its defects sit in the
middle of the difficulty range.

The download script writes an anomalib-style folder layout. It fetches the whole VisA parquet set
regardless of which category you ask for, so the first run takes a while.
"""),

code("""
CATEGORY = "pcb3"
DATA = REPO / "data/visa" / CATEGORY

if not sorted((DATA / "train/good").glob("*.png")):
    # Jupyter does not show a child process's stdout, so this is captured and reprinted when the
    # download finishes. Nothing appears in between. That is not a hang.
    print(f"downloading VisA {CATEGORY}. This pulls the whole parquet set, several GB, so the "
          f"first run takes a while and prints nothing until it is done.", flush=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts/get_visa.py"),
                        "--categories", CATEGORY], capture_output=True, text=True)
    print(r.stdout[-2000:] or "(no output)")
    if r.returncode:
        raise SystemExit(f"get_visa.py failed:\\n{r.stderr[-2000:]}")

for split in ["train/good", "test/good", "test/bad", "ground_truth/bad"]:
    n = len(list((DATA / split).glob("*"))) if (DATA / split).exists() else 0
    print(f"  {split:20s} {n:4d}")
"""),

md("""
### The premise, checked rather than asserted

The whole method rests on the training split containing no defects. That is worth verifying rather
than believing, because if a single defective board leaks in, the model learns it as normal and will
never flag it again.
"""),

code("""
train_files = sorted((DATA / "train/good").glob("*.png"))
masks       = sorted((DATA / "ground_truth/bad").glob("*.png"))
defective_in_train = sorted((DATA / "train").glob("**/bad/*")) if (DATA / "train").exists() else []

print(f"training images        {len(train_files)}")
print(f"of which defective     {len(defective_in_train)}")
print(f"masks available        {len(masks)}  (test split only, scored against, never fitted on)")
assert len(defective_in_train) == 0, "a defective image reached the training split"
"""),

code("""
import matplotlib.pyplot as plt
import cv2

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, f in zip(axes, train_files[:4]):
    ax.imshow(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
    ax.set_title(f.stem, fontsize=8); ax.axis("off")
fig.suptitle("Four of the boards the model will learn from. Every one of them is correct.", y=1.02)
plt.tight_layout(); plt.show()
"""),

md("""
## 2. Two ways anomalib will quietly ruin your evaluation

Neither raises an error. Both cost us real time.

**The split.** `val_split_mode` defaults to `FROM_TEST` with `val_split_ratio=0.5`, so half your test
set is consumed as a validation split. Your metrics are then computed on roughly half the images you
believe you are evaluating on, and the draw is random per run. We saw AUPIMO readings move by more
than 0.4 between runs of a configuration we had not changed.

**The seed.** `seed` is unset by default, which is at least visible. The trap is that setting it to
zero does not fix it either: anomalib checks the seed for truthiness, and zero is falsy, so `seed=0`
silently means unseeded. And `Folder(seed=...)` pins only the data split, so the setup below calls
Lightning's `seed_everything` as well to pin the rest of the run.

Rather than take either on trust, read them out of the installed library.
"""),

code("""
import inspect as _inspect
import anomalib.data.utils.split as _split
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode

sig = _inspect.signature(Folder.__init__)
for k in ["val_split_mode", "val_split_ratio", "seed"]:
    print(f"  {k:18s} default = {sig.parameters[k].default!r}")

hits = [l.strip() for l in Path(_split.__file__).read_text().splitlines() if "manual_seed" in l]
print("\\n  " + (hits[0] if hits else f"no manual_seed line in {_split.__file__}"))
print("  ^ `if seed` is falsy for 0, so seed=0 silently disables seeding")
"""),

md("""
### The corrected setup

`FROM_TRAIN` with a small ratio takes validation from the good training images instead. The full
test set is then tested, there is no leakage, and the normalisation and threshold statistics are
fitted on good images only, which is what a deployment actually has.

One consequence to expect: because the validation split is now all-normal, anomalib warns once per
epoch that its adaptive threshold will be the highest normal score it saw. That warning is correct
and harmless here. It concerns anomalib's own pass/fail threshold, which nothing downstream uses:
AUROC and AUPIMO are threshold-free, and the accept/reject limit in `02_inspect.ipynb` is calibrated
separately on good boards.
"""),

code("""
from lightning.pytorch import seed_everything

seed_everything(1, workers=True)   # Folder's own seed covers the split and nothing else

datamodule = Folder(
    name=f"visa_{CATEGORY}",
    root=DATA,
    normal_dir="train/good",         # the only images used for fitting
    abnormal_dir="test/bad",         # evaluation only, never seen during training
    normal_test_dir="test/good",
    mask_dir="ground_truth/bad",
    train_batch_size=8,
    eval_batch_size=8,

    # The two overrides that matter.
    val_split_mode=ValSplitMode.FROM_TRAIN,
    val_split_ratio=0.05,
    seed=1,
)
datamodule.setup()
print(f"  train {len(datamodule.train_data):4d}   (good only)")
print(f"  val   {len(datamodule.val_data):4d}   (good only, drawn from train)")
print(f"  test  {len(datamodule.test_data):4d}   (the full held-out split, nothing consumed)")
"""),

md("""
## 3. Inside the model

Dinomaly's subtitle is its thesis: *The Less Is More Philosophy in Multi-Class Unsupervised Anomaly
Detection*. The mechanism is reconstruction. A frozen encoder describes the image, a small decoder is
trained to reproduce those descriptions, and because the decoder only ever practised on good boards,
whatever it cannot rebuild was never part of what "good" looks like. **The reconstruction error is
the anomaly map.**

The design then goes out of its way to stop the decoder becoming too capable, because a decoder that
can rebuild anything will happily rebuild a defect and report nothing wrong. Four choices do that:
a dropout bottleneck, linear attention that focuses poorly on purpose, a loss that down-weights
points already reconstructed well, and comparison in fused layer groups rather than layer by layer.

Rather than describe the architecture, print it.
"""),

code("""
from anomalib.metrics import AUPIMO, AUROC, Evaluator
from anomalib.models import Dinomaly

# Dinomaly's default test metrics are image and pixel AUROC and F1. AUPIMO is the one worth ranking
# on, so it has to be asked for. strict=False lets it return NaN instead of raising when a map is
# degenerate, which happens if you cut the step budget down to a smoke test.
model = Dinomaly(evaluator=Evaluator(test_metrics=[
    AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
    AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
    AUPIMO(fields=["anomaly_map", "gt_mask"], strict=False),
]))

enc = model.model.encoder
vit = enc.feature_extractor      # the ViT itself; enc is a TimmFeatureExtractor wrapper

frozen    = sum(p.numel() for p in enc.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"  frozen encoder      {frozen:>12,}   DINOv2 ViT-B/14, not one weight changes")
print(f"  trained             {trainable:>12,}   bottleneck + decoder, fitted on good boards only")
print()
print(f"  width               {vit.embed_dim}  = {vit.blocks[0].attn.num_heads} heads x "
      f"{vit.embed_dim // vit.blocks[0].attn.num_heads} dims per head")
print(f"  patch size          {vit.patch_embed.patch_size}")
print(f"  patch projection    {tuple(vit.patch_embed.proj.weight.shape)}"
      f"  = {vit.patch_embed.proj.weight.numel():,} numbers")
"""),

md("""
A patch *grows* on its way in rather than shrinking. Its `3 x 14 x 14 = 588` raw pixel values are
projected up to 768 features, and the width stays 768 through every block.

The token count is pure geometry. Anomalib resizes the input to 448 pixels and centre-crops it to
392, and `392 / 14 = 28` across and down, so `28 x 28 = 784` patch tokens. Five more are prepended,
one CLS and four registers. Confirm it with a real forward pass.
"""),

code("""
seen = {}
h = model.model.encoder.feature_extractor.blocks[0].register_forward_hook(
    lambda m, i, o: seen.update(shape=tuple(i[0].shape)))
with torch.no_grad():
    model.model.encoder(torch.zeros(1, 3, 392, 392))
h.remove()

b, tokens, width = seen["shape"]
print(f"  sequence entering block 0:  {tokens} x {width}")
print(f"    {tokens - 5} patch tokens  (28 x 28)")
print(f"    1 CLS + 4 register tokens")
"""),

code("""
print("  tapped encoder layers :", model.model.target_layers)
print("  fusion groups         :", model.model.fuse_layer_encoder)
print("  decoder depth         :", len(model.model.bottleneck), "bottleneck /",
      len(model.model.decoder), "decoder blocks")
print("  loss                  :", type(model.model.loss_fn).__name__)
"""),

md("""
Blocks 0 and 1 are still edge and texture detectors, and the last two have specialised toward
DINOv2's own self-supervised objective. The middle is where general, transferable structure lives,
which is why the extractor hands back `blocks.2` through `blocks.9`.

## 4. Train

Only the bottleneck and the decoder are fitted. The encoder never moves.

Anomalib's default budget is 5,000 steps, which is written for a single product and is what we use
here. If you train a unified model across several products the dataset is several times larger and
the same step count gives each image a fraction of its intended exposure, so scale the budget to
keep epochs-per-image constant. Getting this wrong is not a small effect: at 5,000 steps our
four-product model scored mean AUPIMO 0.349 against 0.519 for four separate ones, and at 20,000
steps, which restores the same epochs-per-image, the same configuration reached 0.589 and beat them.
"""),

code("""
from anomalib.engine import Engine

MAX_STEPS = 5_000       # one product. Four products as one model need 20_000 for the same exposure.

engine = Engine(max_steps=MAX_STEPS, accelerator="gpu", devices=1)
engine.fit(model=model, datamodule=datamodule)
"""),

md("""
## 5. Score the held-out split

`AUROC` answers "given one good board and one defective board, how often is the defective one scored
higher". `AUPIMO` is the more informative of the two and is the metric we asked the evaluator for
above: pixel AUROC is weak here because over 99 percent of pixels are ordinary background, so a model
scores well simply by ranking the board area low. AUPIMO measures overlap **per image**, at a
threshold where almost nothing on a good image fires, so one badly localised board cannot hide behind
easy background pixels from other boards. It is scored per defective image; the evaluator reports
the mean.
"""),

code("""
import numpy as np

results = engine.test(model=model, datamodule=datamodule)
for k, v in results[0].items():
    a = np.asarray(torch.as_tensor(v).cpu() if torch.is_tensor(v) else v, dtype=float).ravel()
    note = f"   (mean over {a.size} defective images)" if a.size > 1 else ""
    print(f"  {k:28s} {np.nanmean(a):.4f}{note}")
"""),

md("""
## 6. Save the weights

Only the trained parts are saved. The frozen DINOv2 encoder is not in the file: timm downloads it
whenever the model is constructed, so shipping a copy would add about 330 MB for nothing.
"""),

code("""
ART = REPO / "artifacts"; ART.mkdir(exist_ok=True)
# Deliberately NOT artifacts/detector.pt. That name belongs to the four-product checkpoint that 02
# downloads, and overwriting it would destroy a 234 MB file you would have to fetch again. It would
# also leave your weights paired with thresholds.json, whose limits were calibrated against that
# checkpoint, not yours.
out = ART / f"detector_{CATEGORY}_{MAX_STEPS // 1000}k.pt"
trained = {k: v for k, v in model.state_dict().items() if not k.startswith("model.encoder.")}
torch.save(trained, out)
print(f"wrote {out.relative_to(REPO)}  ({out.stat().st_size / 1048576:.0f} MB)")
print("\\nartifacts/thresholds.json holds limits for the four-product checkpoint, not for this one.")
print("To use these weights for accept/reject you would recalibrate on your own good boards.")
print("\\nNext: 02_inspect.ipynb runs the four-product checkpoint end to end.")
"""),
]
write("01_train.ipynb", train)


# ======================================================================================
# 02 — INSPECT
# ======================================================================================
inspect_nb = [
md("""
# Inspecting a board: score, decision, and where the defect is

`01_train.ipynb` produced a model that scores how far a board sits from what a correct one looks
like. A score is not yet an inspection station. This notebook turns it into two things an operator
can act on:

- **a decision**, by comparing the score against a limit fixed for that product
- **a location**, by taking the highest peaks of the anomaly map

You do not need to have run the training notebook. The cell below downloads the published
checkpoint, which was fitted on all four VisA circuit-board products, 3,614 good boards, no defects
and no masks.

**What you need.** The checkpoint download needs the GitHub release to exist, and the cell below
says so plainly if it does not. A CUDA GPU is optional here, unlike in the training notebook: the
runtime falls back to CPU and tells you which one it picked, but a board takes about 50 ms on an
RTX 5090 and considerably longer without one.
"""),

code("""
import sys, subprocess
from pathlib import Path
""" + FIND_REPO + """
CATEGORY = "pcb3"

# The checkpoint. Captured rather than inherited, because Jupyter does not show fd 1 from a child
# process, and fetch_checkpoint writes the useful part of a failure to stderr.
if not (REPO / "artifacts/detector.pt").exists():
    r = subprocess.run([sys.executable, str(REPO / "scripts/fetch_checkpoint.py")],
                       capture_output=True, text=True)
    print(r.stdout, r.stderr, sep="")
    if r.returncode:
        raise SystemExit("could not fetch the checkpoint, see the message above")

# The data. This notebook is meant to stand alone, so it downloads what it needs rather than
# assuming 01 was run first.
if not sorted((REPO / "data/visa" / CATEGORY / "test/bad").glob("*.png")):
    print(f"downloading VisA {CATEGORY}, several GB on first run, and nothing prints until it "
          f"finishes ...", flush=True)
    r = subprocess.run([sys.executable, str(REPO / "scripts/get_visa.py"),
                        "--categories", CATEGORY], capture_output=True, text=True)
    print(r.stdout[-2000:] or "(no output)")
    if r.returncode:
        raise SystemExit(f"get_visa.py failed:\\n{r.stderr[-2000:]}")

import cv2, numpy as np, matplotlib.pyplot as plt
from pcb_anomaly import load, inspect
from pcb_anomaly.render import boxes, heatmap

STATE = load()
"""),

md("""
## The accept/reject limit

The limit is set **per product**, at the 95th percentile of that product's good-board scores,
computed on a calibration half of the good images. **No defect score is ever consulted when choosing
it**, which is what makes the procedure runnable before you have collected a single defect. That is
the practical difference between this approach and a supervised one: you can commission the station
on day one.
"""),

code("""
for part, limit in STATE["thr"]["per_board_p95"].items():
    print(f"  {part}   limit {limit:.4f}")
"""),

md("""
## One defective board

The heatmap is the model's raw output: how far each patch sits from what the model expects to see
there. The boxes are what an operator would actually be shown, drawn from the peaks of that map
after it has been scaled back up to the size of the photograph.

Peaks are restricted to the board itself. A hot pixel on the background is not a defect in the part,
and letting one win would put a box on the bench rather than on the unit.
"""),

code("""
bad = sorted((REPO / "data/visa" / CATEGORY / "test/bad").glob("*.png"))
img = cv2.imread(str(bad[0]))

r = inspect(img, board=CATEGORY)
print(f"  score     {r['score']:.4f}")
print(f"  limit     {r['threshold']:.4f}")
print(f"  verdict   {r['verdict']}")
print(f"  peaks     {r['peaks']}")
print(f"  time      {r['ms_total']} ms   (first call, includes warm-up; steady state is ~50 ms on a GPU)")
"""),

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));                 axes[0].set_title("the unit under test")
axes[1].imshow(cv2.cvtColor(heatmap(img, r["_raw_map"]), cv2.COLOR_BGR2RGB));   axes[1].set_title("anomaly map")
# boxes() is only meaningful on a reject: candidate boxes are the station saying "look here",
# and on a board that passed there is nothing to look at.
marked = boxes(img, r["peaks"], r["_half"]) if r["verdict"] == "FAIL" else img
axes[2].imshow(cv2.cvtColor(marked, cv2.COLOR_BGR2RGB));              axes[2].set_title(f"regions marked  ({r['verdict']})")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
"""),

md("""
## The board with nothing wrong with it

This is the case that decides whether a station is usable. Finding a defect is the easy half. Not
inventing one is the half that determines whether an operator still trusts the marks in week two.

Note what the heatmap looks like here. Heatmaps are normalised per panel, so on a board with no
defect there is no peak to find and normalisation stretches ordinary noise across the full colour
range. It looks alarming and the score says otherwise. **Read the score, not the colour.**
"""),

code("""
good = sorted((REPO / "data/visa" / CATEGORY / "test/good").glob("*.png"))
img_g = cv2.imread(str(good[0]))
rg = inspect(img_g, board=CATEGORY)

print(f"  score     {rg['score']:.4f}   (limit {rg['threshold']:.4f})")
print(f"  verdict   {rg['verdict']}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img_g, cv2.COLOR_BGR2RGB));        axes[0].set_title("a board that passed")
axes[1].imshow(cv2.cvtColor(heatmap(img_g, rg["_raw_map"]), cv2.COLOR_BGR2RGB));  axes[1].set_title("its anomaly map, normalised")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
"""),

md("""
## Across the whole split

One board proves nothing. Run every held-out image for this product and count. Read the false-alarm
figure as the friendlier of the two numbers: half of these good boards are the calibration half that
set the limit in the first place, so only the other half is genuinely held out from it. The
defective boards are held out from everything.
"""),

code("""
rows = []
for f in bad:
    rows.append(("defective", inspect(cv2.imread(str(f)), board=CATEGORY)))
for f in good:
    rows.append(("good", inspect(cv2.imread(str(f)), board=CATEGORY)))

n_bad  = sum(1 for t, _ in rows if t == "defective")
n_good = sum(1 for t, _ in rows if t == "good")
caught = sum(1 for t, r in rows if t == "defective" and r["verdict"] == "FAIL")
false_alarm = sum(1 for t, r in rows if t == "good" and r["verdict"] == "FAIL")

print(f"  defective boards rejected   {caught:3d} of {n_bad:3d}   ({caught / n_bad:.1%})")
print(f"  good boards falsely rejected {false_alarm:3d} of {n_good:3d}   ({false_alarm / n_good:.1%})")
"""),

md("""
### The limit is a dial, not a constant

Which end of it you want depends on whether a missed defect reaching the customer or an
operator-minute costs you more, and that choice belongs to whoever owns the cost of a miss. Sweeping
it shows the trade directly.
"""),

code("""
scores_bad  = np.array([r["score"] for t, r in rows if t == "defective"])
scores_good = np.array([r["score"] for t, r in rows if t == "good"])

print(f"  {'limit at':>16}  {'value':>8}  {'caught':>8}  {'false alarms':>13}")
for q in [90, 95, 99, 100]:
    thr = np.percentile(scores_good, q)
    print(f"  {'p' + str(q) + ' of good':>16}  {thr:8.4f}  "
          f"{(scores_bad >= thr).mean():7.1%}  {(scores_good >= thr).mean():12.1%}")
"""),

md("""
## What this does not cover

Two honest limits, because results on public data are easy to over-read.

**The model learns what a good unit looks like including how it is lit and framed.** Any variation
you allow in lighting or camera position becomes variation it must treat as normal, and that costs
sensitivity to the small differences that matter. A fixed station with controlled lighting is part
of the method, not an optional extra.

**Some of the separation on this benchmark is not the defect.** On VisA circuit boards, background
brightness alone separates good from defective at 0.68 to 0.77 AUROC. A properly controlled capture
station removes that shortcut, and some of the headline numbers with it.
"""),
]
write("02_inspect.ipynb", inspect_nb)

# validate
for f in sorted(OUT.glob("*.ipynb")):
    nb = json.loads(f.read_text())
    assert nb["nbformat"] == 4 and nb["cells"]
    assert not any(c.get("outputs") for c in nb["cells"]), f"{f.name} ships outputs"
    kinds = [c["cell_type"] for c in nb["cells"]]
    print(f"  {f.name}: valid JSON, {kinds.count('markdown')} md + {kinds.count('code')} code")
