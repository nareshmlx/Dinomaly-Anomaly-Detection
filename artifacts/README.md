# artifacts/

Two files, and only one of them is large.

| file | size | what it is | in git? |
|---|---|---|---|
| `thresholds.json` | 7 KB | one pass/fail limit per product, plus the full operating curve | yes |
| `detector.pt` | 234 MB | the trained Dinomaly bottleneck and decoder | no, run `scripts/fetch_checkpoint.py` |

## Why the detector is not in the repository

Two reasons. It is 234 MB and GitHub caps single files at 100 MB. And the version that would go in git is
already the slim one: the full training checkpoint is 1,268 MB, of which 330 MB is the frozen DINOv2
encoder and the rest is optimiser state. Stripping both leaves only weights that were actually trained
here, and timm downloads the encoder on its own when the model is constructed.

Scores are bit-identical either way. The same board reads 0.080475 from the full checkpoint and 0.080475
from the slim one.

## What each file was fitted on

| | fitted on | labels used |
|---|---|---|
| `detector.pt` | 3,614 good images from the train split | none |
| `thresholds.json` | good-board scores only, one calibration half per product | none |

148 million parameters, of which **zero needed a human**. Nothing here was fitted on a defect, a mask or
an annotation: the detector saw good boards only, and each limit is a percentile of that product's
good-board scores. That is the whole point of the design — detection and localisation need photographs
of correct units and nothing else.

The numbers these files produce, and the protocols behind them, are in [docs/results.md](../docs/results.md).
