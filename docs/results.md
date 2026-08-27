# Results

This repository does two things: it scores a board against a per-product limit, and it points at the
regions that earned the score. Everything below measures those two things.

Every figure carries the protocol that produced it and the sample size it rests on. Intervals are Wilson
95 percent score intervals, because a bare percentage from a small sample is not defensible and this
project got caught out by that more than once. Section 5 lists the ones that collapsed.

Measured on **VisA pcb1 to pcb4**, the complete held-out test split: 400 defective and 402 good images,
none of which the detector or the thresholds ever saw. This demonstrates the method on public consumer
boards. It is not a prediction for any other set of parts.

Sections 1 and 6 are reproducible from this repository with `scripts/measure_metrics.py`. The rest was
measured with training and experiment code that lives elsewhere, so the protocols are stated in enough
detail to judge the numbers rather than take them on faith.

## 1. End to end

802 images, zero failures. The product name is supplied with each image, since the accept/reject limit is
per product, and it comes from the directory the image was read from. With all four products downloaded,
`python scripts/measure_metrics.py` reproduces all of this.

| | result | 95% CI | n |
|---|---|---|---|
| **defective boards rejected** (sensitivity) | **97.25%** | 95.1 to 98.5 | 400 |
| **good boards accepted** (specificity) | **94.03%** | 91.3 to 96.0 | 402 |
| overall accuracy | 95.64% | 94.0 to 96.8 | 802 |
| precision | 94.19% | | 413 rejected |
| F1 | 95.69% | | |

It misses 11 of 400 defective boards and falsely rejects 24 of 402 good ones.

### Per part

100 defective plus 100 or 101 good per part.

| part | what it is | rejects defective | missed | accepts good | false alarms | limit |
|---|---|---|---|---|---|---|
| pcb1 | HC-SR04 ultrasonic sensor | 94.0% | 6 | 94.0% | 6 | 0.0794 |
| pcb2 | dual-IC driver board | 99.0% | 1 | 93.0% | 7 | 0.0758 |
| pcb3 | MH-B infrared sensor | 96.0% | 4 | 95.0% | 5 | 0.0788 |
| pcb4 | TP4056 charger module | **100.0%** | 0 | 94.1% | 6 | 0.0784 |

pcb1 is the weakest board on both axes. See section 5 for why it was once reported as perfect.

## 2. Accept or reject

A threshold, no classifier. The limit is the 95th percentile of that part's good-board scores, computed on
a calibration half of the good images. **Defect scores are never consulted when choosing it.** Recall is
then measured on all 400 defective boards and false alarms on the held-back good half.

That protocol is the point. An earlier version set the threshold just below the lowest defective score,
which uses knowledge of the defects and makes the miss rate look better than it is.

### The operating curve is a dial, not a constant

One global threshold, swept over quantiles of good-board scores:

| threshold at | limit | defects caught | missed | good boards falsely rejected |
|---|---|---|---|---|
| p90 of good | 0.0726 | 99.0% | 4 of 400 | 13.4% |
| p95 of good | 0.0750 | 98.5% | 6 of 400 | 11.4% |
| p99 of good | 0.0866 | 89.8% | 41 of 400 | 4.5% |
| p100 (max) | 0.1075 | 63.5% | 146 of 400 | 0.5% |

**Deployed setting, per-part p95:** recall 97.25 percent, false alarms 4.95 percent. That is why the limits
are per product rather than global: one p95 limit across all four parts buys 1.25 points more recall for
more than double the false alarms.

Why not the maximum? pcb1's worst good board scores 0.0920. Set the limit there and you get zero false
alarms while recall collapses to 77 percent, because one unusual-but-fine board drags the bar up and 23
defects walk through underneath it.

Moving every part from p95 to p90 rescues **6 more defects** and creates **13 more false alarms**. On pcb4
it rescues zero and creates four, since pcb4 already catches all 100. Which end you want depends on whether
an escape or an operator minute costs you more, and that is not a decision this repository should make.

## 3. Where the defect is

No parameters. Arithmetic on the anomaly map, measured on all 400 defective images.

| | result | 95% CI |
|---|---|---|
| **first box contains the defect** | **90.5%** (362 of 400) | 87.2 to 93.0 |
| **one of three boxes does** | **97.0%** (388 of 400) | 94.8 to 98.3 |
| peak lands inside the defect mask | 27.3% (109 of 400) | 23.1 to 31.8 |
| median peak-to-defect distance | 12 px on a 1750 to 1833 px diagonal | p90 is 126 px |

Which candidate held the defect: box 1 on 362 boards, box 2 on 22 more, box 3 on 4 more, none of the three
on 12. So showing only the first box would hide **26 defects the detector actually found**, which is why
`inspect()` returns three candidates rather than one confident rectangle.

**Why "peak inside mask" is only 27 percent while pointing is good.** The masks are thin slivers: a bent
pin is roughly 200 by 20 pixels. Being 12 pixels sideways puts the peak outside a 20-pixel-wide sliver
while still being visually dead on. One encoder patch covers 48 to 56 pixels at full scale, so a 12 pixel
median error is a quarter of a patch, at the granularity limit of the backbone. No tuning improves that
without a finer grid.

Difficulty scales with defect size, as you would expect:

| defect area, percent of frame | first-box hit rate | n |
|---|---|---|
| under 0.2 | 80.9% | 115 |
| 0.2 to 0.5 | 92.2% | 166 |
| 0.5 to 1.0 | 94.3% | 35 |
| over 1.0 | 98.8% | 84 |

## 4. Things that were tried and did not work

Kept because knowing what fails is worth as much as knowing what works.

| idea | result | verdict |
|---|---|---|
| 180 degree pose canonicalisation | AUPIMO 0.633 to 0.283 | refuted, minus 0.350 |
| test-time pose search | plus 0.0003 | no effect |
| more training steps, 20k | identical | no effect |
| **masking the map to the PCB before scoring** | recall 97.25 to 92.00, specificity 94.06 to 97.03 | refuted, see below |

### The masked-score experiment, and what it revealed

The score is the mean of the hottest 1 percent of the map over the **whole frame**, background included,
while the boxes are masked to the PCB. That asymmetry looked like free specificity: mask the background
before scoring and the false alarms should drop.

The premise held. On average **35.1 percent of the hottest pixels sit off the board**, rising to 42.2
percent on good images. But masking made things worse: image AUROC fell from 0.9914 to 0.9779, and the best
masked variant bought 2.97 points of specificity for 5.25 points of recall, which is 6 fewer false alarms
in exchange for 21 more escaped defects.

The number that explains it: **84 of 389 correctly rejected boards have more than 70 percent of their hot
pixels off the board.** For those boards the strongest evidence is not on the part at all. Background
hotness is not noise contaminating the score, it is a label-correlated shortcut the detector leans on, and
removing it removes real predictive power.

It is concentrated rather than uniform. pcb1 loses 0.025 AUROC and pcb2 loses 0.031, while pcb3 slightly
improves and pcb4 is unchanged. pcb1 is also the weakest board in section 1, which is unlikely to be
coincidence.

**Practical read: do not change the score, and do expect these numbers to move on a controlled lightbox,
because a fixed background removes exactly the signal this experiment proved is being used.**

## 5. Numbers that collapsed on a larger sample

Each of these was measured, reported, and then fell when n grew. Always in the same direction.

| claim | on a small sample | on the full sample |
|---|---|---|
| pcb1 image AUROC | 1.0000 (25 good) | **0.9869** (100 good) |
| end-to-end sensitivity | 100% (100 images) | **97.25%** (400) |

Read the interval's lower bound, not the point estimate. To claim above 90 percent with a 95 percent lower
bound above 90 needs about 35 samples if the true rate is 100 percent, about 53 at 98 percent, and about
1,150 if the true rate is 91.7 percent. Interval width shrinks only as one over the square root of n, so a
barely-above-90 system is very expensive to prove.

## 6. Latency

Batch of one, no optimisation, warm-up call discarded, on one RTX 5090. These were timed with extra stages
in the loop that this repository does not ship, so read them as upper bounds on the detection path rather
than as its median. Re-measure on your own card with `python scripts/measure_metrics.py --latency`.

| path | median | p90 |
|---|---|---|
| score and peaks | 43 ms | 47 ms |
| the same, plus rendering the heatmap and the boxes | 60 ms | 72 ms |

Rendering is real work rather than a rounding error: resizing the map to full resolution, colour-mapping
it, drawing three boxes and JPEG-encoding the result is roughly 8 ms of that.

Four runs on the same machine gave medians of 50, 54, 61 and 73 milliseconds, in different sessions with
different GPU tenancy. If you need one number, quote the slowest. The spread is not attributed to a cause,
because that was not measured.

## 7. Resolution sensitivity

Below about 800 pixels of width the verdict stops being trustworthy, and this is why.

A small contamination spot spans about 2 of the 784 patches. Downscaling a 1358 pixel board to 640 loses
about 40 percent of the local detail there, measured as Laplacian variance falling from 3870 to 2269, and
moves the score by roughly 3 percent.

| input width | score | verdict |
|---|---|---|
| 1358, native | 0.0805 | reject, correct |
| 900 | 0.0787 | reject |
| 700 | 0.0790 | reject |
| **600** | 0.0783 | **accept, wrong** |
| **500** | 0.0781 | **accept, wrong** |

JPEG compression is harmless by comparison: quality 60 still gives the right verdict. Upscaling does not
recover it either, and three interpolation methods gave three different answers straddling the limit, so
the right response to a small image is to refuse it rather than to rescue it.

Only borderline boards are affected. The strongest pcb4 defect still scores 0.1895 at 500 pixels. That is
what makes the failure quiet, and worth an explicit guard on input width.

## 8. Lighting and background

Both of these were measured on all 802 test images by perturbing the input and re-scoring. They are the
two questions anyone deploying this will ask, and the answers are very different from each other.

### Lighting: the detector copes, the limits do not

| condition | rejects defective | accepts good | board mask found |
|---|---|---|---|
| baseline | 97.2% | 94.0% | 100% |
| dimmer 10% | 97.0% | 95.0% | 100% |
| dimmer 25% | 96.5% | 95.8% | 100% |
| brighter 10% | 98.2% | 93.3% | 100% |
| **brighter 25%** | 100.0% | **65.7%** | 100% |
| gamma 0.85 | 97.2% | 94.0% | 100% |
| gamma 1.20 | 97.2% | 92.5% | 100% |
| **warm light +15%** | 99.5% | **75.4%** | 100% |
| **cool light -15%** | 99.5% | **73.4%** | **42.3%** |
| vignette 35% | 98.5% | 88.8% | 100% |

Ten percent either way is free. Beyond that the whole score distribution shifts past a limit that did not
move, so specificity collapses while sensitivity rises. The detector degrades into something that rejects
everything, which is at least a safe direction to fail in.

Recalibrating the limits on good boards from the new lighting mostly fixes it, with no retraining and no
labels:

| condition | as shipped | after recalibration |
|---|---|---|
| brighter 25% | 100.0% / 65.8% | 93.2% / 92.6% |
| warm light +15% | 99.5% / 75.7% | **97.5% / 94.1%** |
| cool light -15% | 99.5% / 74.8% | 95.2% / 93.6% |
| vignette 35% | 98.5% / 88.6% | 97.8% / 93.1% |

So lighting is a calibration procedure, not a blocker: photograph 50 good units per part under the new
illumination and re-derive the limits. Two to four points of recall are genuinely lost at the extremes, so
this is not purely a threshold offset.

**Known issue to fix.** Under cool light the board mask fails on 58 percent of images. `pcb_anomaly/board.py`
segments with a blue hue window, and boosting blue makes a grey background read as board, so the segmented
region exceeds the 85 percent plausibility ceiling and the function returns None. It fails open, which is
correct behaviour, but without a mask the peaks can land off the board. Widening the plausibility band or
replacing hue segmentation with a fixed region of interest both fix it.

### Background: changing it breaks everything

Same board pixels, composited onto a different surround using the mask from the original image, so only
what surrounds the board changes.

| background | shipped limits | recalibrated | masked and recalibrated |
|---|---|---|---|
| original grey | 97.2% / 94.1% | 97.2% / 94.1% | 92.0% / 97.0% |
| **solid red** | 100.0% / **0.0%** | **9.0%** / 96.0% | **14.5%** / 97.5% |
| **solid white** | 100.0% / **0.0%** | **8.0%** / 91.1% | **12.2%** / 95.0% |
| **cluttered** | 100.0% / **0.0%** | **2.8%** / 94.1% | **12.5%** / 96.0% |

With the shipped limits, **every board is rejected**, good or bad, on all three new backgrounds. Zero
percent specificity.

Recalibration does not fix this. It restores specificity by lifting the limit so far that almost nothing
trips it, and sensitivity falls to between 3 and 15 percent. The detector stops finding defects.

**Masking the score to the board does not rescue it either**, which is the informative part. If the problem
were only that background pixels enter the score, masking them out would help. It does not, because the
transformer's attention is global: background patches change what the board patches themselves become. The
board's own features are contaminated before any pooling happens.

So the background is not a nuisance variable to be filtered. It is part of what the model learned "normal"
means. There is no calibration procedure for this.

**Two possible fixes, both needing the detector retrained.** Recapture good boards on the new background
and retrain, which is cheap in effort since it needs no labels, about 900 good images per part. Or crop to
the board before inference so the background never reaches the encoder, which is the better fix
architecturally and is what a fixture gives you anyway. Neither has been tested here.

## 9. What these numbers do not cover

**Public consumer boards, not industrial parts.** VisA pcb1 to pcb4 are photographed in a fixed setup.

**Some detection signal comes from capture conditions.** Background brightness alone separates good from
bad at AUROC 0.68 to 0.77 across these boards, and the masking experiment in section 4 puts a firmer number
on it. A controlled lightbox on real parts removes that crutch and some of this performance with it.

**Localisation is scored by box, not by mask.** A hit means the defect falls inside a 0.22-width box, which
is the right unit for pointing an operator at a region and the wrong one for measuring a defect.

**Latency depends on GPU tenancy**, with four measurements spanning 50 to 73 milliseconds, and none of it
measured under concurrent load.
