# How it works

Every number here was read from the loaded model rather than from documentation. Reproduce any of it with
the snippets at the bottom.

## The idea in one paragraph

Imagine an apprentice who has studied 3,614 good boards and never a faulty one. Cut a board into 784 small
tiles, describe each one, then ask the apprentice to write those descriptions back from memory. On a
familiar tile the copy comes out nearly identical. On a tile holding something they have never seen, they
write down what they expected instead, and the mismatch is large. The anomaly map is that mismatch, tile
by tile. Nobody ever showed the system a defect. It measures surprise, not recognition.

## Three names that get confused

| name | what it actually is |
|---|---|
| **DINOv2** | Meta's general vision model, trained on 142 million unlabelled photographs. Encoder only, it has no decoder. Describes image patches and knows nothing about defects. |
| **ViT-B/14** | The architecture DINOv2 uses. "B" fixes 12 blocks and width 768. "14" is the patch size in pixels, not a block count. |
| **Dinomaly** | A method: the frozen DINOv2 encoder plus a decoder that Dinomaly adds. Not a new backbone. The decoder is why the name exists. |

## The models

| component | parameters | state | learned from |
|---|---|---|---|
| DINOv2 ViT-B/14 encoder | 86,582,016 | **frozen**, never updated | Meta, 142 M unlabelled photos |
| Dinomaly bottleneck and decoder | 61,390,848 | trained | 3,614 good boards, **no labels** |

148 million parameters, of which **zero needed a human**. The two stages after the map have no
parameters at all: the verdict is a threshold and the boxes are arithmetic.

## Where the numbers come from

| number | origin |
|---|---|
| 14 | patch size in pixels, from DINOv2 |
| 28 | 392 divided by 14, twice, giving the patch grid |
| 768 | ViT-Base width: 12 attention heads times 64 dims per head. A design choice, not a calculation |
| 784 | 28 times 28 patches |
| 789 | 784 patches plus 1 class token plus 4 register tokens |

A patch grows rather than shrinks. Three channels by 14 by 14 is 588 raw pixel values, and one learned
convolution turns that into 768 features. Width then stays 768 through every block.

## The fixed input size

Anomalib's pre-processor, not the model, is what fixes the size:

```
any input   ->  Resize(448, 448)      bilinear, aspect ratio NOT preserved
            ->  CenterCrop(392, 392)  the outer 6.25 percent of each edge is discarded
            ->  Normalize(ImageNet mean and std)
```

The ViT itself sets `dynamic_img_size=True` and would happily take 224 (a 16 by 16 grid) or 560 (40 by
40). This one cannot, because the trained decoder and all the peak arithmetic assume 784 tokens.
Running at 560 for a finer grid would mean retraining the decoder. That is the real lever if localisation
precision ever needs to improve.

Two consequences worth knowing. The **centre crop is a blind spot**: a defect in the outer 6.25 percent of
the frame is never seen. All 400 VisA defect masks sit well inside the board, so it costs nothing on this
dataset, but on a real part it would be an invisible failure. And **aspect ratio is not preserved**, so a
1358 by 1104 board is squashed by 3.03 in x and 2.46 in y. Harmless for detection since training and
inference squash identically, but any physical measurement taken off the map needs un-squashing per axis.

## One pass, one encoder

The encoder is loaded once and runs once per board: 3 by 392 by 392 pixels in, 789 tokens out, of which
the 784 patch tokens keep their 28 by 28 layout. The decoder tries to write those 784 descriptions back
and the disagreement is the anomaly map. Both answers the runtime gives you, the score and the three
boxes, are read off that one map, so there is nothing a second forward pass could add.

Keeping the layout is why the map is spatial at all. Pooling the 784 tokens into a single vector, which is
what a classifier head would want, answers *how unusual* and throws away *where*.

## How the anomaly map is built

Blocks 2 to 9 are handed out by the feature extractor, 8 of the 12. All 12 always execute, since you
cannot skip the middle of a transformer, and blocks 10 and 11 are simply discarded.

```
blocks 2..9  --average all 8-->  bottleneck MLP  -->  8 decoder blocks
                                                          |
      encoder side, fused into 2 depth groups             |  decoder side, same 2 groups
      (blocks 2-5, blocks 6-9)                            |  (the block list is REVERSED first)
                     |                                    |
                     +------------- 1 - cosine ------------+
                          per tile, over all 768 dims
                                     |
                        two 28x28 grids, averaged
                                     |
                          784 numbers = the anomaly map
```

The 768 dimension collapses at the comparison. Concretely, on one defective board:

```
tile (row 18, col 14), the contamination
  encoder says :  [-0.164, -1.061, -0.370, +0.679, +0.103, ...]   768 numbers
  decoder wrote:  [-0.129, -0.798, -0.140, +0.229, -0.092, ...]   768 numbers
                  cosine similarity 0.9395  ->  1 - 0.9395 = 0.0605

tile (row 3, col 3), quiet board
  cosine similarity 0.9839  ->  0.0161
```

1,204,224 numbers go in and 784 come out. Cosine is used rather than subtraction because it asks whether
the copyist described the same kind of thing, ignoring how bright it was.

The map is then bilinearly upsampled from 28 by 28 to 392 by 392, so the smooth heatmap you see on screen
carries only 784 real values. Because bilinear interpolation is a convex combination, the argmax always
lands on a lattice point, spaced about 52 pixels apart in original coordinates. Apparent pixel precision
is not real.

Note the **reversal**: `decoder_features[::-1]` before fusing, so the decoder's first blocks are graded
against the encoder's last. U-Net logic. The decoder starts at the bottleneck and rebuilds the abstract
deep features first, then recovers the shallow detailed ones.

## The score is not the hottest tile

```python
blurred = gaussian_blur(interpolate(anomaly_map, size=256))     # sigma 4
score   = mean of the top 1 percent of pixels                   # 655 of 65,536, about 8 tiles
```

Deliberately. One freak tile from dust or a glare highlight should not reject a board, so roughly eight
tiles have to agree.

Two asymmetries hide in there. The score comes from a **blurred 256 by 256** version while the map used for
boxes is the **unblurred 392 by 392** one, so the number deciding pass or fail and the picture the operator
sees are not quite the same surface. And the score sees the **whole frame** while the boxes are **masked to
the PCB**, so a hot background patch can reject a board whose three boxes then land on clean substrate.
Masking before scoring was tested and made things worse, which is covered in
[results.md](results.md).

## From map to boxes

```python
a = np.where(board_mask_resized, a, -np.inf)      # background can never win
peaks = topk_nms(a, 3, radius=int(0.22 * 392 / 2))
```

Take the highest value, blank a square around it, repeat. Without the blanking the second peak would be
the tile next door and all three boxes would stack on the same defect. The radius equals the box
half-width, which is what makes the boxes touch but never overlap.

One offset is easy to miss. The map covers the 392 centre crop, not the 448 resize, so a peak at map pixel
`x` sits at `(x + 28) * w / 448` in the original frame. Drop the 28 and every box lands up to 6 percent of
the image width off target.

Showing only the first box would hide 26 defects the detector actually found: the first box is right 90.5
percent of the time and one of the three is right 97.0 percent.

## How the decoder was trained

The loss targets are two fused encoder tensors, `(B, 768, 28, 28)` each, one from blocks 2 to 5 and one
from blocks 6 to 9. The class and register tokens are stripped before the comparison, which is forced
anyway since 789 does not reshape into 28 by 28.

```python
en_ = encoder_features[i].detach()          # gradient cannot reach the encoder
loss += mean(1 - cos(en_.reshape(B, -1), de_.reshape(B, -1)))     # one cosine per image per group

with no_grad():
    point_dist = 1 - cos(en_, de_)          # per tile
thresh = topk(point_dist.flatten(), k=numel * (1 - p))[-1]
de_.register_hook(scale gradients by 0.1 where point_dist < thresh)
```

Three things matter here. The `.detach()` is the mechanical reason the encoder stays frozen even though
the loss touches its output. The scalar loss is a **whole-image** cosine, since `reshape(B, -1)` flattens
768 by 28 by 28 into one 602,112-dimension vector, so locality enters training only through the gradient
hook. And that hook is the whole trick: `p` ramps from 0 to 0.9 over the first 1,000 steps, after which the
90 percent of tiles the decoder already reconstructs well have their gradients cut to 10 percent.

Training concentrates on the hardest 10 percent, which sounds backwards for a reconstruction task and is
deliberate. Anomalib's own docstring says it plainly: this prevents the decoder from learning to
reconstruct anomalous patterns. A decoder that reconstructed everything perfectly would produce a flat map
and detect nothing. The detector is kept intentionally mediocre.

Note the asymmetry with inference: training optimises a whole-image cosine, while inference reads a
per-tile cosine.

## Where each piece runs

| component | device |
|---|---|
| DINOv2 encoder, bottleneck, decoder | CUDA when there is one, CPU otherwise, at seconds per board |
| board finding, peak finding, rendering | CPU, OpenCV and numpy |

Total GPU footprint is 0.59 GB allocated for a 148 M parameter system, so it sits comfortably alongside
other work on the same card.

If you ever wanted cheaper hardware, only the encoder and decoder need a GPU. Everything downstream is
arithmetic on a 28 by 28 grid and would run on a Raspberry Pi.

## Reproducing these numbers

```python
from anomalib.models import Dinomaly
import torch

m = Dinomaly(post_processor=False)
vit = m.model.encoder.feature_extractor
len(vit.blocks), vit.embed_dim, vit.blocks[0].attn.num_heads, vit.patch_embed.patch_size
# -> 12, 768, 12, (14, 14)

# which blocks the extractor hands out
list(m.model.encoder(torch.randn(1, 3, 392, 392)).keys())
# -> ['blocks.2' ... 'blocks.9']

# what fixes the input size
print(m.pre_processor)
# -> Resize(448) -> CenterCrop(392) -> Normalize

# count encoder passes during one real inference
vit.patch_embed.register_forward_hook(lambda mod, i, o: print("pass", tuple(i[0].shape)))
vit.blocks[0].register_forward_hook(lambda mod, i, o: print("  tokens", i[0].shape[1]))
```
