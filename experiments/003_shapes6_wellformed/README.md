# Experiment 003: experiment 002 on well-formed polygons

Identical to [`002_shapes6_unet`](../002_shapes6_unet/README.md) — same model (UNetSurrogate width 16, depth 4,
1.94M params), data sizes (5000 / 1000 / 1000), loss, schedule, augmentation and inspection schedule — except for one
setting: `data.polygon_sampler: "wellformed"` (requires `synthdataseg >= 0.2.0`).

## Why

Experiment 002 learned background, stars and crowns well (val IoU 0.997 / 0.95 / 0.90 at epoch 61) but not the
convex polygons (triangle 0.49, quadrilateral 0.30, pentagon 0.34), with patchy predictions inside single shapes.
The cause was the data: `synthdataseg <= 0.1.0` drew the N-gon classes uniformly at random (Valtr's algorithm),
which produces nearly flat corners and very short edges. Measured on shapes as placed in rendered images (after the
compositor's affine transform), the share showing all N corners — every corner turning ≥ 20° with both adjacent
edges ≥ 5% of the perimeter — was:

| Sampler | triangle | quadrilateral | pentagon |
| :--- | ---: | ---: | ---: |
| `valtr` (002) | 100% | 57% | 32% |
| `wellformed` (003) | 100% | 100% | 100% |

So in 002 many "pentagons" looked like quadrilaterals or triangles, and the labels were partly ambiguous. The
`wellformed` sampler builds random convex N-gons whose corners all turn by ≥ 30° (≤ 155°) and whose edges are all
≥ 10% of the perimeter, keeping variety in proportions and angles.

## What to compare

Same seeds and split sizes as 002, so differences are due to the polygon definition. Expectations to check:
polygon IoUs should rise substantially; star/crown/background should stay where they were; remaining polygon errors
should concentrate on **occluded** shapes (part of a shape hidden behind another changes its visible corner count).

Known remaining ambiguity (not addressed here): neighbouring shapes sometimes get the same plain colour or a very
similar texture, so the boundary between them is not visible.

## Results

| Run | val mIoU | val mIoU(fg) | test mIoU (best ckpt) | test mIoU(fg) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| *pending* | | | | | |
