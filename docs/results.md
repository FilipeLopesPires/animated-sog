# Results

All numbers measured on one desktop (RTX 5080, Windows 11, Chrome) unless noted. PSNR is measured on
rendered images, not on raw attributes.

## TRiGS clips (Open4DGS output) → `.asog`

Source: the six public sample clips of the TRiGS editor (Open4DGS `trigs.runtime` v1, SH degree 3, 60 fps).
Default encoder settings (`qp 2, qa 8, prune 0.3`), soundtrack not included in the size.

| clip | splats | source `.trigsbin` | `.asog` | ratio |
|---|---|---|---|---|
| Charlie (Grade Eterna) | 1.29 M | 319 MB | 28.4 MB | 11.2× |
| Adrianna | 1.39 M | 343 MB | 30.1 MB | 11.4× |
| Natalia | 1.70 M | 422 MB | 38.0 MB | 11.1× |
| Arvin & Steffanie (two people) | 2.53 M | 627 MB | 56.3 MB | 11.1× |
| Mike | 2.83 M | 698 MB | 60.7 MB | 11.5× |
| Miller | 3.30 M | 817 MB | 71.3 MB | 11.5× |

About 31 bytes per splat after pruning, SH degree 3 included.

### Quality against size (Charlie)

Mean PSNR over 6 views × 2 times, rendered in the web player, against the source clip with full SH.

| variant | size | PSNR |
|---|---|---|
| source `.trigsbin` (float32/16) | 319 MB | reference |
| float16 repack without SH (our first attempt) | 118 MB | 36.8 dB |
| `.asog`, qp 0.5, no pruning | 47.8 MB | 46.6 dB |
| `.asog`, qp 1, no pruning | 45.1 MB | 46.5 dB |
| `.asog`, qp 2, no pruning | 42.7 MB | 46.4 dB |
| `.asog`, prune 20 % | 32.7 MB | 45.7 dB |
| **`.asog`, prune 30 % (default)** | **28.9 MB** | **45.3 dB** |
| `.asog`, prune 40 % | 24.6 MB | 44.6 dB |
| `.asog`, prune 50 % | 20.9 MB | 42.7 dB (small detail such as teeth changes) |

Natalia: `.asog` 43.1 dB (no pruning 44.2 dB), float16 repack without SH 36.4 dB.

The quality floor is the SOG part (codebooks and the SH palette); the motion step and the position bits
barely move the score. Pruning beyond ~40 % is where detail starts to go.

Motion accuracy (`asog check`, Rubloff 1 s, qp 2 / qa 8): splat centre error over each splat's visible
life 3.3 mm median, 22 mm p99; rotation error 0.34° median, 0.58° p99. qp 0.25 / qa 0.5 is about 8× more
precise for a larger file.

### Playback (web, PlayCanvas player)

| | load (decode) | fps | longest main-thread task while loading |
|---|---|---|---|
| Rubloff 1 s, 105 k splats, 3.2 MB | 0.17 s | 120 (cap) | none > 50 ms |
| Charlie, 900 k splats, 28.5 MB | 0.9 s | 120 (cap) | 57 ms (WebGPU), 72 ms (WebGL) |

three.js player (WebGL 2, Charlie, 900 k splats, 1060×900): load 0.95 s; 120 fps (cap), a frame with
moving time takes 2.5 ms uncapped; the depth order (CPU worker, visible splats only) refreshes about 25×
a second. Against the PlayCanvas WebGL 2 player at the same camera and time: 49.5 dB over the subject
(Rubloff with SH degree 3: 44.5 dB; for comparison the PlayCanvas player's own WebGL 2 vs WebGPU images
differ by 37.2 dB).

Soundtrack sync (both players): median drift 3–5 ms, max 11 ms, no corrective seeks.

Memory, Mike (2.8 M splats, WebGPU): browser tab peak +325 MB while loading the `.asog`, against +1.33 GB
for the source file in our first player; GPU process +361 MB with SH.

## Flip books → `.asog`

A flip book is one independently trained PLY per frame (see [pipeline.md](pipeline.md)). Rubloff: a
standing person, 900 frames at 30 fps in five lighting segments, 82–110 k splats per frame, SH degree 3.
The numbers below are the last neutral segment (180 frames, 6 s).

| representation | size | resident splats | notes |
|---|---|---|---|
| raw PLY frames | 3.6 GB | 88 k per frame | flickers: every frame is re-tiled |
| SOG per frame | 202 MB | 88 k per frame | |
| training-free 4D field codec, 12 mm (earlier experiment) | 35 MB | — | blurred, own player |
| **`.asog` (distilled with TRiGS, 400 k splats)** | **8.6 MB** | **280 k** | see below |

Distillation run: 23 min on the RTX 5080 (60 k steps, 72 views at 512², keyframe stride 9).
Evaluated against the true flip book on 8 held-out cameras between the training views (`asog flipbook
evaluate`):

| 6 s segment, 400 k splats | PSNR vs flip book | frame-to-frame jitter vs flip book | median splat life |
|---|---|---|---|
| **frames rotated by `asog flipbook stage` (current pipeline)** | **41.0 dB** (worst frame 38.5) | **0.42×** | 33 frames |
| trainer's own `--transform.rotation` (rotates SH bands 2–3 wrongly) | 39.4 dB (worst frame 37.3) | 0.42× | 33 frames |

Encoder settings on the same trained clip (stage-rotated run):

| encoding | size | PSNR vs flip book |
|---|---|---|
| default (prune 30 %) | 8.6 MB | 41.0 dB |
| prune 15 % | 10.2 MB | 42.1 dB |
| no pruning | 11.7 MB | 43.0 dB |
| no pruning, qp 1 / qa 4 | 12.4 MB | 43.3 dB |
| near-lossless (no pruning, qp 0.25, qa 0.5, 16-bit positions, finer times) | 14.8 MB | 43.4 dB |

Flip books are built from many small translucent splats, so pruning costs more here than on the TRiGS
clips; for them `--prune 0 --qp 1 --qa 4` is the better default. Beyond ~43.4 dB the limit is the trained
model, not the encoding.

Jitter is the second temporal difference on the subject's pixels, |x(t) − (x(t−1) + x(t+1)) / 2|: smooth
motion scores near zero, the flip book's frame-to-frame re-tiling scores high. The distilled clip keeps the
motion and loses 58 % of the jitter.
