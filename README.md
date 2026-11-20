# Animated SOG

**Animated SOG** (`.asog`) is a compact file format and toolkit for **4D Gaussian splats**: volumetric
video that plays in the browser. One file holds a clip of moving Gaussians, with spherical harmonics, and
optionally its soundtrack.

- About **11× smaller** than the TRiGS runtime files it is made from (Charlie: 319 MB → 28 MB at 45 dB), and
  a 6-second flip book of per-frame PLYs becomes **8.6 MB** instead of 202 MB of per-frame SOG.
- Streams and decodes while it downloads; a 900 k-splat clip is ready in under a second on a desktop.
- Plays in **PlayCanvas** (WebGPU and WebGL) and **three.js**, with the motion evaluated on the GPU.

How it works: every splat lives for a short time and moves rigidly over its life, the motion model of
**TRiGS** (*Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting*, arXiv 2604.00538). The splats'
static attributes are stored the way **SOG v2** stores them (16-bit log positions, smallest-three
rotations, 256-entry codebooks, an SH palette), so a SOG renderer only has to add the motion. The motion is
stored in an exact 17-value form and quantised to a fixed displacement over each splat's life.

## What's here

| path | what |
|---|---|
| [`docs/format.md`](docs/format.md) | the `.asog` v1 specification |
| [`docs/pipeline.md`](docs/pipeline.md) | flip book (one PLY per frame) → `.asog`, end to end |
| [`docs/results.md`](docs/results.md) | sizes, quality and performance measurements |
| [`python/`](python) | `animated-sog` (`import asog`): encoder, reference decoder, `asog` CLI, flip-book pipeline, tests |
| [`js/`](js) | `animated-sog` npm package: streaming decoder (engine-agnostic), PlayCanvas player, three.js player |
| [`examples/playcanvas`](examples/playcanvas), [`examples/three`](examples/three) | standalone viewer pages |
| [`test-data/`](test-data) | small sample clips |

## Quick start

Encode a TRiGS clip (`.trigsbin` written by Open4DGS):

```
pip install animated-sog[palette]
asog encode clip.trigsbin clip.asog --audio sound.mp3
asog info clip.asog
```

Convert a flip book (a folder of per-frame 3DGS PLYs) — needs a CUDA GPU and the external Open4DGS trainer,
see [docs/pipeline.md](docs/pipeline.md):

```
asog flipbook run frames/ work/ --rotation=-90,0,0
```

View it: serve the repository root with any static server and open
`examples/playcanvas/?src=../../test-data/rubloff-1s.asog` (or `examples/three/`).

Play it in your own PlayCanvas app:

```js
import { AsogPlayer } from "animated-sog/playcanvas";
const player = await AsogPlayer.load(app, "clip.asog", parentEntity, { audio: true });
```

or in three.js (WebGL 2; our own small splat renderer, the motion and SH evaluated on the GPU):

```js
import { AsogThreePlayer } from "animated-sog/three";
const player = await AsogThreePlayer.load(renderer, "clip.asog", scene, { audio: true, listener });
```

## Status

Format v1. Made at [arrival.space](https://arrival.space), where it is being integrated. The players rely on
a few engine internals (PlayCanvas 2.22); see the package READMEs for limits.

## Credits

- TRiGS: Yeom, Nam, Choi et al., *Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting* (2026).
- Open4DGS (ASU Narrative and Emerging Media Lab): the trainer and timeline exporter the flip-book
  pipeline runs as external tools (AGPL-3.0, not part of this repository).
- SOG and the gsplat renderer of the PlayCanvas engine.

## Licence

MIT — see [LICENSE](LICENSE). The Open4DGS trainer and exporter used by the flip-book pipeline are
separate AGPL-3.0 tools and are not part of this repository.
