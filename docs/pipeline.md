# Flip book → `.asog`

A *flip book* is a volumetric video stored as one independently trained 3DGS PLY per frame. Nothing ties a
splat in one frame to a splat in the next, so there is no motion to extract and every frame flickers on its
own. We convert one by **distillation**: the Open4DGS TRiGS trainer renders training images from the PLYs
and fits a single 4D model of splats with short lives and rigid-body motion (TRiGS), which the `asog`
encoder then compresses.

```
PLY frames ──stage──▶ segment, y up ──train──▶ trigs_ckpt.pt (y up) ──flip──▶ ckpt_ydown.pt
           ──export──▶ clip.trigsbin ──encode──▶ clip.asog ──evaluate──▶ PSNR / flicker / jitter
```

Every step is a Python function in `asog.flipbook` and a subcommand of `asog flipbook`; `asog flipbook run`
chains them for one segment in one work directory.

## Licence separation

The trainer ([open4dgs-trigs](https://github.com/ASU-NEM-Labs/open4dgs-trigs)) is **AGPL-3.0**; the timeline
exporter `deliver/make-trigs-timeline.py` lives in [open4dgs-core](https://github.com/ASU-NEM-Labs/open4dgs-core).
They are external tools: this package runs them as subprocesses and never imports their modules, and they
are not dependencies. Install them yourself and point at them:

| setting | what |
|---|---|
| `ASOG_TRIGS_PYTHON` / `--trigs-python` | the interpreter of the TRiGS install, or a wrapper that sets up its environment and passes its arguments on (on Windows a `.bat` that calls `vcvars64.bat`, then the venv's `python.exe %*`). It also runs the exporter (numpy + torch). |
| `ASOG_OPEN4DGS_CORE` / `--open4dgs-core` | an open4dgs-core checkout |

Staging (with its rotation), the checkpoint flip and the evaluation are our own code (`pip install animated-sog[flipbook]`: torch, gsplat
(Apache-2.0), imageio).

## Steps

Measured on an RTX 5080 (16 GB), Windows, with the Rubloff flip book (60 fps capture, z up).

### 1. Stage one segment

```
asog flipbook stage <flipbook dir> <work>/frames --start 0 --count 180 --stride 2 --rotation=-90,0,0
```

Takes frames `start, start+stride, …` of the name-sorted PLYs. A TRiGS splat has **one colour for its whole
life**, so a segment must not contain a lighting or scene cut: split the flip book at cuts and train each
segment on its own. `--stride 2` turns a 60 fps capture into the 30 fps the trainer is told.

The trainer's camera orbit assumes **y up with the feet at y = 0**. `--rotation` ("rx,ry,rz" degrees,
extrinsic XYZ; `-90,0,0` for z-up sources) rotates every frame into that frame as it is staged: positions
and normals, quaternions, and the SH degrees 1–3 through their rotation matrices (one Wigner-D block per
degree, in the 3DGS `f_rest` order and signs), keeping each file's PLY layout. A staged Rubloff frame
rendered from the trainer's 72 views matches the original frame from correspondingly rotated cameras at
110 dB; the trainer's own `--transform.rotation` reaches only 44 dB (it rotates SH degrees 2 and 3 wrongly,
see below), which is why the pipeline never uses it. Without `--rotation` the frames are hard-linked
(copied across drives). Staging 180 Rubloff frames with rotation takes 15 s (3.7 GB).

### 2. Train

```
asog flipbook train <work>/frames <work>/train
```

runs

```
python -m trigs.cli.train_from_plys_upstream --input-dir <frames> --output-dir <run>
    --camera.latitude-steps 6 --config.fps 30
    --config.keyframe-stride 9 --config.target-count 400000 --config.n-steps 60000
    --no-render-debug --no-save-plys
```

- The trainer renders its own ground truth from the PLYs on an orbit: 12 longitudes × 6 latitude rows
  (−60…60°), 512², around the point `(0, height/2, 0)`, so the staged frames must be y up with the feet at
  y = 0 (step 1). `--transform.rotation` stays at its identity default. A source that is not standing on
  its origin also needs a translation (not handled by `stage` yet; the Rubloff frames stand at y ≈ 0.03).
- The ground-truth cache is kept in **system RAM** (the trainer's log says "on GPU", but it moves the uint8
  images to the CPU): about `frames × views × 512² × 4` bytes, 13.6 GB for 180 frames × 72 views.
- `keyframe-stride` picks the frames whose splats seed the 4D initialisation. With stride 3 and 400k
  splats its farthest-point sampling took more than 14 minutes and filled the GPU; with stride 9 training
  starts in seconds.
- 180 frames, 400k splats, 60k steps: **22 minutes**. 30 frames, 150k splats, 30k steps, 10 rows: 7.5 minutes.
- `--no-save-plys` skips writing a student PLY per frame (4.6 GB for 180 frames at 400k); the evaluation
  below renders the `.asog` instead.

### 3. Flip to y down

```
asog flipbook flip <work>/train/trigs_ckpt.pt <work>/ckpt_ydown.pt
```

TRiGS / COLMAP clips are y down, and players rotate them 180° about z to show them. The trainer worked y
up, so the checkpoint is rotated 180° about z: positions, anchors and every twist vector `zeta_*` as
vectors, quaternions pre-multiplied, real SH coefficients of order m multiplied by (−1)^|m|.

**The flipped checkpoint stores quaternions xyzw.** The trainer stores wxyz, but `make-trigs-timeline.py`
(version `2026-09-16.runtime-bin-v2`) copies the checkpoint quaternions verbatim and labels them
`quaternionOrder: "xyzw"`; exported as is, every splat renders mis-rotated ("furry"). Use the flipped
checkpoint only for the export.

### 4. Export

```
asog flipbook export <work>/ckpt_ydown.pt [more segments …] --output <work>/clip.trigsbin
```

runs `make-trigs-timeline.py from-checkpoints --frame-rate 30 --checkpoint <c> --range a:b:a:b …`, one clip
per checkpoint, back to back with hard cuts; frame counts and rate come from the checkpoints. For one
checkpoint this writes the same payload as `from-checkpoint --chunk-duration-seconds <segment length>`.
Several segments give one file with several clips on one timeline.

The exporter's default `--static-storage global` puts static splats in a top-level group that `.asog` v1
cannot hold (a splat has one clip); the encoder refuses such a file. The flip-book trainer makes no static
splats, so this does not arise here; for other checkpoints export with `--static-storage per-clip`.

### 5. Encode

```
asog flipbook encode <work>/clip.trigsbin <work>/clip.asog [--audio sound.mp3]
```

The same as `asog encode` (see the package README for the options).

### 6. Evaluate

```
asog flipbook evaluate <work>/frames <work>/clip.asog --compare-png <work>/compare.png
```

The teacher is the staged frames, i.e. the flip book rotated exactly; `--rotation` (default `0,0,0`) takes
other teacher frames to the trainer's y up, e.g. `-90,0,0` to evaluate against the original z-up flip book.

Renders the teacher PLYs and the `.asog` (decoded and evaluated per frame as a player does) with gsplat from
8 held-out cameras: 4 longitudes offset by half the training step, 2 rows at ±33.3°. Instead of rotating the
splats (and their SH), each side keeps its frame and the cameras are moved. Reports PSNR (student vs
teacher), flicker (mean |frame(t+1) − frame(t)|, full frame and on the subject), jitter (|x(t) − (x(t−1) +
x(t+1))/2| on the subject; near zero for smooth motion, large for splats popping in and out) and the
dynamic splats' visible lives in frames.

### All at once

```
asog flipbook run <flipbook dir> <work> --start 0 --count 180 --stride 2 --rotation=-90,0,0 [--audio sound.mp3]
```

Reuses `<work>/frames` and `<work>/train/trigs_ckpt.pt` when they exist (delete them to redo).

## Results (Rubloff)

| | frames | splats | PSNR | jitter vs teacher | median life | size |
|---|---|---|---|---|---|---|
| segment 5, 6 s | 180 | 400k trained | 43.6 dB ¹ | 0.42× | 28 frames | 8.6 MB `.asog` (202 MB as per-frame SOG) |
| 1-s test | 30 | 150k trained, 105k kept | 45.4 dB ¹ / 42.2 dB ² / 39.6 dB ³ | 0.58× ³ | 25 frames ³ | 3.2 MB |

Both runs were trained before staging rotated frames, with the trainer's `--transform.rotation`.

¹ uncompressed student PLYs against the trainer's own (rotated) teacher renders, the original measurement.
² the same student PLYs against the flip book itself.
³ the `.asog` against the flip book (`asog flipbook evaluate`).

The gap between ¹ and ² is the trainer's fault, not the student's: **the trainer's `--transform.rotation`
rotates SH bands 2 and 3 wrongly.** Rendering a rotated source frame with the trainer's own transform
against the unrotated frame seen from correspondingly rotated cameras (identical by construction) gives
109 dB with SH up to degree 1, 47.7 dB up to degree 2 and 44.3 dB with degree 3. The trainer's ground
truth therefore has slightly wrong view-dependent colour, the student learns it, and the evaluation here,
which never rotates SH, measures against the true flip book. **This is now handled by `stage --rotation`**:
the frames are rotated with exact SH rotations before training and the trainer gets no rotation of its own,
so its ground truth is the flip book (110 dB on the training views). The runs above predate this and have
not been retrained yet.

From ² to ³, quantisation and pruning 30% of the splats cost about 2.6 dB. `asog check` against the
`.trigsbin` gives a centre error over each splat's visible life of 3.3 mm median, 22 mm p99 at the default
`qp 2`, `qa 8`.

## Training on a cloud GPU

`docker/trainer/Dockerfile` builds an image with the Open4DGS trainer and exporter (cloned at fixed
commits, built for compute capability 8.0–12.0) and this package in one environment, with
`ASOG_TRIGS_PYTHON` and `ASOG_OPEN4DGS_CORE` already set. GitHub Actions publishes it as
`ghcr.io/arrival-space/animated-sog-trainer:latest`. The install is `docker/trainer/setup.sh`, which also
sets up a stock CUDA 12.8 machine without the image (`TORCH_CUDA_ARCH_LIST=9.0 bash setup.sh` on an H100,
about 10 minutes, mostly compiling gsplat and pytorch3d). On a pod (e.g. RunPod, with TCP port 22 exposed and
your public key in `PUBLIC_KEY`):

```
rsync -a frames/ root@<pod>:/workspace/frames/          # one PLY per frame
ssh root@<pod> "asog flipbook run /workspace/frames /workspace/run --rotation=-90,0,0 \
    --target-count 800000 --steps 80000 --prune 0 --qp 1 --qa 4"
scp root@<pod>:/workspace/run/clip.asog .
```

A GPU with more memory (and a machine with more RAM for the trainer's image cache) allows larger training
images and more views than a 16 GB card: the cache holds frames × views × width × height × 4 bytes.
