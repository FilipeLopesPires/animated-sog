# Test data

| file | what | size |
|---|---|---|
| `rubloff-1s.asog` | 1 s (30 frames) of the Rubloff flip book, distilled with TRiGS (150 k splats, 105 k after pruning), no sound | 3.2 MB |
| `rubloff-6s.asog` | 6 s (180 frames) of the Rubloff flip book, distilled with TRiGS (400 k splats, encoded `--prune 0 --qp 1 --qa 4`), no sound | 12.4 MB |
| `rubloff-6s-4k.asog` | the same 6 s, distilled with TRiGS at 891 k splats (every 18th frame a keyframe, 1024², 120 views, 150 k steps on an H100), encoded `--prune 0 --qp 1 --qa 4`, no sound; the viewers' default | 30.0 MB |

The Rubloff flip book is [IR-MichaelRF](https://github.com/Infinite-Realities-Research/IR-MichaelRF): Michael Rubloff
([radiancefields.com](https://radiancefields.com/)) captured by [Infinite Realities](https://www.ir-ltd.net/), part of the
[Infinite-Realities Datasets](https://github.com/Infinite-Realities-Research/IR-Datasets). Its terms are in the dataset's
[LICENSE.md](https://github.com/Infinite-Realities-Research/IR-MichaelRF/blob/main/LICENSE.md); please cite both
`ir2025datasets` and `ir2025michaelrf` (BibTeX in the [main README](../README.md#dataset-citations)).
