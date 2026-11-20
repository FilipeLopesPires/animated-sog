"""Animated SOG (`.asog`): compressed 4D Gaussian splat clips.

`encode` turns a TRiGS runtime file (`.trigsbin`) into an `.asog`; `read` decodes one and
`Asog.evaluate(t)` gives the splats at a time. `asog.flipbook` converts per-frame 3DGS PLY sequences
("flip books") through the external Open4DGS TRiGS trainer.
"""

from .decode import Asog, Frame, read
from .encode import EncodeStats, encode

__all__ = ["Asog", "EncodeStats", "Frame", "encode", "read"]
__version__ = "0.1.0"
