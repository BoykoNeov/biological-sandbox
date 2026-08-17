"""Field -> RGBA bytes, mapped in numpy.

The measurement priced this whole path — colormap, wasm->JS, ``putImageData`` —
at **1.6-2.2% of a frame at every grid size tested**, against 98% for the
simulation step. So it is written for clarity, not for speed, and the lookup
table below is a 256-entry ``uint8`` array indexed by a normalized field: one
fancy-index and one reshape.

**These are this project's own colormaps, not matplotlib's.** They are named
rather than numbered so nothing here can be mistaken for viridis or magma, whose
exact tables are a dependency this module deliberately does not have. What they
*do* share with those is the property that matters: for a sequential map the
perceived luminance rises monotonically from one end to the other, so a reader
ranks two patches the same way in colour and in greyscale, and a printed or
colour-blind copy still carries the ordering. ``tests/test_web_bridge.py``
asserts that numerically rather than trusting the stops to have been chosen
well — the same posture as everything else here.

A **diverging** map is exempt from that check and must be: its whole point is a
light middle with two dark ends, so luminance is monotone on each half and not
across. It is offered for fields with a meaningful zero and is the wrong default
for one without.
"""

from __future__ import annotations

import numpy as np

__all__ = ["COLORMAPS", "SEQUENTIAL", "DIVERGING", "lut", "to_rgba", "relative_luminance"]

# Control points as (position in [0, 1], R, G, B) in 8-bit sRGB, interpolated
# componentwise. Positions are uneven on purpose: an evenly-spaced ramp spends
# too much of its range in the dark end, where small differences are hardest to
# see.
_STOPS: dict[str, tuple[tuple[float, int, int, int], ...]] = {
    # Sequential, dark-to-light through purple and orange.
    "ember": (
        (0.00, 0, 0, 4),
        (0.15, 40, 11, 84),
        (0.30, 101, 21, 110),
        (0.50, 159, 42, 99),
        (0.70, 212, 72, 66),
        (0.85, 245, 125, 21),
        (0.95, 252, 193, 60),
        (1.00, 252, 253, 191),
    ),
    # Sequential, dark-to-light through blue and cyan.
    "ice": (
        (0.00, 2, 2, 10),
        (0.25, 17, 45, 110),
        (0.50, 18, 96, 163),
        (0.72, 46, 153, 187),
        (0.88, 137, 204, 207),
        (1.00, 245, 252, 255),
    ),
    # Sequential, neutral. The control case: anything a colour map appears to
    # show that this one does not is the colour map talking.
    "slate": (
        (0.00, 8, 8, 12),
        (0.50, 122, 124, 130),
        (1.00, 246, 248, 252),
    ),
    # Diverging, for fields with a meaningful zero in the middle.
    "tide": (
        (0.00, 5, 48, 97),
        (0.25, 67, 147, 195),
        (0.50, 247, 247, 247),
        (0.75, 214, 96, 77),
        (1.00, 103, 0, 31),
    ),
}

SEQUENTIAL: frozenset[str] = frozenset({"ember", "ice", "slate"})
DIVERGING: frozenset[str] = frozenset({"tide"})


def _build(name: str) -> np.ndarray:
    stops = _STOPS[name]
    positions = np.array([s[0] for s in stops], dtype=float)
    channels = np.array([s[1:] for s in stops], dtype=float)
    grid = np.linspace(0.0, 1.0, 256)
    table = np.empty((256, 3), dtype=float)
    for c in range(3):
        table[:, c] = np.interp(grid, positions, channels[:, c])
    return np.rint(table).astype(np.uint8)


COLORMAPS: dict[str, np.ndarray] = {name: _build(name) for name in _STOPS}


def lut(name: str) -> np.ndarray:
    """The ``(256, 3)`` uint8 lookup table for ``name``."""
    try:
        return COLORMAPS[name]
    except KeyError:
        raise KeyError(f"unknown colormap {name!r}; have {sorted(COLORMAPS)}") from None


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec. 709 relative luminance of 8-bit sRGB, linearized first.

    Linearizing is not a detail: averaging the raw 0-255 channels would call a
    saturated blue and a saturated yellow similarly bright, which is exactly the
    mistake a luminance check exists to catch.
    """
    c = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def to_rgba(
    field: np.ndarray,
    *,
    cmap: str = "ember",
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Map a 2-D field to a flat ``(h*w*4,)`` uint8 RGBA buffer.

    Returns the buffer together with the ``(vmin, vmax)`` **actually used**, so a
    caller can display the range beside the picture. An image drawn with a
    silently autoscaled range shows shape while concealing magnitude: a pattern
    decaying to nothing looks exactly like one at full strength when the scale
    follows it down.

    A constant field is mapped to the middle of the map rather than to one end.
    Either choice is arbitrary; the middle is the one that does not look like a
    result.
    """
    table = lut(cmap)
    values = np.asarray(field, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"field must be 2-D, got shape {values.shape}")

    finite = values[np.isfinite(values)]
    lo = float(vmin) if vmin is not None else (float(finite.min()) if finite.size else 0.0)
    hi = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else 1.0)

    span = hi - lo
    if not np.isfinite(span) or span <= 0.0:
        normalized = np.full(values.shape, 0.5)
    else:
        normalized = (values - lo) / span
    # Non-finite entries would index the table at random after the cast; they are
    # pinned to the low end and are visible as such rather than as noise.
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)

    index = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = table[index]
    rgba[..., 3] = 255
    return rgba.reshape(-1), lo, hi
