"""Climate: temperature, rainfall and moisture layers.

Temperature is a north-south band minus a lapse with altitude. Rainfall comes
from a steady-state moisture simulation: wind carries moist air from the sea and
lakes over the land, rain falls out of it as it travels, and falls out faster
where it is forced uphill. The far side of a mountain therefore stays dry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from .hydrology import detect_lakes
from .noise import fbm

if TYPE_CHECKING:  # world.py imports this module, so only import it for type checks
    from .world import World

# moisture of air entering an unwrapped map from outside
_INFLOW = 0.4
# terrain detail the air ignores: Gaussian sigma as a fraction of the map width
_AIR_SMOOTHING = 0.012


# ------------------------------------------------------------------ resampling
def _area_matrix(n: int, r: int) -> np.ndarray:
    """(r, n) matrix averaging ``n`` cells into ``r`` by exact area overlap."""
    s = n / r
    lo = np.arange(r)[:, None] * s
    hi = lo + s
    j = np.arange(n)[None, :]
    overlap = np.clip(np.minimum(hi, j + 1) - np.maximum(lo, j), 0.0, None)
    return overlap / s


def _linear_matrix(n: int, r: int, wrap: bool) -> np.ndarray:
    """(n, r) matrix interpolating ``r`` cell centres linearly onto ``n`` cells."""
    q = (np.arange(n) + 0.5) * (r / n) - 0.5
    i0 = np.floor(q).astype(np.int64)
    f = q - i0
    m = np.zeros((n, r))
    rows = np.arange(n)
    for idx, weight in ((i0, 1.0 - f), (i0 + 1, f)):
        idx = idx % r if wrap else np.clip(idx, 0, r - 1)
        np.add.at(m, (rows, idx), weight)
    return m


def _blur_matrix(n: int, sigma: float, wrap: bool) -> np.ndarray:
    """(n, n) Gaussian blur with rows summing to 1."""
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    if wrap:
        d = np.minimum(d, n - d)
    m = np.exp(-0.5 * (d / max(sigma, 1e-6)) ** 2)
    return m / m.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- moisture sim
@njit(cache=True, nogil=True)
def _sample(a, y, x, wrap, fill):
    """Bilinear sample of ``a`` at (y, x); cells outside a non-wrapping map read ``fill``."""
    n = a.shape[0]
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    fy = y - y0
    fx = x - x0
    acc = 0.0
    for dy in range(2):
        for dx in range(2):
            wgt = (fy if dy else 1.0 - fy) * (fx if dx else 1.0 - fx)
            if wgt == 0.0:
                continue
            yy = y0 + dy
            xx = x0 + dx
            if wrap:
                acc += wgt * a[yy % n, xx % n]
            elif 0 <= yy < n and 0 <= xx < n:
                acc += wgt * a[yy, xx]
            else:
                acc += wgt * fill
    return acc


@njit(cache=True, nogil=True)
def _simulate(h, water, ux, uy, wrap, rain_out, orographic, evaporation, spread, inflow,
              max_sweeps, tol):
    """Steady-state moist air advected one cell along the dominant wind axis per step.

    Cells are updated in place in wind order, so every cell reads upwind values
    that are already final. On a map that does not loop one sweep is exact. On a
    looping map the first slice reads the last one, so sweeps repeat until the
    change is below ``tol``. They converge, and to a single answer, because
    rain removes a fixed fraction of the air's moisture on every step.

    Each step also mixes the air sideways (across the minor axis, so it still
    only reads the previous slice). Without that, a row of land running along
    the wind would dry out as a perfectly straight stripe.

    Returns (moisture, precipitation density). Density is rain per cell in units
    of "saturated air rains out at rate 1 per unit length".
    """
    n = h.shape[0]
    steplen = np.sqrt(ux * ux + uy * uy)
    rain_step = rain_out * steplen / n
    evap = 1.0 - np.exp(-evaporation * steplen / n)
    m = np.zeros((n, n))
    precip = np.zeros((n, n))
    x_major = abs(ux) >= abs(uy)
    for sweep in range(max_sweeps if wrap else 1):
        change = 0.0
        for a in range(n):
            for b in range(n):
                if x_major:
                    x = a if ux > 0 else n - 1 - a
                    y = b
                else:
                    y = a if uy > 0 else n - 1 - a
                    x = b
                sy = y - uy
                sx = x - ux
                m_in = 0.5 * _sample(m, sy, sx, wrap, inflow)
                if x_major:
                    m_in += 0.25 * (_sample(m, sy + spread, sx, wrap, inflow)
                                    + _sample(m, sy - spread, sx, wrap, inflow))
                else:
                    m_in += 0.25 * (_sample(m, sy, sx + spread, wrap, inflow)
                                    + _sample(m, sy, sx - spread, wrap, inflow))
                dh = max(h[y, x] - _sample(h, sy, sx, wrap, h[y, x]), 0.0)
                rain = m_in * (1.0 - np.exp(-(rain_step + orographic * dh)))
                m_new = m_in - rain + water[y, x] * evap * (1.0 - (m_in - rain))
                change = max(change, abs(m_new - m[y, x]))
                m[y, x] = m_new
                precip[y, x] = rain * n / steplen
        if change < tol:
            break
    return m, precip


def _wind_vector(wind: float) -> tuple[float, float]:
    """(columns, rows) travelled per step by wind coming *from* ``wind`` degrees.

    Degrees run clockwise from north, so 270 is a westerly: air moves east.
    Rows grow southwards. The larger component is scaled to exactly 1.
    """
    a = np.radians(wind)
    dx, dy = -np.sin(a), np.cos(a)
    big = max(abs(dx), abs(dy))
    return float(dx / big), float(dy / big)


@dataclass(frozen=True)
class Climate:
    """Create ``temperature`` (deg C), ``rainfall`` and ``moisture``.

    Temperature runs from ``cold_temp`` to ``warm_temp`` along the north-south
    axis. ``wobble`` bends the bands with low-frequency noise, as a fraction of
    the map height (0 gives perfectly straight bands). On a looping world the
    band is one full cosine cycle in y, with its warm side at ``warm_band`` (a
    fraction of the map height). Without ``wrap`` it ramps from cold at the top
    to warm at the bottom. Altitude above the sea cools by ``lapse`` degrees
    between sea level and the highest point (height 1). That is a property of
    the compressed world, not 6.5 C/km.

    Rainfall is the output of a moisture simulation on a grid of at most
    ``grid`` cells per side, interpolated back up:

    * ``wind``: where the wind blows from, degrees clockwise from north
      (270 is a westerly, air moving east).
    * ``evaporation``: how fast air over open water (ocean, lakes) takes up
      moisture, in e-folds per map width.
    * ``rain_out``: share of moisture that falls as ordinary rain, in e-folds
      per map width. It is also what makes a looping simulation converge to a
      single answer, so it must be positive.
    * ``orographic``: extra rain as the air rises. 1.5 removes about 78% of the
      moisture over a climb from sea level to the highest peak.
    * ``base``: rainfall that needs no moisture at all, relative to the rain of
      fully saturated air.
    * ``spread``: how far the air mixes sideways while crossing the map, as a
      standard deviation in fractions of the map width. It keeps rain from
      forming ruler-straight stripes along the wind.
    * ``contrast``: exponent applied to the simulated rain before adding
      ``base``. 1 keeps it as simulated; below 1 evens out the extremes, so
      windward slopes are wetter than plains but not a hundred times wetter.

    ``rainfall`` is scaled to a mean of 1 over land (everything outside the
    ``ocean`` layer); ``moisture`` squashes it into [0, 1]. Lakes are found
    again from the current heights, never read from an earlier run.
    """

    warm_temp: float = 34.0
    cold_temp: float = 0.0
    lapse: float = 25.0
    warm_band: float = 0.5
    wobble: float = 0.12
    wind: float = 270.0
    evaporation: float = 8.0
    rain_out: float = 0.7
    orographic: float = 1.5
    base: float = 0.1
    contrast: float = 0.6
    spread: float = 0.08
    grid: int = 256
    name: str = "climate"

    def _validate(self) -> None:
        for field_name in ("warm_temp", "cold_temp", "lapse", "warm_band", "wobble", "wind",
                           "evaporation", "rain_out", "orographic", "base", "contrast", "spread"):
            if not np.isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")
        if (self.lapse < 0 or self.wobble < 0 or self.orographic < 0 or self.base < 0
                or self.spread < 0):
            raise ValueError("lapse, wobble, orographic, base and spread must be >= 0")
        if self.evaporation <= 0 or self.rain_out <= 0 or self.contrast <= 0:
            raise ValueError("evaporation, rain_out and contrast must be > 0")
        if self.grid < 8:
            raise ValueError("grid must be >= 8")

    def temperature(self, world: World, height: np.ndarray, sea: float) -> np.ndarray:
        n = world.size
        rows = np.arange(n, dtype=np.float64)[:, None] * np.ones((1, n))
        v = rows / n if world.wrap else rows / (n - 1)
        if self.wobble > 0:
            noise = fbm(n, seed=world.seed_for(self.name + "/temperature"), scale=3.0,
                        octaves=4, warp=0.3, wrap=world.wrap)
            v = v + self.wobble * 2.0 * (noise.astype(np.float64) - 0.5)
        if world.wrap:
            warmth = 0.5 + 0.5 * np.cos(2.0 * np.pi * (v - self.warm_band))
        else:
            warmth = np.clip(v, 0.0, 1.0)
        t = self.cold_temp + (self.warm_temp - self.cold_temp) * warmth
        altitude = np.clip((height - sea) / max(1.0 - sea, 1e-9), 0.0, 1.0)
        return (t - self.lapse * altitude).astype(np.float32)

    def rainfall(self, world: World, height: np.ndarray, sea: float,
                 ocean: np.ndarray | None, lakes: np.ndarray) -> np.ndarray:
        n = world.size
        r = min(n, int(self.grid))
        water = lakes if ocean is None else lakes | ocean
        relief = np.clip((np.maximum(height, sea) - sea) / max(1.0 - sea, 1e-9), 0.0, 1.0)
        if r < n:
            down = _area_matrix(n, r)
            up = _linear_matrix(n, r, world.wrap)
            relief_c = down @ relief @ down.T
            water_c = down @ water.astype(np.float64) @ down.T
        else:
            relief_c, water_c = relief, water.astype(np.float64)
        # single-cell roughness must not count as mountains, or small maps get speckled
        blur = _blur_matrix(r, _AIR_SMOOTHING * r, world.wrap)
        relief_c = blur @ relief_c @ blur.T
        ux, uy = _wind_vector(self.wind)
        _, precip = _simulate(np.ascontiguousarray(relief_c), np.ascontiguousarray(water_c),
                              ux, uy, bool(world.wrap), float(self.rain_out),
                              float(self.orographic), float(self.evaporation),
                              float(self.spread) * np.sqrt(2.0 * r), _INFLOW,
                              2000, 1e-7)
        if r < n:
            precip = np.clip(up @ precip @ up.T, 0.0, None)
        rain = self.base + precip ** self.contrast
        land = np.ones((n, n), dtype=bool) if ocean is None else ~ocean
        if not land.any():
            land = np.ones((n, n), dtype=bool)
        mean = float(rain[land].mean())
        return rain / mean if mean > 0 else np.ones((n, n))

    def __call__(self, world: World) -> None:
        self._validate()
        height = np.clip(np.asarray(world["height"], dtype=np.float64), 0.0, 1.0)
        sea = 0.0 if world.sea_level is None else float(world.sea_level)
        ocean = world.layers.get("ocean")
        ocean = None if ocean is None else np.asarray(ocean).astype(bool)
        lakes = detect_lakes(height, outlets=ocean, wrap=world.wrap).mask
        rain = self.rainfall(world, height, sea, ocean, lakes)
        world["temperature"] = self.temperature(world, height, sea)
        world["rainfall"] = rain.astype(np.float32)
        world["moisture"] = (rain / (rain + 1.0)).astype(np.float32)
