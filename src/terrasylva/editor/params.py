"""Which controls the editor shows for a step, derived from its dataclass fields."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

HIDDEN = {"name", "mask", "table"}


@dataclass(frozen=True)
class Param:
    field: str
    label: str
    kind: str                   # "float", "int" or "bool"
    low: float = 0.0
    high: float = 1.0
    optional: bool = False      # None is allowed and means "automatic"
    fallback: float = 0.0       # starting value when an optional one is switched on
    help: str = ""


# (step, field): label, low, high, help, whole numbers, fallback for optional values
_KNOWN = {
    ("Noise", "scale"): ("Feature count", 1, 12, "How many hills and ranges fit across the world", True, 0),
    ("Noise", "octaves"): ("Detail", 1, 10, "Layers of finer and finer detail", True, 0),
    ("Noise", "ridged"): ("Ridges", 0, 1, "0 rolling hills, 1 sharp mountain ridges", False, 0),
    ("Noise", "warp"): ("Twist", 0, 1, "Bends the shapes so they look less regular", False, 0),
    ("Continents", "scale"): ("Continent count", 1, 8, "Low: a few large ones. High: an archipelago", True, 0),
    ("Continents", "strength"): ("Strength", 0, 1, "How strongly continents shape the heights", False, 0),
    ("Continents", "warp"): ("Coast twist", 0, 1, "How irregular the coasts are", False, 0),
    ("Thermal", "iterations"): ("Amount", 0, 80, "Softens cliffs by letting material slide", True, 0),
    ("Thermal", "talus"): ("Stable slope", 0.1, 2, "Steepness that stays in place", False, 0),
    ("Hydraulic", "drops"): ("Raindrops", 0, 2_000_000, "More drops carve deeper. Automatic: 0.3 per cell", True, 300_000),
    ("Ocean", "land"): ("Land", 0.05, 0.95, "Fraction of the world above the sea", False, 0),
    ("Ocean", "level"): ("Fixed sea level", 0, 1, "Automatic: chosen to match Land", False, 0.35),
    ("Ocean", "min_area"): ("Smallest sea", 0, 0.1, "Smaller basins become lakes instead", False, 0),
    ("Climate", "warm_temp"): ("Warm zone (°C)", -10, 45, "", False, 0),
    ("Climate", "cold_temp"): ("Cold zone (°C)", -40, 30, "", False, 0),
    ("Climate", "lapse"): ("Cooling with height (°C)", 0, 60, "From sea level to the highest peak", False, 0),
    ("Climate", "warm_band"): ("Warm zone position", 0, 1, "Only for looping worlds", False, 0),
    ("Climate", "wobble"): ("Zone irregularity", 0, 0.5, "", False, 0),
    ("Climate", "wind"): ("Wind from (°)", 0, 360, "0 north, 90 east, 180 south, 270 west", False, 0),
    ("Climate", "evaporation"): ("Evaporation", 0.5, 20, "How fast air picks up moisture over water", False, 0),
    ("Climate", "rain_out"): ("Rain along the way", 0.05, 4, "High: the rain stays near the coasts", False, 0),
    ("Climate", "orographic"): ("Mountain rain", 0, 5, "How strongly mountains catch rain", False, 0),
    ("Climate", "base"): ("Minimum rain", 0, 1, "", False, 0),
    ("Climate", "contrast"): ("Wet/dry contrast", 0.1, 1.5, "", False, 0),
    ("Climate", "spread"): ("Rain spread", 0, 0.3, "", False, 0),
    ("Climate", "grid"): ("Simulation grid", 32, 512, "", True, 0),
    ("Rivers", "carve"): ("Carve valleys", 0, 1, "", False, 0),
    ("Rivers", "threshold"): ("River size", 2, 20_000, "Cells a river must drain. Lower: more rivers", True, 1500),
    ("Biomes", "softness"): ("Border softness", 0.02, 1, "", False, 0),
    ("Biomes", "noise"): ("Border irregularity", 0, 1, "", False, 0),
    ("Biomes", "wet_bonus"): ("Green riversides", 0, 1, "", False, 0),
    ("Biomes", "wet_reach"): ("Riverside width", 0, 0.1, "", False, 0),
}

WORLD = (
    Param("seed", "Seed", "int", 0, 999_999, help="The same seed always gives the same world"),
    Param("wrap", "Loop both ways", "bool", help="Leaving one side brings you back on the other"),
    Param("relief", "Mountain height", "float", 0.01, 0.4, help="Height range as a fraction of the world's width"),
    Param("size", "Final size (cells)", "int", 64, 4096, help="Resolution used when exporting"),
    Param("cell_size", "Cell size (m)", "float", 0.25, 20),
)


def params_for(step) -> list[Param]:
    """Controls for a step. Unknown numeric fields get a generic range."""
    out = []
    for f in dataclasses.fields(step):
        if f.name in HIDDEN:
            continue
        value = getattr(step, f.name)
        kind_name = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
        optional = "None" in kind_name
        known = _KNOWN.get((type(step).__name__, f.name))
        if "bool" in kind_name:
            kind = "bool"
        elif "int" in kind_name or "float" in kind_name:
            kind = "int" if "int" in kind_name else "float"
        else:
            continue
        if known:
            label, low, high, text, whole, fallback = known
            if kind == "float" and whole:
                kind = "int"
        else:
            label, text = f.name.replace("_", " ").capitalize(), ""
            base = f.default if isinstance(f.default, (int, float)) else (value or 1)
            low, high = min(0, 2 * base), max(1, 4 * abs(base))
            fallback = base
        # never hide a value that came from a file
        if value is not None and kind != "bool":
            low, high = min(low, value), max(high, value)
        out.append(Param(f.name, label, kind, low, high, optional, fallback, text))
    return out


def coerce(step, field: str, value):
    """Value in the type the dataclass declares (whole-number floats stay floats)."""
    if value is None or isinstance(value, bool):
        return value
    f = next(f for f in dataclasses.fields(step) if f.name == field)
    return int(value) if "int" in str(f.type) else float(value)
