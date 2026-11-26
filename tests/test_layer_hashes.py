"""Stored hashes of two small worlds, to catch changes to the output nobody meant to make.

The hashes are for Windows, since other systems round some maths differently. To update
them, run this file on Windows and paste what it prints over RECORDED_WITH and STORED:

    python tests/test_layer_hashes.py
"""
import hashlib
import platform
import sys

import numba
import numpy as np
import pytest

import terrasylva as ts

RECORDED_WITH = "numpy 2.5.5, numba 0.68.1"
STORED = {
    "default": {
        "base": "4bd74c636e4d6c8f",
        "basin": "39f30640469eb263",
        "biome": "c882f0d20f7146fa",
        "biome_weights": "38ea5f8c6515b662",
        "continents": "912fcaa29037692d",
        "flow": "d63bedf8579004a4",
        "height": "9e0831bb59661cd1",
        "lake_depth": "d9d96b637e6bfcbd",
        "lakes": "45636ff3b8c53df7",
        "moisture": "80130f09ce737fd4",
        "ocean": "50fb967ab5b5fe9d",
        "ocean_depth": "51485c5f482e99fc",
        "rainfall": "14aa8467fe95c7fd",
        "rivers": "a14fac8653a35a58",
        "sinks": "54d0812cfef93af2",
        "splat": "445ff6d7abe90f7a",
        "temperature": "7199ace328e3ea43",
        "uplift": "f35d1bad7e70be47",
        "sea_level": "a655c416c0f30114",
    },
    "classic": {
        "base": "4bd74c636e4d6c8f",
        "biome": "37235839d0b81114",
        "biome_weights": "13a4e1f74d634bae",
        "continents": "912fcaa29037692d",
        "flow": "c48a5c28270ce175",
        "height": "5029d41fd8ed0816",
        "lake_depth": "66066e07469cd6f4",
        "lakes": "456357b00051cf4f",
        "moisture": "8defb7f54019247c",
        "ocean": "337c5e5a4160e78f",
        "ocean_depth": "52397b36a898a879",
        "rainfall": "d77aefaf18d0bea5",
        "rivers": "5331fa50b4fe75a4",
        "splat": "c15b311b5cecbc83",
        "temperature": "9b752d7644bddd26",
        "sea_level": "114888d205530040",
    },
}


def versions() -> str:
    return f"numpy {np.__version__}, numba {numba.__version__}"


def _digest(value) -> str:
    a = np.asarray(value)
    return hashlib.sha256(f"{a.dtype} {a.shape}".encode() + a.tobytes()).hexdigest()[:16]


def layer_hashes() -> dict[str, dict[str, str]]:
    """A short hash of every layer and the sea level, for both pipelines."""
    result = {}
    for name, pipeline in (("default", ts.default_pipeline()), ("classic", ts.classic_pipeline())):
        world = pipeline.run(ts.World(size=64, seed=42))
        result[name] = {layer: _digest(a) for layer, a in sorted(world.layers.items())}
        result[name]["sea_level"] = _digest(world.sea_level)
    return result


# check_skips.py expects exactly this reason outside Windows
@pytest.mark.skipif(sys.platform != "win32", reason="layer hashes are stored for Windows only")
def test_layers_match_stored_hashes():
    now = layer_hashes()
    changed = [f"{name}/{layer}" for name in now
               for layer in sorted(now[name].keys() | STORED.get(name, {}).keys())
               if now[name].get(layer) != STORED.get(name, {}).get(layer)]
    assert not changed, (
        f"output changed: {', '.join(changed)}\n"
        f"Hashes were recorded with {RECORDED_WITH}; this run uses {versions()}.\n"
        "If you meant to change the output, or a numpy or numba update changed it, run "
        "`python tests/test_layer_hashes.py` on Windows and paste what it prints over "
        "RECORDED_WITH and STORED in tests/test_layer_hashes.py. Otherwise, find the "
        "change that caused this.")


if __name__ == "__main__":
    from llvmlite.binding import get_host_cpu_name

    print(f"# {platform.system()}, Python {platform.python_version()}, "
          f"CPU {get_host_cpu_name()}")
    print(f'RECORDED_WITH = "{versions()}"')
    print("STORED = {")
    for name, hashes in layer_hashes().items():
        print(f'    "{name}": {{')
        for layer, digest in hashes.items():
            print(f'        "{layer}": "{digest}",')
        print("    },")
    print("}")
