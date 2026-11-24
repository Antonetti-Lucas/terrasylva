"""Command line interface: ``terrainforge generate --size 512 --seed 7 -o out``."""
from __future__ import annotations

import argparse
import os
import time

from . import export, render
from .terrain import generate


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terrainforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="generate a terrain and export it")
    g.add_argument("--size", type=int, default=512)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--scale", type=float, default=3.0, help="base noise frequency")
    g.add_argument("--octaves", type=int, default=8)
    g.add_argument("--ridged", type=float, default=0.5, help="0 smooth .. 1 ridged")
    g.add_argument("--warp", type=float, default=0.35)
    g.add_argument("--drops", type=int, default=None, help="hydraulic erosion droplets")
    g.add_argument("--thermal", type=int, default=20, help="thermal erosion iterations")
    g.add_argument("--no-carve", action="store_true", help="don't carve river valleys")
    g.add_argument("--river-threshold", type=float, default=None,
                   help="upstream cells needed to count as a river (default 0.4%% of map)")
    g.add_argument("--obj-resolution", type=int, default=256)
    g.add_argument("-o", "--out", default="out", help="output directory")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    t = generate(args.size, seed=args.seed, scale=args.scale, octaves=args.octaves,
                 ridged=args.ridged, warp=args.warp, erosion_drops=args.drops,
                 thermal_iterations=args.thermal, carve=not args.no_carve,
                 river_threshold=args.river_threshold)
    dt = time.perf_counter() - t0
    out = lambda name: os.path.join(args.out, name)  # noqa: E731
    export.save_png16(out("heightmap.png"), t.heightmap)
    export.save_raw16(out("heightmap.r16"), t.heightmap)
    export.save_normal_map(out("normal.png"), t.heightmap)
    export.save_obj(out("terrain.obj"), t.heightmap, max_resolution=args.obj_resolution)
    render.save_preview(out("preview.png"), t.heightmap, t.rivers, lakes=t.lakes)
    render.save_preview(out("preview_base.png"), t.base)
    print(f"generated {args.size}x{args.size} (seed {args.seed}) in {dt:.2f}s -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
