"""Command line interface: ``terrasylva generate --size 512 --seed 7 -o out``."""
from __future__ import annotations

import argparse
import os
import time

from . import export, render
from .world import (Hydraulic, Noise, Pipeline, Rivers, Thermal, World,
                    load_recipe, save_recipe)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="terrasylva", description=__doc__)
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
    g.add_argument("--wrap", action="store_true", help="loop in both directions")
    g.add_argument("--cell-size", type=float, default=1.0, help="metres per cell")
    r = sub.add_parser("run", help="generate from a JSON recipe")
    r.add_argument("recipe")
    for q in (g, r):
        q.add_argument("--obj-resolution", type=int, default=256)
        q.add_argument("-o", "--out", default="out", help="output directory")
    return p


def _export(world: World, out_dir: str, obj_resolution: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, name)  # noqa: E731
    h = world["height"]
    look = dict(relief=world.relief, wrap=world.wrap)
    export.save_png16(out("heightmap.png"), h)
    export.save_raw16(out("heightmap.r16"), h)
    export.save_normal_map(out("normal.png"), h, world.relief, wrap=world.wrap)
    export.save_obj(out("terrain.obj"), h, max_resolution=obj_resolution,
                    vertical_scale=world.relief / world.size)
    render.save_preview(out("preview.png"), h, world.layers.get("rivers"),
                        lakes=world.layers.get("lakes"), **look)
    if "base" in world:
        render.save_preview(out("preview_base.png"), world["base"], **look)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        world, pipeline = load_recipe(args.recipe)
    else:
        world = World(size=args.size, seed=args.seed, cell_size=args.cell_size,
                      wrap=args.wrap)
        steps = [Noise(scale=args.scale, octaves=args.octaves, ridged=args.ridged,
                       warp=args.warp)]
        if args.thermal > 0:
            steps.append(Thermal(iterations=args.thermal))
        if args.drops is None or args.drops > 0:
            steps.append(Hydraulic(drops=args.drops))
        steps.append(Rivers(carve=not args.no_carve, threshold=args.river_threshold))
        pipeline = Pipeline(steps)
    t0 = time.perf_counter()
    result = pipeline.run(world)
    dt = time.perf_counter() - t0
    _export(result, args.out, args.obj_resolution)
    save_recipe(os.path.join(args.out, "recipe.json"), world, pipeline)
    print(f"generated {world.size}x{world.size} (seed {world.seed}) in {dt:.2f}s -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
