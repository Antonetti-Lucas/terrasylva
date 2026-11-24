"""Command line interface: ``terrasylva generate --size 512 --seed 7 -o out``."""
from __future__ import annotations

import argparse
import os
import time

from . import export, render
from .biomes import Biomes, biome_legend, default_biome_table
from .world import (Climate, Continents, Hydraulic, Noise, Ocean, Pipeline, Rivers, Thermal,
                    World, load_recipe, save_recipe)


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
    g.add_argument("--land", type=float, default=None,
                   help="fraction of land (0-1); adds continents and an ocean")
    g.add_argument("--continents", type=float, default=2.0,
                   help="continent frequency: 2 = a few large ones, 5 = archipelago")
    g.add_argument("--cell-size", type=float, default=1.0, help="metres per cell")
    g.add_argument("--biomes", action="store_true",
                   help="add climate and biomes (temperature, rainfall, biome and splat maps)")
    r = sub.add_parser("run", help="generate from a JSON recipe")
    r.add_argument("recipe")
    for q in (g, r):
        q.add_argument("--obj-resolution", type=int, default=256)
        q.add_argument("-o", "--out", default="out", help="output directory")
    return p


def _biome_table(pipeline: Pipeline | None, world: World):
    """The table that made the world's ``biome`` layer, or None if it can't be told."""
    for step in (pipeline.steps if pipeline else []):
        if isinstance(step, Biomes):
            return step.table
    table = default_biome_table()
    return table if world["biome_weights"].shape[-1] == len(table) else None


def _export(world: World, out_dir: str, obj_resolution: int,
            pipeline: Pipeline | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, name)  # noqa: E731
    h = world["height"]
    look = dict(relief=world.relief, wrap=world.wrap)
    export.save_png16(out("heightmap.png"), h)
    export.save_raw16(out("heightmap.r16"), h)
    export.save_normal_map(out("normal.png"), h, world.relief, wrap=world.wrap)
    export.save_obj(out("terrain.obj"), h, max_resolution=obj_resolution,
                    vertical_scale=world.relief / world.size, wrap=world.wrap)
    masks = {k: world.layers.get(k) for k in ("ocean", "lakes", "rivers")}
    export.save_water_map(out("water.png"), h.shape, **masks)
    render.save_preview(out("preview.png"), h, sea_level=world.sea_level,
                        lake_depth=world.layers.get("lake_depth"), **masks, **look)
    if "base" in world:
        render.save_preview(out("preview_base.png"), world["base"], **look)
    for kind in ("temperature", "rainfall"):
        if kind in world:
            render.save_layer_preview(out(f"preview_{kind}.png"), world[kind], kind, h,
                                      ocean=world.layers.get("ocean"), **look)
    if "biome" in world:
        export.save_biome_map(out("biomes.png"), world["biome"])
        if "splat" in world:
            export.save_splat_map(out("splat.png"), world["splat"])
        table = _biome_table(pipeline, world) if "biome_weights" in world else None
        if table is not None:
            export.save_biome_legend(out("biomes.json"), biome_legend(table))
            render.save_preview(out("preview_biomes.png"), h, sea_level=world.sea_level,
                                lake_depth=world.layers.get("lake_depth"), **masks, **look,
                                biome=world["biome"], biome_colors=[b.color for b in table],
                                biome_weights=world["biome_weights"])


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        world, pipeline = load_recipe(args.recipe)
    else:
        world = World(size=args.size, seed=args.seed, cell_size=args.cell_size,
                      wrap=args.wrap)
        steps = [Noise(scale=args.scale, octaves=args.octaves, ridged=args.ridged,
                       warp=args.warp)]
        if args.land is not None:
            steps.append(Continents(scale=args.continents))
        if args.thermal > 0:
            steps.append(Thermal(iterations=args.thermal))
        if args.drops is None or args.drops > 0:
            steps.append(Hydraulic(drops=args.drops))
        if args.land is not None:
            steps.append(Ocean(land=args.land))
        if args.biomes:
            steps.append(Climate())
        steps.append(Rivers(carve=not args.no_carve, threshold=args.river_threshold))
        if args.biomes:
            steps.append(Biomes())
        pipeline = Pipeline(steps)
    t0 = time.perf_counter()
    result = pipeline.run(world)
    dt = time.perf_counter() - t0
    _export(result, args.out, args.obj_resolution, pipeline)
    save_recipe(os.path.join(args.out, "recipe.json"), world, pipeline)
    print(f"generated {world.size}x{world.size} (seed {world.seed}) in {dt:.2f}s -> {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
