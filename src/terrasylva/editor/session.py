"""Editor state and generation, independent of any GUI toolkit."""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Callable

from ..world import STEPS, Pipeline, World, default_pipeline

MIN_SIZE = 64


class Cancelled(Exception):
    """Generation was stopped because its result is no longer wanted."""


@dataclass(frozen=True)
class Item:
    """One step in the list and whether it is switched on."""

    step: object
    enabled: bool = True


@dataclass(frozen=True)
class Document:
    """Everything the user edits. Immutable, so undo is a list of these."""

    size: int = 1024
    seed: int = 1
    cell_size: float = 2.0
    relief: float = 0.1          # height range as a fraction of the world's width
    base_height: float = 0.0     # metres at height 0
    wrap: bool = True
    items: tuple[Item, ...] = ()

    def world(self, size: int | None = None) -> World:
        """Empty world covering the same ground with ``size`` cells per side."""
        n = self.size if size is None else size
        width = self.size * self.cell_size
        return World(size=n, seed=self.seed, cell_size=width / n, wrap=self.wrap,
                     height_range=(self.base_height, self.base_height + self.relief * width))

    def steps(self, size: int | None = None) -> list:
        """Enabled steps, with per-cell counts rescaled to ``size``."""
        n = self.size if size is None else size
        ratio = (n / self.size) ** 2
        out = []
        for item in self.items:
            if not item.enabled:
                continue
            step = item.step
            if getattr(step, "drops", None) is not None:
                step = replace(step, drops=int(round(step.drops * ratio)))
            if getattr(step, "threshold", None) is not None:
                step = replace(step, threshold=max(step.threshold * ratio, 2.0))
            out.append(step)
        return out


def default_document() -> Document:
    return Document(items=tuple(Item(s) for s in default_pipeline().steps))


def replace_item(doc: Document, index: int, item: Item) -> Document:
    items = list(doc.items)
    items[index] = item
    return replace(doc, items=tuple(items))


def add_step(doc: Document, kind: str) -> Document:
    """Append a step of a registered kind, with a name nothing else uses."""
    step = STEPS[kind]()
    taken = {i.step.name for i in doc.items}
    if step.name in taken:
        n = 2
        while f"{step.name} {n}" in taken:
            n += 1
        step = replace(step, name=f"{step.name} {n}")
    return replace(doc, items=doc.items + (Item(step),))


def remove_step(doc: Document, index: int) -> Document:
    return replace(doc, items=doc.items[:index] + doc.items[index + 1:])


def reorder(doc: Document, order: list[int]) -> Document:
    if sorted(order) != list(range(len(doc.items))):
        raise ValueError("order must be a permutation of the steps")
    return replace(doc, items=tuple(doc.items[i] for i in order))


# ---------------------------------------------------------------- history
class History:
    """Undo and redo. Quick successive edits of one control count as one step."""

    MERGE_SECONDS = 1.0

    def __init__(self, doc: Document, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self.reset(doc)

    def reset(self, doc: Document) -> None:
        self.doc = doc
        self._past: list[Document] = []
        self._future: list[Document] = []
        self._key = None
        self._time = -math.inf
        self._saved = doc

    def commit(self, doc: Document, key=None) -> bool:
        """Make ``doc`` current. Returns False if nothing changed."""
        if doc == self.doc:
            return False
        now = self._clock()
        merge = (key is not None and key == self._key and self._past
                 and now - self._time < self.MERGE_SECONDS)
        if not merge:
            self._past.append(self.doc)
        self._future.clear()
        self._key, self._time = key, now
        self.doc = doc
        return True

    def undo(self) -> bool:
        return self._move(self._past, self._future)

    def redo(self) -> bool:
        return self._move(self._future, self._past)

    def _move(self, source: list, target: list) -> bool:
        if not source:
            return False
        target.append(self.doc)
        self.doc = source.pop()
        self._key = None
        return True

    can_undo = property(lambda self: bool(self._past))
    can_redo = property(lambda self: bool(self._future))
    dirty = property(lambda self: self.doc != self._saved)

    def mark_saved(self) -> None:
        self._saved = self.doc
        self._key = None  # an edit after saving must be undoable on its own


# ------------------------------------------------------------- generation
class Generator:
    """Runs documents, reusing the results of steps that did not change.

    Every step starts from a snapshot of the steps before it, so switching a
    step off also drops the layers it made.
    """

    def __init__(self):
        self._cache: list[tuple[str, World]] = []

    def run(self, doc: Document, size: int | None = None, *,
            progress: Callable[[int, int, str], None] | None = None,
            cancelled: Callable[[], bool] = lambda: False) -> World:
        steps = doc.steps(size)
        Pipeline(steps)  # validates names
        world = doc.world(size)
        key = repr(world)
        old, kept = self._cache, []
        reuse = True
        for i, step in enumerate(steps):
            key += repr(step)
            if reuse and i < len(old) and old[i][0] == key:
                world = old[i][1]
            else:
                reuse = False
                if cancelled():
                    raise Cancelled
                if progress:
                    progress(i, len(steps), step.name)
                world = Pipeline([step]).run(world)
            kept.append((key, world))
            # keep finished work even if a later step fails or is cancelled
            self._cache = kept
        if cancelled():
            raise Cancelled
        return world


# ------------------------------------------------------------------ files
def save(path: str, doc: Document) -> None:
    """Write a recipe that ``load_recipe`` and the CLI can also run."""
    world = doc.world()
    data = {
        "world": {"size": world.size, "seed": world.seed, "cell_size": world.cell_size,
                  "height_range": world.height_range, "wrap": world.wrap},
        "steps": [{"step": type(i.step).__name__, **asdict(i.step),
                   **({} if i.enabled else {"enabled": False})} for i in doc.items],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load(path: str) -> Document:
    """Read any recipe made of built-in steps."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        items = []
        for entry in data["steps"]:
            args = dict(entry)
            kind, enabled = args.pop("step"), bool(args.pop("enabled", True))
            if kind not in STEPS:
                raise ValueError(f"unknown step {kind!r}")
            items.append(Item(STEPS[kind](**args), enabled))
        settings = dict(data["world"])
        settings.pop("sea_level", None)  # recomputed by the Ocean step
        world = World(**settings)
        Pipeline([i.step for i in items])
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise ValueError(f"cannot open {path}: {e}") from e
    lo, hi = world.height_range
    return Document(size=world.size, seed=world.seed, cell_size=world.cell_size,
                    relief=(hi - lo) / (world.size * world.cell_size), base_height=lo,
                    wrap=world.wrap, items=tuple(items))
