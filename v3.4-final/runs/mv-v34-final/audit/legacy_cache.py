"""Reuse exact DP subresults within one fully replayed fleet instance."""

from functools import lru_cache
import importlib
import sys

from proofs import V32


class Reuse:
    """Hold bounded caches for a single arm and restore the frozen functions."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.inspected = set()
        self.patches = []
        self.screens = {"full": 0, "identical": 0, "filtered": 0}
        self.screen_support = None

    def __enter__(self):
        return self

    def install(self, module, attr, limit):
        sys.path.insert(0, str(V32))
        target = importlib.import_module(module)
        original = getattr(target, attr)
        cached = lru_cache(maxsize=limit)(original)

        def wrapped(graph, reward, partition, *rest):
            # Projection includes complete effective directed matrix, horizon,
            # depot entry and return vectors. Reversal changes that full key.
            arguments = tuple(tuple(value) if type(value) is list else value for value in rest)
            return cached(graph, tuple(reward), tuple(partition), *arguments)

        self.patches.append((target, attr, original, cached))
        setattr(target, attr, wrapped)

    def install_screen_masks(self):
        target = importlib.import_module("screen2.check")
        original = target.enumerate_masks

        def wrapped(graph, reward, partition, cutoff):
            key = graph, tuple(reward), tuple(partition)
            if self.screen_support is not None:
                cached_key, minimum, complete = self.screen_support
                if key == cached_key and cutoff >= minimum:
                    if cutoff == minimum:
                        self.screens["identical"] += 1
                        return complete
                    # The complete set at the stronger (lower) threshold is a
                    # superset. A mask's support score depends only on its bits.
                    self.screens["filtered"] += 1
                    return {mask for mask in complete if mask_score(mask, reward) >= cutoff}
            self.screens["full"] += 1
            self.screen_support = None
            complete = original(graph, reward, partition, cutoff)
            self.screen_support = key, cutoff, complete
            return complete

        self.patches.append((target, "enumerate_masks", original, None))
        target.enumerate_masks = wrapped

    def inspect(self, kind):
        if not self.enabled:
            return
        modules = []
        if kind in ("zone", "screen2", "fleet-selection-v33"):
            modules.append(("zone.check", "priced", 128))
        if kind == "screen2":
            # One table is up to roughly 150 MiB. Keep one mathematical
            # support for adjacent screens. enumerate_masks reads it only.
            modules.append(("screen2.check", "suffix_table", 1))
        if kind == "maskfleet":
            modules.append(("maskfleet.check", "classes", 32))
        for module, attr, limit in modules:
            if (module, attr) not in self.inspected:
                self.install(module, attr, limit)
                self.inspected.add((module, attr))
        if kind == "screen2" and "screen masks" not in self.inspected:
            self.install_screen_masks()
            self.inspected.add("screen masks")

    def stats(self):
        result = {f"{module.__name__}.{attr}": cached.cache_info()._asdict()
                  for module, attr, _, cached in self.patches if cached is not None}
        if "screen masks" in self.inspected:
            result["screen2.check.enumerate_masks"] = self.screens.copy()
        return result

    def __exit__(self, *_):
        for module, attr, original, cached in reversed(self.patches):
            setattr(module, attr, original)
            if cached is not None:
                cached.cache_clear()
        self.screen_support = None


def mask_score(mask, reward):
    value = 0
    while mask:
        bit = mask & -mask
        value += reward[bit.bit_length() - 1]
        mask ^= bit
    return value
