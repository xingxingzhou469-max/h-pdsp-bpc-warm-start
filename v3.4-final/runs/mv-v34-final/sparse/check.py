#!/usr/bin/env python3
"""Replay every supplier-cover leaf without the optimizer, native code or site packages."""
from argparse import ArgumentParser
from array import array
from fractions import Fraction as F
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparse.model import METHOD, NEG, SCALE, Model, clusters


def priced(model, required, region, reward, prefix):
    if not model.positive:
        raise ValueError("a positive projected progress is necessary for the pricing proof")
    h, n, arc, flags = model.horizon, model.n, model.costs, model.bits
    allowed = [i for i in range(1, n) if not flags[i] or flags[i] & required]
    masks = [m for m in range(required + 1) if not m & ~required]
    by_mask = {m: j for j, m in enumerate(masks)}
    outside = {}
    inside = {}
    widths = {}
    own = {}
    offsets = {}
    frame = 0
    for i in allowed:
        cell = [j for j in allowed if region[i] == region[j]]
        own[i] = 1 << cell.index(i)
        widths[i] = 1 << (len(cell) - 1)
        offsets[i] = frame
        frame += widths[i]
    if (h + 1) * len(masks) * (frame + 2 * n) > 10_000_000:
        raise ValueError("pricing proof exceeds state limit")
    if max(max(map(abs, reward)), max(map(abs, prefix))) * (h + 1) >= 1 << 59:
        raise ValueError("pricing rewards exceed exact integer limits")

    for i in allowed:
        options = []
        self_bit = own[i]
        for local in range(widths[i]):
            split = self_bit - 1
            seen = (local & split) | ((local & ~split) << 1) | self_bit
            candidates = []
            for j in allowed:
                if region[i] != region[j] or own[j] & seen or arc[i][j] > h:
                    continue
                expanded = seen | own[j]
                split = own[j] - 1
                child = (expanded & split) | ((expanded >> 1) & ~split)
                duration = arc[i][j]
                candidates.append((duration, offsets[j] + child - duration * frame, j))
            options.append(sorted(candidates))
        inside[i] = options
        outside[i] = sorted((arc[i][j], j, flags[j], offsets[j] - arc[i][j] * frame,
                             j - arc[i][j] * n)
                            for j in allowed if region[i] != region[j] and arc[i][j] <= h)

    tables = [array("q", [NEG]) * ((h + 1) * frame) for _ in masks]
    without = [array("q", [NEG]) * ((h + 1) * n) for _ in masks]
    denied = [array("h", [-1]) * ((h + 1) * n) for _ in masks]
    # For each remaining supplier set, a cross-cell successor can take its
    # best path unless that path returns immediately to the current station.
    for t in range(h + 1):
        frame_start, n_start = t * frame, t * n
        for index, mask in enumerate(masks):
            table, alt, label = tables[index], without[index], denied[index]
            weights = prefix if mask == required else reward
            for i in allowed:
                if flags[i] & mask:
                    continue
                first = 0 if not mask and arc[i][0] <= t else NEG
                second, forbid = NEG, -1
                for distance, j, bit, offset, other in outside[i]:
                    if distance > t:
                        break
                    if bit and not bit & mask:
                        continue
                    target_mask = by_mask[mask ^ bit]
                    slot = n_start + other
                    if denied[target_mask][slot] == i:
                        suffix = without[target_mask][slot]
                    else:
                        suffix = tables[target_mask][frame_start + offset]
                    if suffix == NEG:
                        continue
                    candidate = suffix + weights[j]
                    if candidate > first:
                        first, second, forbid = candidate, first, j
                    elif candidate > second:
                        second = candidate

                start = frame_start + offsets[i]
                for local, choices in enumerate(inside[i]):
                    keep = NEG
                    for distance, offset, j in choices:
                        if distance > t:
                            break
                        suffix = table[frame_start + offset]
                        if suffix != NEG and suffix + weights[j] > keep:
                            keep = suffix + weights[j]
                    table[start + local] = max(first, keep)
                    if not local:
                        alt[n_start + i] = max(second, keep)
                label[n_start + i] = forbid

    result = 0
    for i in allowed:
        duration = arc[0][i]
        if duration <= h:
            value = tables[by_mask[required ^ flags[i]]][(h - duration) * frame + offsets[i]]
            if value != NEG:
                result = max(result, prefix[i] + value)
    return result


def replay(instance, k, cert):
    method = cert.get("method")
    if method not in (METHOD, "sparse-supplier-fair-v1") or type(cert.get("vehicles")) is not int or cert["vehicles"] != k:
        raise ValueError("wrong supplier cover or fleet size")
    model = Model(instance, k)
    covers = model.covers()
    width = cert.get("cell_width", 4)
    if type(width) is not int or not 1 <= width <= 6:
        raise ValueError("unsupported cell width")
    partition = clusters(model, width)
    if cert.get("partition") != partition or type(cert.get("leaves")) is not list or len(cert["leaves"]) != len(covers):
        raise ValueError("incorrect partition or incomplete cover")
    caps = model.cheap_caps()
    endpoint = F()
    cache = {}
    for leaf, groups in zip(cert["leaves"], covers, strict=True):
        if leaf.get("groups") != list(groups) or any(type(x) is not int for x in leaf["groups"]):
            raise ValueError("cover groups differ")
        limit = model.box(groups, caps)
        line = leaf["line"]
        if line is not None:
            alpha = tuple(F(x) for x in line["alpha"])
            prices, claims = line["prices"], line["route_bounds"]
            if (len(alpha) != len(groups) * (len(groups) - 1) // 2 or any(not 0 <= x <= 1 for x in alpha)
                    or len(prices) != model.n or any(type(x) is not int or x < 0 for x in prices)
                    or any(prices[i] for i in range(model.n) if not model.p[i])
                    or len(claims) != len(groups) or any(type(x) is not int or x < 0 for x in claims)):
                raise ValueError("invalid fair split or station prices")
            for group, a, claim in zip(groups, model.supports(groups, alpha), claims, strict=True):
                rewards = model.rewards(a, prices)
                prefix = tuple(-p for p in prices) if method == METHOD else rewards
                key = (group, rewards, prefix)
                if key not in cache:
                    cache[key] = priced(model, group, partition, rewards, prefix)
                if cache[key] != claim:
                    raise ValueError("conditional route price does not replay")
            limit = min(limit, F(sum(claims) + sum(prices), SCALE))
        if limit != F(leaf["upper"]):
            raise ValueError("supplier partition endpoint differs")
        endpoint = max(endpoint, limit)
    if endpoint != F(cert["upper"]):
        raise ValueError("advertised upper differs from maximum across partitions")
    return endpoint


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--vehicles", type=int)
    args = parser.parse_args()
    model = json.loads(args.instance.read_bytes(), parse_float=F)
    cert = json.loads(args.certificate.read_bytes(), parse_float=F)
    k = model["vehicles"] if args.vehicles is None else args.vehicles
    endpoint = replay(model, k, cert)
    print(f"Exact sparse upper {endpoint} ({float(endpoint):.12g}) from {len(cert['leaves'])} leaves")


if __name__ == "__main__":
    main()
