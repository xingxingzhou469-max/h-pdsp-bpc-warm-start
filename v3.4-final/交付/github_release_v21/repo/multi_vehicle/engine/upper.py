"""Budgeted portfolio of independently replayable upper bounds."""
from fractions import Fraction as F
from time import monotonic_ns

from common import Data, NANO, projection
from pricing_core import Deadline, Pricing, StateCap, envelope, run_upper
from screen import ScreenLimit, screen, threat_width
from supplier_split import build as build_split, cover, without_supplier
from tree import tree_certificate


def _limited(deadline, duration):
    return min(deadline, monotonic_ns() + int(duration))


def _supplier_cover(instance, data, vehicles, stop, emit, log):
    scopes = cover(data, vehicles)
    if scopes is None or monotonic_ns() >= stop:
        return
    low, options, sources = scopes
    models = [dict(instance, vehicles=low),
              *(without_supplier(instance, node, low) for _, node in options)]
    start = monotonic_ns()
    marked = None
    try:
        pricing = Pricing(data, vehicles, "per_vehicle", cycle2=True,
                          markers=tuple(v for _, v in sources), state_cap=460_000)
        marked, _ = pricing.solve(F(1), F(1), (F(0),) * len(data.nodes), F(0),
                                  _limited(stop, (stop - start) * 0.10))
    except (StateCap, Deadline, OverflowError) as error:
        log.append(dict(stage="singleton_population", reason=type(error).__name__))
    proofs = [envelope(Data.from_instance(raw), raw["vehicles"], ()) for raw in models]

    # All scopes start with independently valid boxes. Keep a slice for
    # exact subset and rectangle evaluation after the last pricing scope.
    reserve = max(NANO // 20, (stop - start) // 12)
    for index, raw in enumerate(models):
        left = stop - reserve - monotonic_ns()
        if left <= NANO // 25:
            break
        sub_data = Data.from_instance(raw)
        sub_stop = min(stop - reserve, monotonic_ns() + left // (len(models) - index))

        def observe(cert, _phase, _iteration):
            proofs[index] = cert
            return True

        stages = run_upper(sub_data, raw["vehicles"], "polygon", sub_stop, observe)
        log.extend(dict(item, scope=(options[index - 1][1] if index else "low-active"))
                   for item in stages)
    if monotonic_ns() < stop:
        cert = build_split(data, vehicles, proofs[0], marked, proofs[1:])
        accepted = emit(cert, "active_supplier_cover")
        log.append(dict(stage="active_supplier_cover", leaves=len(cert["leaves"]), accepted=accepted,
                        leaf_bounds=[leaf["upper"] for leaf in cert["leaves"]], score=cert["upper"]))


def _ordered_candidate(data, vehicles, root, deadline, emit, log):
    cap = data.cap
    table = dict(zip(data.nodes, data.caps))
    supports = [s for s in root["supports"] if s["witness"]]
    if not cap or not supports:
        return
    probe = min(supports, key=lambda s: F(envelope(data, vehicles, (s,))["upper"]))
    if sum(table[v] for v in probe["witness"]["route"]) <= cap:
        return
    time_unit, native_load = projection(data, vehicles)
    start = monotonic_ns()
    lines = root["supports"]
    for width in (12, 32):
        if monotonic_ns() >= deadline:
            break
        try:
            pricing = Pricing(data, vehicles, "per_vehicle", cycle2=True, ordered=True,
                              units=(time_unit, max(native_load, cap / width)),
                              state_cap=2_000_000)
            candidate, _ = pricing.solve(F(probe["alpha"]), F(probe["beta"]),
                                         tuple(map(F, probe["visit_prices"])), F(probe["time_price"]),
                                         _limited(deadline, (deadline - start) * 0.55))
        except (StateCap, Deadline, OverflowError) as error:
            log.append(dict(stage="ordered", width=width, reason=type(error).__name__))
            continue
        lines = [*lines, candidate]
        cert = envelope(data, vehicles, lines, "both")
        log.append(dict(stage="ordered", width=width, states=candidate["reached_states"],
                        score=cert["upper"]))
        emit(cert, "ordered_polygon")
        if width == 12 and F(cert["upper"]) >= F(root["upper"]):
            break


def _quick_tree(data, vehicles, root, deadline, emit, log):
    if not root["supports"]:
        return
    probe = min(root["supports"], key=lambda s: F(envelope(data, vehicles, (s,))["upper"]))
    prices = tuple(map(F, probe["visit_prices"]))
    scale = data.q_cap / data.total
    scores = [p + c + scale * pop for p, c, pop in zip(prices, data.caps, data.people)]
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    marks = tuple(sorted(data.nodes[i] for i in ranked[:min(3, len(ranked))]))
    try:
        pricing = Pricing(data, vehicles, "per_vehicle", cycle2=True, markers=marks, state_cap=240_000)
        support, _ = pricing.solve(F(probe["alpha"]), F(probe["beta"]), prices,
                                   F(probe["time_price"]), deadline)
    except (StateCap, Deadline, OverflowError) as error:
        log.append(dict(stage="visit_tree", reason=type(error).__name__))
        return
    if monotonic_ns() < deadline:
        item = tree_certificate(data, vehicles, root, marks, [support])
        if emit(item, "visit_tree"):
            log.append(dict(stage="visit_tree", markers=marks, score=item["upper"]))


def run(instance, vehicles, arm, deadline, emit, fixed_lower=None):
    data = Data.from_instance(instance)
    root = envelope(data, vehicles, ())
    emit(root, "input_box")
    if not vehicles or not data.total or not data.q_cap:
        return []
    if arm not in ("polygon", "joint"):
        raise ValueError(f"unknown upper arm: {arm}")
    best = F(root["upper"])
    log = []
    start = monotonic_ns()
    budget = max(0, deadline - start)
    n = len(data.nodes)
    sources = sum(data.sources)
    short_gate = arm == "joint" and n <= 20
    post_dssr = arm == "joint" and 20 < n <= 40 and vehicles in (2, 3)

    def submit(cert, source):
        nonlocal best
        value = F(cert["upper"])
        if value < best and emit(cert, source):
            best = value
            return True
        return False

    def product_event(cert, phase, iteration):
        nonlocal root
        root = cert
        submit(cert, "polygon")
        return monotonic_ns() < deadline

    def attempt_screen(polygon, threshold, duration, *, stock_grid=False):
        try:
            cert = screen(instance, vehicles, polygon, threshold, _limited(deadline, duration),
                          stock_grid=stock_grid)
        except ScreenLimit as error:
            log.append(dict(stage="route_screen", threshold=str(threshold), stock_grid=stock_grid,
                            reason=str(error)))
            return False
        else:
            submit(cert, "route_screen")
            log.append(dict(stage="route_screen", threshold=str(threshold), stock_grid=stock_grid,
                            score=cert["upper"]))
            return True

    retry_short = False
    if short_gate:
        # Full enumeration/screening has worked on small graphs. One capped
        # attempt leaves the remaining budget available for the polygon.
        retry_short = not attempt_screen(None, fixed_lower or F(0), budget * 0.16)

    polygon_deadline = deadline
    if post_dssr or retry_short:
        polygon_deadline = _limited(deadline, budget * 0.45)

    def after_core(cert):
        if cover(data, vehicles) is not None and sources <= 7:
            _supplier_cover(instance, data, vehicles, _limited(polygon_deadline, budget * 0.47),
                            submit, log)
        elif vehicles >= 3 and 20 < n <= 120:
            # A single short conditional support can pay off on K>=3.
            _quick_tree(data, vehicles, cert, _limited(polygon_deadline, budget * 0.035), submit, log)
        if not post_dssr and 20 < n <= 45 and sources > 7:
            _ordered_candidate(data, vehicles, cert, _limited(polygon_deadline, budget * 0.04),
                               submit, log)

    screened = False
    considered = None

    def stock_screen(threshold, duration, max_width):
        try:
            width = threat_width(data, root, threshold, _limited(deadline, budget * 0.01))
        except ScreenLimit:
            return False
        if width is None or width > max_width or duration <= 0:
            return False
        log.append(dict(stage="route_screen_admitted", width=width, permitted_ns=int(duration)))
        attempt_screen(root, threshold, duration, stock_grid=True)
        return True

    def late_screen(*, exhausted=False):
        nonlocal screened, considered
        now = monotonic_ns()
        if (screened or considered is root or not root["supports"]
                or not exhausted and now - start < budget * 0.74
                or deadline - now < budget * 0.09):
            return
        considered = root
        spare = exhausted and deadline - now >= budget * 0.20
        target_fraction = F(199, 200) if spare else F(997, 1000)
        threshold = min(F(root["upper"]), best) * target_fraction
        duration = min(budget * (0.28 if spare else 0.16), deadline - now - budget * 0.02)
        screened = stock_screen(threshold, duration, 24)

    medium = arm == "joint" and 40 < n <= 60 and vehicles == 2

    search = run_upper(data, vehicles, "polygon", polygon_deadline, product_event,
                       after_core=after_core if arm == "joint" else None,
                       after_slice=late_screen if medium else None)
    if medium:
        late_screen(exhausted=True)
    if (post_dssr or retry_short) and root["supports"]:
        use_stock = root["q_step"] is not None and F(root["q_step"]) == data.q_step
        log.append(dict(stage="post_dssr_screen", upper=root["upper"],
                        remaining_ns=max(0, deadline - monotonic_ns())))
        if retry_short:
            attempt_screen(root, F(fixed_lower or 0), deadline - monotonic_ns(),
                           stock_grid=use_stock)
        else:
            target = best * F(49, 50)
            floor = F(fixed_lower or 0) * F(21, 20)
            if target <= floor < best:
                target = floor
            for proportion in (F(1), F(1, 2), F(1, 4)):
                remaining = deadline - monotonic_ns()
                if remaining < NANO * 2 // 5:
                    break
                step = proportion * (best - target)
                if proportion == F(1, 4):
                    # A broad screen can hit its enumeration quota even with
                    # one quantity row. Use a finer last step near the bound.
                    step = min(step, best / 800)
                threshold = best - step
                if threshold >= best:
                    break
                duration = remaining if proportion == F(1, 4) else remaining * 0.6
                attempt_screen(root, threshold, duration, stock_grid=use_stock)
    elif arm == "joint" and 20 < n <= 40 and vehicles >= 4 and best == data.q_cap and root["supports"]:
        stock_screen(best * F(99, 100), min(budget * 0.20, deadline - monotonic_ns()), 4)
    return [*search, *log]
