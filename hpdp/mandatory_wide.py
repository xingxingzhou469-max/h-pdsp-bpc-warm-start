"""Original-arc proof search for exact objective numerators exceeding signed int64."""

import ctypes
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import sys
from time import monotonic

from .allocation import allocate
from .check import check


Int = ctypes.c_int
Ptr = ctypes.POINTER(Int)
UInt64 = ctypes.c_uint64
Long = ctypes.c_int64
LongPtr = ctypes.POINTER(Long)
class Word(ctypes.Structure):
    _fields_ = (("low",UInt64),("high",Long))


WordPtr = ctypes.POINTER(Word)
ExactWitness = ctypes.CFUNCTYPE(None,Int,Ptr,WordPtr)
ExactCheckpoint = ctypes.CFUNCTYPE(None,WordPtr,UInt64,UInt64,Ptr)
MAX_PRODUCT = (1 << 123)-1


def encode(value):
    if not -2<=value<=MAX_PRODUCT:
        raise OverflowError("Exact objective numerator exceeds native range")
    return Word(value & ((1 << 64)-1),value >> 64)


def decode(word):
    return (word.high << 64)+word.low

INT64_MAX = (1 << 63)-1
MAX_CELLS = 25_000_000


@lru_cache(maxsize=4)
def library(path=None):
    suffix = "dylib" if sys.platform == "darwin" else "so"
    binary = Path(path) if path is not None else Path(__file__).with_name(f"libmandatory_wide.{suffix}")
    native = ctypes.CDLL(str(binary.resolve()))
    types = {
        "create": (ctypes.c_void_p, [Int, Int, Int, Int, Int, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr]),
        "free": (None, [ctypes.c_void_p]),
        "query": (Int, [ctypes.c_void_p, Int, Int, Int, Int, ctypes.c_double, Ptr]),
        "trace": (Int, [ctypes.c_void_p, Int, Int, Int, Int, Ptr, Int]),
        "state_count": (UInt64, [ctypes.c_void_p]),
        "plane_count": (UInt64, [ctypes.c_void_p]),
        "exact_wide_v2": (Int, [ctypes.c_void_p, LongPtr, LongPtr, Ptr, Int, Int, Int, Long,
                             Long, Long, WordPtr, WordPtr, Int, ctypes.c_double, ExactWitness,
                             ExactCheckpoint, WordPtr, ctypes.POINTER(UInt64), ctypes.POINTER(UInt64)]),
    }
    for label, (returns, args) in types.items():
        method = getattr(native, f"hpdp_mandatory_{label}")
        method.restype, method.argtypes = returns, args
    return native


def _array(values):
    return (Int * len(values))(*values)


@dataclass(frozen=True, slots=True)
class Root:
    supplier_mask: int
    first: int
    left: int
    flow: int


class MandatoryGraph:
    def __init__(self, r):
        self.r = r
        self.n = len(r.travel)
        self.k = len(r.p)
        self.donor_bits = [0]*self.n
        for bit, site in enumerate(r.p):
            self.donor_bits[site] = 1 << bit
        self.times = _array([min(r.limit+1, travel + r.service[v])
                             for row in r.travel for v, travel in enumerate(row)])
        self.finish = _array([min(r.limit+1, r.travel[v][0]) if r.pop[v] else r.limit+1
                              for v in range(self.n)])
        self.zeros = tuple(v for v in r.d if any(self.times[u*self.n+v] == 0
                                                 for u in r.nodes if u != v))
        self.roots = []
        for supply in range(1, 1 << self.k):
            group = [v for v in r.p if self.donor_bits[v] & supply]
            flow = min(sum(r.supply[v] for v in group), len(r.d)*r.capacity)
            for v in group:
                left = r.limit-r.travel[0][v]-r.service[v]
                if left >= 0:
                    self.roots.append(Root(supply, v, left, flow))
        # Keep the largest possible reduced sum safely within signed int32.
        # Rounding weights up is part of the relaxation, not input rounding.
        max_weight = ((1 << 31)-2)//(r.limit+self.k+9)
        self.population_unit = max(1, (max(r.pop)+max_weight-1)//max_weight)
        self.weights = [(pop+self.population_unit-1)//self.population_unit for pop in r.pop]

    def fits(self, track_count):
        return (track_count <= 8 and self.n*(self.r.limit+1)*(1 << (self.k+track_count)) <= MAX_CELLS)

    def check_track(self, track):
        if len(track) != len(set(track)) or not set(track) <= set(self.r.d):
            raise ValueError("Explicitly tracked stops must be distinct agencies")
        if not set(self.zeros) <= set(track):
            raise ValueError("Zero-time agency transitions must clear an explicit bit")
        if not self.fits(len(track)):
            raise ValueError("Mandatory table exceeds its configured memory")

    @staticmethod
    def supported(r):
        return 1 <= len(r.p) <= 5 and 2 <= len(r.nodes) < 120 and 0 <= r.limit <= 4096


class MandatoryTable:
    def __init__(self, graph, prices, tracked, no_return, lib_path=None):
        graph.check_track(tracked)
        if (len(prices) != graph.n
                or any(price < 0 or price > weight for price, weight in zip(prices, graph.weights))):
            raise ValueError("Prices must be nonnegative and at most the corresponding weights")
        explicit = [0]*graph.n
        for j, site in enumerate(tracked):
            explicit[site] = 1 << j
            if prices[site]:
                raise ValueError("Explicitly tracked stations must have zero price")
        self.graph = graph
        self.tracked = tuple(tracked)
        self.price_sum = sum(prices)
        self.explicit = tuple(explicit)
        self.tracked_full = (1 << len(tracked))-1
        self.lib = library(lib_path)
        status = Int()
        self.handle = self.lib.hpdp_mandatory_create(
            graph.n, graph.r.limit, graph.k, len(tracked), int(no_return), graph.times,
            _array(graph.donor_bits), _array(explicit), graph.finish,
            _array([weight-price for weight, price in zip(graph.weights, prices)]), ctypes.byref(status))
        if not self.handle or status.value != 1:
            raise RuntimeError(f"Mandatory table initialization failed ({status.value})")

    def close(self):
        if self.handle:
            self.lib.hpdp_mandatory_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()

    @property
    def size(self):
        return (self.lib.hpdp_mandatory_state_count(self.handle),
                self.lib.hpdp_mandatory_plane_count(self.handle))

    def query(self, needed, available, left, last, previous, deadline):
        output = Int()
        status = self.lib.hpdp_mandatory_query(
            self.handle, needed|(available << self.graph.k), left, last, previous,
            max(0.0, deadline-monotonic()), ctypes.byref(output))
        if status == 0:
            return False, None
        if status < 0:
            raise RuntimeError(f"Mandatory table query failed ({status})")
        return True, None if output.value < 0 else output.value

    def root(self, root, deadline):
        needed = root.supplier_mask ^ self.graph.donor_bits[root.first]
        complete, value = self.query(needed, self.tracked_full, root.left, root.first, 0, deadline)
        if not complete or value is None:
            return complete, None
        return True, value + self.price_sum

    def route(self, root):
        needed = root.supplier_mask ^ self.graph.donor_bits[root.first]
        size = root.left+self.graph.k+len(self.tracked)+1
        buffer = (Int*size)()
        steps = self.lib.hpdp_mandatory_trace(
            self.handle, needed|(self.tracked_full << self.graph.k), root.left, root.first, 0,
            buffer, size)
        if steps < 0:
            raise ArithmeticError("Completed mandatory query did not reconstruct")
        return (root.first, *buffer[:steps])


def solve_bound(r, deadline, seed, *, emit=None, tracked=4, no_return=True,
                lib_path=None, pricing_rounds=(9, 4), _analysis=None):
    """Reuse completed context bounds and discard work on interrupted roots."""
    lb = None if seed is None else check(
        r.instance, seed["route"], seed["amounts"], tol=0)["objective"]
    upper = Fraction(r.prior,r.units)
    metrics = dict(phase="mandatory_population_bound", tables=0, roots_completed=0,
                   roots_count=0, states=0, planes=0, tracked=0, timed_out=False,
                   forbid_immediate_return=no_return, first_table_ub_exact=None,
                   population_unit=None, support="supported")
    started = monotonic()

    def row():
        if lb is not None and (upper is None or lb > upper):
            raise ArithmeticError("Mandatory population bound contradicts the checked incumbent")
        return dict(
            status="infeasible" if upper is None else "optimal" if lb == upper else "timeout",
            lb=lb, ub=upper, lb_exact=None if lb is None else str(lb),
            ub_exact=None if upper is None else str(upper),
            certificate="rational_mandatory_population",
            route=None if seed is None else seed["route"],
            amounts=None if seed is None else seed["amounts"],
            metrics={**metrics, "relaxation_seconds": monotonic()-started})

    if lb == upper or monotonic() >= deadline:
        return row()
    if not MandatoryGraph.supported(r):
        metrics["support"] = "node_count_supplier_count_or_exact_horizon"
        return row()
    graph = MandatoryGraph(r)
    metrics.update(roots_count=len(graph.roots), population_unit=graph.population_unit,
                   mandatory_zero_tracks=len(graph.zeros))
    if not graph.fits(len(graph.zeros)):
        metrics["support"] = "zero_time_agency_tracking_or_memory"
        return row()
    bounds = [root.flow*r.people for root in graph.roots]
    if _analysis is not None:
        _analysis.update(graph=graph,bounds=bounds,configuration=None)
    best_table = None
    if not bounds:
        upper = None
        return row()

    order = sorted(r.d, key=lambda v:(-graph.weights[v], v))
    pivot = graph.weights[order[len(order)//4]]
    prices = [max(0, w-pivot) for w in graph.weights]
    first_prices = prices.copy()
    repeated = [0]*graph.n
    seen = set()
    for stage, repeats in enumerate(pricing_rounds):
        track = list(graph.zeros)
        if stage:
            ranked = sorted((v for v in r.d if v not in graph.zeros),
                            key=lambda v:(-repeated[v], -graph.weights[v],v))
            for v in ranked:
                if len(track) >= tracked or not graph.fits(len(track)+1):
                    break
                track.append(v)
        for v in track:
            prices[v] = 0
        for it in range(repeats):
            if monotonic() >= deadline:
                metrics["timed_out"] = True
                return row()
            configuration = (tuple(track), tuple(prices))
            if configuration in seen:
                break
            seen.add(configuration)
            with MandatoryTable(graph,prices,track,no_return,lib_path) as table:
                metrics["tables"] += 1
                metrics["tracked"] = max(metrics["tracked"],len(track))
                interrupted = False
                active = []
                order = sorted(range(len(bounds)),
                               key=lambda i:(bounds[i] is None,-(bounds[i] or 0),i))
                for i in order:
                    if bounds[i] is None:
                        continue
                    root = graph.roots[i]
                    finished, population = table.root(root,deadline)
                    if not finished:
                        interrupted = True
                        break
                    metrics["roots_completed"] += 1
                    if population is None:
                        bounds[i] = None
                    else:
                        cap = min(r.people,population*graph.population_unit)
                        bounds[i] = min(bounds[i],cap*root.flow)
                        active.append((bounds[i], i))
                states, planes = table.size
                metrics["states"] += states
                metrics["planes"] += planes
                metrics["timed_out"] = interrupted
                remaining = [value for value in bounds if value is not None]
                upper = None if not remaining else min(upper,Fraction(max(remaining),r.units))
                if _analysis is not None and not interrupted and upper is not None:
                    rank=(upper,-len(track))
                    if best_table is None or rank<best_table:
                        best_table=rank
                        _analysis["configuration"]=configuration
                if metrics["tables"] == 1 and not interrupted:
                    metrics["first_table_ub_exact"] = None if upper is None else str(upper)
                if emit and monotonic() < deadline:
                    emit(row())
                if interrupted or upper is None or lb == upper:
                    return row()
                if not active:
                    return row()
                _, choice = max(active)
                walk = table.route(graph.roots[choice])
            counts = [0]*graph.n
            for v in walk:
                counts[v] += 1
            step = max(1,pivot//(2+it//2))
            for v in r.d:
                repeated[v] += max(0,counts[v]-1)
                prices[v] = 0 if v in track else max(
                    0,min(graph.weights[v],prices[v]+step*(counts[v]-1)))
        if not stage:
            prices = [(a+b)//2 for a,b in zip(first_prices,prices)]
    return row()


def supported(r):
    """Report whether the wider objective interface is required and safe."""
    stock=sum(r.supply)
    product=r.people*stock
    return (MandatoryGraph.supported(r) and len(r.d)<=63
            and max(stock,r.people,r.capacity)<=INT64_MAX and INT64_MAX<product<=MAX_PRODUCT)


def solve_exact(r, deadline, seed, *, emit=None, tracked=4, equity=True,
                lib_path=None, pricing_seconds=3.0):
    """Use wider thresholds only when the signed int64 objective range is exceeded."""
    stock=sum(r.supply)
    if r.people*stock<=INT64_MAX:
        from .mandatory_walks import solve_exact as solve_standard
        return solve_standard(r,deadline,seed,emit=emit,tracked=tracked,equity=equity,
                              pricing_seconds=pricing_seconds)
    started=monotonic()
    proof={}
    priced_end=min(deadline,started+min(pricing_seconds,0.22*max(0,deadline-started)))
    result=solve_bound(r,priced_end,seed,emit=emit,tracked=tracked,lib_path=lib_path,_analysis=proof)
    if result["status"]!="timeout" or monotonic()>=deadline or not proof.get("configuration"):
        return result
    if not supported(r):
        return {**result,"metrics":{**result["metrics"],"exact_support":"population_supply_scaling_or_agency_count"}}

    graph=proof["graph"]
    bounds=proof["bounds"]
    selected,prices=proof["configuration"]
    # The stored context bounds remain valid when the elementary search uses
    # a different suffix oracle. Track zero-time agencies first for termination.
    if len(r.p) >= 4 and stock > r.capacity and len(selected) < 3 and graph.fits(3):
        ordered = sorted(r.d, key=lambda v: (-r.pop[v], v))
        selected = tuple(dict.fromkeys((*selected, *ordered)))[:3]
        prices = tuple(0 if v in selected else price for v, price in enumerate(prices))
    lb=None if result["lb_exact"] is None else Fraction(result["lb_exact"])
    witness=seed
    tested=expanded=closed=0
    metrics=dict(result["metrics"],phase="mandatory_elementary_search",
                 equity_chain=equity,elementary_tracked=list(selected),search_contexts=0,search_expanded=0,search_allocations=0,
                 native_objective_bits=(r.people*stock).bit_length())
    native_pop=(Long*graph.n)(*r.pop)
    native_supply=(Long*graph.n)(*r.supply)
    native_prices=_array(prices)
    error=[]

    def floor():
        return -1 if lb is None else (lb*r.units)//1

    def row():
        unresolved=[cap for cap in bounds if cap is not None]
        ub=Fraction(max(unresolved),r.units) if unresolved else lb
        if lb is not None and ub is not None:
            ub=max(lb,ub)
        return dict(status="infeasible" if ub is None else "optimal" if ub==lb else "timeout",
                    certificate="rational_mandatory_elementary",
                    lb=lb,ub=ub,lb_exact=None if lb is None else str(lb),
                    ub_exact=None if ub is None else str(ub),
                    route=None if witness is None else witness["route"],
                    amounts=None if witness is None else witness["amounts"],metrics=metrics.copy())

    def publish():
        if emit and monotonic()<deadline:
            emit(row())

    def set_counts(nodes,leaves):
        metrics.update(search_contexts=closed,search_expanded=expanded+nodes,
                       search_allocations=tested+leaves)

    @ExactWitness
    def improvement(length, nodes, answer):
        nonlocal lb,witness
        try:
            if monotonic()>=deadline:
                return
            route=tuple(nodes[:length])
            score,amounts=allocate(r.instance,route)
            if lb is None or score>lb:
                if check(r.instance,route,amounts,tol=0)["objective"]!=score:
                    raise ArithmeticError("Mandatory elementary allocation failed route check")
                if monotonic()>=deadline:
                    return
                lb=score
                witness=dict(route=route,amounts=amounts)
                metrics["improvements"]=metrics.get("improvements",0)+1
                publish()
            if monotonic()<deadline:
                answer[0]=encode(floor())
        except BaseException as exc:
            error.append(exc)

    @ExactCheckpoint
    def progress(answer,nodes,leaves,proceed):
        try:
            if monotonic()>=deadline:
                return
            cap=decode(answer[0])
            bounds[i]=None if cap<0 else min(bounds[i],cap)
            set_counts(nodes,leaves)
            publish()
            if monotonic()<deadline:
                proceed[0]=1
        except BaseException as exc:
            error.append(exc)

    with MandatoryTable(graph,prices,selected,True,lib_path) as table:
        order=sorted((i for i,x in enumerate(bounds) if x is not None),key=lambda i:(-bounds[i],i))
        for i in order:
            if monotonic()>=deadline or bounds[i]<=floor():
                break
            root=graph.roots[i]
            complete,population=table.root(root,deadline)
            if not complete:
                break
            if population is None:
                bounds[i]=None
                closed+=1
                set_counts(0,0)
                publish()
                continue
            cap=min(bounds[i],root.flow*min(r.people,population*graph.population_unit))
            bounds[i]=cap
            if cap<=floor():
                continue
            mask=(root.supplier_mask^graph.donor_bits[root.first])|(table.tracked_full<<graph.k)
            upper=Word()
            native_expanded=UInt64()
            native_evaluated=UInt64()
            incumbent=encode(floor())
            active=encode(cap)
            status=table.lib.hpdp_mandatory_exact_wide_v2(
                table.handle,native_pop,native_supply,native_prices,mask,root.first,root.left,
                r.capacity,root.flow,graph.population_unit,ctypes.byref(incumbent),ctypes.byref(active),int(equity),
                max(0,deadline-monotonic()),improvement,progress,ctypes.byref(upper),
                ctypes.byref(native_expanded),ctypes.byref(native_evaluated))
            if error:
                raise error[0]
            if status<0:
                raise RuntimeError(f"Mandatory elementary search failed ({status})")
            computed=decode(upper)
            bounds[i]=None if computed<0 else min(bounds[i],computed)
            expanded+=native_expanded.value
            tested+=native_evaluated.value
            closed+=status==1
            set_counts(0,0)
            publish()
            if status==0:
                break
    return row()
