"""Exact independent replay of a fleet optional-node 2-matching LP dual."""
import argparse
from fractions import Fraction
from hashlib import sha256
import json
from math import lcm
from pathlib import Path

METHOD = "fleet-cut-dual-v33"
SCALE = 1_000_000
MAX_TILTS = 16


def rat(value):
    return Fraction(str(value)) if type(value) is float else Fraction(value)


def integral(values):
    den = lcm(*(v.denominator for v in values))
    return [int(v * den) for v in values], den


def distances(cost, through=None):
    """Nonnegative directed shortest paths from node 0."""
    n = len(cost)
    d = list(cost[0])
    d[0] = 0
    unseen = set(range(1,n)) if through is None else set(through)
    while unseen:
        u = min(unseen, key=d.__getitem__)
        unseen.remove(u)
        for v in range(1,n):
            d[v] = min(d[v], d[u] + cost[u][v])
    return d


class Model:
    def __init__(self, raw, k):
        if type(k) is not int or k < 0:
            raise ValueError("invalid fleet size")
        stock, pop, service = [list(map(rat, raw[name])) for name in ("supply", "population", "service")]
        travel = [list(map(rat, row)) for row in raw["travel"]]
        cap, limit = rat(raw["capacity"]), rat(raw["limit"])
        n = len(stock)
        if (not n or any(len(t) != n for t in (pop, service, travel, *travel)) or
                min(cap, limit, *stock, *pop, *service, *map(min, travel)) < 0 or
                stock[0] or pop[0] or any(a and b for a, b in zip(stock, pop))):
            raise ValueError("invalid fleet model")
        canonical = [[str(v) for v in a] for a in (stock, pop, service, *travel)]
        self.identity = sha256(json.dumps([k, str(cap), str(limit), canonical], separators=(",", ":")).encode()).hexdigest()
        self.stock, self.qden = integral([min(s, cap) for s in stock])
        self.pop, self.pden = integral(pop)
        self.ptotal = sum(pop)
        self.qtotal = min(sum(stock), cap * sum(bool(p) for p in pop), Fraction(sum(self.stock), self.qden))
        clock = [limit, *service, *(d for row in travel for d in row)]
        ticks, self.tden = integral(clock)
        self.limit, self.service = ticks[0], ticks[1:1+n]
        self.service[0] = 0
        dist = [ticks[1+n+i*n:1+n+(i+1)*n] for i in range(n)]
        arc = [[dist[i][j] + self.service[j] for j in range(n)] for i in range(n)]
        # Project away permitted zero-action prefixes and suffixes; the first
        # supply and last agency of a productive route are preserved.
        head = distances(arc,(i for i in range(1,n) if not stock[i]))
        tail = distances(list(zip(*arc)),(i for i in range(1,n) if not pop[i]))
        start, end = distances(arc), distances(list(zip(*arc)))
        self.edges, self.cost, self.index, self.oriented = [], [], {}, []
        for i in range(n):
            for j in range(i+1,n):
                if i==0:
                    dirs=[]
                    if stock[j] and head[j]+end[j]<=self.limit:
                        dirs.append((0,j,head[j]-self.service[j]))
                    if pop[j] and start[j]+tail[j]<=self.limit:
                        dirs.append((j,0,tail[j]))
                else:
                    dirs=[(u,v,dist[u][v]) for u,v in ((i,j),(j,i))
                          if start[u]+arc[u][v]+end[v]<=self.limit]
                if dirs:
                    self.index[i,j] = len(self.edges)
                    self.edges.append((i,j))
                    self.cost.append(min(d for u,v,d in dirs))
                    self.oriented.append(dirs)
        self.n, self.k = n, k


def integer(x):
    if type(x) is not int:
        raise ValueError("expected integer")
    return x


def mask_sites(mask, n):
    if type(mask) is not int or mask <= 0 or mask >> n or mask & 1:
        raise ValueError("invalid cut shore")
    return [i for i in range(1,n) if mask >> i & 1]


def cut_row(model, key):
    """Return an upper row (right hand side, node terms, edge terms)."""
    kind, mask, arg = key
    n = model.n
    if kind == "link":
        e, u = integer(mask), integer(arg)
        if not 0 <= e < len(model.edges) or u == 0 or u not in model.edges[e]:
            raise ValueError("bad edge activation")
        return 0, [(u,-1)], [(e,1)]
    sites = mask_sites(mask,n)
    if kind == "connect":
        root = integer(arg)
        if root not in sites:
            raise ValueError("missing cut anchor")
        # Removing the physical depot leaves a union of disjoint simple paths.
        edges = [(e,1) for m,v in enumerate(sites) for u in sites[:m]
                 if (e := model.index.get((u,v))) is not None]
        return 0, [(u,-1) for u in sites if u != root], edges
    if kind == "blossom":
        if type(arg) not in (list,tuple) or any(type(e) is not int for e in arg):
            raise ValueError("bad teeth")
        teeth = set(arg)
        if not len(teeth)%2 or len(teeth)!=len(arg):
            raise ValueError("odd number of distinct teeth required")
        boundary = [e for e,(u,v) in enumerate(model.edges) if (mask>>u ^ mask>>v)&1]
        if not teeth.issubset(boundary):
            raise ValueError("a tooth does not cross its shore")
        return len(teeth)-1, [], [(e,1 if e in teeth else -1) for e in boundary]
    raise ValueError("unknown cut")


def dual_rhs(model, line):
    beta = Fraction(line["beta"])
    degree = line["degree"]
    flow = line["flow"]
    time, fleet = map(integer, (line["time"], line["fleet"]))
    if (beta <= 0 or len(degree)!=model.n or degree[0]!=0 or
            len(flow)!=model.n or flow[0]!=0 or any(type(i) is not int for i in [*degree,*flow]) or min(time, fleet)<0):
        raise ValueError("invalid basic dual")
    # Unit coefficients of all rational data live on one exact integer grid.
    den = lcm(2*SCALE*model.tden, model.qden, beta.denominator * model.pden, SCALE*model.tden)
    one = den//SCALE
    tr = time * (den//(SCALE*model.tden))
    nodes = [s*(den//model.qden)+p*beta.numerator*(den//(model.pden*beta.denominator))
             - tr*st - 2*du*one for s,p,st,du in zip(model.stock,model.pop,model.service,degree)]
    # Each used car consumes at most T. Its projected core contributes
    # exactly two depot incidences and obeys directed flow conservation.
    edges = [(degree[i]+degree[j]-(fleet if i==0 else 0))*one
             +(tr*model.limit//2 if i==0 else 0)
             +max((flow[v]-flow[u])*one-tr*d for u,v,d in dirs)
             for (i,j),dirs in zip(model.edges,model.oriented)]
    rhs = 2*model.k*fleet*one
    for kind, mask, arg, mult in line["cuts"]:
        if integer(mult)<0:
            raise ValueError("negative cut multiplier")
        b,x,y = cut_row(model,(kind,mask,arg))
        coef = mult*one
        rhs += b*coef
        for idx, a in x:
            nodes[idx] -= a*coef
        for idx, a in y:
            edges[idx] -= a*coef
    rhs += sum(max(0,v) for v in nodes[1:])+sum(max(0,v) for v in edges)
    return Fraction(rhs,den)


def upper(model, line, verified_cap=None):
    if model.ptotal == 0 or not model.k:
        return Fraction()
    if line is None:
        return model.qtotal
    b = Fraction(line["beta"])
    if "station" in line:
        if __package__:
            from .routeproof import rewards,replay_cap,parity_data
        else:
            from routeproof import rewards,replay_cap,parity_data
        raw_tilts=line.get("root_tilts")
        tilts=[Fraction(v) for v in raw_tilts] if raw_tilts is not None else [Fraction(1)]
        if not tilts or len(tilts)>MAX_TILTS or any(t<=0 for t in tilts):
            raise ValueError("invalid route slopes")
        bounds=[]
        extra=parity_data(model,line)[2]
        for t in tilts:
            nodes,bonus,total=rewards(model,line,SCALE,cut_row,t)
            if verified_cap is None:
                bounds.append(replay_cap(model,line,nodes,bonus,raw_tilts is not None))
        if verified_cap is None:
            cap=(sum(sorted(map(min,zip(*bounds)),reverse=True)[:model.k])
                 if raw_tilts is not None else bounds[0])
        else:
            cap=verified_cap
        if integer(line["route_bound"])!=cap or cap<0:
            raise ValueError("route and cut support does not replay")
        a=Fraction(total+extra+(cap if raw_tilts is not None else model.k*cap),SCALE)
        if any(t!=1 for t in tilts):
            return min(model.qtotal,a*a/(4*b*model.ptotal))
    else:
        a=dual_rhs(model,line)
    # Optimise (a-bP)P for 0<=P<=Ptotal, plus the unconditional quantity cap.
    p = min(model.ptotal, a/(2*b))
    return min(model.qtotal, (a-b*p)*p/model.ptotal)


def replay(instance,k,cert):
    model = Model(instance,k)
    if (cert.get("method") != METHOD or type(cert.get("vehicles")) is not int or
            cert["vehicles"]!=k or cert.get("instance_sha256")!=model.identity or
            type(cert.get("upper")) is not str):
        raise ValueError("invalid bound identity")
    u = upper(model, cert["support"])
    if u != Fraction(cert["upper"]):
        raise ValueError("dual bound does not match")
    return u


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance",type=Path)
    ap.add_argument("certificate",type=Path)
    ap.add_argument("--vehicles",type=int)
    args = ap.parse_args()
    raw = json.loads(args.instance.read_bytes(),parse_float=Fraction)
    cert = json.loads(args.certificate.read_bytes())
    print(replay(raw,raw["vehicles"] if args.vehicles is None else args.vehicles,cert))
