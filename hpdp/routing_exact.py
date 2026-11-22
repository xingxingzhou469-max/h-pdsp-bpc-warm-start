"""Frozen integer routing rows and exact Lagrangian box replay."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Row:
    terms: tuple
    rhs: int
    float_scale: int = 1


@dataclass
class Routing:
    nodes: int
    arcs: dict
    visits: dict
    time: Row
    degrees: tuple
    flow_ceiling: int | None = None
    flow_rows: tuple = ()

    @property
    def flow_column(self):
        return len(self.arcs)+len(self.visits) if self.flow_ceiling else None

    @property
    def columns(self):
        return len(self.arcs)+len(self.visits)+int(self.flow_column is not None)

    def inequalities(self, cuts):
        rows = [self.time, *self.flow_rows]
        for nodes, target in cuts:
            side = frozenset(nodes)
            if target not in side or not side <= self.visits.keys():
                raise ValueError("A connectivity cut must keep its target in a depot-free set")
            crossing = [(column, -1) for (i, j), column in self.arcs.items()
                        if i in side and j not in side]
            rows.append(Row(((self.visits[target], 1), *crossing), 0))
        return rows


def routing(r, *, pickup_blocks=False):
    n = len(r.travel)
    arcs = {}
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    travel_terms = []
    for i in range(n):
        for j in range(n):
            if (i == j or (i == 0 and not r.supply[j]) or (j == 0 and not r.pop[i])
                    or r.travel[i][j]+r.service[j] > r.limit):
                continue
            column = len(arcs)
            arcs[i,j] = column
            outgoing[i].append((column,1))
            incoming[j].append((column,1))
            if r.travel[i][j]:
                travel_terms.append((column,r.travel[i][j]))
    visits = {v:column for column,v in enumerate(r.nodes,len(arcs))}
    degrees = [Row(tuple(outgoing[0]),1),Row(tuple(incoming[0]),1)]
    for v in r.nodes:
        y = (visits[v],-1)
        degrees.extend((Row((*outgoing[v],y),0),Row((*incoming[v],y),0)))
    service = [(visits[v],r.service[v]) for v in r.nodes if r.service[v]]
    time = Row(tuple(travel_terms+service),r.limit,max(1,r.limit,*r.service))
    ceiling = flow_cap(r) if pickup_blocks else None
    flow_rows = ()
    if ceiling:
        f = (len(arcs)+len(visits), ceiling)
        supplies = tuple((visits[i], -r.supply[i]) for i in r.p)
        transitions = tuple((column, -r.capacity) for (i, j), column in arcs.items()
                            if r.supply[i] and r.pop[j])
        flow_rows = (Row((f, *supplies), 0, ceiling),
                     Row((f, *transitions), 0, max(ceiling, r.capacity)))
    return Routing(n,arcs,visits,time,tuple(degrees),ceiling,flow_rows)


def rewards(r, model, a, b):
    if not isinstance(a,int) or not isinstance(b,int) or min(a,b) <= 0:
        raise ValueError("Both slope coefficients must be positive integers")
    if model.flow_ceiling is None:
        return [0]*len(model.arcs)+[a*r.supply[v]+b*r.pop[v] for v in model.visits]
    flow = [] if model.flow_column is None else [a*model.flow_ceiling]
    return [0]*len(model.arcs)+[b*r.pop[v] for v in model.visits]+flow


def _route_reward(r, model, a, b, route):
    """Encode a strictly checked route in the baseline or pickup-block objective."""
    if model.flow_ceiling is None:
        return sum(a*r.supply[v]+b*r.pop[v] for v in route)
    delivered = onboard = 0
    for v in route:
        if r.supply[v]:
            onboard = min(r.capacity, onboard+r.supply[v])
        else:
            delivered += onboard
            onboard = 0
    return a*delivered+b*sum(r.pop[v] for v in route)


def flow_cap(r):
    return min(sum(r.supply),len(r.d)*r.capacity)


def exact_upper(objective, inequalities, equalities, denominator, nonnegative, free):
    """Lagrangian box bound using only the original integer model and integer multipliers."""
    if (type(denominator) is not int or denominator <= 0
            or len(inequalities) != len(nonnegative) or len(equalities) != len(free)
            or any(type(x) is not int or x<0 for x in nonnegative)
            or any(type(x) is not int for x in free)):
        raise ValueError("Invalid exact multiplier certificate")
    reduced = [denominator*c for c in objective]
    rhs = 0
    for rows, multipliers in ((inequalities,nonnegative),(equalities,free)):
        for row, multiplier in zip(rows,multipliers,strict=True):
            if multiplier:
                rhs += row.rhs*multiplier
                for col, coefficient in row.terms:
                    reduced[col] -= coefficient*multiplier
    residual = sum(x for x in reduced if x>0)
    return dict(numerator=rhs+residual, denominator=denominator,
                rhs_numerator=rhs, positive_residual_numerator=residual)
