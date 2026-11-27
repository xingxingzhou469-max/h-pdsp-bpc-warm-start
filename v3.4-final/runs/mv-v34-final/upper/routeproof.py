"""Independent route re-pricing for an undirected fleet-cut dual."""
from array import array
from fractions import Fraction
from functools import reduce
from math import gcd

NEG = -(1 << 61)
MAX_RANK = 5
MAX_CELLS = 18_000_000
MAX_REGION = 6
MAX_HORIZON = 400
MAX_PRICE = (1 << 58) // (MAX_HORIZON + 5)


def clock(m):
    n=m.n
    tick=reduce(gcd,(v for group in m.oriented for _,j,c in group for v in (c+m.service[j],)),m.limit) or 1
    if m.limit//tick>MAX_HORIZON:
        unit=Fraction(m.limit,MAX_HORIZON)
        project=lambda t: int(t//unit)
    else:
        project=lambda t:t//tick
    h=project(m.limit)
    arcs=[[h+1]*n for _ in range(n)]
    head=[h+1]*n
    end=[h+1]*n
    for group in m.oriented:
        for i,j,travel in group:
            t=project(travel+m.service[j])
            if i==0:
                head[j]=t
            elif j==0:
                end[i]=t
            else:
                arcs[i][j]=t
    return h,arcs,head,end


def rewards(m,line,scale,cut_row,tilt=Fraction(1)):
    n=m.n
    pi=line["station"]
    if len(pi)!=n or pi[0]!=0 or any(type(v) is not int or v<0 for v in pi):
        raise ValueError("invalid fleet station prices")
    beta=Fraction(line["beta"])
    if beta<=0 or tilt<=0:
        raise ValueError("invalid route slope")
    nodes=[-(-(tilt*Fraction(s,m.qden)+beta*Fraction(p,m.pden)/tilt)*scale//1)-v for s,p,v in zip(m.stock,m.pop,pi)]
    edges=[0]*len(m.edges)
    constant=sum(pi)
    for e,v in line["edge"]:
        if type(e) is not int or not 0<=e<len(edges) or type(v) is not int or v<0:
            raise ValueError("invalid fleet edge price")
        edges[e]-=v
        constant+=v
    for kind,mask,arg,mu in line["cuts"]:
        if type(mu) is not int or mu<0:
            raise ValueError("invalid route cut price")
        b,x,y=cut_row(m,(kind,mask,arg))
        constant+=b*mu
        for j,v in x:
            nodes[j]-=mu*v
        for e,v in y:
            edges[e]-=mu*v
    bonus=[[0]*n for _ in range(n)]
    for e,(i,j) in enumerate(m.edges):
        bonus[i][j]=bonus[j][i]=edges[e]
    if max(map(abs,nodes),default=0)+max(map(abs,edges),default=0)>MAX_PRICE:
        raise ValueError("bounded integer pricing range exceeded")
    return nodes,bonus,constant


def parity_data(m,line):
    cuts=line.get("parity",[])
    if not isinstance(cuts,list) or len(cuts)>MAX_RANK:
        raise ValueError("unsupported parity count")
    tags=[0]*m.n
    mus=[]
    constant=0
    for b,(mask,mu) in enumerate(cuts):
        if (type(mask) is not int or mask<=0 or mask>>m.n or mask&1 or
                mask.bit_count()<3 or not mask.bit_count()%2 or type(mu) is not int or mu<0):
            raise ValueError("invalid odd station rank cut")
        for i in range(1,m.n):
            if mask>>i&1:tags[i]|=1<<b
        mus.append(mu)
        constant+=(mask.bit_count()//2)*mu
    if sum(mus)>MAX_PRICE:
        raise ValueError("parity price exceeded integer pricing range")
    patterns=1<<len(mus)
    discount=[[sum(v for b,v in enumerate(mus) if (tags[i]&p)>>b&1)
               for i in range(m.n)] for p in range(patterns)]
    return tags,discount,constant


def pricing_groups(n,regions):
    if len(regions)!=n or any(type(v) is not int or not 0<=v<n for v in regions):
        raise ValueError("invalid pricing partition")
    groups={}
    for i,r in enumerate(regions):
        groups.setdefault(r,[]).append(i)
    if max(map(len,groups.values()))>MAX_REGION:
        raise ValueError("unsupported region size")
    return groups


def rank_limit(n,h,groups):
    frame=sum(len(g)*(1<<(len(g)-1)) for g in groups.values())
    fitting=MAX_CELLS//((h+1)*(frame+2*n))
    return min(MAX_RANK,fitting.bit_length()-1)


def replay_cap(m,line,nodes,bonus,per_root=False):
    h,arcs,head,end=clock(m)
    n=m.n
    if n>350 or any(0 in row for row in arcs):
        raise ValueError("route DP requires positive projected internal arcs and <=350 sites")
    regions=line["partition"]
    groups=pricing_groups(n,regions)
    tags,discount,_=parity_data(m,line)
    patterns=len(discount)
    if patterns.bit_length()-1>rank_limit(n,h,groups):
        raise ValueError("too many replay states")
    own={i:1<<k for g in groups.values() for k,i in enumerate(g)}
    offsets=[]
    frame=0
    inner=[]
    outer=[]
    for i,r in enumerate(regions):
        offsets.append(frame)
        frame+=1<<(len(groups[r])-1)
    def compact(mask,bit):
        return (mask&(bit-1))|((mask>>1)&~(bit-1))
    for i,r in enumerate(regions):
        moves=[]
        for local in range(1<<(len(groups[r])-1)):
            mask=(local&(own[i]-1))|((local&~(own[i]-1))<<1)|own[i]
            moves.append(sorted((arcs[i][j], j, offsets[j]+compact(mask|own[j],own[j]),nodes[j]+bonus[i][j])
                                for j in groups[r] if not mask&own[j] and arcs[i][j]<=h))
        inner.append(moves)
        outer.append(sorted((arcs[i][j],j,offsets[j],nodes[j]+bonus[i][j])
                            for j in range(n) if regions[j]!=r and arcs[i][j]<=h))
    table=array("q",[NEG])*((h+1)*patterns*frame)
    other=array("q",[NEG])*((h+1)*patterns*n)
    denied=array("i",[-1])*((h+1)*patterns*n)
    for t in range(h+1):
        for p in range(patterns):
            for i in range(n):
                best=bonus[i][0] if end[i]<=t else NEG
                second,block=NEG,-1
                for d,j,off,w in outer[i]:
                    if d>t: break
                    tm=(t-d)*patterns+(p^tags[j])
                    v=other[tm*n+j] if denied[tm*n+j]==i else table[tm*frame+off]
                    if v==NEG: continue
                    v+=w-discount[p][j]
                    if v>best:
                        second,best,block=best,v,j
                    elif v>second:
                        second=v
                for loc,moves in enumerate(inner[i]):
                    inside=NEG
                    for d,j,off,w in moves:
                        if d>t: break
                        v=table[((t-d)*patterns+(p^tags[j]))*frame+off]
                        if v!=NEG:
                            inside=max(inside,v+w-discount[p][j])
                    table[(t*patterns+p)*frame+offsets[i]+loc]=max(inside,best)
                    if loc==0:
                        other[(t*patterns+p)*n+i]=max(inside,second)
                        denied[(t*patterns+p)*n+i]=block
    by_root=[0]*n
    for i in range(1,n):
        if head[i]<=h:
            suffix=table[((h-head[i])*patterns+tags[i])*frame+offsets[i]]
            if suffix!=NEG:
                by_root[i]=max(0,nodes[i]+bonus[0][i]+suffix)
    return by_root if per_root else max(by_root)
