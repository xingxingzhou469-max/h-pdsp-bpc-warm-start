"""Restricted packing dual with both route prices and fleet cut/edge prices."""
import ctypes as ct
from collections import Counter
from fractions import Fraction
import json
from math import ceil
from pathlib import Path
from time import monotonic
from check import MAX_TILTS, SCALE, cut_row
from routeproof import MAX_HORIZON, MAX_REGION, clock, parity_data, pricing_groups, rank_limit, rewards

WARM_BYTES=1_048_576
WARM_ROUTES=512
WARM_VISITS=32_768


def read_warm(m,folder):
    """Load search hints from the same instance. Claimed bounds are never reused."""
    try:
        with (folder/"warm.json").open("rb") as source:
            payload=source.read(WARM_BYTES+1)
        if len(payload)>WARM_BYTES:
            return None
        data=json.loads(payload)
        if data["schema"]!=1 or data["instance_sha256"]!=m.identity:
            return None
        line,paths,elapsed=data["prices"],data["columns"],data["price_seconds"]
        pricing_groups(m.n,line["partition"])
        rewards(m,line,SCALE,cut_row)
        parity_data(m,line)
        if type(elapsed) not in (int,float) or not 0<elapsed<=3600:
            return None
        if (not isinstance(paths,list) or len(paths)>WARM_ROUTES or
                any(not isinstance(r,list) or not r or len(r)>MAX_HORIZON+1 or
                    any(type(v) is not int or not 1<=v<m.n for v in r) for r in paths) or
                sum(map(len,paths))>WARM_VISITS):
            return None
        return line,paths,elapsed
    except (OSError,ValueError,TypeError,KeyError,ZeroDivisionError):
        return None


class Pricing:
    def __init__(self,m):
        h,arcs,head,end=clock(m)
        if any(0 in row for row in arcs) or m.n>350:
            raise ValueError("resource model outside native price scope")
        i,v,ptr=ct.c_int,ct.c_int64,ct.POINTER
        self.fn=ct.CDLL(str(Path(__file__).with_name("kernel.so"))).link_price
        self.fn.argtypes=[i,i,i,v,*([ptr(i)]*4),ptr(v),ptr(v),i,ptr(i),ptr(v),ptr(v),ptr(v),ptr(i),ptr(i)]
        self.fn.restype=i
        self.n,self.h=m.n,h
        self.push=(i*(m.n*m.n))(*(d for row in arcs for d in row)),(i*m.n)(*head),(i*m.n)(*end)
        self.root_bound=(v*m.n)()
        self.lengths=(i*m.n)()
        self.paths=(i*(m.n*(h+1)))()

    def price(self,nodes,bonus,part,end,tags,discount):
        n=self.n
        remaining=int((end-monotonic())*1e9)
        if remaining<=0:
            return None
        value=ct.c_int64()
        status=self.fn(n,self.h,0,remaining,*self.push,(ct.c_int*n)(*part),
                       (ct.c_int64*n)(*nodes),(ct.c_int64*(n*n))(*(c for row in bonus for c in row)),
                       len(discount),(ct.c_int*n)(*tags),
                       (ct.c_int64*(n*len(discount)))(*(u for row in discount for u in row)),
                       ct.byref(value),self.root_bound,self.lengths,self.paths)
        if status in (2,3):
            return None
        if status:
            raise RuntimeError("native route-cut kernel reported "+str(status))
        return list(self.root_bound),[self.paths[i*(self.h+1):i*(self.h+1)+z] for i,z in enumerate(self.lengths) if z]


def partition(m,width):
    n=m.n
    group=list(range(n))
    nodes={i:[i] for i in range(n)}
    near=sorted((d,u,v) for (u,v),d in zip(m.edges,m.cost) if u)
    for _,u,v in near:
        a,b=group[u],group[v]
        if a!=b and len(nodes[a])+len(nodes[b])<=width:
            for j in nodes[b]:
                group[j]=a
            nodes[a].extend(nodes.pop(b))
    return group


def compact_seed(m,line,beta):
    q=[s/m.qden+float(beta)*p/m.pden for s,p in zip(m.stock,m.pop)]
    time=line["time"]/SCALE
    st=[v-time*d/m.tden-2*du/SCALE for v,d,du in zip(q,m.service,line["degree"])]
    ed=[(line["degree"][i]+line["degree"][j]-(line["fleet"] if i==0 else 0))/SCALE+
        (time*m.limit/(2*m.tden) if i==0 else 0)+
        max((line["flow"][v]-line["flow"][u])/SCALE-time*d/m.tden for u,v,d in dirs)
        for (i,j),dirs in zip(m.edges,m.oriented)]
    for a,b,c,num in line["cuts"]:
        _,xs,ys=cut_row(m,(a,b,c))
        for i,v in xs: st[i]-=v*num/SCALE
        for e,v in ys: ed[e]-=v*num/SCALE
    return [max(0,round(v*SCALE)) for v in st],[(e,round(v*SCALE)) for e,v in enumerate(ed) if v>1e-6]


class Dual:
    def __init__(self,m,beta,edges,cuts,hi,np):
        self.m=m
        self.es={e:i+m.n for i,e in enumerate(edges)}
        self.cuts=[]
        self.nodes=[[] for _ in range(m.n)]
        self.arc=[[] for _ in m.edges]
        cost=[1.]*(m.n+len(edges))
        for key in cuts:
            idx=len(cost)
            rhs,x,y=cut_row(m,key)
            cost.append(float(rhs))
            self.cuts.append((key,idx))
            for j,v in x:self.nodes[j].append((idx,v))
            for e,v in y:self.arc[e].append((idx,v))
        self.fleet=len(cost)
        cost.append(m.k)
        self.lp=hi.Highs()
        self.lp.setOptionValue("output_flag",False)
        self.lp.setOptionValue("threads",1)
        self.lp.setOptionValue("presolve","off")
        self.lp.setOptionValue("solver","simplex")
        self.lp.addVars(len(cost),np.zeros(len(cost)),np.repeat(hi.kHighsInf,len(cost)))
        self.lp.changeColsCost(len(cost),np.arange(len(cost),dtype=np.int32),cost)
        self.lp.changeColBounds(0,0,0)
        self.seen=set()
        self.paths=[]
        self.rank=[]
        self.weight=[ceil(SCALE*(Fraction(s,m.qden)+beta*Fraction(p,m.pden)))/SCALE
                     for s,p in zip(m.stock,m.pop)]
        self.hi=hi

    def add(self,paths):
        lows,offset,cols,vals=[],[],[],[]
        for route in paths:
            if not route: continue
            path=tuple(route)
            if path in self.seen: continue
            self.seen.add(path)
            row=Counter(path)
            value=sum(self.weight[j] for j in path)
            for i,q in list(row.items()):
                for j,v in self.nodes[i]:row[j]+=v*q
            for a,b in zip((0,*path),(*path,0)):
                e=self.m.index.get((min(a,b),max(a,b)))
                if e is None:break
                id_=self.es.get(e)
                if id_ is not None:row[id_]+=1
                for j,v in self.arc[e]:row[j]+=v
            else:
                for mask,col in self.rank:
                    row[col]=sum((mask>>u)&1 for u in path)//2
                row[self.fleet]=1
                self.paths.append(path)
                lows.append(value)
                offset.append(len(cols))
                for j,v in row.items():
                    if v:
                        cols.append(j)
                        vals.append(v)
        if lows:
            self.lp.addRows(len(lows),lows,[self.hi.kHighsInf]*len(lows),len(vals),offset,cols,vals)
        return len(lows)

    def add_rank(self,mask):
        col=self.lp.getNumCol()
        ids,vals=[],[]
        for i,path in enumerate(self.paths):
            count=sum((mask>>u)&1 for u in path)//2
            if count:
                ids.append(i)
                vals.append(count)
        self.lp.addCol(mask.bit_count()//2,0,self.hi.kHighsInf,len(ids),ids,vals)
        self.rank.append((mask,col))

    def violated_rank(self):
        import numpy as np
        lam=self.lp.getSolution().row_dual
        chosen=[(l,p) for l,p in zip(lam,self.paths) if l>1e-7]
        if not chosen:return None
        chosen.sort(key=lambda lp:-lp[0])
        weights=np.asarray([w for w,_ in chosen])
        a=np.zeros((len(chosen),self.m.n),dtype=np.int16)
        for i,(_,path) in enumerate(chosen):
            for j,c in Counter(path).items():a[i,j]=c
        usage=(weights[:,None]*a).sum(axis=0)
        active=[j for j in range(1,self.m.n) if usage[j]>.70]
        active.sort(key=lambda j:-usage[j])
        existing={mask for mask,_ in self.rank}
        best,score=None,.0001
        def try_mask(mask):
            nonlocal best,score
            size=mask.bit_count()
            if size<3 or not size%2 or mask in existing:return
            sites=[j for j in active if mask>>j&1]
            v=(weights*(a[:,sites].sum(axis=1)//2)).sum()-size//2
            if v>score+1e-8 or (best and abs(v-score)<1e-8 and size<best.bit_count()):
                best,score=mask,v
        # Seek a GF(2) dependence of odd cardinality among almost-tight station
        # rows. Dropping a few least-used columns allows weighted near-dependencies.
        signs={}
        for j in active:
            signs[j]=sum((int(a[t,j])&1)<<t for t in range(len(weights)))
        for drop in (0,1,2,3,5,8):
            take=max(1,len(weights)-drop)
            odd=1<<take
            for ordering in (active,sorted(active,key=lambda j:(-round(usage[j],5),-j))):
                echelon={}
                for j in ordering:
                    v=(signs[j]&(odd-1)) | odd
                    mask=1<<j
                    while v:
                        bit=(v&-v).bit_length()-1
                        if bit not in echelon:
                            echelon[bit]=v,mask
                            break
                        vv,mm=echelon[bit]
                        v^=vv
                        mask^=mm
                    if v==odd:
                        try_mask(mask)
                    elif v==0 and take in echelon:
                        try_mask(mask^echelon[take][1])
        # Also score small triples, whose price can survive later route changes.
        subset=active[:90]
        if len(subset)>=3:
            sig=1.-2.*(a[:,subset]%2)
            usage=usage[subset]
            for i,j in enumerate(subset[:-2]):
                ss=sig[:,i+1:]
                corr=np.einsum("rj,rk,r->jk",ss,ss,weights*sig[:,i],optimize=False)
                vio=(usage[i]+usage[i+1:,None]+usage[None,i+1:]-(weights.sum()-corr)/2)/2-1
                np.fill_diagonal(vio,-1000)
                idx=np.unravel_index(vio.argmax(),vio.shape)
                if vio[idx]>score-1e-8:
                    try_mask((1<<j)|(1<<subset[i+1+idx[0]])|(1<<subset[i+1+idx[1]]))
        return best

    def optimize(self,end):
        self.lp.setOptionValue("time_limit",max(.001,end-monotonic()+self.lp.getRunTime()))
        self.lp.run()
        sol=self.lp.getSolution()
        if not sol.value_valid: return None
        z=[max(0,round(v*SCALE)) for v in sol.col_value]
        return z[:self.m.n],dict((e,z[i]) for e,i in self.es.items()),[z[i] for _,i in self.cuts],[z[i] for _,i in self.rank]


def hybrid(m,graph,warm,columns,deadline,offer,*,deep=False,saved=None,checkpoint=None):
    if monotonic()>=deadline-.08:
        return
    import highspy as hi
    import numpy as np
    try:
        pricer=Pricing(m)
    except (ValueError,OSError):
        return
    warm_prices,warm_paths,warm_time=saved if saved else (None,[],0)
    if warm_prices and warm_time*2>deadline-monotonic():
        warm_prices=None
    cut_key=lambda a,b,c:(a,b,tuple(c) if a=="blossom" else c)
    rank=[]
    if warm_prices:
        part=warm_prices["partition"]
        beta=Fraction(warm_prices["beta"])
        edges=dict(warm_prices["edge"])
        if graph:
            _,new_edges=compact_seed(m,graph,beta)
            for edge,_ in new_edges:
                edges.setdefault(edge,0)
        graph_mu={cut_key(a,b,c):0 for a,b,c,_ in graph["cuts"]} if graph else {}
        graph_mu.update((cut_key(a,b,c),v) for a,b,c,v in warm_prices["cuts"])
        keys,mult=list(graph_mu),list(graph_mu.values())
        rank=warm_prices.get("parity",[])
        initial=(warm_prices["station"],edges,mult,[v for _,v in rank])
    elif graph:
        beta=Fraction(warm["beta"] if warm is not None else graph["beta"])
        pi,ed=compact_seed(m,graph,beta)
        keys=[cut_key(a,b,c) for a,b,c,_ in graph["cuts"]]
        mult=[v for _,_,_,v in graph["cuts"]]
        edges=dict(ed)
        part=partition(m,5)
        if warm is not None:
            supplied=warm["partition"]
            if len(supplied)==m.n-1 and max(Counter(supplied).values())<=MAX_REGION:
                part=[0]+[v+1 for v in supplied]
            unit=m.qtotal/(1<<22)
            initial=([0]+[max(0,round(v*unit*SCALE)) for v in warm["prices"]],
                      dict.fromkeys(edges,0),[0]*len(mult),[])
        else:
            initial=(pi,edges,mult,[])
    else:
        return
    max_rank=rank_limit(m.n,pricer.h,pricing_groups(m.n,part)) if deep else 2
    if len(rank)>max_rank:
        rank=rank[:max_rank]
        initial=(*initial[:3],[v for _,v in rank])
    restricted=Dual(m,beta,edges,keys,hi,np)
    if columns: restricted.add(columns[:1800])
    if warm_paths: restricted.add(warm_paths)
    for mask,_ in rank:
        restricted.add_rank(mask)
    center=None
    best=None
    proposal=initial
    iteration=0
    finish=deadline
    deadline=monotonic()+max(0,deadline-monotonic())*(.73 if deep else .83)
    best_line=None
    chosen=None
    seconds_per_pattern=0
    saved_at=0

    def save_state():
        nonlocal saved_at
        line,_,price_time=best_line
        routes=[]
        visits=0
        for path in reversed(restricted.paths[-WARM_ROUTES:]):
            if visits+len(path)>WARM_VISITS:
                break
            routes.append(path)
            visits+=len(path)
        checkpoint(dict(schema=1,instance_sha256=m.identity,prices=line,
                        columns=routes,price_seconds=price_time))
        saved_at=monotonic()

    while monotonic()<deadline-.04:
        stations,incident,mu,parity=proposal
        line=dict(beta=str(beta),station=stations,edge=[(e,p) for e,p in incident.items() if p],
                  cuts=[[*key,v] for key,v in zip(keys,mu) if v],partition=part,
                  parity=[[mask,v] for (mask,_),v in zip(restricted.rank,parity) if v])
        try:
            nodes,bonus,const=rewards(m,line,SCALE,cut_row)
            tags,discount,extra=parity_data(m,line)
        except ValueError:
            break
        const+=extra
        began=monotonic()
        if deep and seconds_per_pattern*len(discount)*1.5>deadline-began-.04:
            break
        priced=pricer.price(nodes,bonus,part,deadline-.015,tags,discount)
        if priced is None:break
        price_time=monotonic()-began
        seconds_per_pattern=price_time/len(discount)
        caps,paths=priced
        vmax=max(caps)
        val=const+m.k*vmax
        root=sum(sorted(caps,reverse=True)[:m.k])
        line.update(root_tilts=["1"],route_bound=root)
        offer(line,root)
        if chosen is None or const+root<chosen:
            chosen,best_line=const+root,(line,caps,price_time)
        if best is None or val<best:
            center,best=proposal,val
        added=restricted.add(paths)
        if checkpoint and monotonic()>saved_at+.6 and monotonic()<finish-.03:
            save_state()
        if monotonic()>=deadline-.04:break
        nxt=restricted.optimize(deadline-.03)
        if nxt is None:break
        difference=val/SCALE-restricted.lp.getObjectiveValue()
        # Every additional positive cut doubles the table. Reserve enough for at
        # least two complete prices at the maximum possible active rank.
        next_rank_fits=(not deep or
                        2.5*seconds_per_pattern*(1<<(len(restricted.rank)+1))<deadline-monotonic()-.16)
        if (len(restricted.rank)<max_rank and next_rank_fits and iteration>=2 and
                (not added or difference<max(1e-6,val/SCALE*.002)) and monotonic()<deadline-.16):
            mask=restricted.violated_rank()
            if mask is not None:
                restricted.add_rank(mask)
                center=(*center[:3],[*center[3],0])
                nxt=restricted.optimize(deadline-.03)
                if nxt is None:break
                added+=1
        ns,ne,nm,nr=nxt
        st,pe,du,pr=center
        if iteration%5==4 or not added:
            proposal=nxt
        else:
            a,b=(3,1) if iteration%3==2 else (1,2)
            mix=lambda v,w: (a*v+b*w)//(a+b)
            proposal=([mix(v,w) for v,w in zip(ns,st)],{e:mix(ne[e],pe[e]) for e in edges},
                      [mix(v,w) for v,w in zip(nm,du)],[mix(v,w) for v,w in zip(nr,pr)])
        if not added and proposal==center:
            break
        iteration+=1


    # Each root carries the route's own optimal AM-GM slope. Re-pricing a
    # handful of common slopes and taking their minimum per root is rigorous.
    if best_line is None:return
    if checkpoint and monotonic()<finish-.03:
        save_state()
    line,caps,price_time=best_line
    all_caps=[caps]
    slopes=[Fraction(1)]
    tags,discount,_=parity_data(m,line)
    tilts=("24/25","25/24","9/10","10/9","49/50","50/49","4/5","5/4")
    if deep:
        tilts+=("19/20","20/19","17/20","20/17","2/3","3/2","13/10")
    for t in map(Fraction,tilts[:MAX_TILTS-1]):
        if monotonic()>finish-.05:break
        if deep and 1.5*price_time>finish-monotonic()-.035:break
        nodes,bonus,_=rewards(m,line,SCALE,cut_row,t)
        began=monotonic()
        priced=pricer.price(nodes,bonus,part,finish-.02,tags,discount)
        if priced is None:break
        price_time=monotonic()-began
        caps,_=priced
        if not any(v<min(row[i] for row in all_caps) for i,v in enumerate(caps)):
            continue
        all_caps.append(caps)
        slopes.append(t)
        total=sum(sorted(map(min,zip(*all_caps)),reverse=True)[:m.k])
        use=list(range(len(slopes)))
        for ix in range(len(slopes)-1,-1,-1):
            subset=[i for i in use if i!=ix]
            if subset and sum(sorted(map(min,zip(*(all_caps[i] for i in subset))),reverse=True)[:m.k])==total:
                use=subset
        trial={**line,"root_tilts":[str(slopes[i]) for i in use],"route_bound":total}
        offer(trial,total)
