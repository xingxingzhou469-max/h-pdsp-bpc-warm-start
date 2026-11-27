"""Fleet-wide prize-collecting cycle-cover pricing with optional subtour and parity cuts."""
from time import monotonic, monotonic_ns
START = monotonic()
import argparse
from fractions import Fraction
import json
from pathlib import Path
import os
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check import METHOD, SCALE, Model, cut_row, upper


def write_atomic(path, payload, byte_limit=None):
    data=json.dumps(payload,separators=(",",":"))+"\n"
    if byte_limit is not None and len(data)>byte_limit:
        return
    tmp = path.with_suffix(".tmp")
    tmp.write_text(data)
    os.replace(tmp,path)


class Master:
    def __init__(self, model, hi):
        import numpy as np
        self.np, self.hi, self.model = np, hi, model
        self.lp = hi.Highs()
        for opt,value in (("output_flag",False),("threads",1),("solver","simplex"),("presolve","off")):
            self.lp.setOptionValue(opt,value)
        self.keys, self.added = [], set()
        n = model.n
        self.groups=[]
        dirs=[]
        for group in model.oriented:
            self.groups.append(list(range(len(dirs),len(dirs)+len(group))))
            dirs.extend(group)
        m=len(dirs)
        self.lp.addVars(n+m,np.zeros(n+m),np.ones(n+m))
        self.lp.changeColBounds(0,0,0)
        self.lp.changeObjectiveSense(hi.ObjSense.kMaximize)
        incident=[[] for _ in range(n)]
        flow=[[] for _ in range(n)]
        for e,(u,v,_) in enumerate(dirs):
            incident[u].append(e)
            incident[v].append(e)
            flow[u].append((n+e,1))
            flow[v].append((n+e,-1))
        rows=[(0,0,[(u,2)]+[(n+e,-1) for e in incident[u]]) for u in range(1,n)]
        rows.extend((0,0,flow[u]) for u in range(1,n))
        rows.append((-hi.kHighsInf,2*model.k,[(n+e,1) for e in incident[0]]))
        rows.append((-hi.kHighsInf,0,
                     [(i,float(d/model.tden)) for i,d in enumerate(model.service) if i and d]+
                     [(n+e,(d-(model.limit/2 if not u or not v else 0))/model.tden)
                      for e,(u,v,d) in enumerate(dirs)]))
        self.pairs=set()
        self.rows(rows)

    def rows(self,rows):
        if not rows:
            return
        beg,indices,values,lows,highs = [],[],[],[],[]
        for lo,up,terms in rows:
            beg.append(len(values))
            for i,v in terms:
                indices.append(i)
                values.append(v)
            lows.append(lo)
            highs.append(up)
        self.lp.addRows(len(rows),lows,highs,len(values),beg,indices,values)

    def insert(self,new):
        rows = []
        for key in new:
            if key in self.added:
                continue
            b,x,y = cut_row(self.model,key)
            rows.append((-self.hi.kHighsInf,b,x+[(self.model.n+t,v) for i,v in y for t in self.groups[i]]))
            self.keys.append(key)
            self.added.add(key)
        self.rows(rows)
        return len(rows)

    def run(self,beta,deadline):
        n = self.model.n
        m = self.model
        weight = [s/m.qden+float(beta)*p/m.pden for s,p in zip(m.stock,m.pop)]
        self.lp.changeColsCost(n,self.np.arange(n,dtype=self.np.int32),weight)
        self.lp.setOptionValue("time_limit",max(.001,deadline-monotonic()+self.lp.getRunTime()))
        self.lp.run()
        sol = self.lp.getSolution()
        if not sol.dual_valid or not sol.value_valid:
            return None
        z = [int(round(v*SCALE)) for v in sol.row_dual]
        line = dict(beta=str(beta),degree=[0,*z[:n-1]],flow=[0,*z[n-1:2*n-2]],fleet=max(0,z[2*n-2]),time=max(0,z[2*n-1]),
                    cuts=[[*key,v] for key,v in zip(self.keys,z[2*n:]) if v>0])
        arcs=sol.col_value[n:]
        y = [sum(arcs[e] for e in group) for group in self.groups]
        return line,sol.col_value[:n],y


def shores(n,nx,edges,values):
    g=nx.Graph()
    g.add_nodes_from(range(n))
    for (u,v),value in zip(edges,values):
        if value>1e-9:
            g.add_edge(u,v,capacity=value)
    tree=nx.gomory_hu_tree(g,capacity="capacity",flow_func=nx.algorithms.flow.shortest_augmenting_path)
    subtree = [1<<i for i in range(n)]
    links=list(nx.bfs_edges(tree,0))
    for u,v in reversed(links):
        yield subtree[v]
        subtree[u] |= subtree[v]


def separate(model,x,y,nx,deadline):
    eps = 1e-6
    plus = [(e,u,v,max(0,min(1,ye))) for e,((u,v),ye) in enumerate(zip(model.edges,y)) if ye>1e-8]
    proposals = {}
    for e,u,v,ye in plus:
        for j in (u,v):
            if j and (ye>x[j]+eps or y[e]>1+eps):
                proposals["link",e,j] = ye-x[j]
    if monotonic()>=deadline:
        return sorted(proposals,key=proposals.get,reverse=True)[:100]
    edges = [(u,v) for _,u,v,_ in plus]
    for mode in ("connect","blossom"):
        vals=[t if mode=="connect" else min(t,1-t) for _,_,_,t in plus]
        if mode=="blossom" and sum(vals)<eps:
            continue
        for mask in shores(model.n,nx,edges,vals):
            if mask.bit_count()<2:
                continue
            boundary = [(e,t) for e,u,v,t in plus if (mask>>u ^ mask>>v)&1]
            if mode=="connect":
                root=max((i for i in range(1,model.n) if mask>>i&1),key=x.__getitem__)
                violation=x[root]-sum(v for e,v in boundary)/2
                if violation>eps:
                    proposals["connect",mask,root]=violation
            else:
                teeth={e for e,t in boundary if t>.5}
                cost=sum(min(t,1-t) for e,t in boundary)
                if not len(teeth)%2 and boundary:
                    e,t=min(boundary,key=lambda item:abs(1-2*item[1]))
                    cost+=abs(1-2*t)
                    teeth.symmetric_difference_update([e])
                if len(teeth)%2 and cost<1-eps:
                    proposals["blossom",mask,tuple(sorted(teeth))]=1-cost
        if monotonic()>=deadline:
            break
    return sorted(proposals,key=proposals.get,reverse=True)[:140]


def solve(raw,k,out,deadline,warm_file=None,column_file=None,*,deep=False,warm_upper=None,save_upper=False):
    final_deadline=deadline
    m=Model(raw,k)
    out.mkdir(parents=True,exist_ok=True)
    best=None
    graph_best=None
    def offer(line,cap=None):
        nonlocal best,graph_best
        score=upper(m,line,cap)
        if line and "station" not in line and (graph_best is None or score<graph_best[0]):
            graph_best=score,line
        if (best is None or score<best) and monotonic()<final_deadline:
            write_atomic(out/"best.json",dict(method=METHOD,vehicles=k,instance_sha256=m.identity,
                         upper=str(score),support=line,emitted_monotonic_ns=monotonic_ns()))
            best=score
    offer(None)
    if not m.qtotal or not m.ptotal or not k or m.n<3 or monotonic()>=deadline:
        return
    import highspy as hi
    import networkx as nx
    warm=None
    columns=[]
    if warm_file and warm_file.is_file():
        data=json.loads(warm_file.read_bytes())
        choices=[s for s in data.get("supports",[]) if "partition" in s and len(s["prices"])==m.n-1]
        if choices:
            lines=sorted((Fraction(s["beta"]), m.qtotal*(sum(s["prices"])+k*s["route_bound"])/(1<<22),i)
                         for i,s in enumerate(choices))
            points={Fraction(),m.ptotal}
            for slope,a,_ in lines:
                points.update((a/(2*slope),(a-m.qtotal)/slope))
                for t,b,_ in lines:
                    if t!=slope:points.add((a-b)/(slope-t))
            caps=lambda p:min(m.qtotal,*(a-t*p for t,a,_ in lines))
            p=max((v for v in points if 0<=v<=m.ptotal),key=lambda v:v*caps(v))
            desired=caps(p)/p if p else lines[0][0]
            low=max((row for row in lines if row[0]<=desired),default=lines[0])
            high=min((row for row in lines if row[0]>=desired),default=lines[-1])
            if high[0]>low[0]:
                z=(desired-low[0])/(high[0]-low[0])
                near=choices[low[2] if z<=Fraction(1,2) else high[2]]
                warm={**near,"beta":str(desired.limit_denominator(1000000) or desired),
                      "prices":[round((1-z)*v+z*w) for v,w in zip(choices[low[2]]["prices"],choices[high[2]]["prices"])]}
            else:
                warm=choices[low[2]]
    if column_file and column_file.is_file():
        data=json.loads(column_file.read_bytes())
        columns=[r for r in data.get("routes",[]) if isinstance(r,list) and all(type(v) is int and 1<=v<m.n for v in r)]
    master=Master(m,hi)
    beta=(m.qtotal/m.ptotal).limit_denominator(1000000) or m.qtotal/m.ptotal
    if warm is not None: beta=Fraction(warm["beta"])
    stop=deadline
    native=Path(__file__).with_name("kernel.so").is_file()
    saved=None
    if native and deep and warm_upper:
        from hybrid import read_warm
        saved=read_warm(m,warm_upper)
    if native:
        if deep:
            remaining=max(0,stop-monotonic())
            compact_seconds=min(6,remaining*.10) if saved else min(18,remaining*.25)
            deadline=monotonic()+compact_seconds
        else:
            deadline=monotonic()+max(0,stop-monotonic())*(.32 if warm else .48)
    previous=None
    iteration=0
    while monotonic()<deadline-.06:
        ans=master.run(beta,deadline-.03)
        if ans is None:
            break
        line,x,y=ans
        offer(line)
        if monotonic()>=deadline-.08:
            break
        new=separate(m,x,y,nx,deadline-.05)
        count=master.insert(new)
        p=sum(a*b for a,b in zip(x,m.pop))/m.pden
        q=sum(a*b for a,b in zip(x,m.stock))/m.qden
        if not p or not q:
            break
        ratio=q/p
        if iteration==0 or (iteration%4==0 and count) or not count:
            target=(float(beta)*ratio)**.5
            if not count and previous and (float(beta)-previous[0])*(ratio-previous[1])<0:
                # The LP has adjacent support faces; solve their intersection.
                pp,qq=previous[2:]
                if abs(pp-p)>1e-7:
                    crossing=(q-qq)/(pp-p)
                    if min(previous[0],float(beta))+1e-7<crossing<max(previous[0],float(beta))-1e-7:
                        target=crossing
            new_beta=Fraction(target).limit_denominator(100000) or beta
            previous=float(beta),ratio,p,q
            if not count and abs(float(new_beta/beta)-1)<1e-5:
                break
            beta=new_beta
        iteration+=1


    if (graph_best or saved) and stop>monotonic()+.06 and native:
        from hybrid import WARM_BYTES, hybrid
        graph=graph_best[1] if graph_best else None
        checkpoint=(lambda data: write_atomic(out/"warm.json",data,WARM_BYTES)) if deep or save_upper else None
        hybrid(m,graph,warm,columns,stop,offer,deep=deep,saved=saved,checkpoint=checkpoint)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("instance",type=Path)
    ap.add_argument("--vehicles",type=int)
    ap.add_argument("--seconds",type=float,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--warm-zone",type=Path)
    ap.add_argument("--columns",type=Path)
    ap.add_argument("--deep",action="store_true",help="enable time-adaptive route pricing for later cycles")
    ap.add_argument("--warm-upper",type=Path,help="previous upper worker directory, used with --deep")
    ap.add_argument("--save-upper",action="store_true",help="save reusable prices and columns for a later cycle")
    args=ap.parse_args()
    if args.warm_upper and not args.deep:
        ap.error("--warm-upper requires --deep")
    raw=json.loads(args.instance.read_bytes(),parse_float=Fraction)
    solve(raw,raw["vehicles"] if args.vehicles is None else args.vehicles,args.out,START+args.seconds,
          args.warm_zone,args.columns,deep=args.deep,warm_upper=args.warm_upper,save_upper=args.save_upper)
