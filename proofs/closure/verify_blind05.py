"""Independently verify the compact blind-05 optimum from raw JSON and a small certificate (stdlib)."""
import argparse
from fractions import Fraction as Q
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path


def require(condition,message):
    if not condition:
        raise ValueError(message)


def check_time(model,sites,cert):
    require(tuple(cert["sites"])==sites,"Certificate uses a different visit set")
    prices=cert["prices"]
    scale=cert["time_scale"]
    denominator=cert["denominator"]
    require(type(scale) is int and scale>0 and type(denominator) is int and denominator>0
            and len(prices)==len(sites) and all(type(x) is int for x in prices),"Invalid scales or prices")
    travel=model["travel"]
    def integer_time(x):
        y=Q(x)*scale
        require(y.denominator==1,"Wrong integer time scale")
        return y.numerator
    # Prim scan of the cut at every insertion; no mathematical optimizer or project module is imported.
    reached={0}
    total=0
    while len(reached)<len(sites):
        cost,_,j=min((denominator*integer_time(min(travel[sites[i]][sites[j]],travel[sites[j]][sites[i]]))
                      +prices[i]+prices[j],i,j)
                     for i in reached for j in range(len(sites)) if j not in reached)
        reached.add(j)
        total+=cost
    total+=min(denominator*integer_time(travel[0][v])+prices[i] for i,v in enumerate(sites) if model["supply"][v])
    total+=min(denominator*integer_time(travel[v][0])+prices[i] for i,v in enumerate(sites) if model["population"][v])
    numerator=total-2*sum(prices)
    service=sum(integer_time(model["service"][v]) for v in sites)
    lower=Q(service-(-numerator//denominator),scale)
    require(numerator==cert["travel_numerator"] and service==cert["service_scaled"]
            and lower*scale==cert["lower_scaled"],"Claimed tree bound is incorrect")
    return lower


def verify(model_path,certificate_path):
    raw=model_path.read_bytes()
    data=json.loads(raw,parse_float=Q)
    cert=json.loads(certificate_path.read_bytes())
    require(sha256(raw).hexdigest()==cert["input_sha256"],"Original model SHA-256 differs")
    nodes=set(range(1,len(data["service"])))
    supply=data["supply"]
    pop=data["population"]
    capacity=Q(data["capacity"])
    total_people=sum(pop)
    require(all((supply[i]>0) != (pop[i]>0) for i in nodes) and total_people>0 and capacity>0,
            "Expected one positive role for each station")
    route=cert["witness"]["route"]
    quantities=[Q(x) for x in cert["witness"]["amounts"]]
    require(len(route)==len(quantities)==len(set(route)) and set(route)<=nodes
            and supply[route[0]]>0 and pop[route[-1]]>0,"Invalid route")
    held=Q(0)
    peak=Q(0)
    deliveries={i:Q(0) for i in nodes if pop[i]}
    for i,amount in zip(route,quantities,strict=True):
        require(amount>=0,"Negative quantity")
        if supply[i]:
            require(amount<=supply[i],"Pickup exceeds supply")
            held+=amount
        else:
            deliveries[i]=amount
            held-=amount
        require(0<=held<=capacity,"Load falls outside vehicle capacity")
        peak=max(peak,held)
    require(held==0,"Return load is nonzero")
    driving=sum(data["travel"][i][j] for i,j in zip((0,*route),(*route,0)))
    service=sum(data["service"][i] for i in route)
    require(driving+service<=data["limit"],"Late return")
    delivered=sum(deliveries.values())
    witness=delivered-sum(abs(pop[j]*deliveries[i]-pop[i]*deliveries[j])
                          for i,j in combinations(deliveries,2))/total_people
    require(witness==Q(cert["claimed_optimum"]),"Incorrect witness objective")

    def upper(sites):
        stock=sum(min(capacity,supply[i]) for i in sites)
        flow=min(stock,sum(bool(pop[i]) for i in sites)*capacity)
        return Q(flow*sum(pop[i] for i in sites),total_people)

    special=sorted(i for i in nodes if upper(nodes-{i})>witness)
    proofs={tuple(t["sites"]):t for t in cert["time_certificates"]}
    verified=[]
    for count in range(len(special)+1):
        for omitted in combinations(special,count):
            sites=tuple(sorted(nodes-set(omitted)))
            if upper(sites)<=witness:
                continue
            require(sites in proofs,"Missing time certificate for an improving visit set")
            lower=check_time(data,sites,proofs[sites])
            require(lower>data["limit"],"The tour set has not been excluded")
            verified.append(dict(omitted=omitted,lower_time=str(lower)))
    require(len(verified)==len(proofs)==len(cert["time_certificates"]),"Unexpected or duplicate proofs")
    return dict(strict_optimum=str(witness),delivered=str(delivered),driving=str(driving),service=str(service),
                peak=str(peak),input_sha256=cert["input_sha256"],exceptional_single_omissions=special,
                checked_time_certificates=verified)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model",type=Path)
    parser.add_argument("certificate",type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.model,args.certificate),ensure_ascii=False))
