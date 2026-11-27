"""Evaluate the exact distinct-station subset closure of a support from this run."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import os
from time import monotonic_ns
from check import bound


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance",type=Path)
    parser.add_argument("--vehicles",type=int,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seconds",type=float,required=True)
    parser.add_argument("--warm-zone",type=Path,required=True)
    args=parser.parse_args()
    raw=json.loads(args.instance.read_bytes(),parse_float=F)
    parent=json.loads(args.warm_zone.read_bytes())
    score=bound(raw,args.vehicles,parent,verify=False)
    doc=dict(method="fleet-selection-v33",vehicles=args.vehicles,upper=str(score),parent=parent,
             emitted_monotonic_ns=monotonic_ns())
    args.out.mkdir(parents=True,exist_ok=True)
    p=args.out/"best.json"
    p.with_suffix(".part").write_text(json.dumps(doc,separators=(",",":")))
    os.replace(p.with_suffix(".part"),p)


if __name__=="__main__":
    main()
