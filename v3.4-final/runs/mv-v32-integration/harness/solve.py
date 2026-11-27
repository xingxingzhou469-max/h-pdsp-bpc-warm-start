"""A one-core anytime portfolio. The primal worker keeps its state between proof epochs."""
from time import monotonic_ns, time_ns

ENTERED_MONO, ENTERED_WALL = monotonic_ns(), time_ns()

import argparse
from datetime import datetime
from fractions import Fraction as F
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep

RUN=Path(__file__).resolve().parents[1]
NS=10**9

def read(path):
    return json.loads(path.read_bytes())

def atomic(path,data):
    tmp=path.with_suffix('.part')
    tmp.write_bytes(data)
    os.replace(tmp,path)

def encode(obj):
    return (json.dumps(obj,separators=(',',':'),ensure_ascii=False)+'\n').encode()

def single_env():
    env=os.environ.copy()
    env.update(dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                              'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','HIGHS_NUM_THREADS'),'1'))
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
    return env

class Worker:
    def __init__(self,kind,folder,argv,env):
        self.kind=kind
        self.folder=folder
        self.path=folder/'best.json'
        folder.mkdir(parents=True,exist_ok=False)
        self.log=(folder/'worker.log').open('wb')
        self.proc=subprocess.Popen(list(map(str,argv)),env=env,stdout=self.log,stderr=subprocess.STDOUT)
        self.identity=None
        self.stopped=False
        self.cut=False

    def pause(self):
        if self.proc.poll() is None and not self.stopped:
            self.proc.send_signal(signal.SIGSTOP)
            # Acknowledge the stop before another heavy worker starts.
            pid,status=os.waitpid(self.proc.pid,os.WUNTRACED)
            if not os.WIFSTOPPED(status):
                self.proc.returncode=os.waitstatus_to_exitcode(status)
            self.stopped=True

    def resume(self):
        if self.proc.poll() is None and self.stopped:
            self.proc.send_signal(signal.SIGCONT)
            self.stopped=False

    def close(self):
        if self.proc.poll() is None:
            self.cut=True
            self.proc.kill()
            self.proc.wait()
        self.log.close()

class Timeline:
    def __init__(self,out,input_path,k,endpoints,plan,seed=7):
        self.out=out
        self.k=k
        self.source=input_path
        self.points=endpoints
        self.limit=endpoints[-1]
        self.plan=plan
        self.seed=seed
        self.keep={}
        self.counter=0
        self.workers=[]
        self.cached=None
        self.processes=[]
        self.watch=set()
        self.env=single_env()
        out.mkdir(parents=True,exist_ok=False)
        raw=input_path.read_bytes()
        (out/'input.json').write_bytes(raw)
        self.fingerprint=sha256(raw).hexdigest()
        (out/'events').mkdir()
        self.ledger=(out/'events.jsonl').open('a',buffering=1)
        self.tick=int(os.environ.get('ASTRA_COLD_MONO_NS',ENTERED_MONO))
        self.wall=int(os.environ.get('ASTRA_COLD_WALL_NS',ENTERED_WALL))
        data=json.loads(raw,parse_float=F)
        self.instance_sites=range(1,len(data['supply']))
        # All setup that could use instance information occurs inside the timed run.
        u=min(sum(F(str(x)) for x in data['supply']),F(str(data['capacity']))*sum(F(str(x))>0 for x in data['population'])) if k else F()
        self.admit('L',encode({'lower':'0','witness':{'routes':[],'amounts':[]},'source':'idle'}))
        self.admit('box',encode({'upper':str(u)}))
        self.samples=[]

    def admit(self,kind,buf):
        doc=json.loads(buf)
        lower=kind=='L'
        value=F(doc['lower'] if lower else doc['upper'])
        key='L' if lower else 'U'
        if (key in self.keep and (value<=self.keep[key][0] if lower else value>=self.keep[key][0])):
            return
        path=self.out/'events'/f'{self.counter:04d}-{kind}.json'
        atomic(path,buf)
        mono,wall=monotonic_ns()-self.tick,time_ns()-self.wall
        if max(mono,wall)>self.limit*NS:
            path.unlink()
            return
        self.counter+=1
        r=dict(kind=kind,score=str(value),file=str(path.relative_to(self.out)),sha256=sha256(buf).hexdigest(),
               elapsed_ns=mono,wall_ns=wall)
        self.ledger.write(json.dumps(r,separators=(',',':'))+'\n')
        self.keep[key]=(value,path)
        if 'L' in self.keep and 'U' in self.keep and self.keep['L'][0]>self.keep['U'][0]:
            raise ValueError("emitted L exceeds emitted U")

    def inspect(self,worker):
        try:
            st=worker.path.stat()
        except FileNotFoundError:
            return
        token=(st.st_ino,st.st_mtime_ns,st.st_size)
        if token==worker.identity:
            return
        worker.identity=token
        buf=worker.path.read_bytes()
        self.admit('L' if worker.kind in ('primal','levels','couple','window','prize') else worker.kind,buf)

    def start(self,kind,duration,number,persistent=False):
        folder=self.out/f'work-{number:02d}-{kind}'
        args=[sys.executable,'-B',RUN/kind/'solve.py',self.source,'--vehicles',self.k,'--seconds',
              86400 if persistent else max(.03,duration-.15),'--out',folder]
        if kind in ('primal','levels','window','prize'):
            args+=['--seed',str(self.seed)]
        if kind in ('levels','couple','window','prize') and 'L' in self.keep:
            args+=['--warm',self.keep['L'][1]]
        if kind=='levels':
            guides=[w.folder/'guide.json' for w in self.workers if (w.folder/'guide.json').is_file()]
            if guides:
                args+=['--guide',guides[-1]]
        if kind=='fair':
            proofs=[w.path for w in self.workers if w.kind=='graph' and w.path.exists()]
            if proofs:
                args+=['--graph-proof',proofs[-1]]
        if kind in ('couple','maskfleet','branchprice'):
            cols=[w.folder/'columns.json' for w in self.workers if w.kind=='zone' and (w.folder/'columns.json').is_file()]
            if not cols and kind=='couple':
                fallback=self.out/'empty-columns.json'
                atomic(fallback,encode({'routes':[]}))
                cols=[fallback]
            if cols:
                args+=['--columns',cols[-1]]
        if kind in ('ng','zone','maskfleet','branchprice','screen2'):
            # Only hints generated earlier on this cold trajectory are visible.
            allowed=('route',) if kind=='ng' else ('zone','route') if kind=='zone' else ('zone',)
            refs=[w.path for w in self.workers if w.kind in allowed and w.path.exists()]
            if refs:
                args+=['--warm-zone' if kind=='maskfleet' else '--warm-route',refs[-1]]
        worker=Worker(kind,folder,args,self.env)
        self.workers.append(worker)
        return worker

    def sample(self):
        text=subprocess.check_output(['ps','-axo','pid=,ppid=,pcpu=,comm='],text=True)
        ids={w.proc.pid for w in self.workers}|{os.getpid()}
        top=[]
        for line in text.splitlines():
            fields=line.split(None,3)
            if len(fields)==4 and int(fields[0]) not in ids and float(fields[2])>=20:
                top.append([float(fields[2]),int(fields[0]),Path(fields[3]).name])
        self.samples.append(dict(seconds=(monotonic_ns()-self.tick)/NS,external=sorted(top,reverse=True)[:6]))

    def run(self):
        number=0
        cursor=0
        next_sample=2*NS
        blocks=self.plan['opening']
        expansion=1
        try:
            while cursor<self.limit:
                for kind,length in blocks:
                    if kind=='fleet-proof':
                        kind='screen2' if self.k==2 and len(self.instance_sites)<=128 else 'maskfleet'
                    if cursor>=self.limit:
                        break
                    planned=cursor+length
                    stop=self.tick+int(min(planned,self.limit)*NS)
                    if monotonic_ns()>=stop:
                        cursor=planned
                        continue
                    at=(monotonic_ns()-self.tick)/NS
                    if kind=='primal' and self.cached:
                        worker=self.cached
                        worker.resume()
                    else:
                        worker=self.start(kind,length,number,persistent=(kind=='primal'))
                        if kind=='primal':
                            self.cached=worker
                    while monotonic_ns()<stop:
                        self.inspect(worker)
                        status=worker.proc.poll()
                        if status not in (None,0):
                            raise RuntimeError(f"{kind} exit {status}: {worker.folder/'worker.log'}")
                        if status==0:
                            break
                        if monotonic_ns()-self.tick>=next_sample:
                            self.sample()
                            next_sample+=10*NS
                        sleep(.036)
                    if worker is self.cached:
                        worker.pause()
                    else:
                        worker.close()
                    self.inspect(worker)
                    self.processes.append(dict(kind=kind,began_s=at,ended_s=(monotonic_ns()-self.tick)/NS,
                                               stopped_by_scheduler=worker.cut,exit=worker.proc.returncode))
                    number+=1
                    cursor=planned
                    # If a proof stage finished early, give its unused wall time
                    # to the resident primal (never idle until the next epoch).
                    if self.cached and monotonic_ns()+80_000_000<stop:
                        extra_start=(monotonic_ns()-self.tick)/NS
                        self.cached.resume()
                        while monotonic_ns()<stop:
                            self.inspect(self.cached)
                            sleep(.035)
                        self.cached.pause()
                        self.inspect(self.cached)
                        self.processes.append(dict(kind='primal-remainder',began_s=extra_start,
                                                   ended_s=(monotonic_ns()-self.tick)/NS))
                blocks=[(kind,length*expansion) for kind,length in self.plan['repeat']]
                expansion*=2
        finally:
            for worker in self.workers:
                worker.close()
            self.ledger.close()
        manifest=dict(case=self.source.stem,hardware='local',vehicles=self.k,seed=self.seed,checkpoints=self.points,
                      input_sha256=self.fingerprint,events_sha256=sha256((self.out/'events.jsonl').read_bytes()).hexdigest(),
                      event_count=self.counter,plan=self.plan,started_wall_ns=self.wall,
                      wall_elapsed_s=(time_ns()-self.wall)/NS,epochs=self.processes,load_samples=self.samples)
        atomic(self.out/'run.json',encode(manifest))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('instance',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--plan',type=Path)
    p.add_argument('--seconds',type=int,default=300)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--checkpoints',type=int,nargs='*',default=[15,30,60,120,300])
    args=p.parse_args()
    if args.seconds<=0:
        p.error("a positive budget is required")
    if not 0<=args.seed<2**32:
        p.error("seed must be an unsigned 32-bit integer")
    def terminate(sig,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,terminate)
    source=args.instance.resolve()
    k=read(source)['vehicles']
    plan=args.plan
    if plan is None:
        plan=Path(__file__).with_name('policy_k2.json' if k==2 else 'policy_k3p.json')
        if not plan.is_file():
            plan=Path(__file__).with_name('policy.json')
    t=Timeline(args.out.resolve(),source,k,sorted({s for s in args.checkpoints if 0<s<args.seconds}|{args.seconds}),read(plan),args.seed)
    t.run()
if __name__=='__main__':
    main()
