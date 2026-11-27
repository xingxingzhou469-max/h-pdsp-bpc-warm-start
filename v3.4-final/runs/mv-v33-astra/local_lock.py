"""Atomically reserve the shared M4, run one command under caffeinate, and release it."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LOCK = Path("/tmp/hpdp_m4_cpu.lock")

def main(argv):
    if not argv:
        raise SystemExit("usage: python3 locked.py COMMAND [ARGS ...]")
    while True:
        try:
            fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o644)
            break
        except FileExistsError:
            time.sleep(4)
    owner=os.fstat(fd)
    proc=None
    try:
        with os.fdopen(fd,"w") as stream:
            stream.write(f"owner=main\npid={os.getpid()}\nphase=astra-breakthrough\nstarted={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
            stream.flush()
            os.fsync(stream.fileno())
        print("ACQUIRED",LOCK,time.strftime("%Y-%m-%d %H:%M:%S %Z"),flush=True)
        def stop(sig,frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop)
        env=os.environ.copy()
        env.update(dict.fromkeys(("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","HIGHS_NUM_THREADS"),"1"))
        env.update(PYTHONDONTWRITEBYTECODE="1",PYTHONHASHSEED="0",ASTRA_LOCK_OWNER=str(os.getpid()))
        proc=subprocess.Popen(["caffeinate","-is",*argv],env=env,start_new_session=True)
        return proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid,signal.SIGCONT)
            os.killpg(proc.pid,signal.SIGTERM)
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
        if LOCK.exists() and LOCK.stat().st_ino==owner.st_ino and LOCK.stat().st_dev==owner.st_dev:
            LOCK.unlink()
            print("RELEASED",time.strftime("%Y-%m-%d %H:%M:%S %Z"),flush=True)
        else:
            raise RuntimeError("lock ownership changed")
if __name__=="__main__":
    sys.exit(main(sys.argv[1:]))
