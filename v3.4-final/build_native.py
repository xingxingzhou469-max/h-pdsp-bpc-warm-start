"""Build the native libraries for the archived v3.4-final solver."""
import os
from pathlib import Path
import platform
from shlex import quote
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
V21 = ROOT / "交付/github_release_v21/repo"
V32 = ROOT / "runs/mv-v32-integration"
V34 = ROOT / "runs/mv-v34-final"


def main():
    system = platform.system()
    if system not in ("Darwin", "Linux"):
        raise SystemExit("Use macOS or Linux for the native build")
    env = dict(os.environ, V33_V32=str(V32), V33_V21=str(V21), PYTHONDONTWRITEBYTECODE="1")
    compiler = shutil.which(env.get("CXX", "c++"))
    if compiler is None:
        raise SystemExit("A C++20 compiler is required. Set CXX to its executable path.")
    commands = [[sys.executable, "-B", V21 / "multi_vehicle/build_native.py"], ["sh", V32 / "build.sh"]]
    commands.extend(["sh", V34 / part / "build.sh"] for part in ("lower", "upper", "subsequence", "sparse"))
    with TemporaryDirectory(prefix="hpdp-v34-cxx-") as folder:
        if system == "Linux":
            shim = Path(folder) / "cxx"
            shim.write_text(f'#!/bin/sh\nexec {quote(compiler)} "$@" -Wno-error\n', encoding="utf-8")
            shim.chmod(0o700)
            env["CXX"] = str(shim)
        for command in commands:
            subprocess.run(list(map(str, command)), cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
