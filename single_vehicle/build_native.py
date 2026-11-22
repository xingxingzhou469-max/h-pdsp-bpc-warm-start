"""Build the native V8 components with a local C++ compiler."""

import ctypes
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


HPDP = Path(__file__).resolve().parents[1] / "hpdp"
LIBRARIES = (
    "fragment",
    "large_sets",
    "mandatory_walks",
    "mandatory_wide",
    "order_frontier",
    "population_walks",
    "sparse_sets",
    "weighted_prefix",
)


def library_name(component):
    if sys.platform not in ("darwin", "linux"):
        raise ValueError("Building V8 native code requires macOS or Linux")
    suffix = "dylib" if sys.platform == "darwin" and component != "fragment" else "so"
    return f"lib{component}.{suffix}"


def require_native():
    try:
        for component in LIBRARIES:
            ctypes.CDLL(str(HPDP / library_name(component)))
        worker = HPDP / "chain-worker"
        if not worker.is_file() or not os.access(worker, os.X_OK):
            raise OSError(f"Missing executable {worker.name}")
    except OSError as error:
        raise RuntimeError(
            f"Run python3 single_vehicle/build_native.py before running V8 ({error})"
        ) from error


def build_native():
    compiler = os.environ.get("CXX", "c++")
    flags = ["-O3", "-Wall", "-Wextra"]
    with TemporaryDirectory(prefix=".native-", dir=HPDP) as directory:
        work = Path(directory)
        for component in LIBRARIES:
            artifact = work / library_name(component)
            shared = "-dynamiclib" if sys.platform == "darwin" and component == "mandatory_wide" else "-shared"
            install_name = [f"-Wl,-install_name,@rpath/{artifact.name}"] if sys.platform == "darwin" else []
            subprocess.run(
                [compiler, "-std=c++17", *flags, shared, "-fPIC",
                 *install_name, str(HPDP / f"{component}.cpp"), "-o", str(artifact)],
                check=True,
            )
            ctypes.CDLL(str(artifact))
        subprocess.run(
            [compiler, "-std=c++20", *flags, str(HPDP / "chain_oracle.cpp"),
             "-o", str(work / "chain-worker")],
            check=True,
        )
        for artifact in work.iterdir():
            artifact.replace(HPDP / artifact.name)


if __name__ == "__main__":
    try:
        build_native()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"V8 native build failed: {error}")
    print("Built 8 V8 shared libraries and the chain-worker executable.")
