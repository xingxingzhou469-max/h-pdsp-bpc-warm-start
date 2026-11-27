"""Build native C++17 extensions for the multi-vehicle H-PDSP v2.1 engine."""

import ctypes
from hashlib import sha256
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


MV_ROOT = Path(__file__).resolve().parent
ENGINE = MV_ROOT / "engine"
BUILD = Path(os.environ.get("NG_BB_BUILD", MV_ROOT / "build"))


def _compiler(prefer=("c++", "clang++")):
    env = os.environ.get("CXX")
    if env:
        return env
    for candidate in prefer:
        found = shutil.which(candidate)
        if found:
            return found
    return prefer[0]


def _rebuild_spec():
    source = ENGINE / "primal_vendor" / "rebuild.cpp"
    flags = ("-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic", "-fPIC", "-shared", "-Werror")
    if os.environ.get("NG_BB_SANITIZE") == "1":
        flags += ("-g", "-fsanitize=address,undefined", "-fno-sanitize-recover=all")
    fingerprint = sha256(source.read_bytes() + repr(flags).encode()).hexdigest()
    target = BUILD / f"rebuild-{fingerprint[:20]}.so"
    return source, list(flags), target, ("clang++", "c++")


def _price_spec():
    source = ENGINE / "upper_src" / "price.cpp"
    flags = ["-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared"]
    if os.environ.get("NG_BB_SANITIZE") == "1":
        flags += ["-g", "-pedantic", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    digest = sha256(source.read_bytes()).hexdigest()
    key = sha256((digest + repr(flags)).encode()).hexdigest()[:16]
    target = BUILD / f"price-{key}.so"
    return source, flags, target, ("c++", "clang++")


def require_native():
    """Verify both multi-vehicle native libraries are built and loadable."""
    try:
        for spec in (_rebuild_spec, _price_spec):
            _, _, target, _ = spec()
            if not target.is_file():
                raise OSError(f"Missing native library {target.name}")
            ctypes.CDLL(str(target))
    except OSError as error:
        raise RuntimeError(
            f"Run python3 multi_vehicle/build_native.py before running multi-vehicle v2.1 ({error})"
        ) from error


def build_native():
    if sys.platform not in ("darwin", "linux"):
        raise ValueError("Building multi-vehicle v2.1 native code requires macOS or Linux")
    BUILD.mkdir(parents=True, exist_ok=True)
    built = []
    for spec in (_rebuild_spec, _price_spec):
        source, flags, target, prefer = spec()
        if not target.is_file():
            compiler = _compiler(prefer)
            with tempfile.TemporaryDirectory(dir=BUILD, prefix=".compile-") as tmp:
                artifact = Path(tmp) / target.name
                subprocess.run([compiler, *flags, str(source), "-o", str(artifact)], check=True)
                ctypes.CDLL(str(artifact))
                os.replace(artifact, target)
        else:
            ctypes.CDLL(str(target))
        built.append(target.name)
    return built


if __name__ == "__main__":
    try:
        names = build_native()
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(f"Multi-vehicle v2.1 native build failed: {error}")
    print(f"Built multi-vehicle v2.1 native libraries in {BUILD}: {', '.join(names)}")
