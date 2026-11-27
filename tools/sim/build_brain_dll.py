"""Build script for the pure C++17 real-car brain DLL (tools/sim/build/sim_brain.dll).

Compiles tools/sim/cpp_brain_bridge.cpp together with the 7 pure C++17 zero-ROS2/zero-OpenCV
modules from src/line_follow_ws/src/line_follower/src/ using g++ with -Wall -Wextra -Werror.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


SIM_DIR = Path(__file__).resolve().parent
REPO_ROOT = SIM_DIR.parent.parent
LINE_FOLLOWER_DIR = REPO_ROOT / "src" / "line_follow_ws" / "src" / "line_follower"
INCLUDE_DIR = LINE_FOLLOWER_DIR / "include"
SRC_DIR = LINE_FOLLOWER_DIR / "src"

PURE_MODULE_NAMES = [
    "ground_projector.cpp",
    "obstacle_memory.cpp",
    "ackermann_cbf_shield.cpp",
    "corridor_planner.cpp",
    "finish_guard.cpp",
    "track_state_machine.cpp",
    "motion_controller.cpp",
]

BRIDGE_CPP = SIM_DIR / "cpp_brain_bridge.cpp"
BUILD_DIR = SIM_DIR / "build"
DLL_NAME = "sim_brain.dll" if os.name == "nt" else "libsim_brain.so"
DLL_PATH = BUILD_DIR / DLL_NAME


def get_source_files() -> List[Path]:
    sources = [BRIDGE_CPP] + [SRC_DIR / name for name in PURE_MODULE_NAMES]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(f"Missing required C++ source file: {path}")
    return sources


def get_header_files() -> List[Path]:
    header_dir = INCLUDE_DIR / "line_follower"
    if not header_dir.is_dir():
        raise FileNotFoundError(f"Missing C++ header directory: {header_dir}")
    return sorted(header_dir.glob("*.hpp"))


def is_dll_stale(dll_path: Path = DLL_PATH) -> bool:
    if not dll_path.is_file():
        return True
    dll_mtime = dll_path.stat().st_mtime
    for dep in get_source_files() + get_header_files() + [Path(__file__).resolve()]:
        if dep.stat().st_mtime > dll_mtime:
            return True
    return False


def build_dll(force: bool = False, verbose: bool = True) -> Path:
    if not force and not is_dll_stale(DLL_PATH):
        if verbose:
            print(f"[build_brain_dll] Up-to-date: {DLL_PATH}")
        return DLL_PATH

    gxx = shutil.which("g++")
    if not gxx:
        raise RuntimeError(
            "g++ compiler not found on PATH. Install MinGW-w64 g++ and ensure it is on PATH."
        )

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    sources = get_source_files()

    cmd: List[str] = [
        gxx,
        "-std=c++17",
        "-O2",
        "-shared",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-DLINE_FOLLOWER_NO_OPENCV",
        f"-I{INCLUDE_DIR}",
    ]
    if os.name == "nt":
        cmd.extend(["-static-libgcc", "-static-libstdc++"])
    else:
        cmd.append("-fPIC")

    cmd.extend([str(src) for src in sources])
    cmd.extend(["-o", str(DLL_PATH)])

    if verbose:
        print(f"[build_brain_dll] Compiling {DLL_PATH.name} ({len(sources)} TUs)...")

    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        msg = (
            f"g++ build failed (exit {proc.returncode}):\n"
            f"CMD: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
        raise RuntimeError(msg)

    if verbose:
        if proc.stderr.strip():
            print(proc.stderr.strip())
        print(f"[build_brain_dll] Built successfully: {DLL_PATH}")
    return DLL_PATH


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build tools/sim/build/sim_brain.dll")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if up-to-date")
    parser.add_argument("--quiet", action="store_true", help="Suppress status output")
    args = parser.parse_args(argv)
    build_dll(force=args.force, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
