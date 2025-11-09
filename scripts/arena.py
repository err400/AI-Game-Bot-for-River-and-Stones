#!/usr/bin/env python3
"""
Arena runner for River & Stones.

Builds agents as needed, starts the web server, and runs many games across
board sizes against multiple opponents:

  - greedy:    main student_cpp vs greedy_river (both side assignments)
  - random:    main student_cpp vs random (both side assignments)
  - duplicate: main student_cpp vs duplicate_folder (Python student by default; optional C++)
  - self:      main student_cpp vs main student_cpp

Examples:
  python scripts/arena.py --games-per-size 10 --sizes small medium large --matchups greedy random duplicate self --log

By default uses port 8080 and writes logs to logs/value_data.jsonl (when --log).
"""
import argparse
import os
import sys
import time
import json
import shlex
import signal
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict

try:
    # Prefer stdlib; avoid extra deps
    import urllib.request as _url
except Exception:
    _url = None


REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable or "python3"


def run(cmd: List[str], cwd: Path | None = None, env: Dict[str, str] | None = None, check: bool = True) -> int:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(map(shlex.quote, cmd))}")
    return p.returncode


def build_main_student_cpp():
    print("[build] main student_cpp via cmake-rebuild …")
    run(["make", "cmake-rebuild"], cwd=REPO_ROOT)


def build_greedy_if_needed():
    so_candidates = list((REPO_ROOT / "build_greedy").glob("greedy_river_module*.so"))
    if so_candidates:
        print(f"[build] greedy module present: {so_candidates[0].name}")
        return
    print("[build] building greedy_river_module in build_greedy/ …")
    # Configure and build with CMake
    run(["cmake", "-S", ".", "-B", "build_greedy", "-DCMAKE_BUILD_TYPE=Release", "-DPYBIND11_FINDPYTHON=ON"], cwd=REPO_ROOT)
    run(["cmake", "--build", "build_greedy", "-j"], cwd=REPO_ROOT)


def build_duplicate_student_cpp_if_needed():
    # Only needed if using duplicate C++ agent
    print("[build] duplicate_folder student_cpp via cmake-rebuild …")
    run(["make", "cmake-rebuild"], cwd=REPO_ROOT / "duplicate_folder")


def start_server(port: int) -> subprocess.Popen:
    env = os.environ.copy()
    cmd = [PY, str(REPO_ROOT / "client_server" / "web_server.py"), str(port)]
    print(f"[srv] starting web server on :{port} …")
    p = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Give it a moment to bind
    time.sleep(1.0)
    return p


def stop_process(p: subprocess.Popen):
    if p and p.poll() is None:
        try:
            if sys.platform.startswith('win'):
                p.terminate()
            else:
                p.send_signal(signal.SIGINT)
                time.sleep(0.2)
                if p.poll() is None:
                    p.terminate()
                    time.sleep(0.2)
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


def http_create_game(port: int, board_size: str) -> None:
    url = f"http://localhost:{port}/api/create_game"
    data = json.dumps({"board_size": board_size}).encode("utf-8")
    if _url is None:
        # Last-resort: call curl
        run(["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", data.decode("utf-8")], check=False)
        return
    req = _url.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with _url.urlopen(req, timeout=5) as _:
            return
    except Exception as e:
        print(f"[warn] create_game failed ({e}); retrying after short delay …")
        time.sleep(0.5)
        # one more try
        try:
            with _url.urlopen(req, timeout=5) as _:
                return
        except Exception as e2:
            print(f"[error] create_game failed again: {e2}")


def spawn_bot(cmd: List[str], env_overrides: Dict[str, str] | None = None) -> subprocess.Popen:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    return subprocess.Popen(cmd, cwd=str(REPO_ROOT))


def run_one_game(port: int,
                 board_size: str,
                 circle_cmd: List[str],
                 square_cmd: List[str],
                 circle_env: Dict[str, str] | None,
                 square_env: Dict[str, str] | None,
                 timeout_sec: int) -> bool:
    # New game
    http_create_game(port, board_size)
    # Start bots
    pc = spawn_bot(circle_cmd, circle_env)
    ps = spawn_bot(square_cmd, square_env)
    start = time.time()
    ok = True
    try:
        while True:
            if pc.poll() is not None and ps.poll() is not None:
                break
            if time.time() - start > timeout_sec:
                ok = False
                print("[warn] game timeout; terminating bots …")
                break
            time.sleep(0.5)
    finally:
        stop_process(pc)
        stop_process(ps)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Run many games vs multiple opponents across sizes")
    ap.add_argument("--games-per-size", type=int, default=5, help="Games per matchup per board size")
    ap.add_argument("--sizes", nargs="*", default=["small", "medium", "large"], choices=["small", "medium", "large"], help="Board sizes to play")
    ap.add_argument("--matchups", nargs="*", default=["greedy", "random", "duplicate", "self"], choices=["greedy", "random", "duplicate", "self"], help="Which opponent sets to run")
    ap.add_argument("--use-duplicate-cpp", action="store_true", help="Use duplicate_folder C++ agent instead of Python student for 'duplicate' matchup (builds duplicate)")
    ap.add_argument("--log", action="store_true", help="Enable feature logging for main student_cpp bots (RS_LOG_VALUE=1)")
    ap.add_argument("--log-path", default=str(REPO_ROOT / "logs" / "value_data.jsonl"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=int, default=300, help="Per-game timeout in seconds")
    ap.add_argument("--root-noise", type=float, default=None, help="Mix root priors with uniform by this epsilon (RS_ROOT_NOISE)")
    ap.add_argument("--c-puct", type=float, default=None, help="Override c_puct (RS_C_PUCT)")
    ap.add_argument("--k0", type=int, default=None, help="Override progressive widening K0 (RS_K0)")
    ap.add_argument("--k1", type=float, default=None, help="Override progressive widening K1 (RS_K1)")
    ap.add_argument("--adversarial-circle", action="store_true", help="Make circle student_cpp select adversarially at root")
    ap.add_argument("--adversarial-square", action="store_true", help="Make square student_cpp select adversarially at root")
    ap.add_argument("--circle-module-path", default="", help="Override module path for circle student (snapshot dir with student_agent_module*.so)")
    ap.add_argument("--square-module-path", default="", help="Override module path for square student (snapshot dir with student_agent_module*.so)")
    args = ap.parse_args()

    # Builds
    build_main_student_cpp()
    if "greedy" in args.matchups:
        build_greedy_if_needed()
    if args.use_duplicate_cpp and "duplicate" in args.matchups:
        build_duplicate_student_cpp_if_needed()

    # Ensure logs dir exists if logging
    if args.log:
        Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)

    # Start server
    srv = start_server(args.port)
    try:
        # Prepare command templates
        def main_bot(side: str, strategy: str) -> List[str]:
            return [PY, str(REPO_ROOT / "client_server" / "bot_client.py"), side, str(args.port), "--strategy", strategy]

        def dup_bot(side: str, strategy: str) -> List[str]:
            return [PY, str(REPO_ROOT / "duplicate_folder" / "client_server" / "bot_client.py"), side, str(args.port), "--strategy", strategy]

        # Per-process env overrides for logging
        def env_for_side(strategy: str, enable_log: bool, side_label: str) -> Dict[str, str] | None:
            env: Dict[str, str] = {}
            if enable_log:
                env["RS_LOG_VALUE"] = "1"
                env["RS_LOG_PATH"] = args.log_path
            # Root noise only relevant for student_cpp
            if args.root_noise is not None and strategy == "student_cpp":
                env["RS_ROOT_NOISE"] = str(args.root_noise)
            # Search overrides
            if args.c_puct is not None and strategy == "student_cpp":
                env["RS_C_PUCT"] = str(args.c_puct)
            if args.k0 is not None and strategy == "student_cpp":
                env["RS_K0"] = str(args.k0)
            if args.k1 is not None and strategy == "student_cpp":
                env["RS_K1"] = str(args.k1)
            # Adversarial per side
            if strategy == "student_cpp":
                if side_label == 'circle' and args.adversarial_circle:
                    env["RS_ADVERSARIAL"] = "1"
                if side_label == 'square' and args.adversarial_square:
                    env["RS_ADVERSARIAL"] = "1"
            # Module path overrides (snapshots)
            if strategy == "student_cpp":
                if side_label == 'circle' and args.circle_module_path:
                    env["RS_STUDENT_MODULE_PATH"] = args.circle_module_path
                if side_label == 'square' and args.square_module_path:
                    env["RS_STUDENT_MODULE_PATH"] = args.square_module_path
            return env if env else None

        # Scenarios
        scenarios: List[Tuple[str, List[str], List[str], bool, bool]] = []
        # tuple: (label, circle_cmd, square_cmd, circle_logs, square_logs)

        if "greedy" in args.matchups:
            scenarios.append(("mcts(circle) vs greedy(square)", main_bot("circle", "student_cpp"), main_bot("square", "greedy_river"), True, False))
            scenarios.append(("greedy(circle) vs mcts(square)", main_bot("circle", "greedy_river"), main_bot("square", "student_cpp"), False, True))

        if "random" in args.matchups:
            scenarios.append(("mcts(circle) vs random(square)", main_bot("circle", "student_cpp"), main_bot("square", "random"), True, False))
            scenarios.append(("random(circle) vs mcts(square)", main_bot("circle", "random"), main_bot("square", "student_cpp"), False, True))

        if "duplicate" in args.matchups:
            if args.use_duplicate_cpp:
                scenarios.append(("mcts(circle) vs duplicate_cpp(square)", main_bot("circle", "student_cpp"), dup_bot("square", "student_cpp"), True, False))
                scenarios.append(("duplicate_cpp(circle) vs mcts(square)", dup_bot("circle", "student_cpp"), main_bot("square", "student_cpp"), False, True))
            else:
                scenarios.append(("mcts(circle) vs duplicate_py(square)", main_bot("circle", "student_cpp"), dup_bot("square", "student"), True, False))
                scenarios.append(("duplicate_py(circle) vs mcts(square)", dup_bot("circle", "student"), main_bot("square", "student_cpp"), False, True))

        if "self" in args.matchups:
            scenarios.append(("self: mcts(circle) vs mcts(square)", main_bot("circle", "student_cpp"), main_bot("square", "student_cpp"), True, True))

        # Run
        total = 0
        for size in args.sizes:
            for label, ccmd, scmd, clog, slog in scenarios:
                # Split games evenly between the two side-assignments by design (each scenario is one assignment)
                print(f"\n=== {size} | {label} ===")
                for gi in range(1, args.games_per_size + 1):
                    print(f"[run] game {gi}/{args.games_per_size} …")
                    ok = run_one_game(
                        port=args.port,
                        board_size=size,
                        circle_cmd=ccmd,
                        square_cmd=scmd,
                        circle_env=env_for_side(ccmd[-1], args.log and clog, 'circle'),
                        square_env=env_for_side(scmd[-1], args.log and slog, 'square'),
                        timeout_sec=args.timeout,
                    )
                    total += 1
                    if not ok:
                        print("[warn] game ended due to timeout")
                    time.sleep(0.5)
        print(f"\nDone. Games attempted: {total}")
    finally:
        stop_process(srv)


if __name__ == "__main__":
    main()
