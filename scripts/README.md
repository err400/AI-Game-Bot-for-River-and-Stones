Arena (batch matches)
=====================

Run many games across board sizes vs multiple opponents and optionally log features for training.

Quick start
-----------
- Build and run 5 games per size vs greedy, random, duplicate (Python), and self-play, enabling logging:

  python scripts/arena.py --games-per-size 5 --sizes small medium large --matchups greedy random duplicate self --log

Common options
--------------
- Use duplicate C++ agent instead of Python student:

  python scripts/arena.py --matchups duplicate --use-duplicate-cpp

- Change port (default 8080):

  python scripts/arena.py --port 8080

- Change per-game timeout (seconds, default 300):

  python scripts/arena.py --timeout 240

- Change log path (when --log is set):

  python scripts/arena.py --log --log-path logs/value_data.jsonl

Notes
-----
- The script builds main student_cpp. It builds the greedy module on demand. It builds the duplicate C++ agent only when `--use-duplicate-cpp` is set.
- Logging is enabled only for main (repo root) student_cpp bots, which support `RS_LOG_VALUE`/`RS_LOG_PATH`.
- Games are run sequentially under a single server instance; you can watch at http://localhost:8080.

