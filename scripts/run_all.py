"""Reproduce every result in report/ from cached data (no network needed).

  python scripts/run_all.py            # cached data
  python scripts/run_all.py --refresh  # re-download SPY chain and SPY/VIX/IRX history first
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
refresh = ["--refresh"] if "--refresh" in sys.argv else []
steps = [
    ("Part 1: pricing engine", ["run_pricing.py"]),
    ("Part 2: implied volatility", ["run_iv.py", *refresh]),
    ("Part 3.1: hedging simulation", ["run_hedging_sim.py"]),
    ("Part 3.2: historical VRP study", ["run_historical.py", *refresh]),
]
for title, (script, *args) in steps:
    print(f"\n=== {title} ===", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *args], check=True)
