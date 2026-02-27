"""REDFORGE Demo Runner - AMD Slingshot Competition.

Launches the live attack dashboard (FastAPI + WebSocket).

Usage:
  python -m redforge.demo.run_demo          # Launch live dashboard
  python -m redforge.demo.run_demo --port 9000  # Custom port
"""

import sys
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def launch_dashboard(port: int = 8765):
    """Launch the live FastAPI dashboard."""
    print("\n" + "=" * 60)
    print("  LAUNCHING REDFORGE LIVE DASHBOARD")
    print("=" * 60)
    print(f"  Opening browser at http://localhost:{port}")
    print("  Click 'Launch Attack' in the dashboard to start")
    print("  Press Ctrl+C to stop\n")

    subprocess.run(
        [sys.executable, "-m", "redforge.demo.app", "--port", str(port)],
        cwd=str(ROOT),
    )


def main():
    parser = argparse.ArgumentParser(description="REDFORGE Demo Runner")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port")
    args = parser.parse_args()

    print("""
    ====================================================
           REDFORGE - LIVE DEMO
           AMD Slingshot Competition 2026
    ====================================================
    """)

    launch_dashboard(args.port)


if __name__ == "__main__":
    main()
