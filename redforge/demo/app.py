"""REDFORGE Live Dashboard — FastAPI + WebSocket real-time attack viewer.

Replaces the Streamlit dashboard with a proper web app that shows
attacks happening live via WebSocket push.

Usage:
  python -m redforge.demo.app                # Runs on http://localhost:8765
  python -m redforge.demo.app --port 9000    # Custom port
"""

import os
import sys
import json
import asyncio
import logging
import argparse
import threading
import time
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("redforge.demo")

app = FastAPI(title="REDFORGE Live Dashboard")

# Global state
connected_clients: list[WebSocket] = []
attack_log: list[dict] = []
run_state = {"running": False, "phase": "idle"}

STATIC_DIR = Path(__file__).parent / "static"


# ── WebSocket broadcast ──────────────────────────────────────────────

async def broadcast(event: dict):
    """Push an event to all connected WebSocket clients."""
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


def sync_broadcast(event: dict):
    """Broadcast from sync code (the attack runner thread)."""
    attack_log.append(event)
    loop = _get_loop()
    if loop and loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(event), loop)


_loop_ref = None

def _set_loop(loop):
    global _loop_ref
    _loop_ref = loop

def _get_loop():
    return _loop_ref


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the main dashboard page."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint — sends live attack events to the browser."""
    await ws.accept()
    connected_clients.append(ws)
    logger.info(f"Client connected ({len(connected_clients)} total)")

    # Send current state (replay existing events so late-joining clients catch up)
    for event in attack_log:
        await ws.send_json(event)

    # Send run state
    await ws.send_json({"type": "state", "data": run_state})

    try:
        while True:
            # Keep connection alive; handle incoming commands
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("command") == "start_attack":
                if not run_state["running"]:
                    # Start attack in background thread
                    thread = threading.Thread(target=_run_attack_sequence, daemon=True)
                    thread.start()
            elif msg.get("command") == "reset":
                attack_log.clear()
                run_state["running"] = False
                run_state["phase"] = "idle"
                await broadcast({"type": "reset"})
    except WebSocketDisconnect:
        connected_clients.remove(ws)
        logger.info(f"Client disconnected ({len(connected_clients)} total)")


@app.get("/api/reports")
async def get_reports():
    """Return list of existing benchmark reports."""
    reports_dir = ROOT / "reports"
    if not reports_dir.exists():
        return {"reports": []}
    files = sorted(reports_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files[:10]:
        try:
            data = json.loads(f.read_text())
            result.append({"filename": f.name, "summary": _extract_summary(data)})
        except Exception:
            pass
    return {"reports": result}


def _extract_summary(report: dict) -> dict:
    """Pull key metrics from a report JSON."""
    return {
        "engagement_id": report.get("engagement_id", "unknown")[:8],
        "exploits": len(report.get("confirmed_exploits", [])),
        "risk_score": report.get("risk_assessment", {}).get("composite_risk_score", 0),
        "coverage": report.get("coverage_summary", {}).get("overall_percentage", 0),
        "timestamp": report.get("report_metadata", {}).get("generated_at", ""),
    }


# ── Live Attack Runner ────────────────────────────────────────────────

def _run_attack_sequence():
    """Execute attacks against mock target and push events in real time."""
    from redforge.benchmarks.mock_target import app as mock_app
    from redforge.benchmarks.run_benchmark import (
        ATTACK_SCENARIOS, send_to_target, check_success,
    )
    from redforge.core.state import create_initial_aso
    from redforge.core.mdp import update_coverage_state, compute_coverage_percentage
    from redforge.graph.attack_graph import create_attack_graph, add_node, add_edge, mark_node_vulnerable, serialize_graph
    from redforge.graph.nash import compute_nash_equilibrium
    from redforge.reporting.risk_scorer import compute_composite_risk_score

    run_state["running"] = True
    run_state["phase"] = "starting"

    # Start mock server if not already running
    import httpx
    try:
        r = httpx.get("http://127.0.0.1:8899/health", timeout=2.0)
        server_was_running = True
    except Exception:
        server_was_running = False
        import uvicorn
        def run_server():
            uvicorn.run(mock_app, host="127.0.0.1", port=8899, log_level="error")
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        time.sleep(1.5)

    sync_broadcast({
        "type": "phase",
        "data": {"phase": "mock_server", "message": "Mock target server ready on :8899"},
    })

    # Init ASO
    target_config = {
        "endpoint_url": "http://127.0.0.1:8899/v1/chat/completions",
        "model_name": "mock-vulnerable-llm",
        "target_type": "openai_compatible",
        "api_key_env_var": "NONE",
        "authorization_scope": {"depth": "full", "query_budget": 100},
    }
    aso = create_initial_aso(target_config, query_budget=100)

    run_state["phase"] = "attacking"
    sync_broadcast({
        "type": "phase",
        "data": {"phase": "attacking", "message": f"Running {len(ATTACK_SCENARIOS)} attack scenarios..."},
    })

    results = []
    total_scenarios = len(ATTACK_SCENARIOS)
    turn_offset = 0

    for idx, scenario in enumerate(ATTACK_SCENARIOS):
        name = scenario["name"]
        prompts = scenario["prompts"]
        keywords = scenario["success_keywords"]
        expect_failure = scenario.get("expect_failure", False)
        session_id = f"scenario_{name.replace(' ', '_').lower()}"

        # Broadcast: starting this scenario
        sync_broadcast({
            "type": "attack_start",
            "data": {
                "index": idx,
                "total": total_scenarios,
                "name": name,
                "category": scenario["category"],
                "owasp": scenario["owasp"],
                "severity": scenario["severity"],
                "num_prompts": len(prompts),
            },
        })

        history = []
        all_responses = []

        for i, prompt in enumerate(prompts):
            result = send_to_target(prompt, history, session_id=session_id)
            response = result["response"]
            all_responses.append(response)
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": response})

            # Broadcast each turn
            sync_broadcast({
                "type": "attack_turn",
                "data": {
                    "index": idx,
                    "turn": i,
                    "prompt": prompt,
                    "response": response[:300],
                    "tokens": result.get("tokens_used", 0),
                },
            })
            time.sleep(0.4)  # Pacing for visual effect

        # Judge
        final_response = all_responses[-1] if all_responses else ""
        success, score = check_success(final_response, keywords)

        if expect_failure:
            blocked = not success
            verdict = "BLOCKED" if blocked else "BYPASSED"
            display_success = blocked  # Control passing = success
        else:
            verdict = "EXPLOITED" if success else "BLOCKED"
            display_success = success

        # Update OWASP coverage
        if not expect_failure:
            update_coverage_state(aso, scenario["owasp"],
                                  severity=scenario["severity"] if success else None)

        result_data = {
            "name": name,
            "success": success and not expect_failure,
            "verdict": verdict,
            "score": score,
            "category": scenario["category"],
            "owasp": scenario["owasp"],
            "severity": scenario["severity"] if (success and not expect_failure) else "none",
            "is_control": expect_failure,
            "control_passed": blocked if expect_failure else None,
        }
        results.append(result_data)

        # Broadcast result
        sync_broadcast({
            "type": "attack_result",
            "data": result_data,
        })
        turn_offset += len(prompts)
        time.sleep(0.3)

    # Build attack graph
    run_state["phase"] = "analysis"
    sync_broadcast({
        "type": "phase",
        "data": {"phase": "analysis", "message": "Computing attack graph & Nash equilibrium..."},
    })

    graph = create_attack_graph()
    entry_id = add_node(graph, "user_input", "user_input")
    for r in results:
        node_id = add_node(graph, r["name"], "guardrail",
                           vulnerability=r["success"],
                           vulnerability_severity=r["severity"] if r["success"] else None)
        edge_type = "exploitable_transition" if r["success"] else "constraint"
        add_edge(graph, entry_id, node_id, edge_type, confirmed=r["success"])
        if r["success"]:
            mark_node_vulnerable(graph, node_id, r["severity"])

    nash = compute_nash_equilibrium(graph)
    nash_value = nash.get("equilibrium_value", 0) if nash else 0

    exploits_found = sum(1 for r in results if r["success"])
    control_passed = any(r.get("control_passed") for r in results if r["is_control"])
    coverage = compute_coverage_percentage(aso)

    # Final summary
    summary = {
        "total_scenarios": total_scenarios,
        "exploits_found": exploits_found,
        "control_passed": control_passed,
        "risk_score": round(10.0 * exploits_found / max(1, total_scenarios - 1), 1),
        "coverage": round(coverage, 1),
        "nash_value": round(nash_value, 4),
        "results": results,
    }

    run_state["phase"] = "complete"
    run_state["running"] = False

    sync_broadcast({
        "type": "summary",
        "data": summary,
    })

    logger.info(f"Attack sequence complete: {exploits_found}/{total_scenarios - 1} exploits, Nash={nash_value:.4f}")


# ── Startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    _set_loop(asyncio.get_event_loop())
    logger.info("REDFORGE Dashboard ready — open http://localhost:8765")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import uvicorn
    parser = argparse.ArgumentParser(description="REDFORGE Live Dashboard")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"""
    ====================================================
           REDFORGE — LIVE ATTACK DASHBOARD
           AMD Slingshot Competition 2026
    ====================================================
      Open http://{args.host}:{args.port} in your browser
      Click "Launch Attack" to start real-time demo
    ====================================================
    """)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
