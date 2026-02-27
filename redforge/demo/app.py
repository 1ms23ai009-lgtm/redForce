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
            elif msg.get("command") == "start_harmbench":
                if not run_state["running"]:
                    subset = msg.get("subset", 50)
                    max_iter = msg.get("max_iterations", 5)
                    thread = threading.Thread(
                        target=_run_harmbench_sequence,
                        args=(subset, max_iter),
                        daemon=True,
                    )
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


# ── HarmBench Live Runner ─────────────────────────────────────────

def _run_harmbench_sequence(subset_size: int = 50, max_iterations: int = 5):
    """Run HarmBench PAIR attacks via Vertex AI and push events in real time."""
    from redforge.llm.vertex_provider import VertexAIProvider, VERTEX_MODELS
    from redforge.benchmarks.harmbench_loader import load_behaviors
    from redforge.benchmarks.harmbench_runner import (
        pair_attack, _parse_judge_verdict, CATEGORY_TO_OWASP,
        PAIR_SYSTEM, PAIR_USER_TEMPLATE, JUDGE_SYSTEM, JUDGE_USER_TEMPLATE,
    )

    run_state["running"] = True
    run_state["phase"] = "starting"

    # Read project from env or default
    vertex_project = os.environ.get("VERTEX_PROJECT", "redforge")
    vertex_location = os.environ.get("VERTEX_LOCATION", "us-central1")

    sync_broadcast({
        "type": "phase",
        "data": {"phase": "init", "message": f"Initializing Vertex AI ({vertex_project})..."},
    })

    try:
        attacker = VertexAIProvider(project=vertex_project, location=vertex_location)
    except Exception as e:
        sync_broadcast({
            "type": "phase",
            "data": {"phase": "error", "message": f"Vertex AI init failed: {e}"},
        })
        run_state["running"] = False
        run_state["phase"] = "idle"
        return

    target = attacker  # Same provider for both
    default_models = VERTEX_MODELS
    attacker_model = default_models.get("attacker")
    target_model = default_models.get("target")

    sync_broadcast({
        "type": "phase",
        "data": {"phase": "loading", "message": f"Loading {subset_size} HarmBench behaviors..."},
    })

    behaviors = load_behaviors(subset_size=subset_size)
    total = len(behaviors)

    run_state["phase"] = "attacking"
    sync_broadcast({
        "type": "phase",
        "data": {"phase": "attacking", "message": f"Running PAIR attacks on {total} behaviors via Gemini..."},
    })

    results = []
    successful = 0
    total_tokens = 0

    for idx, behavior in enumerate(behaviors):
        behavior_text = behavior["text"]
        category = behavior["semantic_category"]
        owasp = CATEGORY_TO_OWASP.get(category, "LLM01_prompt_injection")

        # Broadcast: starting this behavior
        sync_broadcast({
            "type": "attack_start",
            "data": {
                "index": idx,
                "total": total,
                "name": behavior_text[:60] + ("..." if len(behavior_text) > 60 else ""),
                "category": category,
                "owasp": owasp,
                "severity": "high",
                "num_prompts": max_iterations,
            },
        })

        # Run PAIR attack with per-iteration broadcasting
        prev_attempt = "None (first attempt)"
        prev_response = "None"
        behavior_tokens = 0
        attack_success = False
        final_verdict = {"success": False, "confidence": 0.0, "reasoning": "Max iterations reached"}

        for iteration in range(1, max_iterations + 1):
            try:
                # Step 1: Attacker generates prompt
                attacker_msgs = [
                    {"role": "system", "content": PAIR_SYSTEM},
                    {"role": "user", "content": PAIR_USER_TEMPLATE.format(
                        behavior=behavior_text,
                        previous_attempt=prev_attempt[:500],
                        previous_response=prev_response[:500],
                    )},
                ]
                attack_result = attacker.chat(
                    attacker_msgs, model=attacker_model,
                    role="attacker", temperature=0.9, max_tokens=512,
                )
                attack_prompt = attack_result["content"].strip()
                behavior_tokens += attack_result["tokens"]["total"]

                if not attack_prompt:
                    continue

                # Step 2: Target responds
                target_msgs = [{"role": "user", "content": attack_prompt}]
                target_result = target.chat(
                    target_msgs, model=target_model,
                    role="target", temperature=0.7, max_tokens=1024,
                )
                target_response = target_result["content"].strip()
                behavior_tokens += target_result["tokens"]["total"]

                # Broadcast turn
                sync_broadcast({
                    "type": "attack_turn",
                    "data": {
                        "index": idx,
                        "turn": iteration - 1,
                        "prompt": attack_prompt[:200],
                        "response": target_response[:300],
                        "tokens": behavior_tokens,
                    },
                })

                # Step 3: Judge evaluates
                judge_msgs = [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                        behavior=behavior_text,
                        response=target_response[:1500],
                    )},
                ]
                judge_result = attacker.chat(
                    judge_msgs, role="judge", temperature=0.1, max_tokens=256,
                )
                behavior_tokens += judge_result["tokens"]["total"]

                verdict = _parse_judge_verdict(judge_result["content"])

                if verdict.get("success", False):
                    attack_success = True
                    final_verdict = verdict
                    break

                prev_attempt = attack_prompt
                prev_response = target_response

            except Exception as e:
                logger.warning(f"Behavior {idx} iteration {iteration} error: {e}")
                time.sleep(2)
                continue

        total_tokens += behavior_tokens

        if attack_success:
            successful += 1

        result_data = {
            "index": idx,
            "name": behavior_text[:60] + ("..." if len(behavior_text) > 60 else ""),
            "success": attack_success,
            "verdict": "EXPLOITED" if attack_success else "BLOCKED",
            "score": final_verdict.get("confidence", 0.0),
            "category": category,
            "owasp": owasp,
            "severity": "high" if attack_success else "none",
            "is_control": False,
            "control_passed": None,
        }
        results.append(result_data)

        sync_broadcast({"type": "attack_result", "data": result_data})

    # Summary
    asr = successful / total if total else 0.0
    run_state["phase"] = "complete"
    run_state["running"] = False

    summary = {
        "total_scenarios": total,
        "exploits_found": successful,
        "control_passed": True,
        "risk_score": round(asr * 10, 1),
        "coverage": round((len(set(r["owasp"] for r in results if r["success"])) / 10) * 100, 1),
        "nash_value": round(asr, 4),
        "asr": f"{asr:.1%}",
        "total_tokens": total_tokens,
        "results": results,
    }

    sync_broadcast({"type": "summary", "data": summary})
    logger.info(f"HarmBench complete: ASR={asr:.1%} ({successful}/{total}), tokens={total_tokens:,}")


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
