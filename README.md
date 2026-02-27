<p align="center">
  <img src="https://img.shields.io/badge/REDFORGE-v0.1.0-ff4040?style=for-the-badge&labelColor=0a0a0f" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/tests-73%20passed-40ff70?style=for-the-badge" alt="tests"/>
  <img src="https://img.shields.io/badge/HarmBench%20ASR-86%25-ff4040?style=for-the-badge" alt="harmbench"/>
  <img src="https://img.shields.io/badge/cost-%240%20(Groq%20Free)-40ccff?style=for-the-badge" alt="cost"/>
</p>

<h1 align="center">REDFORGE</h1>
<h3 align="center">Autonomous AI Red-Teaming System</h3>
<p align="center"><i>Hierarchical RL + Multi-Agent Framework for LLM Security Assessment</i></p>
<p align="center"><b>AMD Slingshot Competition 2026</b></p>

---

## What is REDFORGE?

REDFORGE is a fully autonomous red-teaming system that discovers vulnerabilities in Large Language Models without human intervention. It combines:

- **7 specialist attack agents** that each target different vulnerability classes
- **Hierarchical Reinforcement Learning** that learns which attacks work best
- **Game-theoretic analysis** (Nash Equilibrium) to find optimal attack paths
- **A 3-layer judge ensemble** that verifies exploits with high confidence
- **Real-time visualization** of attacks as they happen

```
Target LLM ←── REDFORGE Orchestrator
                    ├── Jailbreak Agent
                    ├── Prompt Injection Agent
                    ├── Data Exfiltration Agent
                    ├── Tool Abuse Agent
                    ├── Memory Poisoning Agent
                    ├── Social Engineering Agent
                    └── Cross-Agent Injection Agent
                            │
                    RL Policy (PPO) ←→ Nash Equilibrium
                            │
                    Judge Ensemble → Verified Exploits → Report
```

## Key Results

| Metric | Value |
|--------|-------|
| Mock Target Exploits | **8/8** (all vulnerabilities found) |
| HarmBench ASR (50 behaviors) | **86%** (43/50 successful) |
| Risk Score | **10.0 / 10.0** |
| OWASP LLM Top 10 Coverage | **50%** (5/10 categories tested) |
| Control Test | **PASSED** (refuses direct harmful requests) |
| Cost | **$0** (Groq free tier) |
| Unit Tests | **73/73 passing** |

## Architecture

```
redforge/
├── agents/            # 7 specialist attack agents + orchestrator
├── benchmarks/        # Mock target server + HarmBench evaluation
├── core/              # Engagement manager, ASO state, MDP
├── demo/              # Live web dashboard (FastAPI + WebSocket)
├── digest/            # Strategic attack digest generation
├── graph/             # Attack graph, Nash equilibrium, path analysis
├── judge/             # 3-layer ensemble (classifier + LLM + human)
├── llm/               # Groq provider, multi-key rotation
├── reporting/         # Risk scoring + executive report generation
├── rl/                # PPO trainer, high/low policies, replay buffer
├── safety/            # Kill switch, sandboxing, SHA-256 audit trail
├── strategy_library/  # ChromaDB vector store for attack strategies
├── target/            # Target connector (OpenAI, Anthropic, custom)
└── tests/             # 73 unit tests
```

### Core Components

**Multi-Agent Attack System** — Seven specialist agents, each targeting a different OWASP LLM Top 10 vulnerability class. The orchestrator coordinates them based on RL policy decisions.

**Hierarchical RL** — A 67-dimensional state space feeds into a high-level policy (which agent to deploy) and low-level policies (how to craft the attack). Trained via PPO with shaped rewards.

**Nash Equilibrium Analysis** — Models the red-team engagement as a two-player zero-sum game. Computes mixed-strategy Nash equilibria to find attack paths that are optimal even against an adaptive defender.

**Judge Ensemble** — Three verification layers: (1) BART-MNLI zero-shot classifier for fast screening, (2) LLM-based rubric judge for detailed analysis, (3) human escalation queue for edge cases.

**Safety Controls** — SHA-256 hash-chained audit log, real-time kill switch (budget, severity, ethics triggers), and sandboxed execution.

## Quick Start

### 1. Install dependencies

```bash
pip install -r redforge/requirements.txt
```

### 2. Run the live demo (no API keys needed)

```bash
python -m redforge.demo.app
```

Open **http://localhost:8765** in your browser and click **Launch Attack**. Watch REDFORGE find all 8 vulnerabilities in the mock target in real time.

### 3. Run the mock benchmark (CLI)

```bash
python -m redforge.benchmarks.run_benchmark --self-hosted
```

### 4. Run HarmBench evaluation (needs Groq keys)

```bash
# Copy and fill in your Groq API keys
cp redforge/.env.example redforge/.env

# Run on 50 behaviors (free, ~5 minutes)
python -m redforge.main harmbench --subset 50
```

### 5. Run tests

```bash
python -m pytest redforge/tests/ -v
```

## Live Dashboard

The real-time web dashboard shows attacks happening live via WebSocket:

- **Live Attack Feed** — Each prompt sent, response received, and verdict (EXPLOITED/BLOCKED)
- **Metrics** — Exploits found, risk score, OWASP coverage, Nash equilibrium value
- **Scenario Results** — Table updating in real time as each attack completes
- **OWASP Coverage Bars** — Visual breakdown of which vulnerability categories were hit

```bash
python -m redforge.demo.app --port 8765
```

## CLI Reference

```bash
# Full engagement against a target
python -m redforge.main engage \
  --target-url https://api.example.com/v1/chat/completions \
  --target-model gpt-4 \
  --target-type openai \
  --depth full \
  --budget 100

# Check engagement status
python -m redforge.main status --engagement-id <id>

# Manage strategy library
python -m redforge.main library --list-strategies

# Train RL policy
python -m redforge.main train --episodes 1000

# HarmBench benchmark
python -m redforge.main harmbench --subset 50
```

## Groq Integration (Free Tier)

REDFORGE uses Groq's free API for zero-cost operation:

- **Multi-key rotation** — Distribute rate limits across up to 10 API keys
- **Named key pools** — Separate pools for attacker/judge vs. target to avoid interference
- **Auto rate-limit handling** — Round-robin rotation with backoff on 429 errors
- **Models used**: `llama-3.1-8b-instant` (attacker/target), `llama-3.3-70b-versatile` (judge)

```bash
# In your .env file:
GROQ_API_KEY_1=gsk_your-key-1
GROQ_API_KEY_2=gsk_your-key-2
GROQ_API_KEY_3=gsk_your-key-3
```

## Attack Categories

| Agent | OWASP Category | Technique |
|-------|---------------|-----------|
| Jailbreak | LLM01 Prompt Injection | Roleplay framing, fictional scenarios |
| Prompt Injection | LLM01 Prompt Injection | Authority impersonation, base64 encoding |
| Data Exfiltration | LLM06 Sensitive Info | Indirect extraction, encoding tricks |
| Tool Abuse | LLM07 Insecure Plugin | Path traversal, command injection |
| Memory Poisoning | LLM03 Training Data | Persistent context manipulation |
| Social Engineering | LLM09 Overreliance | Multi-turn gradual escalation |
| Cross-Agent | LLM08 Excessive Agency | Inter-agent prompt injection |

## Safety & Ethics

REDFORGE is built for **authorized security testing only**.

- All engagements require explicit authorization scope configuration
- SHA-256 hash-chained audit trail for complete traceability
- Real-time kill switch with configurable triggers (budget, severity, ethics)
- Sandboxed execution environment
- Human escalation queue for ambiguous results

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| RL Training | PyTorch + Stable-Baselines3 |
| Graph Analysis | NetworkX + SciPy |
| Strategy Store | ChromaDB |
| LLM Provider | Groq (OpenAI-compatible) |
| Live Dashboard | FastAPI + WebSocket |
| Testing | Pytest (73 tests) |

## License

This project was built for the AMD Slingshot Competition 2026.
