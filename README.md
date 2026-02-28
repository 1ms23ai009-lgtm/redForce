<p align="center">
  <img src="https://img.shields.io/badge/REDFORGE-v2.0.0-ff4040?style=for-the-badge&labelColor=0a0a0f" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/HarmBench%20ASR-99.8%25-ff4040?style=for-the-badge" alt="harmbench"/>
  <img src="https://img.shields.io/badge/LLM%20Provider-Multi--Provider-blueviolet?style=for-the-badge" alt="provider"/>
</p>

<h1 align="center">REDFORGE</h1>
<h3 align="center">Autonomous AI Red-Teaming System</h3>
<p align="center"><i>3-Phase Adaptive Security Assessment for Any Agentic AI Application</i></p>

---

## What is REDFORGE?

REDFORGE is a fully autonomous red-teaming system that discovers vulnerabilities in **any** agentic AI application without human intervention. Point it at a chat API and it automatically:

1. **Probes the target** to discover what it has (tools, RAG, memory, guards, agents)
2. **Selects relevant attacks** based on what it found
3. **Reports findings** with actionable remediation

It adapts to whatever it finds — chatbot, RAG app, tool-using agent, or multi-agent system.

```
┌──────────────────────────────────────────────────────────────┐
│                         REDFORGE v2                          │
│                                                              │
│  Phase 1: RECON          Phase 2: ATTACK        Phase 3: REPORT
│  ─────────────          ──────────────          ──────────────
│  "What does this        "Attack what we        "Here's what's
│   app have?"             found"                 broken & how
│                                                 to fix it"
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ Probe target │    │ 9 specialist      │    │ Risk score │  │
│  │ Discover:    │───►│ attack agents     │───►│ OWASP map  │  │
│  │ - Tools?     │    │ selected by recon │    │ Remediation│  │
│  │ - RAG?       │    │                   │    │ Report     │  │
│  │ - Memory?    │    │ RL picks best     │    │            │  │
│  │ - Guards?    │    │ strategy per      │    │            │  │
│  │ - Agents?    │    │ attack surface    │    │            │  │
│  └──────────────┘    └───────────────────┘    └────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Key Features

- **9 specialist attack agents** targeting OWASP LLM Top 10 vulnerability classes
- **Automatic reconnaissance** that profiles any target before attacking
- **Adaptive attack planning** — only deploys agents relevant to the target's attack surface
- **Pipeline Judge** that detects tool abuse, data leaks, memory manipulation (not just harmful text)
- **Hierarchical RL** that learns optimal attack strategies
- **Game-theoretic analysis** (Nash Equilibrium) for attack/defense optimization
- **Actionable remediation** — specific fixes for every finding
- **Real-time dashboard** with recon visualization, attack feed, and OWASP coverage
- **Multi-provider LLM support** — Vertex AI (Gemini), Groq, OpenAI, Anthropic
- **Vulnerability Lab** — intentionally vulnerable demo app for testing

## Benchmark Results

### HarmBench — Gemini 2.0 Flash (400 Behaviors)

| Metric | Value |
|--------|-------|
| Behaviors Tested | **400** |
| Exploits Found | **399** |
| Attack Success Rate | **99.8%** |
| Risk Score | **10.0 / 10** |
| OWASP Coverage | **20%** (LLM01, LLM06) |
| Total Tokens | **1.2M** |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r redforge/requirements.txt
```

### 2. Authenticate with Google Cloud (for Vertex AI)

```bash
gcloud auth application-default login
```

### 3. Run Against the Vulnerability Lab

```bash
# Terminal 1: Start the vulnerable demo app
python -m redforge.labs.vulnerable_app

# Terminal 2: Attack it with REDFORGE
python -m redforge.main engage \
  --target-url http://localhost:8900/chat \
  --provider vertex \
  --vertex-project YOUR_GCP_PROJECT \
  --depth full \
  --budget 200
```

### 4. Or Use the Live Dashboard

```bash
python -m redforge.demo.app
# Open http://localhost:8765 and watch live attacks
```

### 5. Run HarmBench Benchmark

```bash
python -m redforge.main harmbench \
  --provider vertex \
  --vertex-project YOUR_GCP_PROJECT \
  --subset 500
```

## Architecture

```
redforge/
├── recon/             # Phase 1: Target reconnaissance & profiling
│   ├── profiler.py           # Orchestrates all probes
│   └── probes.py             # 7 probe types
├── agents/            # Phase 2: 9 specialist attack agents
│   ├── jailbreak.py
│   ├── prompt_injection.py
│   ├── exfiltration.py
│   ├── tool_abuse.py
│   ├── memory_poison.py
│   ├── social_engineering.py
│   ├── cross_agent.py
│   ├── rag_poisoning.py      # NEW in v2
│   ├── guardrail_bypass.py   # NEW in v2
│   └── orchestrator.py       # LangGraph coordinator
├── judge/             # Phase 3: Multi-layer verdict ensemble
│   ├── pipeline_judge.py     # Detects tool abuse, data leaks
│   ├── classifier.py         # BART-MNLI zero-shot
│   ├── llm_judge.py          # LLM rubric judge
│   └── ensemble.py           # 4-layer voting
├── labs/              # Vulnerability Lab (intentionally vulnerable apps)
│   └── vulnerable_app.py     # Gemini-powered demo target
├── core/              # Engagement lifecycle, state, MDP
├── demo/              # FastAPI + WebSocket live dashboard
├── digest/            # Strategic attack digest generation
├── graph/             # Attack graph, Nash equilibrium
├── llm/               # Multi-provider LLM support (Vertex, Groq, OpenAI)
├── reporting/         # Risk scoring + remediation report
├── rl/                # Hierarchical RL (PPO policy)
├── safety/            # Kill switch, audit trail, sandbox
├── strategy_library/  # ChromaDB vector store
└── target/            # Target connector (API clients)
```

## Attack Agents

| Agent | OWASP | Techniques |
|-------|-------|-----------|
| **Jailbreak** | LLM01 | PAIR method, roleplay, fictional framing |
| **Prompt Injection** | LLM01 | Authority impersonation, encoding, delimiter attacks |
| **Exfiltration** | LLM06 | System prompt extraction, API key leakage, differential testing |
| **Tool Abuse** | LLM07/08 | SQL injection, path traversal, SSRF, code execution |
| **Memory Poison** | LLM03 | History injection, instruction injection, context overflow |
| **Social Engineering** | LLM09 | Multi-turn gradual escalation, trust building |
| **Cross-Agent** | LLM08 | Agent-to-agent prompt injection, routing confusion |
| **RAG Poisoning** | LLM01/06 | Keyword manipulation, document boundary confusion |
| **Guardrail Bypass** | LLM01/02 | Encoding tricks, multilingual bypass, token smuggling |

## How It Works

### Phase 1: Reconnaissance

REDFORGE probes the target to discover:
- Available tools (database, file, code execution, API)
- RAG/retrieval system (documents, vectors, access)
- Memory system (persistent, session-based)
- Input/output guards (content filters, validation)
- LLM identity (model, role, system prompt hints)
- Multi-agent architecture (if present)

### Phase 2: Attack

Based on recon, REDFORGE selects relevant agents:
- RL policy decides which agent to deploy
- Each agent has 5 iterations to find vulnerability
- PAIR method refines prompts based on feedback
- All attacks are tracked in attack graph

### Phase 3: Report

REDFORGE generates:
- Risk score (0-10 scale)
- Confirmed exploits with reproducible traces
- OWASP LLM Top 10 coverage map
- Remediation recommendations per finding
- Nash equilibrium attack/defense analysis

## The Vulnerability Lab

`redforge/labs/vulnerable_app.py` is an intentionally vulnerable FastAPI app for demos:

```
Input Guard (weak) → RAG (no access) → Memory (injectable)
        ↓                  ↓                    ↓
    System Prompt + Gemini LLM + Tools (SQL, file, code RCE)
        ↓
    Output Guard (weak)
```

**Vulnerabilities:**
- Keyword blocklist (bypassable with synonyms)
- RAG with secret documents (no document-level access control)
- Injectable conversation history
- SQL tool (no parameterization)
- File tool (no path validation)
- Code calculator (eval-based RCE)

Perfect for testing REDFORGE end-to-end.

## LLM Providers

REDFORGE supports multiple LLM providers:

```bash
# Vertex AI (Gemini)
python -m redforge.main engage \
  --target-url http://localhost:8900/chat \
  --provider vertex \
  --vertex-project my-gcp-project

# Groq (free tier)
python -m redforge.main engage \
  --target-url http://localhost:8900/chat \
  --provider groq
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Orchestration** | LangGraph StateGraph |
| **Reconnaissance** | Custom probe framework |
| **Attack Agents** | 9 specialist modules |
| **Judge** | BART-MNLI + LLM + Human escalation |
| **RL** | PyTorch + Stable-Baselines3 (PPO) |
| **Graph** | NetworkX + Nash equilibrium solver |
| **LLM** | Vertex AI (Gemini), Groq, OpenAI, Anthropic |
| **Dashboard** | FastAPI + WebSocket + vanilla JS |
| **Storage** | ChromaDB (strategy library) |
| **Audit** | SHA-256 hash-chained logs |

## Safety & Ethics

REDFORGE is designed for **authorized security testing only**:
- Explicit authorization scope required
- SHA-256 hash-chained audit trail
- Real-time kill switch (budget, severity, ethics)
- Sandboxed execution
- Human escalation queue for ambiguous verdicts

## License

MIT
