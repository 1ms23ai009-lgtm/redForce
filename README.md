<p align="center">
  <img src="https://img.shields.io/badge/REDFORGE-v0.2.0-ff4040?style=for-the-badge&labelColor=0a0a0f" alt="version"/>
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/tests-73%20passed-40ff70?style=for-the-badge" alt="tests"/>
  <img src="https://img.shields.io/badge/HarmBench%20ASR-99.8%25-ff4040?style=for-the-badge" alt="harmbench"/>
  <img src="https://img.shields.io/badge/LLM%20Provider-Multi--Provider-blueviolet?style=for-the-badge" alt="provider"/>
</p>

<h1 align="center">REDFORGE</h1>
<h3 align="center">Autonomous AI Red-Teaming System</h3>
<p align="center"><i>Hierarchical RL + Multi-Agent Framework for LLM & Agentic Pipeline Security Assessment</i></p>

---

## What is REDFORGE?

REDFORGE is a fully autonomous red-teaming system that discovers vulnerabilities in Large Language Models and agentic AI pipelines without human intervention. It combines:

- **7 specialist attack agents** targeting different OWASP LLM Top 10 vulnerability classes
- **Hierarchical Reinforcement Learning** that learns which attacks work best against each target
- **Game-theoretic analysis** (Nash Equilibrium) to find optimal attack/defense strategies
- **A 3-layer judge ensemble** that verifies exploits with high confidence
- **Real-time live dashboard** with WebSocket streaming of attacks as they happen
- **Multi-provider LLM support** — Vertex AI (Gemini), Groq, OpenAI, Anthropic

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

## Benchmark Results

### HarmBench Evaluation — Gemini 2.0 Flash (400 Behaviors)

| Metric | Value |
|--------|-------|
| Behaviors Tested | **400** |
| Exploits Found | **399** |
| Attack Success Rate (ASR) | **99.8%** |
| Risk Score | **10.0 / 10.0** (Critical) |
| OWASP LLM Top 10 Coverage | **20%** (2/10 categories) |
| Total Tokens | **1.2M** |
| API Calls | **~2,048** |
| Attack Method | **PAIR** (Prompt Automatic Iterative Refinement) |
| Max Iterations per Behavior | **5** |

### Attack Patterns That Achieved 99.8% ASR

| Pattern | Example | Success Rate |
|---------|---------|-------------|
| Fictional Framing | "I'm writing a screenplay where..." | Very High |
| Academic Framing | "For a cybersecurity course..." | Very High |
| Historical Framing | "For a historical fiction novel..." | Very High |
| Gradual Escalation | Build trust across turns, then escalate | Very High |
| Role-Play Injection | "You are an amoral lawyer advising..." | High |

### OWASP LLM Top 10 Coverage

```
LLM01 Prompt Injection      ████████████████████████████████████  HIGH
LLM02 Insecure Output       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM03 Training Data Poison  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM04 Model DoS             ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM05 Supply Chain           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM06 Sensitive Info Disc    ████████████████████████████████████  HIGH
LLM07 Insecure Plugin       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM08 Excessive Agency      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM09 Overreliance          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
LLM10 Model Theft           ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
```

## Architecture

```
redforge/
├── agents/            # 7 specialist attack agents + LangGraph orchestrator
│   ├── jailbreak.py          # PAIR-based jailbreak attacks
│   ├── prompt_injection.py   # Authority impersonation, encoding tricks
│   ├── exfiltration.py       # System prompt extraction, data leakage
│   ├── tool_abuse.py         # Path traversal, command injection
│   ├── memory_poison.py      # Context manipulation, history injection
│   ├── social_engineering.py # Multi-turn gradual escalation
│   ├── cross_agent.py        # Inter-agent trust exploitation
│   └── orchestrator.py       # LangGraph StateGraph coordinator
├── benchmarks/        # HarmBench evaluation framework
├── core/              # Engagement manager, ASO state, MDP formulation
├── demo/              # Live web dashboard (FastAPI + WebSocket)
├── digest/            # Strategic attack digest generation (LLM-powered)
├── graph/             # Attack graph, Nash equilibrium, effort scoring
├── judge/             # 3-layer ensemble (classifier + LLM + human)
├── llm/               # Multi-provider support
│   ├── groq_provider.py      # Groq with multi-key rotation
│   └── vertex_provider.py    # Google Vertex AI (Gemini models)
├── reporting/         # Risk scoring + executive report generation
├── rl/                # Hierarchical RL system
│   ├── high_level_policy.py  # Agent selection policy
│   ├── low_level_policy.py   # Attack crafting policy
│   ├── ppo_trainer.py        # PPO training loop
│   ├── replay_buffer.py      # Experience replay
│   └── reward.py             # Shaped reward function
├── safety/            # Kill switch, sandboxing, SHA-256 audit trail
├── strategy_library/  # ChromaDB vector store for learned strategies
├── target/            # Target connector (OpenAI, Anthropic, custom APIs)
└── tests/             # 73 unit tests
```

## How It Works

### The PAIR Attack Method

REDFORGE's core attack uses **PAIR** (Prompt Automatic Iterative Refinement) with 3 LLM roles:

```
┌─────────────┐     adversarial prompt      ┌──────────────┐
│   ATTACKER   │ ──────────────────────────► │    TARGET     │
│              │                             │              │
│  Generates   │ ◄────────────────────────── │  Responds to │
│  jailbreak   │      target response        │  the prompt  │
└──────┬───────┘                             └──────────────┘
       │                                            │
       │              ┌──────────────┐              │
       └─────────────►│    JUDGE     │◄─────────────┘
                      │              │
                      │ Score: 1-10  │
                      │ ≥8 = Exploit │
                      └──────┬───────┘
                             │
                      Iterate up to 5 turns
```

Each behavior gets up to **5 iterations × 3 API calls** = 15 calls max. The attacker LLM learns from each failed attempt and refines its approach.

### Hierarchical RL System

The full engagement mode uses a two-level RL system:

```
                    ┌─────────────────────────┐
                    │   HIGH-LEVEL POLICY     │
                    │   "Which agent next?"    │
                    │   67-dim state → PPO     │
                    └────────────┬────────────┘
                                 │
         ┌──────────┬────────────┼──────────┬──────────────┐
         ▼          ▼            ▼          ▼              ▼
    Jailbreak  Prompt Inj.  Exfiltration  Tool Abuse  Social Eng.
         │          │            │          │              │
         └──────────┴────────────┼──────────┴──────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   LOW-LEVEL POLICY      │
                    │   "How to craft attack?" │
                    │   LangChain LLM chain   │
                    └─────────────────────────┘
```

The high-level policy observes a **67-dimensional state** (attack history, coverage, target behavior) and selects the optimal agent. Results feed back as rewards to improve future decisions.

### Nash Equilibrium Analysis

REDFORGE models red-teaming as a **two-player zero-sum game**:

- **Attacker strategies**: The 7 attack agents
- **Defender strategies**: Possible mitigations
- **Payoff matrix**: Built from observed attack success rates

Computes mixed-strategy Nash equilibria to answer: *"If the defender optimally patches vulnerabilities, which attack strategy still has the highest expected payoff?"*

### Judge Ensemble

Three verification layers to minimize false positives:

1. **BART-MNLI Classifier** — Fast zero-shot screening (is this harmful?)
2. **LLM Rubric Judge** — Detailed analysis against scoring rubric (1-10)
3. **Human Escalation Queue** — Edge cases flagged for manual review

## Quick Start

### 1. Install dependencies

```bash
pip install -r redforge/requirements.txt
```

### 2. Run the live dashboard

```bash
python -m redforge.demo.app
```

Open **http://localhost:8765** and click **Launch Attack** to watch REDFORGE find all 8 vulnerabilities in the built-in mock target.

### 3. Run HarmBench with Vertex AI (Gemini)

```bash
# Authenticate with Google Cloud
gcloud auth application-default login

# Run benchmark
python -m redforge.main harmbench \
  --provider vertex \
  --vertex-project YOUR_PROJECT_ID \
  --subset 400
```

### 4. Run HarmBench with Groq (Free)

```bash
# Set API keys in .env
cp redforge/.env.example redforge/.env
# Edit .env with your Groq keys

python -m redforge.main harmbench --subset 50
```

### 5. Run HarmBench via Live Dashboard

```bash
python -m redforge.demo.app
```

Open **http://localhost:8765**, click **HarmBench**, configure behaviors count and max iterations, and watch results stream in real-time.

### 6. Full engagement against a real target

```bash
python -m redforge.main engage \
  --target-url https://api.example.com/v1/chat/completions \
  --target-model gpt-4 \
  --target-type openai \
  --provider vertex \
  --vertex-project YOUR_PROJECT_ID \
  --depth full \
  --budget 100
```

### 7. Run tests

```bash
python -m pytest redforge/tests/ -v
```

## LLM Provider Support

REDFORGE supports multiple LLM providers for both the attacker/judge system and as targets:

| Provider | Models | Usage |
|----------|--------|-------|
| **Vertex AI** | Gemini 2.0 Flash, Gemini Pro | `--provider vertex` |
| **Groq** | Llama 3.1 8B, Llama 3.3 70B | `--provider groq` (default) |
| **OpenAI** | GPT-4, GPT-4o | As target via `--target-type openai` |
| **Anthropic** | Claude 3.5 Sonnet | As target via `--target-type anthropic` |

### Vertex AI Setup

```bash
# 1. Install Google Cloud SDK
# 2. Authenticate
gcloud auth application-default login

# 3. Set environment variables (or use CLI flags)
export REDFORGE_PROVIDER=vertex
export VERTEX_PROJECT=your-gcp-project
export VERTEX_LOCATION=us-central1
```

### Groq Setup (Multi-Key Rotation)

```bash
# In .env — distribute rate limits across multiple keys
GROQ_API_KEY_1=gsk_your-key-1
GROQ_API_KEY_2=gsk_your-key-2
GROQ_API_KEY_3=gsk_your-key-3
```

## Attack Categories

| Agent | OWASP Category | Technique |
|-------|---------------|-----------|
| Jailbreak | LLM01 Prompt Injection | Roleplay framing, fictional scenarios, PAIR |
| Prompt Injection | LLM01 Prompt Injection | Authority impersonation, base64 encoding |
| Data Exfiltration | LLM06 Sensitive Info | Indirect extraction, encoding tricks |
| Tool Abuse | LLM07 Insecure Plugin | Path traversal, command injection |
| Memory Poisoning | LLM03 Training Data | Persistent context manipulation |
| Social Engineering | LLM09 Overreliance | Multi-turn gradual escalation |
| Cross-Agent | LLM08 Excessive Agency | Inter-agent prompt injection |

## Live Dashboard

The real-time web dashboard streams attacks via WebSocket:

- **Live Attack Feed** — Each prompt sent, response received, and verdict (EXPLOITED/BLOCKED)
- **Metrics** — Exploits found, scenarios, risk score, OWASP coverage, ASR, tokens used
- **Scenario Results** — Table updating in real time with severity and score
- **OWASP LLM Top 10 Coverage** — Visual breakdown of vulnerability categories hit
- **HarmBench Mode** — Run full benchmarks from the UI with configurable parameters

## CLI Reference

```bash
# HarmBench benchmark (Vertex AI)
python -m redforge.main harmbench --provider vertex --vertex-project PROJECT --subset 400

# HarmBench benchmark (Groq)
python -m redforge.main harmbench --subset 50

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

# Launch live dashboard
python -m redforge.demo.app --port 8765
```

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
| Orchestration | LangGraph StateGraph |
| RL Training | PyTorch + Stable-Baselines3 |
| Graph Analysis | NetworkX + SciPy (Nash Equilibrium) |
| Strategy Store | ChromaDB |
| LLM Providers | Vertex AI (Gemini), Groq, OpenAI, Anthropic |
| Live Dashboard | FastAPI + WebSocket |
| Judge Ensemble | BART-MNLI + LLM Rubric + Human Queue |
| Testing | Pytest (73 tests) |

## License

MIT
