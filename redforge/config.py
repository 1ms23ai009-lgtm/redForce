"""Configuration and constants for REDFORGE."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RedForgeConfig:
    # LLM settings
    attacker_model: str = "gpt-4o"
    judge_model: str = "claude-3-5-sonnet-20241022"
    digest_model: str = "gpt-4o-mini"
    graph_extractor_model: str = "gpt-4o-mini"

    # RL settings
    rl_policy_path: str = "models/high_level_policy.pt"
    use_pretrained_policy: bool = False

    # Attack settings
    max_pair_iterations: int = 10
    tap_branches: int = 3
    nash_update_frequency: int = 5
    digest_mode: str = "llm"  # "llm" or "algorithmic"

    # Safety settings
    default_query_budget: int = 100
    default_max_hours: float = 2.0
    default_authorized_depth: str = "medium"  # "shallow", "medium", "deep", "full"

    # Strategy library
    chroma_persist_dir: str = "data/strategy_library"
    novelty_threshold: float = 0.7

    # Reporting
    reports_dir: str = "reports"

    # Logging
    audit_log_dir: str = "logs/audit"
    log_level: str = "INFO"

    # Provider selection: "groq" or "vertex"
    provider: str = field(default_factory=lambda: os.getenv("REDFORGE_PROVIDER", "groq"))

    # Groq settings (free tier)
    groq_attacker_model: str = "llama-3.3-70b-versatile"
    groq_judge_model: str = "llama-3.3-70b-versatile"
    groq_target_model: str = "llama-3.1-8b-instant"
    groq_rpm: int = 30     # requests per minute per key
    groq_rpd: int = 14400  # requests per day per key

    # Vertex AI settings (Google Cloud)
    vertex_project: str = field(default_factory=lambda: os.getenv("VERTEX_PROJECT", ""))
    vertex_location: str = field(
        default_factory=lambda: os.getenv("VERTEX_LOCATION", "us-central1")
    )
    vertex_attacker_model: str = "gemini-2.0-flash"
    vertex_judge_model: str = "gemini-2.0-flash"
    vertex_target_model: str = "gemini-2.0-flash"
    vertex_digest_model: str = "gemini-2.0-flash"
    vertex_graph_model: str = "gemini-2.0-flash"

    # API keys (loaded from environment)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []

        if self.provider not in ("groq", "vertex"):
            errors.append(
                f"Unknown provider: {self.provider}. Must be 'groq' or 'vertex'."
            )

        # Validate provider-specific settings
        if self.provider == "vertex":
            if not self.vertex_project:
                errors.append(
                    "VERTEX_PROJECT must be set when using Vertex AI provider. "
                    "Set it in .env or pass --vertex-project."
                )
        else:
            # Need at least one LLM provider for Groq mode
            from redforge.llm.key_manager import load_groq_keys_from_env
            groq_keys = load_groq_keys_from_env()
            has_groq = len(groq_keys) > 0
            has_openai = bool(self.openai_api_key)

            if not has_groq and not has_openai:
                errors.append(
                    "No LLM API keys set. Provide OPENAI_API_KEY or "
                    "GROQ_API_KEY_1..N in your .env file"
                )

        if self.digest_mode not in ("llm", "algorithmic"):
            errors.append(f"Invalid digest_mode: {self.digest_mode}")
        if self.default_authorized_depth not in (
            "shallow",
            "medium",
            "deep",
            "full",
        ):
            errors.append(
                f"Invalid authorized_depth: {self.default_authorized_depth}"
            )
        if self.default_query_budget < 1:
            errors.append("query_budget must be >= 1")
        return errors


# Strategy category constants
STRATEGY_CATEGORIES = [
    "jailbreak",
    "prompt_injection_direct",
    "prompt_injection_indirect",
    "cross_agent_injection",
    "data_exfiltration",
    "tool_abuse",
    "memory_poisoning",
    "social_engineering",
]

# Action space mapping
ACTION_TO_STRATEGY = {
    0: "jailbreak",
    1: "prompt_injection_direct",
    2: "prompt_injection_indirect",
    3: "cross_agent_injection",
    4: "data_exfiltration",
    5: "tool_abuse",
    6: "memory_poisoning",
    7: "social_engineering",
}

STRATEGY_TO_ACTION = {v: k for k, v in ACTION_TO_STRATEGY.items()}

# Agent routing map
STRATEGY_TO_AGENT = {
    "jailbreak": "jailbreak_specialist",
    "prompt_injection_direct": "prompt_injection_specialist",
    "prompt_injection_indirect": "prompt_injection_specialist",
    "cross_agent_injection": "cross_agent_specialist",
    "data_exfiltration": "exfiltration_specialist",
    "tool_abuse": "tool_abuse_specialist",
    "memory_poisoning": "memory_poison_specialist",
    "social_engineering": "social_engineering_specialist",
}

# Severity levels
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
SEVERITY_ORDER = {s: i for i, s in enumerate(SEVERITY_LEVELS)}

# Judge verdicts
VERDICT_CONFIRMED = "CONFIRMED"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_FALSE_POSITIVE = "FALSE_POSITIVE"
VERDICT_FAILED = "FAILED"

# Effort score weights
EFFORT_WEIGHT_MSG = 0.4
EFFORT_WEIGHT_TOK = 0.3
EFFORT_WEIGHT_COST = 0.3

# Nash equilibrium parameters
LAMBDA_ATTACKER = 2.0
LAMBDA_DEFENDER = 1.0

# Attack graph node count limits
def max_graph_nodes(conversation_turns: int) -> int:
    """Compute max nodes based on conversation length."""
    if conversation_turns < 20:
        limit = 6
    elif conversation_turns <= 50:
        limit = 10
    elif conversation_turns <= 100:
        limit = 15
    else:
        limit = 20
    return max(4, min(25, limit))
