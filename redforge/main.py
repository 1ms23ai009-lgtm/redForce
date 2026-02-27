"""REDFORGE - Autonomous AI Red-Teaming System.

Entry point with CLI interface.
"""

import argparse
import json
import logging
import os
import sys

from redforge.config import RedForgeConfig
from redforge.core.engagement import EngagementManager
from redforge.strategy_library.library import StrategyLibrary
from redforge.rl.high_level_policy import HighLevelPolicy, PolicyNetwork
from redforge.rl.ppo_trainer import PPOTrainer


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_engage(args):
    """Run a red-team engagement."""
    config = RedForgeConfig()

    if args.attacker_model:
        config.attacker_model = args.attacker_model

    target_config = {
        "endpoint_url": args.target_url,
        "model_name": args.target_model or "unknown",
        "api_key_env_var": args.api_key_env_var or "TARGET_API_KEY",
        "target_type": args.target_type or "openai_compatible",
        "tool_schemas": [],
        "memory_system": None,
        "agent_topology": None,
    }

    customer_config = {
        "depth": args.depth or config.default_authorized_depth,
        "query_budget": args.budget or config.default_query_budget,
    }

    if args.categories:
        customer_config["categories"] = args.categories.split(",")

    manager = EngagementManager(
        target_config=target_config,
        customer_config=customer_config,
        config=config,
    )

    print(f"Starting engagement: {manager.aso['engagement_id']}")
    print(f"Target: {args.target_url}")
    print(f"Budget: {customer_config['query_budget']} queries")
    print(f"Depth: {customer_config['depth']}")
    print()

    report = manager.run()

    # Save report
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")
    else:
        print(json.dumps(report, indent=2, default=str))

    # Print summary
    summary = report.get("executive_summary", {})
    print(f"\n--- Engagement Summary ---")
    print(f"Total queries: {summary.get('total_queries_used', 0)}")
    print(f"Exploits found: {summary.get('total_exploits', 0)}")
    print(f"Risk score: {summary.get('overall_risk_score', 0)}/10")
    print(f"OWASP coverage: {summary.get('owasp_coverage_percentage', 0)}%")


def cmd_status(args):
    """Check engagement status."""
    print(f"Status check for engagement: {args.engagement_id}")
    print("Note: Real-time status requires the engagement to be running.")
    print("Check the reports directory for completed engagement reports.")

    config = RedForgeConfig()
    report_path = os.path.join(
        config.reports_dir, f"report_{args.engagement_id}.json"
    )
    if os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
        summary = report.get("executive_summary", {})
        print(f"\nCompleted engagement found:")
        print(f"  Exploits: {summary.get('total_exploits', 0)}")
        print(f"  Risk score: {summary.get('overall_risk_score', 0)}/10")
    else:
        print(f"\nNo completed report found at {report_path}")


def cmd_library(args):
    """Manage the strategy library."""
    config = RedForgeConfig()
    library = StrategyLibrary(persist_dir=config.chroma_persist_dir)

    if args.stats:
        stats = library.get_stats()
        print("Strategy Library Statistics:")
        print(f"  Total strategies: {stats.get('total', 0)}")
        print(f"  Average success rate: {stats.get('avg_success_rate', 0):.1%}")
        print(f"  Novel strategies: {stats.get('novel_strategies', 0)}")
        categories = stats.get("categories", {})
        if categories:
            print("\n  By category:")
            for cat, info in sorted(categories.items()):
                print(
                    f"    {cat}: {info['count']} entries, "
                    f"avg success: {info['avg_success_rate']:.1%}"
                )

    elif args.list_strategies:
        entries = library.list_all()
        if not entries:
            print("Strategy library is empty.")
            return
        print(f"Strategy Library ({len(entries)} entries):\n")
        for entry in entries:
            print(
                f"  [{entry.category}] {entry.technique_name} "
                f"(success: {entry.success_rate:.1%}, used: {entry.times_used}x)"
            )

    else:
        print("Use --list-strategies or --stats")


def cmd_train(args):
    """Train the RL policy."""
    import numpy as np
    import torch

    config = RedForgeConfig()
    policy = PolicyNetwork()

    if args.resume and os.path.exists(args.resume):
        policy.load_state_dict(
            torch.load(args.resume, map_location="cpu", weights_only=True)
        )
        print(f"Resumed from {args.resume}")

    trainer = PPOTrainer(
        policy=policy,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    from redforge.rl.high_level_policy import STATE_DIM

    # Simple training environment simulation
    def env_step(action):
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        # Simulate reward based on action diversity and random outcomes
        reward = np.random.choice(
            [1.0, 0.3, 0.1, -0.1, -0.3],
            p=[0.05, 0.1, 0.15, 0.5, 0.2],
        )
        done = np.random.random() < 0.02  # ~2% chance of episode end
        return next_state, reward, done, {}

    save_path = args.save_to or config.rl_policy_path
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    print(f"Training for {args.episodes} episodes...")
    metrics = trainer.train_episodes(
        env_step_fn=env_step,
        n_episodes=args.episodes,
        save_path=save_path,
        save_every=max(1, args.episodes // 10),
    )

    print(f"\nTraining complete. Policy saved to {save_path}")
    if metrics:
        last = metrics[-1]
        print(f"  Final policy loss: {last.get('policy_loss', 0):.4f}")
        print(f"  Final value loss: {last.get('value_loss', 0):.4f}")
        print(f"  Total updates: {last.get('total_updates', 0)}")


def cmd_harmbench(args):
    """Run HarmBench evaluation using Groq (free)."""
    from redforge.benchmarks.harmbench_runner import run_harmbench

    key_split = None
    if args.key_split:
        parts = args.key_split.split(",")
        key_split = (int(parts[0]), int(parts[1]))

    result = run_harmbench(
        subset_size=args.subset,
        max_pair_iterations=args.max_iterations,
        target_model=args.target_model,
        attacker_model=args.attacker_model,
        key_split=key_split,
        output_dir=args.output_dir,
    )

    print(f"\nASR: {result['asr']:.1%}")
    print(f"Cost: ${result['cost']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="REDFORGE - Autonomous AI Red-Teaming System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # engage command
    engage_parser = subparsers.add_parser("engage", help="Run a red-team engagement")
    engage_parser.add_argument(
        "--target-url", required=True, help="Target API endpoint URL"
    )
    engage_parser.add_argument(
        "--target-model", default=None, help="Target model name"
    )
    engage_parser.add_argument(
        "--target-type",
        default="openai_compatible",
        choices=["openai", "anthropic", "openai_compatible", "langchain", "custom_http"],
        help="Target API type",
    )
    engage_parser.add_argument(
        "--api-key-env-var",
        default="TARGET_API_KEY",
        help="Environment variable for target API key",
    )
    engage_parser.add_argument(
        "--depth",
        default="medium",
        choices=["shallow", "medium", "deep", "full"],
        help="Authorization depth",
    )
    engage_parser.add_argument(
        "--budget", type=int, default=100, help="Query budget"
    )
    engage_parser.add_argument(
        "--categories", default=None, help="Comma-separated authorized categories"
    )
    engage_parser.add_argument(
        "--attacker-model", default=None, help="Attacker LLM model"
    )
    engage_parser.add_argument(
        "--output", "-o", default=None, help="Output report file path"
    )

    # status command
    status_parser = subparsers.add_parser("status", help="Check engagement status")
    status_parser.add_argument(
        "--engagement-id", required=True, help="Engagement UUID"
    )

    # library command
    library_parser = subparsers.add_parser(
        "library", help="Manage strategy library"
    )
    library_parser.add_argument(
        "--list-strategies", action="store_true", help="List all strategies"
    )
    library_parser.add_argument(
        "--stats", action="store_true", help="Show library statistics"
    )

    # train command
    train_parser = subparsers.add_parser("train", help="Train the RL policy")
    train_parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    train_parser.add_argument(
        "--save-to", default=None, help="Path to save trained policy"
    )
    train_parser.add_argument(
        "--resume", default=None, help="Path to resume training from"
    )

    # harmbench command
    hb_parser = subparsers.add_parser(
        "harmbench", help="Run HarmBench evaluation (FREE via Groq)"
    )
    hb_parser.add_argument(
        "--subset", type=int, default=50,
        help="Number of behaviors to test (default: 50)",
    )
    hb_parser.add_argument(
        "--max-iterations", type=int, default=5,
        help="Max PAIR iterations per behavior (default: 5)",
    )
    hb_parser.add_argument(
        "--target-model", default=None,
        help="Groq model to attack (default: llama-3.1-8b-instant)",
    )
    hb_parser.add_argument(
        "--attacker-model", default=None,
        help="Groq model for attack generation (default: llama-3.3-70b-versatile)",
    )
    hb_parser.add_argument(
        "--key-split", default=None,
        help="Key split as 'redforge,target' (e.g. '3,2')",
    )
    hb_parser.add_argument(
        "--output-dir", default="reports",
        help="Output directory for reports",
    )

    args = parser.parse_args()
    setup_logging()

    if args.command == "engage":
        cmd_engage(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "library":
        cmd_library(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "harmbench":
        cmd_harmbench(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
