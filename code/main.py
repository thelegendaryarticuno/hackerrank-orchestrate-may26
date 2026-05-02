#!/usr/bin/env python3
"""Terminal entry point for the HackerRank Orchestrate support agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import SupportAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the support triage agent.")
    parser.add_argument(
        "--input",
        default="support_tickets/support_tickets.csv",
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default="support_tickets/output.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--corpus",
        default="data",
        help="Support corpus root.",
    )
    parser.add_argument(
        "--provider",
        choices=["template", "api", "comparison"],
        default="template",
        help="Response generator. template is deterministic; api is optional; comparison runs both and compares.",
    )
    parser.add_argument(
        "--api-model",
        default=None,
        help="Optional API model name for OpenAI-compatible providers.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-ticket diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.provider == "comparison":
        print("=== Running Template Provider ===")
        path_out = Path(args.output)
        out_template = path_out.with_name(f"{path_out.stem}_template{path_out.suffix}")
        agent_tmpl = SupportAgent(
            corpus_root=Path(args.corpus),
            provider="template",
            api_model=args.api_model,
            debug=args.debug,
        )
        sum_tmpl = agent_tmpl.run_csv(Path(args.input), out_template)
        
        print("\n=== Running API Provider ===")
        out_api = path_out.with_name(f"{path_out.stem}_api{path_out.suffix}")
        agent_api = SupportAgent(
            corpus_root=Path(args.corpus),
            provider="api",
            api_model=args.api_model,
            debug=args.debug,
        )
        sum_api = agent_api.run_csv(Path(args.input), out_api)
        
        print("\n=== Comparison Summary ===")
        print(f"Input: {args.input}")
        print(f"Total Rows: {sum_tmpl['rows']}")
        print(f"Template - Replied: {sum_tmpl['replied']} | Escalated: {sum_tmpl['escalated']}")
        print(f"API      - Replied: {sum_api['replied']} | Escalated: {sum_api['escalated']}")
        print(f"Outputs saved to: {out_template} and {out_api}")
        return

    agent = SupportAgent(
        corpus_root=Path(args.corpus),
        provider=args.provider,
        api_model=args.api_model,
        debug=args.debug,
    )
    summary = agent.run_csv(Path(args.input), Path(args.output))
    print("Support triage complete")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Rows: {summary['rows']}")
    print(f"Replied: {summary['replied']}")
    print(f"Escalated: {summary['escalated']}")
    print(f"Provider: {summary['provider']}")
    if summary.get("api_fallbacks"):
        print(f"API fallbacks: {summary['api_fallbacks']}")


if __name__ == "__main__":
    main()
