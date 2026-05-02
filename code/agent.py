"""Top-level support triage orchestration."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from classifier import classify_request_type, infer_product_area, normalize_company
from corpus import load_corpus
from llm import build_provider
from retrieval import BM25Index, confidence
from risk import assess_risk, decide_status
from validator import validate


OUTPUT_FIELDS = [
    "issue",
    "subject",
    "company",
    "response",
    "product_area",
    "status",
    "request_type",
    "justification",
]


@dataclass(frozen=True)
class TicketResult:
    row: dict[str, str]
    diagnostics: dict[str, Any]


class SupportAgent:
    def __init__(
        self,
        *,
        corpus_root: Path,
        provider: str = "template",
        api_model: str | None = None,
        debug: bool = False,
    ) -> None:
        _load_env()
        self.corpus_root = corpus_root
        self.debug = debug
        self.chunks = load_corpus(corpus_root)
        self.index = BM25Index(self.chunks)
        self.provider = build_provider(provider, api_model)
        self.api_fallbacks = 0

    def run_csv(self, input_path: Path, output_path: Path) -> dict[str, Any]:
        with input_path.open(newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))

        results = [self.process_ticket(row) for row in rows]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(result.row for result in results)

        if self.debug:
            for idx, result in enumerate(results, 1):
                diag = result.diagnostics
                line = (
                    f"{idx:02d} {result.row['status']} {result.row['request_type']} "
                    f"{result.row['product_area']} risk={diag['risk_score']} "
                    f"conf={diag['retrieval_confidence']}"
                )
                if diag.get("api_fallback"):
                    line += f" api_fallback={diag['api_fallback'][:140]}"
                print(line)

        replied = sum(1 for result in results if result.row["status"] == "replied")
        escalated = sum(1 for result in results if result.row["status"] == "escalated")
        return {
            "rows": len(results),
            "replied": replied,
            "escalated": escalated,
            "provider": self.provider.name,
            "api_fallbacks": self.api_fallbacks,
        }

    def process_ticket(self, raw_row: dict[str, str]) -> TicketResult:
        issue = _get(raw_row, "Issue", "issue")
        subject = _get(raw_row, "Subject", "subject")
        raw_company = _get(raw_row, "Company", "company")
        ticket_text = f"{subject}\n{issue}".strip()

        company = normalize_company(raw_company, ticket_text)
        search_company = None if company == "None" else company
        results = self.index.search(ticket_text, company=search_company, top_k=8)
        retrieval_confidence = confidence(results, search_company)
        request_type = classify_request_type(ticket_text)
        product_area = infer_product_area(company, results, ticket_text)
        risk = assess_risk(ticket_text, company, request_type, retrieval_confidence)
        safe_informational = _safe_informational(ticket_text, request_type, risk.score)
        status = decide_status(risk, retrieval_confidence, safe_informational)
        if request_type == "invalid" and not risk.prompt_injection and risk.score < 60:
            status = "replied"
            if company == "None" and any(term in ticket_text.lower() for term in ["actor", "movie", "weather", "recipe"]):
                product_area = "conversation_management"
            if ticket_text.lower().strip() in {"thank you for helping me", "thank you", "thanks"}:
                product_area = ""

        draft = self.provider.generate(
            ticket_text=ticket_text,
            company=company if company != "None" else "the relevant product",
            product_area=product_area,
            status=status,
            request_type=request_type,
            risk_reasons=risk.reasons,
            results=results,
        )
        if draft.fallback_reason:
            self.api_fallbacks += 1

        validated = validate(
            draft=draft,
            ticket_text=ticket_text,
            company=company if company != "None" else "the relevant product",
            product_area=product_area,
            status=status,
            request_type=request_type,
            risk=risk,
            retrieval_confidence=retrieval_confidence,
            results=results,
        )
        if validated.changed:
            status = "escalated"
            draft = validated.draft

        row = {
            "issue": issue,
            "subject": subject,
            "company": raw_company,
            "response": draft.response,
            "product_area": product_area,
            "status": status,
            "request_type": request_type,
            "justification": draft.justification,
        }
        diagnostics = {
            "risk_score": risk.score,
            "risk_band": risk.band,
            "risk_reasons": risk.reasons,
            "retrieval_confidence": retrieval_confidence,
            "top_evidence": results[0].chunk.path if results else "",
            "validation_changed": validated.changed,
            "validation_issues": validated.issues,
            "api_fallback": draft.fallback_reason,
        }
        return TicketResult(row, diagnostics)


def _get(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()] or ""
    return ""


def _load_env() -> None:
    """Load .env without requiring python-dotenv inside the hackathon venv."""
    if load_dotenv is not None:
        load_dotenv()

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for path in dict.fromkeys(candidates):
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _safe_informational(ticket_text: str, request_type: str, risk_score: int) -> bool:
    lower = ticket_text.lower()
    if request_type == "invalid":
        return False
    if risk_score >= 60:
        return False
    action_terms = [
        "restore",
        "refund",
        "increase my score",
        "ncrease my score",
        "ban",
        "delete all",
        "change",
        "update it",
        "remove them",
        "pause our subscription",
    ]
    if any(term in lower for term in action_terms):
        return False
    return True
