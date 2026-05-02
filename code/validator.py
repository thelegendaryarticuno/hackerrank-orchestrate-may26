"""Schema and safety validation for generated rows."""

from __future__ import annotations

from dataclasses import dataclass

from llm import Draft, TemplateProvider
from retrieval import SearchResult
from risk import RiskAssessment


ALLOWED_STATUS = {"replied", "escalated"}
ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}


@dataclass(frozen=True)
class ValidationResult:
    draft: Draft
    changed: bool
    issues: tuple[str, ...]


def validate(
    *,
    draft: Draft,
    ticket_text: str,
    company: str,
    product_area: str,
    status: str,
    request_type: str,
    risk: RiskAssessment,
    retrieval_confidence: str,
    results: list[SearchResult],
) -> ValidationResult:
    issues: list[str] = []
    if status not in ALLOWED_STATUS:
        issues.append("invalid status")
    if request_type not in ALLOWED_REQUEST_TYPES:
        issues.append("invalid request_type")
    if not draft.response.strip():
        issues.append("empty response")
    if not draft.justification.strip():
        issues.append("empty justification")
    if _unsafe_claim(draft.response):
        issues.append("unsafe action claim")
    if _reveals_internals(draft.response):
        issues.append("reveals internal logic or documents")
    if status == "replied" and retrieval_confidence == "weak":
        issues.append("reply with weak evidence")
    if status == "replied" and risk.score >= 60:
        issues.append("reply with high risk")
    if risk.prompt_injection and "internal" in draft.response.lower():
        issues.append("prompt injection response leak")

    if not issues:
        return ValidationResult(draft, False, ())

    fallback = TemplateProvider().generate(
        ticket_text=ticket_text,
        company=company,
        product_area=product_area,
        status="escalated",
        request_type=request_type,
        risk_reasons=tuple(issues) + risk.reasons,
        results=results,
    )
    return ValidationResult(fallback, True, tuple(issues))


def _unsafe_claim(text: str) -> bool:
    lower = text.lower()
    banned = [
        "i have refunded",
        "we have refunded",
        "i restored",
        "we restored",
        "i changed your score",
        "we changed your score",
        "i updated your account",
        "we updated your account",
        "i banned",
        "we banned",
        "your refund is approved",
    ]
    return any(term in lower for term in banned)


def _reveals_internals(text: str) -> bool:
    lower = text.lower()
    banned = [
        "risk score",
        "retrieved document",
        "internal rule",
        "hidden policy",
        "chain of thought",
        "bm25",
    ]
    return any(term in lower for term in banned)
