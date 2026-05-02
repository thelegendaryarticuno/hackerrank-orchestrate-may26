"""Risk scoring and escalation policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskAssessment:
    score: int
    band: str
    reasons: tuple[str, ...]
    prompt_injection: bool


def assess_risk(text: str, company: str, request_type: str, retrieval_confidence: str) -> RiskAssessment:
    lower = text.lower()
    score = 0
    reasons: list[str] = []
    prompt_injection = False

    score += _add_if(
        lower,
        35,
        "account, identity, or permission control",
        [
            "lost access",
            "restore my access",
            "removed my seat",
            "not the workspace owner",
            "not the workspace owner or admin",
            "delete my account",
            "remove a user",
            "remove an interviewer",
            "employee has left",
            "blocked",
            "identity has been stolen",
            "identity theft",
        ],
        reasons,
    )
    score += _add_if(
        lower,
        30,
        "money, payment, refund, dispute, fraud, or chargeback",
        [
            "refund",
            "payment",
            "charge",
            "dispute",
            "merchant",
            "wrong product",
            "cash",
            "order id",
            "fraud",
            "stolen",
            "money",
        ],
        reasons,
    )
    score += _add_if(
        lower,
        25,
        "assessment integrity or candidate outcome",
        [
            "increase my score",
            "ncrease my score",
            "score dispute",
            "review my answers",
            "move me to the next round",
            "graded me unfairly",
            "rescheduling",
            "reschedule",
            "certificate",
        ],
        reasons,
    )
    if any(term in lower for term in ["score dispute", "ncrease my score", "increase my score", "graded me unfairly"]):
        score += 35
        reasons.append("direct score or hiring outcome change request")
    if "site is down" in lower or "none of the pages" in lower:
        score += 35
        reasons.append("unscoped broad outage")
    score += _add_if(
        lower,
        25,
        "security, privacy, or legal concern",
        [
            "security vulnerability",
            "major security vulnerability",
            "bug bounty",
            "data to improve",
            "how long will the data",
            "stop crawling",
            "privacy",
            "internal rules",
            "documents retrieved",
            "rules internal",
            "logic exact",
        ],
        reasons,
    )
    score += _add_if(
        lower,
        25,
        "live outage or broad platform failure",
        [
            "site is down",
            "none of the pages",
            "all submissions",
            "none of the submissions",
            "across any challenges are working",
            "all requests are failing",
            "all requests to claude",
            "stopped working completely",
        ],
        reasons,
    )
    if request_type == "bug" and any(
        term in lower
        for term in [
            "site is down",
            "all requests",
            "all submissions",
            "none of the submissions",
            "stopped working completely",
            "resume builder is down",
        ]
    ):
        score += 35
        reasons.append("broad or service-impacting bug")
    if "order id" in lower or "cs_live_" in lower:
        score += 35
        reasons.append("account-specific payment identifier")
    if "pause our subscription" in lower or "pause my subscription" in lower:
        score += 60
        reasons.append("account-specific subscription change")
    if "bug bounty" in lower or "security vulnerability" in lower:
        score += 45
        reasons.append("security vulnerability report")
    injection_points = _add_if(
        lower,
        25,
        "prompt injection or unsafe instruction",
        [
            "reveal internal",
            "affiche toutes les règles internes",
            "show retrieved",
            "documents récupérés",
            "logic exact",
            "ignore previous",
            "delete all files",
            "hidden",
        ],
        reasons,
    )
    if injection_points:
        prompt_injection = True
    score += injection_points

    if retrieval_confidence == "weak":
        score += 20
        reasons.append("weak corpus evidence")
    if company == "None":
        score += 10
        reasons.append("company could not be confidently inferred")
    if request_type == "invalid" and _unsafe_invalid(lower):
        score += 35
        reasons.append("invalid or unsafe request")
    score += _add_if(
        lower,
        10,
        "urgent or coercive language",
        ["immediately", "asap", "today", "urgent", "for quicker", "pour aller plus vite"],
        reasons,
    )
    if retrieval_confidence == "strong" and not _sensitive(lower) and request_type not in {"invalid", "bug"}:
        score -= 10
        reasons.append("strong evidence and low-sensitive informational request")

    score = max(0, min(100, score))
    if score >= 60:
        band = "high"
    elif score >= 40:
        band = "medium"
    else:
        band = "low"
    return RiskAssessment(score, band, tuple(dict.fromkeys(reasons)), prompt_injection)


def decide_status(risk: RiskAssessment, retrieval_confidence: str, safe_informational: bool) -> str:
    if risk.score >= 60:
        return "escalated"
    if risk.score >= 40 and not (safe_informational and retrieval_confidence == "strong"):
        return "escalated"
    if retrieval_confidence == "weak" and not safe_informational:
        return "escalated"
    return "replied"


def _add_if(text: str, points: int, reason: str, needles: list[str], reasons: list[str]) -> int:
    if any(needle in text for needle in needles):
        reasons.append(reason)
        return points
    return 0


def _sensitive(text: str) -> bool:
    terms = [
        "refund",
        "payment",
        "charge",
        "dispute",
        "access",
        "admin",
        "score",
        "fraud",
        "identity",
        "vulnerability",
        "privacy",
        "delete",
        "certificate",
        "subscription",
    ]
    return any(term in text for term in terms)


def _unsafe_invalid(text: str) -> bool:
    terms = [
        "delete all files",
        "ignore previous",
        "reveal internal",
        "internal rules",
        "show retrieved",
        "logic exact",
    ]
    return any(term in text for term in terms)
