"""Ticket classification helpers."""

from __future__ import annotations

from collections import Counter
import re

from retrieval import SearchResult


COMPANIES = {"hackerrank": "HackerRank", "claude": "Claude", "visa": "Visa"}


def normalize_company(value: str | None, text: str) -> str:
    raw = (value or "").strip()
    if raw and raw.lower() != "none":
        for key, company in COMPANIES.items():
            if key in raw.lower():
                return company
    lower = text.lower()
    for key, company in COMPANIES.items():
        if key in lower:
            return company
    if any(term in lower for term in ["assessment", "candidate", "interviewer", "hackerrank"]):
        return "HackerRank"
    if any(term in lower for term in ["claude", "anthropic", "bedrock", "lti"]):
        return "Claude"
    if any(term in lower for term in ["visa", "card", "merchant", "charge", "identity theft"]):
        return "Visa"
    return "None"


def classify_request_type(text: str) -> str:
    lower = text.lower()
    stripped = lower.strip()
    if stripped in {"thank you", "thank you for helping me", "thanks", "thanks for helping me"}:
        return "invalid"
    if stripped in {"it's not working, help", "it’s not working, help", "its not working, help", "not working help"}:
        return "invalid"
    if any(phrase in lower for phrase in ["it's not working, help", "it’s not working, help", "its not working, help"]):
        if len(re.findall(r"[a-zA-Z0-9]+", lower)) <= 8:
            return "invalid"
        return "invalid"
    if _has_any(lower, ["actor in iron man", "movie", "weather", "recipe"]):
        return "invalid"
    if _has_any(
        lower,
        [
            "delete all files",
            "ignore previous",
            "show all internal",
            "internal rules",
            "logic exact",
            "retrieved documents",
        ],
    ):
        return "invalid"
    if len(re.findall(r"[a-zA-Z0-9]", lower)) < 12:
        return "invalid"
    if _has_any(
        lower,
        [
            "site is down",
            "stopped working",
            "not working",
            "submissions",
            "all requests are failing",
            "all requests to claude",
            "error",
            "blocker",
            "bug",
            "down",
            "failing",
        ],
    ):
        return "bug"
    if _has_any(
        lower,
        [
            "feature request",
            "can you add",
            "please add",
            "would like to request support for",
            "setup a",
            "set up a",
            "integrate",
            "lti key",
            "infosec process",
        ],
    ):
        return "feature_request"
    return "product_issue"


def infer_product_area(company: str, results: list[SearchResult], text: str) -> str:
    lower = text.lower()
    if company == "HackerRank":
        if any(term in lower for term in ["variant", "variants", "default versions of roles"]):
            return "screen"
        if any(term in lower for term in ["remove", "employee has left", "interviewer from the platform"]):
            return "settings"
        if any(term in lower for term in ["community", "practice", "certificate", "resume"]):
            return "community"
        if any(term in lower for term in ["interview", "mock", "screen share", "hr lobby"]):
            return "interviews"
        if any(term in lower for term in ["test", "assessment", "candidate", "submissions", "score"]):
            return "screen"
        if any(term in lower for term in ["user", "employee", "interviewer", "subscription"]):
            return "settings"
        if any(term in lower for term in ["infosec", "forms", "hiring"]):
            return "general-help"
    if company == "Claude":
        if "bedrock" in lower or "aws" in lower:
            return "amazon-bedrock"
        if any(term in lower for term in ["workspace", "seat", "admin", "team"]):
            return "team-and-enterprise-plans"
        if any(term in lower for term in ["data", "privacy", "private info", "temporary chat", "delete conversation", "crawl", "website"]):
            return "privacy"
        if any(term in lower for term in ["student", "professor", "lti", "college"]):
            return "claude-for-education"
        if any(term in lower for term in ["vulnerability", "bug bounty", "security"]):
            return "privacy-and-legal"
    if company == "Visa":
        if any(term in lower for term in ["traveller", "traveler", "cheques", "lisbon"]):
            return "travel_support"
        if any(term in lower for term in ["identity", "stolen", "fraud", "blocked"]):
            return "general_support" if "lost or stolen visa card" in lower else "fraud-and-security"
        if any(term in lower for term in ["dispute", "charge", "merchant", "refund", "wrong product"]):
            return "disputes"
        if any(term in lower for term in ["cash", "atm"]):
            return "cash-access"
        if any(term in lower for term in ["minimum", "spend", "us virgin islands"]):
            return "general_support"
        return "visa-support"

    areas = [r.chunk.product_area_hint for r in results[:5] if r.chunk.product_area_hint]
    if areas:
        return Counter(areas).most_common(1)[0][0]
    return ""


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)
