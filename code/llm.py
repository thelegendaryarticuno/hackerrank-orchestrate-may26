"""Response generation providers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import request, error

from retrieval import SearchResult


@dataclass
class Draft:
    response: str
    justification: str
    used_api: bool = False
    fallback_reason: str = ""


class TemplateProvider:
    name = "template"

    def generate(
        self,
        *,
        ticket_text: str,
        company: str,
        product_area: str,
        status: str,
        request_type: str,
        risk_reasons: tuple[str, ...],
        results: list[SearchResult],
    ) -> Draft:
        if request_type == "invalid" and status == "replied":
            return Draft(
                response=(
                    "I cannot help with that request because it is outside the supported "
                    "HackerRank, Claude, and Visa support scope."
                ),
                justification="Replied with an out-of-scope response because the request is not a supported product issue.",
            )
        if status == "escalated":
            return Draft(
                response=_escalation_response(risk_reasons, request_type),
                justification=_justification(status, product_area, risk_reasons, results),
            )
        answer = _extract_answer(company, results, ticket_text)
        response = (
            f"Hi,\n\nBased on the available {company} support documentation, {answer}\n\n"
            "If this does not resolve the issue or requires account-specific action, "
            "please contact the appropriate support team."
        )
        return Draft(
            response=response,
            justification=_justification(status, product_area, risk_reasons, results),
        )


class APIProvider(TemplateProvider):
    name = "api"

    def __init__(self, model: str | None = None) -> None:
        self.providers = _discover_apis(model)
        self.fallback = TemplateProvider()

    def generate(self, **kwargs: Any) -> Draft:
        fallback = self.fallback.generate(**kwargs)
        if not self.providers:
            fallback.fallback_reason = "no API key/base URL configured"
            return fallback
        if kwargs["status"] == "escalated":
            return fallback

        evidence = []
        for result in kwargs["results"][:3]:
            chunk = result.chunk
            evidence.append(
                {
                    "title": chunk.title,
                    "area": chunk.product_area_hint,
                    "excerpt": chunk.body[:1200],
                }
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You write concise support replies. Use only the provided evidence. "
                    "Do not add outside knowledge, promises, refunds, account changes, or hidden logic. "
                    "Return strict JSON with response and justification."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "ticket": kwargs["ticket_text"],
                        "company": kwargs["company"],
                        "product_area": kwargs["product_area"],
                        "request_type": kwargs["request_type"],
                        "evidence": evidence,
                    },
                    ensure_ascii=True,
                ),
            },
        ]
        errors: list[str] = []
        for provider in self.providers:
            for model in provider["models"]:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 350,
                    "response_format": {"type": "json_object"},
                }
                try:
                    data = _post_chat_completion(provider["base_url"], provider["api_key"], payload)
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    response = str(parsed.get("response", "")).strip()
                    justification = str(parsed.get("justification", "")).strip()
                    if response and justification:
                        return Draft(response=response, justification=justification, used_api=True)
                    errors.append(f"{provider['name']}/{model}: empty JSON fields")
                except Exception as exc:  # API is optional; deterministic fallback is intentional.
                    errors.append(f"{provider['name']}/{model}: {str(exc)[:120]}")
        if errors:
            fallback.fallback_reason = " | ".join(errors)[:500]
        return fallback


def build_provider(name: str, model: str | None = None) -> TemplateProvider:
    if name == "api":
        return APIProvider(model)
    return TemplateProvider()


def _discover_apis(requested_model: str | None) -> list[dict[str, object]]:
    providers: list[dict[str, object]] = []
    if os.getenv("GITHUB_TOKEN"):
        providers.append(
            {
                "name": "github",
                "base_url": os.getenv("GITHUB_MODELS_API_BASE_URL") or "https://models.github.ai/inference",
                "api_key": os.getenv("GITHUB_TOKEN", ""),
                "models": _model_candidates("github", requested_model),
            }
        )
    if os.getenv("GROQ_API_KEY"):
        providers.append(
            {
                "name": "groq",
                "base_url": os.getenv("GROQ_API_BASE_URL") or "https://api.groq.com/openai/v1",
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "models": _model_candidates("groq", requested_model),
            }
        )
    return providers


def _model_candidates(provider_name: str, requested: str | None) -> list[str]:
    if requested:
        return [requested]
    if provider_name == "groq":
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
        ]
    if provider_name == "github":
        return [
            "openai/gpt-4o-mini",
            "gpt-4o-mini",
            "openai/gpt-4.1-mini",
            "gpt-4.1-mini",
        ]
    return []


def _post_chat_completion(base_url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "hackerrank-orchestrate-support-agent/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"API HTTP {exc.code}: {body}") from exc


def _escalation_response(risk_reasons: tuple[str, ...], request_type: str) -> str:
    if request_type == "invalid":
        return (
            "I cannot help with that request because it is outside the supported product scope "
            "or could be unsafe. Please provide a product-specific support issue for HackerRank, "
            "Claude, or Visa if you need help."
        )
    reason = risk_reasons[0] if risk_reasons else "sensitive or unsupported details"
    return (
        "Thanks for reaching out. I cannot safely resolve this directly because it involves "
        f"{reason}. I am escalating this to the appropriate support team. Please include relevant "
        "account, workspace, assessment, transaction, or error details in the official support "
        "channel, but do not share passwords or sensitive credentials."
    )


def _extract_answer(company: str, results: list[SearchResult], ticket_text: str) -> str:
    if not results:
        return "I could not find enough relevant documentation to answer this safely."
    query_terms = {
        term
        for term in ticket_text.lower().replace("_", " ").split()
        if len(term.strip(".,?!:;()[]")) > 3
    }
    candidates: list[tuple[int, int, str]] = []
    for result_index, result in enumerate(results[:5]):
        chunk = result.chunk
        if "related articles" in chunk.heading.lower():
            continue
        paragraphs = [p.strip(" -\n\t") for p in chunk.body.split("\n\n") if len(p.split()) >= 8]
        if not paragraphs:
            paragraphs = [chunk.body[:900]]
        for paragraph_index, paragraph in enumerate(paragraphs[:6]):
            lower = paragraph.lower()
            if lower.startswith("related articles"):
                continue
            overlap = sum(1 for term in query_terms if term.strip(".,?!:;()[]") in lower)
            title_overlap = sum(1 for term in query_terms if term.strip(".,?!:;()[]") in chunk.title.lower())
            score = overlap * 3 + title_overlap * 2 - result_index - paragraph_index
            candidates.append((score, result_index, paragraph))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [candidates[0][2]]
        if len(candidates) > 1 and candidates[1][0] > 2:
            selected.append(candidates[1][2])
    else:
        selected = [results[0].chunk.body[:700]]
    answer = " ".join(selected)
    answer = " ".join(answer.split())
    if len(answer) > 850:
        answer = answer[:847].rsplit(" ", 1)[0] + "..."
    return answer


def _justification(
    status: str,
    product_area: str,
    risk_reasons: tuple[str, ...],
    results: list[SearchResult],
) -> str:
    evidence = ""
    if results:
        top = results[0].chunk
        evidence = f" Top evidence: {top.title} ({top.product_area_hint})."
    if status == "escalated":
        reason = "; ".join(risk_reasons[:3]) or "risk or insufficient evidence"
        return f"Escalated because the request involves {reason}.{evidence}"
    return f"Replied because the request maps to {product_area} with supporting corpus evidence.{evidence}"
