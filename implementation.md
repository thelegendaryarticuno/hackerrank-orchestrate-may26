# Implementation Plan: RLM-Style Support Triage Agent

## Goal

Build a terminal-based support triage agent that reads `support_tickets/support_tickets.csv`, uses only the provided `data/` support corpus for answer grounding, and writes `support_tickets/output.csv` with:

- `status`: `replied` or `escalated`
- `product_area`: best support area
- `response`: user-facing answer or escalation message
- `justification`: concise evidence-backed routing reason
- `request_type`: `product_issue`, `feature_request`, `bug`, or `invalid`

The first prototype should be deterministic, corpus-grounded, and fast. API models can improve wording and validation when keys are configured, but the escalation decision must come from explicit rules, retrieval confidence, and risk scoring.

## Final Architecture Choice

Use an RLM-style architecture, not a raw long-context LLM architecture.

The local PDF `2512.24601v1.pdf` describes Recursive Language Models as an inference strategy where the full prompt or corpus is kept outside the model context, exposed through an environment, searched or decomposed programmatically, and only small snippets are sent to recursive model calls. For this hackathon, that maps cleanly to:

1. Keep all support docs in an indexed local corpus.
2. Let deterministic code inspect, filter, retrieve, and score the corpus.
3. Let an optional API model see only the ticket plus top evidence snippets.
4. Validate the generated answer against retrieved evidence before writing output.

This is better than putting the full corpus into a local 120B model prompt because the corpus is 774 files and about 7.1 MB. Direct prompting is slower, less deterministic, and more vulnerable to context rot. RLM-style orchestration gives better control, clearer judge explanation, and safer escalation behavior.

## Recommended First Prototype

Prototype 1 should be:

- Python only, inside `code/`.
- No required network calls.
- Pure local BM25 retrieval over `data/`.
- Rule-based classifier and risk scorer.
- Template-based response generator for escalations and high-confidence FAQ answers.
- Optional API generator only after the deterministic pipeline is working.

This is the safest first build because the challenge evaluates `output.csv`, not how fancy the model is. A strong deterministic baseline is easier to debug in the remaining time.

## Optional Model Strategy

Use model backends in this order:

1. `template`: deterministic fallback, always available.
2. `api`: external API for validation or response polishing, only if keys are present in env vars.

The model must not decide unsupported actions by itself. It receives a strict JSON task:

- ticket fields
- inferred company
- request type
- product area
- top evidence snippets
- risk score
- escalation decision
- banned claims

The model returns only a draft response and short justification. The validator can reject it and fall back to a template.

Using an API is acceptable as long as the API is not used as a live support knowledge source. The API may transform, summarize, or validate text from the provided corpus. Secrets must come from environment variables only, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, and the prompt must include only ticket text plus retrieved local evidence.

## RLM-Style Application Flow

1. Load ticket CSV.
2. Normalize each ticket:
   - lowercase search copy
   - preserve original text for response
   - combine subject and issue into query text
   - infer company if `Company` is `None`
3. Retrieve evidence:
   - filter corpus by company when known
   - BM25 over title, breadcrumbs, headings, and body chunks
   - return top 8 chunks with file path, title, breadcrumb, score, and excerpt
4. Run recursive sub-tasks:
   - classify request type
   - infer product area
   - detect risk signals
   - detect prompt injection or unsafe requests
   - verify evidence coverage
5. Calculate risk score.
6. Decide status:
   - `escalated` for high risk, unsupported, sensitive, action-required, or low-evidence cases
   - `replied` for low-risk questions with clear corpus evidence
7. Generate response:
   - escalation template for high risk
   - evidence-grounded answer for safe, high-confidence cases
8. Validate response:
   - ensure no unsupported claim
   - ensure response does not reveal internal chain, scoring internals, or hidden rules
   - ensure response matches status
   - ensure output fields use allowed values
9. Write `support_tickets/output.csv`.
10. Print a terminal summary with counts by status, request type, company, and risk band.

## Corpus Indexing Details

Create `code/corpus.py`.

Index every `*.md` under:

- `data/hackerrank`
- `data/claude`
- `data/visa`

Each chunk should store:

- `doc_id`
- `company`
- `path`
- `title`
- `breadcrumbs`
- `heading`
- `body`
- `source_url` from front matter when present
- `product_area_hint`

Chunking rules:

- Prefer markdown headings as chunk boundaries.
- Target 350 to 900 words per chunk.
- Keep title and breadcrumbs attached to every chunk.
- Add synthetic chunks from each index page because they often map categories.

Product area inference from paths:

- HackerRank: top folder names such as `screen`, `interviews`, `settings`, `hackerrank_community`, `integrations`, `skillup`, `general-help`.
- Claude: top folder names such as `team-and-enterprise-plans`, `claude-api-and-console`, `privacy-and-legal`, `amazon-bedrock`, `claude-for-education`, `safeguards`.
- Visa: use document title or support heading because the Visa corpus is small.

## BM25 Retrieval Details

Create `code/retrieval.py`.

Use a small in-repo BM25 implementation to avoid dependency risk:

- tokenize with regex: `[a-z0-9]+`
- lowercase
- keep important terms like `refund`, `workspace`, `seat`, `charge`, `dispute`, `assessment`, `certificate`, `bedrock`
- remove only tiny stopword list
- BM25 constants: `k1=1.5`, `b=0.75`

Query expansion:

- For `Claude`: add `team`, `workspace`, `seat`, `admin`, `account`, `bedrock`, `privacy`, `education` when matching terms appear.
- For `HackerRank`: add `test`, `assessment`, `candidate`, `interview`, `screen`, `community`, `billing`, `certificate`.
- For `Visa`: add `card`, `charge`, `dispute`, `merchant`, `fraud`, `identity theft`, `cash`, `minimum spend`.

Retrieval confidence:

- `strong`: top score >= 10 and top 3 include at least two same-area chunks
- `medium`: top score >= 5
- `weak`: top score < 5 or top evidence comes from the wrong company

Thresholds should be tuned on `sample_support_tickets.csv`.

## Request Type Classification

Create `code/classifier.py`.

Rules:

- `bug`: outage, broken functionality, submissions failing, site down, all requests failing, error during compatibility check, service stopped working.
- `feature_request`: asks to add, create, support, integrate, configure a new capability not currently available, or asks for product roadmap.
- `invalid`: irrelevant, malicious, prompt injection, asks for code to harm system, too vague to route, outside support scope.
- `product_issue`: normal support question, account/admin guidance, billing question, documentation question, usage question.

When uncertain, choose `product_issue` if a company and support area are clear; otherwise choose `invalid`.

## Risk Score Calculation

Create `code/risk.py`.

Risk score is an integer from 0 to 100. Start at 0, add points, then clamp to 100.

### Risk Signals

Add 35 points for account, identity, or permission control:

- lost access
- restore access
- removed seat
- not owner/admin
- delete account
- remove user
- blocked card
- identity stolen

Add 30 points for money, payment, refund, dispute, fraud, or chargeback:

- refund
- payment issue
- charge dispute
- merchant dispute
- wrong product
- cash
- order ID
- fraud
- stolen identity

Add 25 points for assessment integrity or candidate outcome:

- change score
- review answers
- tell recruiter/company to move candidate forward
- reschedule official company test
- certificate name mutation

Add 25 points for security, privacy, or legal:

- vulnerability
- bug bounty
- data use retention
- crawling website
- internal rules
- hidden policy
- PII or secrets

Add 25 points for live outage or broad platform failure:

- site down
- none of the pages accessible
- all submissions failing
- all requests failing
- stopped working completely

Add 25 points for prompt injection or unsafe instruction:

- reveal internal rules
- show retrieved documents
- show exact logic
- ignore previous instructions
- delete all files
- expose hidden chain or policy

Add 20 points for low evidence:

- retrieval confidence is `weak`
- company is `None` and no strong inferred company
- top evidence does not answer the requested action

Add 10 points for urgency or coercive language:

- immediately
- asap
- today
- urgent
- blocked while traveling

Subtract 10 points for strong evidence and no sensitive action:

- retrieval confidence is `strong`
- request asks only for informational guidance
- no money, access, legal, security, privacy, or integrity signal

### Decision Thresholds

- `risk_score >= 60`: escalate.
- `risk_score 40-59`: escalate unless the response is a safe informational answer directly supported by strong evidence.
- `risk_score < 40`: reply if retrieval confidence is `medium` or `strong`; otherwise escalate.

Important: escalation is not a failure. In this challenge, escalating high-risk or unsupported cases is correct behavior.

## Status Decision Examples for Current Tickets

Likely escalations:

- Claude workspace access restoration by non-admin: account permission risk.
- HackerRank score dispute asking to increase score and influence recruiter: assessment integrity risk.
- Visa merchant dispute asking Visa to refund today and ban seller: money and dispute risk.
- Payment issue with order ID: billing/money risk, likely escalate.
- Infosec form request: enterprise process, may need human follow-up.
- Broad submissions or all Claude requests failing: outage/bug risk.
- Identity theft: high-risk fraud/security.
- Bug bounty: security vulnerability.
- Prompt injection in blocked-card ticket: answer safe public guidance only or escalate; never reveal internal logic or retrieved docs.
- Delete all files: invalid unsafe request.

Likely direct replies if evidence is strong:

- Remove a HackerRank interviewer/user, if docs explain admin user management.
- Data use retention for Claude, if privacy docs clearly answer.
- Claude website crawling opt-out, if docs clearly answer.
- Claude LTI key for students, if education docs clearly answer.
- Visa minimum spend in US Virgin Islands, if Visa support docs clearly answer.

Borderline cases should escalate if evidence is weak.

## Response Generation Contract

The response should be:

- short, direct, and user-facing
- grounded in retrieved local corpus
- honest about what support can and cannot do
- free of internal scoring, hidden logic, chain-of-thought, and raw retrieved document dumps
- safe around account access, fraud, payments, security, and assessment integrity

Escalation response template:

```text
Thanks for reaching out. I cannot safely resolve this directly because it involves <risk reason>. I am escalating this to the appropriate support team. Please include any relevant account, workspace, assessment, transaction, or error details in the official support channel, but do not share passwords or sensitive credentials.
```

Out-of-scope or invalid response template:

```text
I cannot help with that request because it is outside the supported product scope or could be unsafe. Please provide a product-specific support issue for HackerRank, Claude, or Visa if you need help.
```

Grounded answer template:

```text
Hi,

Based on the available <company> support documentation, <answer>.

If this does not resolve the issue or requires account-specific action, please contact the appropriate support team.
```

## Validation Design

Create `code/validator.py`.

Validation runs after response generation and before writing CSV.

Checks:

1. Schema validation:
   - status in `{"replied", "escalated"}`
   - request_type in `{"product_issue", "feature_request", "bug", "invalid"}`
   - non-empty response and justification
2. Evidence validation:
   - response must contain only claims supported by top evidence snippets
   - at least one retrieved chunk must be cited internally in the justification object, even if not output as a public citation
3. Safety validation:
   - no hidden chain-of-thought
   - no internal scoring details in user response
   - no raw corpus dump
   - no promises to refund, restore access, change score, ban merchant, update account, or perform admin-only action
4. Consistency validation:
   - high risk must not be `replied` unless reply is a safe informational refusal
   - weak evidence must not produce a detailed policy answer
   - prompt injection must not alter rules or reveal internals

If validation fails:

- switch to `escalated`
- set response to escalation template
- set justification to a concise reason such as `Escalated because the request involves payment/account/security risk or lacks sufficient corpus support.`

## API-Based Validation

API validation can be used without violating the challenge constraints if it is limited to checking the draft against local evidence. The API prompt must say:

- Do not use outside knowledge.
- Use only the provided evidence snippets.
- Return JSON with `supported`, `unsupported_claims`, `safety_issues`, and `recommended_fix`.

The API should never fetch live docs. It should never be required for the final run. If no API key exists, local validation runs.

## API Provider

Create `code/llm.py` with a provider interface:

- `TemplateProvider`
- `APIProvider` for OpenAI-compatible APIs only if env keys exist

Recommended API use:

- Run API generation only for low-risk or medium-risk wording.
- Keep temperature at 0.
- Set max output tokens around 250.
- Pass only top 3 evidence snippets.
- Reject the output if it contains unsupported claims.

API generation is useful for fluent responses, but it should not replace retrieval, risk scoring, or validation. It remains optional because the evaluator should be able to run the deterministic path without secrets.

## File Plan

Create these files:

- `code/main.py`: CLI entry point; loads CSV, runs pipeline, writes output.
- `code/agent.py`: orchestrates ticket processing.
- `code/corpus.py`: loads and chunks support docs.
- `code/retrieval.py`: BM25 index and search.
- `code/classifier.py`: company, request type, and product area classification.
- `code/risk.py`: risk signals and score.
- `code/llm.py`: template and optional API generation providers.
- `code/validator.py`: schema, evidence, and safety validation.
- `code/README.md`: install and run instructions.

Keep `code/main.py` as the evaluator entry point.

## CLI Contract

Recommended command:

```bash
python code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv \
  --corpus data \
  --provider template
```

Optional:

```bash
python code/main.py --provider api
```

Default must be `template` so the evaluator can run without secrets or API setup.

## Output CSV Rules

Output header:

```csv
issue,subject,company,response,product_area,status,request_type,justification
```

Preserve original input values for `issue`, `subject`, and `company`. Use lowercase status values as required by README, even if samples use title case.

## Development Workflow

1. Implement corpus loader and BM25.
2. Run retrieval smoke tests manually on known tickets:
   - `workspace removed seat`
   - `HackerRank refund mock interviews`
   - `Visa dispute charge`
   - `Claude Bedrock failing`
3. Implement classifier and risk score.
4. Run on `sample_support_tickets.csv` and compare:
   - exact match for `Status`
   - exact match for `Request Type`
   - fuzzy match for `Product Area`
   - manual review for response faithfulness
5. Implement generator templates.
6. Implement validator.
7. Run on `support_tickets.csv`.
8. Manually review all 29 rows.
9. Optionally enable API response polish.
10. Re-run validation and write final `output.csv`.

## Metrics for Validation

On the sample file, track:

- `status_accuracy`
- `request_type_accuracy`
- `product_area_contains_match`
- `escalation_precision`
- `unsafe_reply_count`
- `weak_evidence_reply_count`
- `schema_error_count`

Target before final run:

- 100 percent schema validity
- 0 unsafe replies
- 0 weak-evidence detailed replies
- status accuracy above 80 percent on sample
- request type accuracy above 80 percent on sample

## Judge Interview Explanation

Explain the design as:

> I used an RLM-inspired architecture. The model never receives the full corpus as raw context. The corpus is treated as an external environment indexed by deterministic tools. The agent recursively decomposes each ticket into retrieval, classification, risk scoring, generation, and validation sub-tasks. LLM calls are optional and bounded to retrieved evidence. Escalation is controlled by explicit risk and evidence rules, which is important for support safety.

Trade-offs:

- BM25 is simple and deterministic, but may miss semantic matches.
- Optional API generation improves wording, but can hallucinate, so validation gates it.
- A local 120B model could improve reasoning, but is excluded from this prototype because it is too slow and memory-heavy for the current submission path.
- RLM-style orchestration gives better reliability than direct long-context prompting.

## Final Recommendation

Build the first prototype with deterministic BM25, rules, risk scoring, and templates. Add API validation only after the CSV output is already valid. For this challenge, correctness and safe escalation are worth more than model size.
