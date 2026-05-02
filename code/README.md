# Support Triage Agent

This directory contains the terminal-based HackerRank Orchestrate support agent.

## Approach

The agent uses an RLM-style pipeline:

1. Load and chunk the local `data/` support corpus.
2. Retrieve evidence with deterministic BM25.
3. Classify company, product area, and request type.
4. Calculate a risk score for account, payment, security, assessment integrity, outage, prompt-injection, and weak-evidence cases.
5. Generate a grounded response with templates, or optionally with an API model.
6. Validate the response before writing CSV output.

The default path does not require network access or API keys.

## Run Providers

The agent supports three provider modes for response generation:
1. `template` (Default): Deterministic, rule-based text matching and orchestration. No external APIs needed.
2. `api`: Generates dynamic responses using a Large Language Model API.
3. `comparison`: Runs both methods concurrently and outputs a comparative summary.

### 1. Template Provider (Default)

From the repo root:

```bash
python code/main.py \
  --input support_tickets/support_tickets.csv \
  --output support_tickets/output.csv \
  --corpus data \
  --provider template
```

For diagnostics:

```bash
python code/main.py --provider template --debug
```

### 2. API Provider

The API provider is optional and only sees the ticket plus locally retrieved evidence snippets. It must not be used as a live support knowledge source.

Supported environment variables:

- `GROQ_API_KEY`
- `GROQ_API_BASE_URL`
- `GITHUB_TOKEN`
- `GITHUB_MODELS_API_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_API_BASE_URL`
- `MODEL_NAME` or `GITHUB_MODEL`

Run:

```bash
python code/main.py --provider api --api-model <model-name>
```

If no usable API key is configured, the agent falls back to deterministic templates.

### 3. Comparison Mode

To run both the default template and API providers sequentially and compare the orchestration results (e.g., number of replied vs escalated tickets):

```bash
python code/main.py --provider comparison --debug
```

This will automatically save `output_template.csv` and `output_api.csv` in the output directory, alongside a printed comparison summary in the terminal.

## Files

- `main.py`: CLI entry point.
- `agent.py`: end-to-end orchestration.
- `corpus.py`: markdown corpus loader and chunker.
- `retrieval.py`: BM25 implementation.
- `classifier.py`: company, product area, and request-type rules.
- `risk.py`: risk scoring and escalation thresholds.
- `llm.py`: template and optional API response generation.
- `validator.py`: schema and safety validation.

## Setup 

### Cloning the repository

```bash

git clone 