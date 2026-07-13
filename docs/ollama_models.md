# Ollama Model Research

This note summarizes which models are worth using through Ollama for this
project, and how that maps onto the team's shared GPU machine.

## Models confirmed on the shared GPU machine

The team's GPU/LLM usage guide lists two models explicitly reachable at
`http://10.15.33.66:11435` (VPN required):

- `qwen2.5-coder:14b` — used for coding and SQL generation tasks.
- `qwen3:32b` — used for larger, more complex generation tasks.

Run `curl -s http://10.15.33.66:11435/api/tags` while on VPN to get the
authoritative, up-to-date list; other models may already be installed.

These two match the LogRouter paper's own setup almost exactly (the paper
uses `Qwen2.5-14B-Instruct`, `Qwen3-32B`, and `Qwen2.5-Coder-14B` on a single
RTX 8000), which is a good sign that this shared machine was provisioned
with the same paper in mind.

## General model recommendations (2026)

| Model | Parameters | VRAM (Q4) | Strongest at |
|---|---|---|---|
| Llama 3.3 70B | 70B | ~40GB | General-purpose, highest quality |
| Qwen 2.5 32B | 32B | ~20GB | General-purpose, multilingual |
| Qwen 2.5 Coder 32B | 32B | ~20GB | Code generation |
| DeepSeek R1 32B / 14B | 32B / 14B | ~20GB / ~9GB | Chain-of-thought reasoning |
| qwen3-coder:30b (MoE) | 30B / ~3B active | ~18GB | Agentic coding, long context |
| gpt-oss:20b (MoE) | 20.9B / ~3.6B active | ~14-16GB | General + agentic, fits 16GB |
| Llama 3.1 8B | 8B | ~5GB | Budget general-purpose, RAG grounding |
| nomic-embed-text | 137M | ~0.5GB | Embeddings for RAG |

All sizes assume Q4_K_M quantization, Ollama's default. Benchmark figures
are community-reported and vary by prompt length and hardware; treat them
as rough guidance rather than exact numbers.

## Recommendations for this project

- **Everyday development and quick iteration:** use a lighter model to
  avoid tying up the shared GPU. `Config.ollama_light_model` defaults to
  `qwen2.5-coder:7b`; confirm with `/api/tags` whether this (or a similarly
  sized model) is actually installed on the shared machine, and adjust if
  not.
- **Coding / SQL generation path:** `qwen2.5-coder:14b` (already on the
  shared machine, matches the paper's coder model).
- **Complex semantic synthesis:** `qwen3:32b` (already on the shared
  machine, matches the paper's Level-2 "large" generator).
- **Embeddings for the semantic branch (if implemented later):**
  `nomic-embed-text`, the same embedding model used in the paper.

## Log analysis suitability

Prior work on LLM-based log analysis (LogGPT, LLMLogAnalyzer, LogLLM,
AD-LLM) shows that general-purpose mid-to-large models (Qwen 2.5 class,
Llama 3.x class) are capable of interpreting log templates, summarizing
incidents, and answering factual questions about logs, without requiring a
specialized fine-tuned model. Reasoning-focused models (DeepSeek R1 class)
add value specifically for multi-step root-cause analysis, at a latency
cost. This matches the LogRouter paper's own design: a smaller model for
routine queries, a larger model reserved for queries whose complexity score
crosses a threshold.

## Practical notes

- Always check `qwen2.5-coder:14b`-vs-`qwen2.5-coder:7b`-style tradeoffs
  with a small local test set before committing to a default, since the
  shared GPU is used by multiple people at once.
- Use `keep_alive` conservatively (the guide's examples use `"5m"`) so
  models are unloaded from the shared GPU's memory when idle.
- Prefer the smallest model that meets the accuracy bar for a given task;
  this is the same principle the LogRouter paper's Level-2 router encodes
  as a complexity threshold rather than always using the largest model.
