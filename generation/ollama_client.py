"""HTTP client for the Ollama service.

Wraps the embedding endpoint (nomic-embed-text) and the generation endpoints
(Qwen2.5-14B-Instruct, Qwen3-32B, and Qwen2.5-Coder-14B).

Reference:
    LogRouter paper, Section III-A (Hardware configuration).
"""
