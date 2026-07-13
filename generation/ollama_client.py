"""HTTP client for the Ollama service.

Wraps the model listing endpoint (``/api/tags``) and the chat and
generation endpoint (``/api/chat``) exposed by Ollama, whether it is
running locally or on the shared GPU machine described in the team's
GPU/LLM usage guide.

Reference:
    LogRouter paper, Section III-A (Hardware configuration).
"""

from typing import Any, Optional

import requests


class OllamaClient:
    """Thin HTTP client around an Ollama server's REST API.

    Attributes:
        host: Base URL of the Ollama server, for example
            ``http://10.15.33.66:11435``.
        timeout: Timeout in seconds applied to every HTTP request.
    """

    def __init__(self, host: str, timeout: int = 600) -> None:
        """Initializes the client.

        Args:
            host: Base URL of the Ollama server.
            timeout: Timeout in seconds applied to every HTTP request.
        """
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[dict[str, Any]]:
        """Lists the models currently available on the Ollama server.

        Returns:
            list[dict[str, Any]]: The ``models`` entries returned by the
            server's ``/api/tags`` endpoint.
        """
        response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("models", [])

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: Optional[dict[str, Any]] = None,
        format: Optional[dict[str, Any]] = None,
        stream: bool = False,
        keep_alive: str = "5m",
    ) -> dict[str, Any]:
        """Sends a chat completion request to the Ollama server.

        Args:
            model: Name of the model to use, for example
                ``qwen2.5-coder:14b``.
            messages: Conversation history as a list of role/content
                dictionaries.
            options: Optional generation parameters such as temperature,
                num_ctx, top_p, top_k, repeat_penalty, num_predict, seed,
                and stop.
            format: Optional JSON schema forcing structured output.
            stream: Whether to request a streamed response.
            keep_alive: How long the model should stay loaded in memory
                after the request completes, for example ``"5m"``.

        Returns:
            dict[str, Any]: The parsed JSON response from the server,
            including ``message.content`` and timing and token
            statistics.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": [],
            "format": format,
            "options": options or {},
            "stream": stream,
            "keep_alive": keep_alive,
        }
        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
