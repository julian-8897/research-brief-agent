"""Deterministic OpenAI-compatible server for the packaged release smoke.

It drives the real provider adapter and agent loop through semantic search,
full-text reading, and final synthesis without using a paid provider key.
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = "0.0.0.0"
PORT = 8081


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _paper_from_messages(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "{}")
        except (TypeError, json.JSONDecodeError):
            continue
        papers = payload.get("papers")
        if isinstance(papers, list) and papers and isinstance(papers[0], dict):
            paper = papers[0]
            if paper.get("id"):
                return paper
    return None


def completion_for(payload: dict[str, Any]) -> dict[str, Any]:
    messages = payload.get("messages") or []
    tool_messages = [message for message in messages if message.get("role") == "tool"]
    paper = _paper_from_messages(messages)

    if not tool_messages:
        content = None
        tool_calls = [
            _tool_call(
                "smoke-search",
                "search_papers",
                {
                    "query": (
                        "retrieval augmented generation source grounded technical "
                        "decision briefs"
                    ),
                    "k": 2,
                },
            )
        ]
        finish_reason = "tool_calls"
    elif tool_messages[-1].get("tool_call_id") == "smoke-search" and paper:
        content = None
        tool_calls = [
            _tool_call(
                "smoke-fulltext",
                "get_full_text",
                {"paper_ids": [paper["id"]]},
            )
        ]
        finish_reason = "tool_calls"
    else:
        paper_id = paper.get("id", "unknown") if paper else "unknown"
        title = (
            paper.get("title", "retrieved evidence") if paper else "retrieved evidence"
        )
        content = (
            "# Decision Memo\n\n"
            "## Recommendation\n"
            "Proceed with a bounded pilot of retrieval-augmented technical briefs, "
            "with citation checks and latency budgets as release criteria.\n\n"
            "## Evidence\n"
            f"The packaged smoke read the retrieved paper *{title}* in full "
            f"before synthesis [{paper_id}].\n\n"
            "## Tradeoffs and risks\n"
            "Retrieval improves source visibility but adds indexing, PDF extraction, "
            "provider, and network failure modes. This single-paper smoke is limited "
            "evidence and does not establish answer quality.\n\n"
            "## Next actions\n"
            "Repeat with a live provider and the seeded benchmark corpus, then compare "
            "citation grounding, latency, token usage, and cost against release limits."
        )
        tool_calls = None
        finish_reason = "stop"

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": f"chatcmpl-smoke-{time.time_ns()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": payload.get("model", "smoke-agent"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 120 + 20 * len(tool_messages),
            "completion_tokens": 24 if tool_calls else 180,
            "total_tokens": 144 + 20 * len(tool_messages),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid JSON request"})
            return
        self._json(200, completion_for(payload))

    def log_message(self, format: str, *args: Any) -> None:
        print(json.dumps({"event": "smoke_provider_request", "message": format % args}))

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(json.dumps({"event": "smoke_provider_started", "port": PORT}))
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
