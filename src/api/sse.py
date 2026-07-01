from __future__ import annotations

import json
from typing import Any

SSE_EVENT_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "started": frozenset({"event", "message"}),
    "retrieval_complete": frozenset({"event", "returned", "latency_ms"}),
    "llm_turn": frozenset({"event", "turn", "tools_requested"}),
    "tool_call": frozenset({"event", "name", "arguments"}),
    "tool_result": frozenset({"event", "name"}),
    "discovery_budget_reached": frozenset({"event", "reason", "message"}),
    "evidence_required": frozenset(
        {
            "event",
            "reason",
            "required_full_text_papers",
            "full_text_fetched",
            "candidate_ids",
            "message",
        }
    ),
    "warning": frozenset({"event", "code", "message"}),
    "degraded": frozenset({"event", "reason", "message"}),
    "error": frozenset({"event", "stage", "message", "type"}),
    "synthesis_complete": frozenset({"event", "llm_calls"}),
    "final": frozenset({"event", "data"}),
}

SSE_EVENT_TYPES = frozenset(SSE_EVENT_REQUIRED_FIELDS)


def validate_sse_event(event: dict[str, Any]) -> dict[str, Any]:
    event_type = event.get("event")
    if not isinstance(event_type, str):
        raise ValueError("SSE event must include string field 'event'")
    required = SSE_EVENT_REQUIRED_FIELDS.get(event_type)
    if required is None:
        raise ValueError(f"Unknown SSE event type: {event_type}")
    missing = sorted(required - event.keys())
    if missing:
        raise ValueError(
            f"SSE event '{event_type}' missing required fields: {', '.join(missing)}"
        )
    return event


def format_sse_event(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(validate_sse_event(event))}\n\n"
