"""LLM-based triple extractor using Anthropic tool-use API."""

import anthropic
from pydantic import BaseModel

EXTRACTION_SYSTEM_PROMPT = (
    "You extract factual subject-predicate-object triples from personal conversation text. "
    "Focus on people, places, food/activity preferences, biographical facts, and events. "
    "Return only triples you are confident about."
)

_TOOL_SCHEMA = {
    "name": "extract_triples",
    "description": "Extract factual triples from the given text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string"},
                        "predicate": {"type": "string"},
                        "object": {"type": "string"},
                        "confidence": {"type": "number"},
                        "subject_type": {"type": "string"},
                        "object_type": {"type": "string"},
                    },
                    "required": ["subject", "predicate", "object"],
                },
            }
        },
        "required": ["triples"],
    },
}


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    subject_type: str = "entity"
    object_type: str = "entity"


def extract_triples(text: str, client: anthropic.Anthropic) -> list[Triple]:
    """Extract triples from *text* using the Anthropic tool-use API.

    Returns an empty list on any parse error or API failure.
    """
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=EXTRACTION_SYSTEM_PROMPT,
            tools=[_TOOL_SCHEMA],
            tool_choice={"type": "tool", "name": "extract_triples"},
            messages=[{"role": "user", "content": text}],
        )
        for block in response.content:
            if block.type == "tool_use":
                raw_triples = block.input.get("triples", [])
                result = []
                for item in raw_triples:
                    try:
                        result.append(Triple(**item))
                    except Exception:
                        continue
                return result
        return []
    except Exception:
        return []
