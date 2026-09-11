"""
src/meeting/summarizer.py
─────────────────────────────────────────────────────────────────────────────
LLM-based meeting analysis: summary, key points, decisions, action items,
topics, and open questions.

WHY USE AN LLM AFTER TRANSCRIPTION?
──────────────────────────────────────
Whisper produces a word-for-word transcript.  An LLM understands *meaning*:
it can:
  • Identify what was actually decided (not just discussed)
  • Extract concrete tasks with owners and deadlines
  • Summarise a 10,000-word transcript into a readable paragraph
  • Identify questions that were raised but not answered

WHY STRUCTURED OUTPUTS?
────────────────────────
We instruct the LLM to return JSON matching our Pydantic schema.  Benefits:
  1. Reliable downstream consumption — no fragile regex parsing.
  2. Validation at parse time — if the LLM returns wrong types, Pydantic raises.
  3. Null values are explicit — easier to distinguish "unknown" from "empty".

HOW WE PREVENT HALLUCINATION:
───────────────────────────────
1. The system prompt explicitly prohibits inventing information.
2. Null / "Not specified" is encouraged over a guess.
3. Every action item and decision requires an evidence field (verbatim quote).
4. Long transcripts are chunked and summarised in stages to avoid context
   overflow pushing the model to guess.

CHUNKING STRATEGY (Phase 9):
──────────────────────────────
If the transcript exceeds max_chunk_words (default 3000):
  1. Split into overlapping chunks by word count.
  2. Summarise each chunk independently.
  3. Feed the chunk summaries to a final "reduce" call.

This is a Map-Reduce summarisation pattern.
"""

from __future__ import annotations


import json
import logging
from typing import Optional

from openai import OpenAI

from src.config import settings
from src.models import (
    ActionItem,
    AlignedSegment,
    Decision,
    MeetingSummary,
    Topic,
)
from src.utils import chunk_text_by_words, count_words

logger = logging.getLogger(__name__)

# ── LLM client (lazily initialised) ────────────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not settings.llm_available:
        raise RuntimeError(
            "No LLM API key is set. "
            "Add NVIDIA_API_KEY to your .env file (free at build.nvidia.com)."
        )
    # The OpenAI SDK supports custom base_url — this is how NVIDIA NIM works:
    # same SDK, different endpoint + key.
    kwargs: dict = {"api_key": settings.active_api_key}
    if settings.active_base_url:
        kwargs["base_url"] = settings.active_base_url
        logger.info(f"LLM client: NVIDIA NIM | {settings.active_base_url}")
    else:
        logger.info("LLM client: OpenAI")
    _client = OpenAI(**kwargs)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a professional meeting analyst. Your job is to extract structured information from meeting transcripts and return it as a JSON object.

CRITICAL RULES:
1. NEVER invent information not present in the transcript.
2. If something is unknown, use null instead of a guess.
3. Only extract action items where someone explicitly commits to doing something.
4. Only extract decisions that were explicitly stated, not just discussed.
5. Return ONLY a valid JSON object. No markdown fences, no extra text, no explanation.
6. The "summary" field MUST be a non-empty string describing the meeting.
7. The "key_points" field MUST contain 3 to 7 bullet strings in the list. NEVER return an empty list for "key_points". Do NOT only describe them in "summary" - put each key point as an item in "key_points".

Your output must be a single JSON object with exactly these fields:
- "title": a short meeting title string, or null if not determinable
- "summary": a 2-4 sentence paragraph summarising the meeting (required, never empty)
- "key_points": a list of 3-7 concise bullet strings describing major discussion points (required, never empty)
- "decisions": a list of objects with fields: decision (string), timestamp (string or null), evidence (string or null)
- "action_items": a list of objects with fields: task (string), owner (string or null), deadline (string or null), timestamp (string or null), confidence (number 0 to 1), evidence (string or null)
- "open_questions": a list of strings for unresolved questions raised
- "topics": a list of objects with fields: topic (string), start (string or null), end (string or null)
- "participants": a list of speaker label strings found in the transcript

Example output:
{"title": "Weekly Team Sync", "summary": "The team discussed the Q3 roadmap and agreed on a new launch date.", "key_points": ["Q3 roadmap reviewed", "Launch moved to October", "Stakeholder communications planned"], "decisions": [{"decision": "Launch date moved to October 15", "timestamp": "00:05:00", "evidence": "We agreed to move the launch to October 15th"}], "action_items": [{"task": "Update project plan", "owner": "Alice", "deadline": "Friday", "timestamp": null, "confidence": 0.9, "evidence": "Alice will update the plan"}], "open_questions": ["Will marketing be ready?"], "topics": [{"topic": "Launch Planning", "start": "00:00:00", "end": "00:10:00"}], "participants": ["Alice", "Bob"]}"""

CHUNK_SUMMARY_PROMPT = """Summarise this portion of a meeting transcript. Include:
- Key points discussed
- Any decisions made
- Any action items mentioned

Be concise. This summary will be combined with summaries of other chunks.
Return plain text, not JSON."""

REDUCE_PROMPT = """Below are summaries of successive chunks of a single meeting transcript.
Combine them into a single structured meeting analysis following the JSON schema.
Do not invent any information not present in the summaries.

{summaries}"""


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def _transcript_to_text(segments: list[AlignedSegment]) -> str:
    """Convert aligned segments to timestamped text for the LLM."""
    lines: list[str] = []
    for seg in segments:
        prefix = f"{seg.speaker}: " if seg.speaker else ""
        lines.append(f"[{seg.timestamp_str}] {prefix}{seg.text.strip()}")
    return "\n".join(lines)


def _call_llm(
    user_content: str,
    system_content: str = SYSTEM_PROMPT,
    max_tokens: int = 4096,
) -> str:
    """Make a single LLM call and return the response text."""
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            # Assistant prefix nudge: forces model to start generating JSON immediately
            {"role": "assistant", "content": "{"},
        ],
        max_tokens=max_tokens,
        temperature=0.1,  # Low temperature → more deterministic, less hallucination
    )
    raw = response.choices[0].message.content.strip()
    # Re-add the opening brace we injected as the assistant prefix
    if not raw.startswith("{"):
        raw = "{" + raw
    logger.debug(f"LLM raw response (first 300 chars): {raw[:300]!r}")
    return raw


def _parse_summary_json(raw_json: str) -> MeetingSummary:
    """
    Parse LLM JSON response into a MeetingSummary model.

    Handles:
    • Extra whitespace / newlines around JSON
    • Markdown code blocks (```json ... ```)
    • Stray conversational preamble or postscript from the LLM
    • Topics, decisions, or action items formatted as strings or dicts with alternate keys
    • Missing or None values in required fields
    """
    cleaned = raw_json.strip()

    # Strip markdown fences if the model added them despite instructions
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove opening fence line (e.g. ``` or ```json)
        lines = lines[1:]
        # Remove closing fence line if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # If the LLM included introductory text or trailing commentary, extract the JSON object
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    data = json.loads(cleaned)

    # ── Parse decisions robustly ─────────────────────────────────────────────
    decisions: list[Decision] = []
    for d in data.get("decisions", []) or []:
        if isinstance(d, str) and d.strip():
            decisions.append(Decision(decision=d.strip()))
        elif isinstance(d, dict):
            decision_text = d.get("decision") or d.get("text") or d.get("description") or ""
            if str(decision_text).strip():
                decisions.append(
                    Decision(
                        decision=str(decision_text).strip(),
                        timestamp=d.get("timestamp"),
                        evidence=d.get("evidence"),
                    )
                )

    # ── Parse action items robustly ──────────────────────────────────────────
    action_items: list[ActionItem] = []
    for a in data.get("action_items", []) or []:
        if isinstance(a, str) and a.strip():
            action_items.append(ActionItem(task=a.strip()))
        elif isinstance(a, dict):
            task_text = a.get("task") or a.get("action") or a.get("description") or ""
            if str(task_text).strip():
                action_items.append(
                    ActionItem(
                        task=str(task_text).strip(),
                        owner=a.get("owner"),
                        deadline=a.get("deadline"),
                        timestamp=a.get("timestamp"),
                        confidence=a.get("confidence"),
                        evidence=a.get("evidence"),
                    )
                )

    # ── Parse topics robustly ────────────────────────────────────────────────
    topics: list[Topic] = []
    for t in data.get("topics", []) or []:
        if isinstance(t, str) and t.strip():
            topics.append(Topic(topic=t.strip()))
        elif isinstance(t, dict):
            topic_name = t.get("topic") or t.get("name") or t.get("title") or ""
            if str(topic_name).strip():
                topics.append(
                    Topic(
                        topic=str(topic_name).strip(),
                        start=t.get("start"),
                        end=t.get("end"),
                    )
                )

    # Helper to clean string lists
    def _clean_str_list(items: list | None) -> list[str]:
        if not items or not isinstance(items, list):
            return []
        return [str(x).strip() for x in items if x is not None and str(x).strip()]

    summary_text = data.get("summary") or ""
    key_points = _clean_str_list(data.get("key_points"))
    if not key_points and summary_text:
        key_points = extract_fallback_key_points(summary_text)

    return MeetingSummary(
        title=data.get("title"),
        summary=summary_text,
        key_points=key_points,
        decisions=decisions,
        action_items=action_items,
        open_questions=_clean_str_list(data.get("open_questions")),
        topics=topics,
        participants=_clean_str_list(data.get("participants")),
    )


def extract_fallback_key_points(summary_text: str) -> list[str]:
    """Extract key points from summary text when the LLM omits them from the array."""
    if not summary_text or not summary_text.strip():
        return []
    import re
    points: list[str] = []
    # 1. Look for explicit 'key points included / were ...' pattern
    match = re.search(r'key points? (?:included|were|are)\s+([^.]*)', summary_text, re.IGNORECASE)
    if match:
        clause = match.group(1).strip()
        splits = re.split(r',|\band the\b|\band\b', clause, flags=re.IGNORECASE)
        for part in splits:
            cleaned = re.sub(r'^(?:the|a|an)\s+', '', part.strip(), flags=re.IGNORECASE).strip()
            if len(cleaned) > 8:
                points.append(cleaned[0].upper() + cleaned[1:])

    # 2. If no explicit clause or too few points, extract core informative sentences
    if len(points) < 2:
        sentences = [s.strip() for s in re.split(r'\.(?:\s+|$)', summary_text) if s.strip()]
        for s in sentences:
            s_lower = s.lower()
            if any(skip in s_lower for skip in ['decisions made', 'action items included', 'participants included', 'open questions included']):
                continue
            if len(s) > 15 and s not in points:
                points.append(s)

    return points[:7]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def analyze_meeting(
    segments: list[AlignedSegment],
    max_chunk_words: Optional[int] = None,
) -> MeetingSummary:
    """
    Analyse a meeting transcript and return a structured MeetingSummary.

    Parameters
    ----------
    segments       : aligned transcript segments (with speaker labels)
    max_chunk_words: max words per chunk.  Defaults to settings.max_chunk_words.

    Returns
    -------
    MeetingSummary

    Strategy
    ─────────
    • If the transcript is short enough, analyse in a single LLM call.
    • If it's too long, use Map-Reduce chunking:
        1. Split into chunks
        2. Summarise each chunk (plain text)
        3. Feed chunk summaries to a final structured analysis call
    """
    if not settings.llm_available:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    max_words = max_chunk_words or settings.max_chunk_words
    transcript_text = _transcript_to_text(segments)
    word_count = count_words(transcript_text)

    logger.info(
        f"Starting LLM analysis: {word_count} words | "
        f"model={settings.openai_model}"
    )

    if word_count <= max_words:
        # ── Single-pass analysis ──────────────────────────────────────────────
        logger.info("Transcript fits in one chunk. Running single-pass analysis.")
        raw = _call_llm(
            f"Analyse this meeting transcript:\n\n{transcript_text}"
        )
    else:
        # ── Map-Reduce analysis ───────────────────────────────────────────────
        chunks = chunk_text_by_words(transcript_text, max_words)
        logger.info(
            f"Transcript too long ({word_count} words). "
            f"Splitting into {len(chunks)} chunks."
        )

        chunk_summaries: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Summarising chunk {i}/{len(chunks)}...")
            summary = _call_llm(
                f"Meeting transcript (part {i} of {len(chunks)}):\n\n{chunk}",
                system_content=CHUNK_SUMMARY_PROMPT,
                max_tokens=1024,
            )
            chunk_summaries.append(f"--- Part {i} ---\n{summary}")

        combined = "\n\n".join(chunk_summaries)
        prompt = REDUCE_PROMPT.format(summaries=combined)
        logger.info("Running final reduce analysis over chunk summaries...")
        raw = _call_llm(prompt)

    try:
        summary = _parse_summary_json(raw)
        logger.info(
            f"Analysis complete: "
            f"{len(summary.action_items)} action items | "
            f"{len(summary.decisions)} decisions | "
            f"{len(summary.key_points)} key points"
        )
        return summary
    except Exception as exc:
        logger.error(f"Failed to parse LLM response: {exc}\nRaw response:\n{raw}")
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc
