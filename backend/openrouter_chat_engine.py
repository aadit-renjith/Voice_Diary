import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name("apikey.env")
load_dotenv(ENV_PATH)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:5173")
APP_NAME = os.getenv("OPENROUTER_APP_NAME", "Voice Diary")

SYSTEM_PROMPT = """You are a warm, empathetic diary companion embedded in a Voice Diary app.
Your job is to help the user reflect on their day through a natural, caring conversation.

RULES:
1. Ask ONE question at a time. Keep questions short and conversational.
2. Start with a broad opener like "How was your day?" then go deeper based on their answers.
3. If the user gives a short or dismissive answer (e.g. "fine", "nothing", "ok"), gently pivot:
   - Acknowledge their response without being pushy
   - Try a different angle: "What was the best part of today?" or "Did anything surprise you?"
4. If the user shares something emotional, respond with empathy FIRST, then ask a follow-up.
5. Track what you have learned: their mood, key events, interactions, and feelings.
6. After gathering enough context (usually 4-8 exchanges), wrap up naturally.

RESPONSE FORMAT - you MUST respond with valid JSON only, no markdown:
{
  "reply": "Your conversational message to the user",
  "is_complete": false,
  "summary": null,
  "detected_topics": ["topic1", "topic2"]
}

When is_complete is true, provide a brief emotional summary in the "summary" field describing
the user's day and emotional state in 2-3 sentences. Set is_complete to true only when you feel
you have a good understanding of the user's current state (mood + context + at least one event).

IMPORTANT: You must ONLY output raw JSON. No markdown code blocks. No extra text."""


def _strip_code_fences(raw_text: str) -> str:
    raw = raw_text.strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1:] if first_newline != -1 else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    return raw.strip()


def _extract_json_object(raw_text: str) -> str | None:
    text = _strip_code_fences(raw_text)

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _normalize_response(result: dict, raw_text: str) -> dict:
    topics = result.get("detected_topics", [])
    if isinstance(topics, str):
        topics = [part.strip() for part in re.split(r"[,;\n]", topics) if part.strip()]
    elif not isinstance(topics, list):
        topics = []

    return {
        "reply": str(result.get("reply", "")).strip() or raw_text.strip(),
        "is_complete": bool(result.get("is_complete", False)),
        "summary": result.get("summary"),
        "detected_topics": topics,
    }


def _parse_response(raw_text: str) -> dict:
    json_candidate = _extract_json_object(raw_text)
    if json_candidate:
        return _normalize_response(json.loads(json_candidate), raw_text)

    cleaned = _strip_code_fences(raw_text)
    return {
        "reply": cleaned,
        "is_complete": False,
        "summary": None,
        "detected_topics": [],
    }


def _friendly_error_message(error: Exception) -> str:
    message = str(error)
    lower = message.lower()

    if "quota" in lower or "429" in message:
        return (
            "The diary chat hit an OpenRouter or provider rate limit. "
            "If you are using a ':free' model, switch to a different model or retry shortly."
        )

    if "api key" in lower or "401" in message or "403" in message:
        return "OpenRouter rejected the request. Check OPENROUTER_API_KEY and OPENROUTER_MODEL in backend/apikey.env."

    if "connection refused" in lower or "actively refused it" in lower:
        return "The diary chat could not reach OpenRouter. A local proxy or network setting is blocking the request."

    return "Sorry, I had a moment. Could you say that again?"


def _to_openrouter_messages(history: list, user_message: str = None) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in history:
        parts = msg.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        if not text:
            continue

        role = "assistant" if msg.get("role") == "model" else "user"
        messages.append({"role": role, "content": text})

    if user_message is not None:
        messages.append({"role": "user", "content": user_message})

    return messages


def _call_openrouter(messages: list) -> str:
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    request = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": APP_URL,
            "X-Title": APP_NAME,
        },
        method="POST",
    )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    try:
        with opener.open(request, timeout=45) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8")
        except Exception:
            detail = str(error)
        raise RuntimeError(f"{error.code} {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(str(error.reason)) from error

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {body}")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not content:
        raise RuntimeError(f"OpenRouter returned an empty response: {body}")

    return content


class ChatEngine:
    """Manages a conversational session with an OpenRouter-backed model."""

    def __init__(self):
        self.sessions = {}

    def _get_or_create_session(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def _append_message(self, history: list, role: str, text: str):
        history.append({"role": role, "parts": [{"text": text}]})

    def get_serializable_history(self, session_id: str) -> list:
        history = self.sessions.get(session_id, [])
        return [
            {
                "role": msg.get("role"),
                "parts": [{"text": (msg.get("parts", [{}])[0].get("text", ""))}],
            }
            for msg in history
        ]

    def chat(self, session_id: str, user_message: str, emotion: str = None) -> dict:
        history = self._get_or_create_session(session_id)

        if not OPENROUTER_API_KEY:
            return {
                "reply": "Chat is not configured yet. Add OPENROUTER_API_KEY to backend/apikey.env and restart the backend.",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        context_prefix = ""
        if emotion:
            context_prefix = f"[The user's voice emotion was detected as: {emotion}] "

        full_user_message = context_prefix + user_message
        messages = _to_openrouter_messages(history, full_user_message)

        try:
            raw_text = _call_openrouter(messages)
            result = _parse_response(raw_text)
            reply_text = result.get("reply", "")

            self._append_message(history, "user", full_user_message)
            self._append_message(history, "model", reply_text)

            return {
                "reply": reply_text,
                "is_complete": result.get("is_complete", False),
                "summary": result.get("summary"),
                "detected_topics": result.get("detected_topics", []),
            }
        except Exception as error:
            print(f"ChatEngine error: {error}")
            return {
                "reply": _friendly_error_message(error),
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def get_opening_message(self, session_id: str) -> dict:
        history = self._get_or_create_session(session_id)

        if not OPENROUTER_API_KEY:
            opening = "Chat is not configured yet. Add OPENROUTER_API_KEY to backend/apikey.env and restart the backend."
            self._append_message(history, "model", opening)
            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        if len(history) > 0:
            return {
                "reply": "We're already chatting! Go ahead.",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        try:
            raw_text = _call_openrouter(
                _to_openrouter_messages(
                    history,
                    "The user just opened the diary chat. Generate your opening message.",
                )
            )
            result = _parse_response(raw_text)
            opening = result.get("reply", "Hey! How was your day?")

            self._append_message(history, "model", opening)

            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }
        except Exception as error:
            print(f"Opening message error: {error}")
            opening = _friendly_error_message(error)
            self._append_message(history, "model", opening)
            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def reset_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]
