import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load API key
load_dotenv("apikey.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

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
6. After gathering enough context (usually 4–8 exchanges), wrap up naturally.

RESPONSE FORMAT — you MUST respond with valid JSON only:
{
  "reply": "Your conversational message to the user",
  "is_complete": false,
  "summary": null,
  "detected_topics": ["topic1", "topic2"]
}

When is_complete is true, provide a brief emotional summary describing
the user's day and emotional state in 2–3 sentences.
"""


def _parse_response(raw_text: str) -> dict:
    """Parse JSON from Gemini response, handling markdown code fences."""
    raw = raw_text.strip()

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    return json.loads(raw)


class ChatEngine:
    """Manages a conversational session with Gemini AI."""

    def __init__(self):
        self.sessions = {}

    def _get_or_create_session(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def chat(self, session_id: str, user_message: str, emotion: str = None) -> dict:

        history = self._get_or_create_session(session_id)

        context_prefix = ""
        if emotion:
            context_prefix = f"[User voice emotion detected: {emotion}] "

        full_user_message = context_prefix + user_message

        try:

            prompt = SYSTEM_PROMPT + "\n\nConversation history:\n"

            for msg in history:
                prompt += msg + "\n"

            prompt += f"\nUser: {full_user_message}\n"

            response = model.generate_content(prompt)

            raw_text = response.text

            result = _parse_response(raw_text)

            history.append(f"User: {full_user_message}")
            history.append(f"AI: {raw_text}")

            return {
                "reply": result.get("reply", ""),
                "is_complete": result.get("is_complete", False),
                "summary": result.get("summary"),
                "detected_topics": result.get("detected_topics", []),
            }

        except json.JSONDecodeError:

            fallback_reply = raw_text.strip() if 'raw_text' in locals() else "I'm here to listen. How was your day?"

            history.append(f"User: {full_user_message}")
            history.append(f"AI: {fallback_reply}")

            return {
                "reply": fallback_reply,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        except Exception as e:

            print("ChatEngine error:", e)

            return {
                "reply": "Sorry, I had a moment. Could you say that again?",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def get_opening_message(self, session_id: str) -> dict:

        history = self._get_or_create_session(session_id)

        if len(history) > 0:
            return {
                "reply": "We're already chatting! Go ahead.",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        try:

            prompt = SYSTEM_PROMPT + "\n\nGenerate an opening diary conversation message."

            response = model.generate_content(prompt)

            raw_text = response.text

            result = _parse_response(raw_text)

            opening = result.get("reply", "Hey! How was your day?")

            history.append(f"AI: {opening}")

            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        except Exception as e:

            print("Opening message error:", e)

            opening = "Hey there! 👋 How's your day going?"

            history.append(f"AI: {opening}")

            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def reset_session(self, session_id: str):

        if session_id in self.sessions:
            del self.sessions[session_id]