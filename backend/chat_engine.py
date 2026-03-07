<<<<<<< HEAD
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json

# Load API key
load_dotenv("apikey.env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")
=======
import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv("apikey.env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440

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
<<<<<<< HEAD
6. After gathering enough context (usually 4–8 exchanges), wrap up naturally.

RESPONSE FORMAT — you MUST respond with valid JSON only:
=======
6. After gathering enough context (usually 4-8 exchanges), wrap up naturally.

RESPONSE FORMAT — you MUST respond with valid JSON only, no markdown:
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
{
  "reply": "Your conversational message to the user",
  "is_complete": false,
  "summary": null,
  "detected_topics": ["topic1", "topic2"]
}

<<<<<<< HEAD
When is_complete is true, provide a brief emotional summary describing
the user's day and emotional state in 2–3 sentences.
"""
=======
When is_complete is true, provide a brief emotional summary in the "summary" field describing
the user's day and emotional state in 2-3 sentences. Set is_complete to true only when you feel
you have a good understanding of the user's current state (mood + context + at least one event).

IMPORTANT: You must ONLY output raw JSON. No markdown code blocks. No extra text."""
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440


def _parse_response(raw_text: str) -> dict:
    """Parse JSON from Gemini response, handling markdown code fences."""
    raw = raw_text.strip()
<<<<<<< HEAD

=======
    # Strip markdown code fences if present
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()
<<<<<<< HEAD

=======
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
    return json.loads(raw)


class ChatEngine:
    """Manages a conversational session with Gemini AI."""

    def __init__(self):
<<<<<<< HEAD
        self.sessions = {}
=======
        self.sessions = {}  # session_id -> conversation history
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440

    def _get_or_create_session(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def chat(self, session_id: str, user_message: str, emotion: str = None) -> dict:
<<<<<<< HEAD

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
=======
        """
        Send a user message and get an AI response.

        Args:
            session_id: Unique session identifier
            user_message: The user's transcribed message
            emotion: Optional detected emotion from the SER model

        Returns:
            dict with keys: reply, is_complete, summary, detected_topics
        """
        history = self._get_or_create_session(session_id)

        # Build the context-aware user message
        context_prefix = ""
        if emotion:
            context_prefix = f"[The user's voice emotion was detected as: {emotion}] "

        full_user_message = context_prefix + user_message

        # Build contents for the API call
        contents = []
        for msg in history:
            contents.append(msg)
        contents.append({"role": "user", "parts": [{"text": full_user_message}]})

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                },
            )

            raw_text = response.text
            result = _parse_response(raw_text)

            # Add to history
            history.append({"role": "user", "parts": [{"text": full_user_message}]})
            history.append({"role": "model", "parts": [{"text": raw_text}]})
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440

            return {
                "reply": result.get("reply", ""),
                "is_complete": result.get("is_complete", False),
                "summary": result.get("summary"),
                "detected_topics": result.get("detected_topics", []),
            }

        except json.JSONDecodeError:
<<<<<<< HEAD

            fallback_reply = raw_text.strip() if 'raw_text' in locals() else "I'm here to listen. How was your day?"

            history.append(f"User: {full_user_message}")
            history.append(f"AI: {fallback_reply}")

=======
            # If Gemini didn't return valid JSON, treat the raw text as the reply
            fallback_reply = raw_text.strip() if 'raw_text' in dir() else "I'm here to listen. How was your day?"
            history.append({"role": "user", "parts": [{"text": full_user_message}]})
            history.append({"role": "model", "parts": [{"text": fallback_reply}]})
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
            return {
                "reply": fallback_reply,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        except Exception as e:
<<<<<<< HEAD

            print("ChatEngine error:", e)

=======
            print(f"ChatEngine error: {e}")
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
            return {
                "reply": "Sorry, I had a moment. Could you say that again?",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def get_opening_message(self, session_id: str) -> dict:
<<<<<<< HEAD

        history = self._get_or_create_session(session_id)

=======
        """Generate the first AI message to start the conversation."""
        history = self._get_or_create_session(session_id)

        # Only generate opening if session is fresh
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
        if len(history) > 0:
            return {
                "reply": "We're already chatting! Go ahead.",
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        try:
<<<<<<< HEAD

            prompt = SYSTEM_PROMPT + "\n\nGenerate an opening diary conversation message."

            response = model.generate_content(prompt)

            raw_text = response.text

            result = _parse_response(raw_text)

            opening = result.get("reply", "Hey! How was your day?")

            history.append(f"AI: {opening}")
=======
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {"role": "user", "parts": [{"text": "The user just opened the diary chat. Generate your opening message."}]}
                ],
                config={
                    "system_instruction": SYSTEM_PROMPT,
                },
            )

            raw_text = response.text
            result = _parse_response(raw_text)
            opening = result.get("reply", "Hey! How was your day? 😊")

            # Store as model message in history (don't include the system prompt message)
            history.append({"role": "model", "parts": [{"text": raw_text}]})
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440

            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

        except Exception as e:
<<<<<<< HEAD

            print("Opening message error:", e)

            opening = "Hey there! 👋 How's your day going?"

            history.append(f"AI: {opening}")

=======
            print(f"Opening message error: {e}")
            opening = "Hey there! 👋 How's your day going?"
            history.append({"role": "model", "parts": [{"text": opening}]})
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
            return {
                "reply": opening,
                "is_complete": False,
                "summary": None,
                "detected_topics": [],
            }

    def reset_session(self, session_id: str):
<<<<<<< HEAD

        if session_id in self.sessions:
            del self.sessions[session_id]
=======
        """Clear a conversation session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
>>>>>>> 1199f8bd0fc876cc006503db54f3268eaccb9440
