import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class QuickReply:
    handled: bool
    reply: str
    intent: str  # "greeting" | "goodbye" | "thanks" | "help" | "unknown"


class ChatIntentService:
    """
    Handles conversational intents that do NOT require document evidence:
    greetings, goodbyes, thanks, etc.
    """

    GREETING = re.compile(r"^\s*(hi|hello|hey|yo|good\s(morning|afternoon|evening)|السلام عليكم|مرحبا|هلا)\b", re.I)
    GOODBYE = re.compile(r"^\s*(bye|goodbye|see you|see ya|take care|later|مع السلامه|سلام)\b", re.I)
    THANKS = re.compile(r"^\s*(thanks|thank you|thx|much appreciated|شكرا|مشكور)\b", re.I)
    HELP = re.compile(r"\b(help|what can you do|how do you work|what do you know)\b", re.I)

    @classmethod
    def quick_reply(cls, message: str) -> QuickReply:
        msg = (message or "").strip()
        low = msg.lower()

        if not msg:
            return QuickReply(True, "Say hi 👋 or ask me anything about Samah’s experience, projects, and skills.", "help")

        # Greeting
        if cls.GREETING.search(msg):
            return QuickReply(
                True,
                "Hi! 👋 Ask me anything about Samah — skills, projects, experience, or how to contact her.",
                "greeting"
            )

        # Goodbye
        if cls.GOODBYE.search(msg):
            return QuickReply(True, "Bye! 👋 If you need anything else about Samah’s profile, I’m here.", "goodbye")

        # Thanks
        if cls.THANKS.search(msg):
            return QuickReply(True, "You’re welcome! 😊 Want to know anything else about Samah?", "thanks")

        # Help / capabilities
        if cls.HELP.search(low):
            return QuickReply(
                True,
                "I can answer questions about Samah’s background using her uploaded CV and documents. "
                "Try: “What are her strongest technical skills?”, “Which projects used Django?”, or “How can I contact her?”",
                "help"
            )

        return QuickReply(False, "", "unknown")