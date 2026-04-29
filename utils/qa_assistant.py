"""
AgroSage - LangChain Q&A helper
"""

from __future__ import annotations

from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage


_BASE_PROMPT = (
    "You are AgroSage, an agronomy assistant for farmers and agronomy teams. "
    "Give clear, practical guidance with concise steps. "
    "If you are unsure, say so and suggest verifying with local experts or data. "
    "Avoid giving medical or legal advice. "
    "When discussing pesticides or treatments, provide high-level best practices "
    "and remind the user to follow local regulations and product labels."
)

_APP_CONTEXT = (
    "AgroSage modules:\n"
    "- Crop Recommender: predicts crops from soil N-P-K, temperature, humidity, pH, rainfall.\n"
    "- Disease Detector: classifies leaf images with CNN/ViT and confidence analysis.\n"
    "- Weather Analyst: historical climate trends and forecast-style summaries.\n"
)

_CONTEXT_FOCUS: Dict[str, str] = {
    "General agronomy": "Focus on practical agronomy guidance, crop planning, and farm workflows.",
    "Crop recommendation": "Focus on interpreting soil nutrients, pH, rainfall, and crop suitability.",
    "Disease detection": "Focus on leaf disease symptoms, prevention, and safe scouting steps.",
    "Weather and climate": "Focus on weather impacts, climate trends, and seasonal planning.",
    "AgroSage app help": "Explain how to use AgroSage features, inputs, and outputs.",
}


def build_system_prompt(focus: str) -> str:
    """Build the system prompt for the selected focus area."""
    focus_line = _CONTEXT_FOCUS.get(focus, _CONTEXT_FOCUS["General agronomy"])
    return f"{_BASE_PROMPT}\n\n{_APP_CONTEXT}\n{focus_line}"


def build_messages(question: str, focus: str) -> list:
    """Return LangChain messages for a single-turn Q&A response."""
    system_prompt = build_system_prompt(focus)
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question.strip()),
    ]
