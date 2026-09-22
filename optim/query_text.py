"""Remove explicit list markers without stripping digits from technical terms."""
import re


def strip_list_marker(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^(?:[-*•]\s+)", "", text)
    # Require punctuation; '3NF', '2026 trends' and decimal versions are content.
    return re.sub(r"^(?:\d+[.)、](?!\d)\s*|[（(]\d+[）)]\s*)", "", text).strip()
