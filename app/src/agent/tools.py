"""
Tools available to the travel agent.
Each tool takes a single destination string so tool schemas stay small enough
for the small local models used in the workshop.
"""

import difflib
import logging
import os
import re
from datetime import date
from typing import List, Optional

from langchain_core.tools import BaseTool, tool

logger = logging.getLogger(__name__)

# Destinations in the KB that are in the southern hemisphere (seasons are inverted).
SOUTHERN_HEMISPHERE = {
    "brisbane", "buenos-aires", "canberra", "johannesburg", "melbourne",
    "perth", "sao-paulo", "sydney", "wellington",
}

# Destinations in the KB with a tropical climate (wet/dry seasons instead of four seasons).
TROPICAL = {
    "bangkok", "bengaluru", "chennai", "jakarta", "manila", "mumbai",
    "petaling-jaya", "pune", "singapore",
}

_NORTHERN_SEASONS = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
_OPPOSITE_SEASON = {"winter": "summer", "summer": "winter", "spring": "autumn", "autumn": "spring"}


def normalize_destination(name: str) -> str:
    """Normalize a destination name to the KB title format (e.g. 'New York' -> 'new-york')."""
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def load_known_destinations(destinations_path: str) -> List[str]:
    """Return the KB destination titles (HTML file stems in the destinations directory)."""
    if not os.path.isdir(destinations_path):
        logger.warning(f"Destinations path does not exist: {destinations_path}")
        return []
    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(destinations_path)
        if filename.endswith(".html")
    )


def match_known_destination(text: str, known_destinations: List[str]) -> Optional[str]:
    """Find a known destination mentioned in free text (longest match wins)."""
    normalized_text = f"-{normalize_destination(text)}-"
    matches = [d for d in known_destinations if f"-{d}-" in normalized_text]
    return max(matches, key=len) if matches else None


def season_for(destination: str, today: Optional[date] = None) -> str:
    """Describe the current season at a destination."""
    today = today or date.today()
    slug = normalize_destination(destination)
    if slug in TROPICAL:
        return f"Today is {today.isoformat()}. {destination} has a tropical climate with wet and dry seasons rather than four seasons."
    season = _NORTHERN_SEASONS[today.month]
    hemisphere = "northern"
    if slug in SOUTHERN_HEMISPHERE:
        season = _OPPOSITE_SEASON[season]
        hemisphere = "southern"
    return f"Today is {today.isoformat()}. It is {season} in {destination} ({hemisphere} hemisphere)."


def build_tools(rag_pipeline, known_destinations: List[str]) -> List[BaseTool]:
    """Create the agent tools bound to the shared RAG pipeline and KB destination list."""

    @tool
    def validate_destination(destination: str) -> str:
        """Check whether a destination exists in the travel knowledge base."""
        slug = normalize_destination(destination)
        if slug in known_destinations:
            return f"VALID: '{destination}' is in the knowledge base as '{slug}'."
        found = match_known_destination(destination, known_destinations)
        if found:
            return f"VALID: '{destination}' matches knowledge base destination '{found}'."
        suggestions = difflib.get_close_matches(slug, known_destinations, n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        return f"INVALID: '{destination}' is not in the knowledge base.{hint}"

    @tool
    def search_destination_kb(destination: str) -> str:
        """Search the travel knowledge base for facts about a destination."""
        known = match_known_destination(destination, known_destinations)
        retrieval = rag_pipeline.retrieve_context(known or destination)
        # For unknown destinations, only accept context that actually mentions the destination;
        # loosely related vector-search results would let the model improvise an answer.
        if not retrieval.contexts or (not known and not retrieval.destination_found):
            return "NO INFORMATION: the knowledge base returned no content for this destination."
        return "\n\n".join(retrieval.contexts)

    @tool
    def get_current_season(destination: str) -> str:
        """Get today's date and the current season at a destination."""
        return season_for(destination)

    return [validate_destination, search_destination_kb, get_current_season]
