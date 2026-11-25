"""Label handling: category vocabulary and colour normalisation.

The Fashion Product Images dataset has 40+ raw colour names ("Navy Blue",
"Teal", "Off White", ...). Many are near-duplicates that a photo cannot tell
apart reliably, so they are merged into a smaller set of colour groups.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# Raw dataset colour -> colour group. Anything not listed maps to "other".
COLOUR_GROUPS: dict[str, str] = {
    "black": "black",
    "charcoal": "black",
    "white": "white",
    "off white": "white",
    "cream": "white",
    "grey": "grey",
    "grey melange": "grey",
    "steel": "grey",
    "silver": "grey",
    "blue": "blue",
    "navy blue": "blue",
    "turquoise blue": "blue",
    "teal": "blue",
    "red": "red",
    "maroon": "red",
    "burgundy": "red",
    "rust": "red",
    "pink": "pink",
    "magenta": "pink",
    "rose": "pink",
    "peach": "pink",
    "green": "green",
    "olive": "green",
    "lime green": "green",
    "sea green": "green",
    "fluorescent green": "green",
    "yellow": "yellow",
    "mustard": "yellow",
    "gold": "yellow",
    "orange": "orange",
    "coral": "orange",
    "brown": "brown",
    "coffee brown": "brown",
    "tan": "brown",
    "khaki": "brown",
    "beige": "beige",
    "skin": "beige",
    "nude": "beige",
    "mushroom brown": "beige",
    "taupe": "beige",
    "purple": "purple",
    "lavender": "purple",
    "mauve": "purple",
    "multi": "multi",
}

COLOUR_CLASSES: list[str] = sorted(set(COLOUR_GROUPS.values()) | {"other"})


def normalise_colour(raw: object) -> str:
    """Map a raw dataset colour name to its colour group."""
    if not isinstance(raw, str) or not raw.strip():
        return "other"
    return COLOUR_GROUPS.get(" ".join(raw.lower().split()), "other")


@dataclass
class LabelMaps:
    """Index <-> name mappings for both prediction heads."""

    categories: list[str]
    colours: list[str] = field(default_factory=lambda: list(COLOUR_CLASSES))

    @property
    def cat2idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.categories)}

    @property
    def col2idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.colours)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"categories": self.categories, "colours": self.colours}, indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabelMaps":
        data = json.loads(Path(path).read_text())
        return cls(categories=data["categories"], colours=data["colours"])
