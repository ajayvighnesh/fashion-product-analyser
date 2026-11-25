"""Combine what the image shows with what the product text says.

Product titles in the dataset look like "Nike Men Navy Blue Running Shoes".
Simple, transparent rules pull colour, gender and category out of the text,
then the result is cross-checked against the vision model's prediction so
contradictions (e.g. title says "red", photo is clearly blue) get flagged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .labels import COLOUR_GROUPS

GENDER_WORDS = {
    "men": "men", "mens": "men", "man": "men", "male": "men",
    "women": "women", "womens": "women", "woman": "women", "female": "women", "ladies": "women",
    "boys": "boys", "boy": "boys", "girls": "girls", "girl": "girls",
    "unisex": "unisex", "kids": "kids",
}


def _norm(text: str) -> str:
    return " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) + " "


@dataclass
class TextAttributes:
    colours: list[str] = field(default_factory=list)   # colour groups, in order of appearance
    raw_colours: list[str] = field(default_factory=list)
    gender: str | None = None
    category: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def extract_text_attributes(text: str, categories: list[str] | None = None) -> TextAttributes:
    t = _norm(text or "")
    attrs = TextAttributes()

    # Longest colour names first, so "navy blue" wins over "blue".
    found: list[tuple[int, str]] = []
    taken = [False] * len(t)
    for name in sorted(COLOUR_GROUPS, key=len, reverse=True):
        for m in re.finditer(rf" {re.escape(name)} ", t):
            s, e = m.start() + 1, m.end() - 1
            if not any(taken[s:e]):
                found.append((s, name))
                for i in range(s, e):
                    taken[i] = True
    for _, name in sorted(found):
        attrs.raw_colours.append(name)
        group = COLOUR_GROUPS[name]
        if group not in attrs.colours:
            attrs.colours.append(group)

    for word in t.split():
        if word in GENDER_WORDS:
            attrs.gender = GENDER_WORDS[word]
            break

    if categories:
        best = None
        for cat in categories:
            c = cat.lower()
            forms = {c, c + "s"}
            if c.endswith("es"):
                forms.add(c[:-2])  # "watches" -> "watch"
            if c.endswith("s"):
                forms.add(c[:-1])  # "shirts" -> "shirt"
            for form in forms:
                if f" {form} " in t and (best is None or len(cat) > len(best)):
                    best = cat
        attrs.category = best
    return attrs


@dataclass
class ProductRecord:
    category: str
    colour: str
    gender: str | None
    image_confidence: dict
    text_attributes: dict
    conflicts: list[str]
    warnings: list[str]
    needs_review: bool

    def to_dict(self) -> dict:
        return asdict(self)


def merge(prediction: dict, text: TextAttributes, min_conf: float = 0.5) -> ProductRecord:
    """Merge image prediction and text attributes into one product record.

    Rules: a confident image prediction wins for colour (titles are often
    wrong or generic). If the image is unsure and the text names a colour, the
    text is used. Any disagreement is listed in `conflicts`.
    """
    conflicts: list[str] = []
    colour = prediction["colour"]
    if text.colours and colour not in text.colours:
        conflicts.append(f"colour: image says '{colour}' "
                         f"({prediction['colour_confidence']:.0%}), text says '{text.raw_colours[0]}'")
        if prediction["colour_confidence"] < min_conf:
            colour = text.colours[0]

    category = prediction["category"]
    if text.category and text.category != category:
        conflicts.append(f"category: image says '{category}' "
                         f"({prediction['category_confidence']:.0%}), text says '{text.category}'")
        if prediction["category_confidence"] < min_conf:
            category = text.category

    warnings = list(prediction.get("warnings", []))
    return ProductRecord(
        category=category, colour=colour, gender=text.gender,
        image_confidence={"category": prediction["category_confidence"],
                          "colour": prediction["colour_confidence"]},
        text_attributes=text.to_dict(), conflicts=conflicts, warnings=warnings,
        needs_review=bool(conflicts or warnings),
    )
