"""Stable Diffusion prompt drafting for design variants.

The template keeps prompts consistent (studio product shot, same framing), so
generated variants are comparable with each other and with the original.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .labels import COLOUR_CLASSES, COLOUR_GROUPS

# Every colour word a variant might use: group names plus raw names ("olive", "navy blue").
COLOUR_NAMES = (set(COLOUR_CLASSES) | set(COLOUR_GROUPS)) - {"other", "multi"}

STYLE = ("professional e-commerce product photo, centered, plain white background, "
         "soft studio lighting, high detail, sharp focus, 4k")
NEGATIVE = ("person, model, mannequin, hands, text, watermark, logo, brand name, "
            "blurry, lowres, deformed, cropped, duplicate, extra objects, busy background")

MAX_VARIANT_CHARS = 120


@dataclass
class SDPrompt:
    prompt: str
    negative_prompt: str
    variant: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_variant(variant: str) -> str:
    v = " ".join((variant or "").replace("\n", " ").split())
    # The variant is free text from a user or an LLM: keep it short and drop
    # prompt-weighting syntax so it cannot dominate the template.
    for ch in "()[]{}<>:|":
        v = v.replace(ch, " ")
    v = " ".join(v.split())
    return v[:MAX_VARIANT_CHARS].rstrip(" ,.")


def draft_prompt(category: str, colour: str, gender: str | None = None,
                 variant: str = "") -> SDPrompt:
    if not category or not category.strip():
        raise ValueError("category is required to draft a prompt")
    variant = _clean_variant(variant)
    colour_words = "" if colour in ("other", "multi", "", None) else colour
    if colour == "multi":
        colour_words = "multicoloured"
    who = {"men": "men's", "women": "women's", "boys": "boys'", "girls": "girls'",
           "kids": "kids'", "unisex": "unisex"}.get(gender or "", "")

    # If the variant names a new colour, it replaces the original colour.
    v_words = " " + " ".join(re.sub(r"[^a-z ]", " ", variant.lower()).split()) + " "
    if any(f" {c} " in v_words for c in COLOUR_NAMES):
        colour_words = ""

    subject = " ".join(w for w in (colour_words, who, category.strip().lower()) if w)
    parts = [subject]
    if variant:
        parts.append(variant)
    parts.append(STYLE)
    return SDPrompt(prompt=", ".join(parts), negative_prompt=NEGATIVE, variant=variant)
