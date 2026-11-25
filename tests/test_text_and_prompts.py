import pytest

from fashion_analyser.labels import COLOUR_CLASSES, LabelMaps, normalise_colour
from fashion_analyser.prompts import MAX_VARIANT_CHARS, draft_prompt
from fashion_analyser.text_attributes import extract_text_attributes, merge

CATS = ["Tshirts", "Shirts", "Casual Shoes", "Sports Shoes", "Watches", "Handbags"]


# ---- labels -------------------------------------------------------------------
@pytest.mark.parametrize("raw,group", [
    ("Navy Blue", "blue"), ("  navy   BLUE ", "blue"), ("Off White", "white"),
    ("Maroon", "red"), ("Unknownish", "other"), (None, "other"), (float("nan"), "other"), ("", "other"),
])
def test_normalise_colour(raw, group):
    assert normalise_colour(raw) == group


def test_every_group_is_a_class():
    assert "other" in COLOUR_CLASSES and "blue" in COLOUR_CLASSES


def test_label_maps_roundtrip(tmp_path):
    lm = LabelMaps(categories=["a", "b"])
    lm.save(tmp_path / "l.json")
    assert LabelMaps.load(tmp_path / "l.json").cat2idx == {"a": 0, "b": 1}


# ---- text attributes ------------------------------------------------------------
def test_longest_colour_wins():
    a = extract_text_attributes("Turtle Men Navy Blue Shirt", CATS)
    assert a.raw_colours == ["navy blue"] and a.colours == ["blue"]
    assert a.gender == "men" and a.category == "Shirts"


def test_multiple_colours_kept_in_order():
    a = extract_text_attributes("Women Black & White Striped Tshirt", CATS)
    assert a.colours == ["black", "white"] and a.gender == "women" and a.category == "Tshirts"


def test_longer_category_preferred():
    assert extract_text_attributes("Nike Men Sports Shoes", CATS).category == "Sports Shoes"


def test_no_false_colour_inside_words():
    # "tan" inside "Titan", "red" inside "Fred" must not count
    a = extract_text_attributes("Titan Men Watch by Fred", CATS)
    assert a.colours == [] and a.category == "Watches"


@pytest.mark.parametrize("text", ["", "   ", "!!!", None])
def test_empty_text(text):
    a = extract_text_attributes(text, CATS)
    assert a.colours == [] and a.gender is None and a.category is None


def _pred(cat="Shirts", cc=0.9, col="blue", lc=0.9, warnings=()):
    return {"category": cat, "category_confidence": cc, "colour": col,
            "colour_confidence": lc, "warnings": list(warnings)}


def test_merge_agreement_is_clean():
    rec = merge(_pred(), extract_text_attributes("Men Navy Blue Shirt", CATS))
    assert rec.conflicts == [] and not rec.needs_review and rec.colour == "blue"


def test_merge_confident_image_wins_but_flags_conflict():
    rec = merge(_pred(lc=0.95), extract_text_attributes("Men Red Shirt", CATS))
    assert rec.colour == "blue" and rec.needs_review and "colour" in rec.conflicts[0]


def test_merge_unsure_image_defers_to_text():
    rec = merge(_pred(col="grey", lc=0.3), extract_text_attributes("Men Red Shirt", CATS))
    assert rec.colour == "red" and rec.needs_review


def test_merge_quality_warning_triggers_review():
    rec = merge(_pred(warnings=["image looks blurry"]), extract_text_attributes("", CATS))
    assert rec.needs_review and rec.conflicts == []


# ---- prompts ------------------------------------------------------------------------
def test_basic_prompt():
    p = draft_prompt("Tshirts", "blue", "men", "linen, summer")
    assert p.prompt.startswith("blue men's tshirts, linen, summer,")
    assert "watermark" in p.negative_prompt


def test_variant_colour_replaces_original():
    p = draft_prompt("Tshirts", "blue", None, "olive green version")
    assert not p.prompt.startswith("blue") and "olive green version" in p.prompt


@pytest.mark.parametrize("variant", ["olive green, linen", "in Olive.", "maroon/suede"])
def test_variant_colour_with_punctuation_replaces_original(variant):
    assert not draft_prompt("Tshirts", "blue", "women", variant).prompt.startswith("blue")


def test_variant_is_sanitised_and_capped():
    p = draft_prompt("Shirts", "red", None, "(masterpiece:1.9) [ignore] {all}\n" + "x" * 500)
    assert not any(ch in p.variant for ch in "()[]{}:")
    assert len(p.variant) <= MAX_VARIANT_CHARS


@pytest.mark.parametrize("colour", ["other", "", None])
def test_unknown_colour_is_left_out(colour):
    assert draft_prompt("Watches", colour).prompt.startswith("watches,")


def test_multi_colour_wording():
    assert draft_prompt("Handbags", "multi").prompt.startswith("multicoloured handbags")


@pytest.mark.parametrize("cat", ["", "   "])
def test_missing_category_raises(cat):
    with pytest.raises(ValueError):
        draft_prompt(cat, "blue")
