"""LangChain agent on top of the vision model.

The agent answers questions about a product and drafts design-variant
prompts. It never guesses attributes itself: it has to call tools, and every
tool is backed by deterministic code (vision model, text rules, prompt
template), so answers are traceable.

    from fashion_analyser.agent import build_agent
    agent = build_agent(analyser, model="openai:gpt-4o-mini")   # or "ollama:llama3.1"
    agent.invoke({"messages": [{"role": "user", "content":
        "Analyse data/images/15970.jpg (title: 'Turtle Men Navy Blue Shirt') "
        "and draft a summer variant in olive green."}]})

`analyse_product()` runs the same pipeline without an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from .predict import Analyser
from .prompts import draft_prompt
from .text_attributes import extract_text_attributes, merge

SYSTEM_PROMPT = """You are a product-data assistant for a fashion design team.

Rules:
- Never state a product attribute (category, colour, gender) you did not get from a tool.
- To look at an image, call analyse_product with the image path and the product text (if any).
- If the result has conflicts or warnings, say so plainly and recommend a human check.
- To create a design variant, call draft_design_prompt with the attributes from analyse_product
  and a short variant description. Return the prompt and negative prompt exactly as the tool gives them.
- If a tool returns an error, explain it to the user; do not invent a result.
- Keep answers short and structured."""


def analyse_product(analyser: Analyser, image, product_text: str = "") -> dict:
    """Vision prediction + text attributes + conflict check, as one record."""
    pred = analyser.predict(image).to_dict()
    text = extract_text_attributes(product_text, analyser.labels.categories)
    record = merge(pred, text).to_dict()
    record["top_categories"] = pred["top_categories"]
    record["top_colours"] = pred["top_colours"]
    record["dominant_rgb"] = pred["dominant_rgb"]
    return record


def make_tools(analyser: Analyser, allowed_dir: str | Path | None = None):
    root = Path(allowed_dir).resolve() if allowed_dir else None

    @tool
    def analyse_product_tool(image_path: str, product_text: str = "") -> str:
        """Analyse a product photo. Returns category, colour and gender with confidences,
        attributes found in the product text, and any conflicts or quality warnings.

        Args:
            image_path: path to the product image file.
            product_text: optional product title or description.
        """
        p = Path(image_path).expanduser().resolve()
        if root and root not in p.parents and p != root:
            return json.dumps({"error": f"access outside {root} is not allowed"})
        try:
            return json.dumps(analyse_product(analyser, p, product_text))
        except Exception as e:  # surfaced to the LLM, which reports it
            return json.dumps({"error": f"{type(e).__name__}: {e}"})

    @tool
    def draft_design_prompt(category: str, colour: str, variant: str,
                            gender: str | None = None) -> str:
        """Draft a Stable Diffusion prompt for a design variant of a product.

        Args:
            category: product category from analyse_product (e.g. 'Tshirts').
            colour: colour from analyse_product (e.g. 'blue').
            variant: short description of the change, e.g. 'olive green, linen, summer'.
            gender: optional, from analyse_product.
        """
        try:
            return json.dumps(draft_prompt(category, colour, gender, variant).to_dict())
        except ValueError as e:
            return json.dumps({"error": str(e)})

    analyse_product_tool.name = "analyse_product"
    return [analyse_product_tool, draft_design_prompt]


def build_agent(analyser: Analyser, model="openai:gpt-4o-mini", allowed_dir=None):
    """Create the agent. `model` is a LangChain chat model or a "provider:model" string."""
    from langchain.agents import create_agent

    if isinstance(model, str):
        from langchain.chat_models import init_chat_model
        model = init_chat_model(model, temperature=0)
    return create_agent(model, make_tools(analyser, allowed_dir), system_prompt=SYSTEM_PROMPT)
