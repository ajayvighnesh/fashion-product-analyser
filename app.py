"""Gradio demo: upload a product photo, optionally paste its title, get attributes
and a design-variant prompt.

    python app.py --run-dir runs/vit
"""

from __future__ import annotations

import argparse
import json

import gradio as gr
import numpy as np

from fashion_analyser.agent import analyse_product
from fashion_analyser.predict import Analyser
from fashion_analyser.prompts import draft_prompt


def swatches(rgbs) -> str:
    cells = "".join(
        f'<div style="display:inline-block;margin-right:8px;text-align:center">'
        f'<div style="width:40px;height:40px;border-radius:6px;border:1px solid #ccc;'
        f'background:rgb{tuple(c)}"></div><small>{share:.0%}</small></div>'
        for c, share in rgbs)
    return f"<div>{cells}</div>"


def build_ui(analyser: Analyser) -> gr.Blocks:
    def run(image: np.ndarray, text: str, variant: str):
        if image is None:
            return "Upload an image first.", "", "", ""
        rec = analyse_product(analyser, image, text or "")
        pred = analyser.predict(image)
        status = "Needs human review" if rec["needs_review"] else "Looks consistent"
        notes = "\n".join(f"- {c}" for c in rec["conflicts"] + rec["warnings"]) or "- none"
        summary = (f"### {rec['colour']} {rec['category']}"
                   f"{' (' + rec['gender'] + ')' if rec['gender'] else ''}\n"
                   f"**{status}**\n\nIssues:\n{notes}")
        prompt = ""
        if variant.strip():
            prompt = json.dumps(draft_prompt(rec["category"], rec["colour"], rec["gender"],
                                             variant).to_dict(), indent=2)
        return summary, swatches(pred.dominant_rgb), json.dumps(rec, indent=2), prompt

    with gr.Blocks(title="Fashion Product Analyser") as demo:
        gr.Markdown("# Fashion Product Analyser\nCategory and colour from the photo, "
                    "cross-checked against the product text, plus a Stable Diffusion prompt "
                    "for a design variant.")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="numpy", label="Product photo")
                txt = gr.Textbox(label="Product title (optional)",
                                 placeholder="e.g. Puma Men Black Running Shoes")
                var = gr.Textbox(label="Design variant (optional)",
                                 placeholder="e.g. olive green, linen, summer collection")
                btn = gr.Button("Analyse", variant="primary")
            with gr.Column():
                out_md = gr.Markdown()
                out_sw = gr.HTML(label="Dominant colours")
                out_prompt = gr.Code(label="Stable Diffusion prompt", language="json")
                out_json = gr.Code(label="Full record", language="json")
        btn.click(run, [img, txt, var], [out_md, out_sw, out_json, out_prompt])
    return demo


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="training output directory")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    build_ui(Analyser.from_checkpoint(a.run_dir, a.device)).launch()
