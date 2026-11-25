"""Optional: render a drafted prompt with Stable Diffusion (needs a GPU in practice).

    pip install diffusers accelerate
    python -m fashion_analyser.generate --prompt-json prompt.json --out variant.png
"""

from __future__ import annotations

import argparse
import json


def generate(prompt: str, negative_prompt: str, out: str,
             model_id: str = "stabilityai/stable-diffusion-2-1-base",
             steps: int = 30, seed: int = 0) -> str:
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    g = torch.Generator(device).manual_seed(seed)
    image = pipe(prompt, negative_prompt=negative_prompt, num_inference_steps=steps,
                 generator=g).images[0]
    image.save(out)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-json", required=True, help="output of draft_design_prompt")
    p.add_argument("--out", default="variant.png")
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-2-1-base")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    d = json.loads(open(a.prompt_json).read())
    print(generate(d["prompt"], d["negative_prompt"], a.out, a.model_id, a.steps, a.seed))


if __name__ == "__main__":
    main()
