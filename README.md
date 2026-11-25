# Multimodal Fashion Product Analyser

![tests](https://github.com/ajayvighnesh/fashion-product-analyser/actions/workflows/tests.yml/badge.svg)

Turn a product photo and its title into clean, structured product data, and use
that data to draft Stable Diffusion prompts for new design variants.

- **Computer vision:** a pretrained Hugging Face backbone (ViT by default), fine-tuned in PyTorch with
  two heads that predict **product category** and **colour**. OpenCV handles preprocessing and
  image-quality checks.
- **Text + image fusion:** colour, gender and category are pulled from the product title and
  cross-checked against the photo. Contradictions are flagged for human review.
- **GenAI agent:** a LangChain agent answers questions about a product and drafts design-variant
  prompts. It may only state attributes it got from a tool, so its answers are traceable.
- **Tested with edge cases:** 64 tests cover corrupt files, transparent PNGs, grayscale and 16-bit
  images, tiny or extreme-aspect photos, blur, under/over-exposure, conflicting titles, prompt-injection
  characters in variant text, and a full train, predict and agent run.

```mermaid
flowchart LR
    A[Product photo] --> B[OpenCV<br/>load · quality check · pad · resize]
    B --> C[ViT backbone<br/>fine-tuned]
    C --> D[Category head]
    C --> E[Colour head]
    T[Product title] --> F[Text attribute rules]
    D & E & F --> G[Merge + conflict check]
    G --> H{LangChain agent}
    H -->|tool| G
    H -->|tool| I[Prompt template]
    I --> J[Stable Diffusion<br/>optional]
```

## Example

Input: a photo plus the title `"Turtle Men Navy Blue Shirt"`; variant request `"olive green, linen, summer"`.

The product record (field names are real; the numbers depend on your trained model):

```json
{
  "category": "Shirts",
  "colour": "blue",
  "gender": "men",
  "image_confidence": {"category": 0.93, "colour": 0.88},
  "text_attributes": {"colours": ["blue"], "raw_colours": ["navy blue"], "gender": "men", "category": "Shirts"},
  "conflicts": [],
  "warnings": [],
  "needs_review": false
}
```

The drafted prompt (this is exactly what the code produces for these inputs):

```text
men's shirts, olive green, linen, summer, professional e-commerce product photo, centered,
plain white background, soft studio lighting, high detail, sharp focus, 4k
negative: person, model, mannequin, hands, text, watermark, logo, brand name, blurry, ...
```

The original colour is dropped because the variant names a new one.

## Quick start

```bash
git clone https://github.com/ajayvighnesh/fashion-product-analyser
cd fashion-product-analyser
python -m venv .venv && source .venv/bin/activate
pip install -e ".[demo,data,dev]"

pytest -q                                   # 64 tests, no download needed (~10 s on CPU)

python scripts/download_data.py             # Fashion Product Images (Small), ~44k products
python -m fashion_analyser.train --data-dir data --out-dir runs/vit --epochs 5
python app.py --run-dir runs/vit            # Gradio demo at http://127.0.0.1:7860
```

**Train on a free GPU:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ajayvighnesh/fashion-product-analyser/blob/main/notebooks/train_colab.ipynb)
[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) installs everything, downloads the data,
trains with mixed precision on a T4, and shows results, confusion matrices, example predictions
and photo/title conflict rates.

Training on a laptop CPU: add `--max-samples 4000 --epochs 3` for a quicker run.

Training writes to `runs/<name>/`: `model.pt`, `labels.json`, `metrics.json` (per-epoch history and
test-set accuracy and macro-F1 for both heads) and confusion matrices as CSV.

### Using the agent

```bash
pip install -e ".[llm]"
export OPENAI_API_KEY=...      # or run a local model with Ollama, no key needed
```

```python
from fashion_analyser.predict import Analyser
from fashion_analyser.agent import build_agent

analyser = Analyser.from_checkpoint("runs/vit")
agent = build_agent(analyser, model="openai:gpt-4o-mini", allowed_dir="data")  # or "ollama:llama3.1"
reply = agent.invoke({"messages": [{"role": "user", "content":
    "Analyse data/images/15970.jpg, title 'Turtle Men Navy Blue Shirt'. "
    "Is the data consistent? Then draft a summer variant in olive green linen."}]})
print(reply["messages"][-1].content)
```

Without an LLM, the same pipeline is one call: `analyse_product(analyser, image, title)`.

To render a drafted prompt: `pip install -e ".[generate]"`, then
`python -m fashion_analyser.generate --prompt-json prompt.json` (a GPU is recommended).

## Design decisions

| Decision | Why |
|---|---|
| One backbone, two heads | Category and colour share visual features; one forward pass, half the compute of two models. |
| 40+ raw colour names merged into 14 groups | "Navy Blue" vs "Blue" or "Off White" vs "Cream" can't be told apart reliably from an 80×60 photo, and the labels are inconsistent anyway. Mapping in `labels.py`. |
| No hue or colour-jitter augmentation | Colour is a label; changing hue would teach the model wrong answers. Only flips, crops and a small brightness change. |
| Pad to square on white, never stretch | Stretching distorts shape, which is the main cue for category. |
| Frozen-backbone warm-up, then full fine-tune | Random heads produce large gradients that would damage pretrained features in epoch 1. |
| Class-weighted loss + label smoothing | The dataset is imbalanced (many t-shirts, few belts); macro-F1 is the selection metric so rare classes count. |
| Confident image beats text for colour; unsure image defers to text | Titles are often generic or wrong, but a low-confidence prediction shouldn't override an explicit title. Either way, the disagreement is reported. |
| Agent tools are deterministic code | The LLM chooses *what* to do; it never invents attributes. Every number in an answer comes from the model or the rules. |
| Variant text is sanitised and capped | It comes from a user or an LLM; removing SD weighting syntax like `(word:1.9)` stops it from overpowering the template. |

## Edge cases tested

| Case | Behaviour |
|---|---|
| Corrupt file, empty bytes, text file named `.jpg`, missing file | `ImageLoadError`; the agent reports it instead of crashing |
| Transparent PNG | Composited onto white (not black) |
| Grayscale, 16-bit PNG | Converted to 8-bit RGB |
| Image under 32 px, aspect ratio over 3:1 | Predicted, with a warning |
| Blurry, very dark, almost all white | Warning; `needs_review = true` |
| Title colour contradicts photo | Conflict listed with both values and confidence |
| "Titan" (contains "tan"), "Fred" (contains "red") | Not read as colours (whole-word matching) |
| "Navy Blue" in title | Matched as navy blue, not as plain blue |
| Variant like `(masterpiece:1.9) [ignore]` or 500 chars | Syntax removed, capped at 120 chars |
| Truncated image inside the training set | Replaced by a blank image; training continues |
| Agent asked to read a file outside the data folder | Refused |

## Project structure

```
src/fashion_analyser/
  preprocess.py        OpenCV loading, quality checks, padding, augmentation, dominant colours
  labels.py            category vocabulary, colour grouping
  data.py              styles.csv loading, stratified split, PyTorch Dataset
  model.py             backbone + two heads, save/load
  train.py             two-phase fine-tuning, metrics, confusion matrices
  predict.py           Analyser: image -> Prediction (+ warnings)
  text_attributes.py   title parsing and image/text conflict check
  prompts.py           Stable Diffusion prompt template
  agent.py             LangChain tools and agent
  generate.py          optional Stable Diffusion rendering
app.py                 Gradio demo
scripts/               dataset download, synthetic data for tests
tests/                 64 pytest tests
```

## Data

[Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
by Param Aggarwal: 44k e-commerce product photos with category, colour, gender and title metadata.
The download script also supports the Hugging Face mirror (`ashraq/fashion-product-images-small`).
The data isn't included in this repo.

## Limitations

- The images are small (80×60), so fine detail such as fabric or pattern isn't learnable from this dataset.
- Colour labels in the dataset are noisy; colour accuracy is capped by label quality.
- Title parsing is rule-based and English-only. That's fine for this dataset, but it would need an NER
  model or LLM extraction for free-form descriptions.
- Stable Diffusion output still needs a human to check it. The prompt template controls framing, not design quality.

## License

MIT
