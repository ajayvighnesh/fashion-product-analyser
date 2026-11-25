"""End-to-end: data -> training -> checkpoint -> inference -> agent, on synthetic data."""

import json

import cv2
import numpy as np
import pytest
import torch
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from fashion_analyser.agent import analyse_product, build_agent, make_tools
from fashion_analyser.data import ProductImageDataset, load_metadata, split
from fashion_analyser.labels import LabelMaps
from fashion_analyser.model import MultiTaskClassifier
from fashion_analyser.predict import Analyser
from fashion_analyser.train import train


def test_metadata_and_split(synth_dir):
    df = load_metadata(synth_dir, min_per_class=10)
    assert set(df["category"]) == {"Tshirts", "Shoes", "Watches"}
    assert set(df["colour"]) <= {"blue", "red", "black", "green", "yellow"}
    tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(te["category"]) == set(df["category"])  # stratified


def test_metadata_skips_rows_without_images(synth_dir, tmp_path):
    import shutil
    d = tmp_path / "d"
    shutil.copytree(synth_dir, d)
    first = sorted((d / "images").glob("*.jpg"))[0]
    first.unlink()
    df = load_metadata(d, min_per_class=10)
    assert int(first.stem) not in set(df["id"])


def test_dataset_survives_corrupt_image(synth_dir, tmp_path):
    import shutil
    d = tmp_path / "d"
    shutil.copytree(synth_dir, d)
    df = load_metadata(d, min_per_class=10)
    open(df.loc[0, "image_path"], "wb").write(b"garbage")
    ds = ProductImageDataset(df, LabelMaps(sorted(df["category"].unique())), image_size=64)
    x, yc, yl = ds[0]
    assert x.shape == (3, 64, 64)


def test_forward_and_checkpoint_roundtrip(tiny_vit_config, tmp_path):
    m = MultiTaskClassifier.from_backbone_config(tiny_vit_config, 5, 13).eval()
    x = torch.randn(2, 3, 64, 64)
    lc, ll = m(x)
    assert lc.shape == (2, 5) and ll.shape == (2, 13)
    m.save(tmp_path)
    m2 = MultiTaskClassifier.load(tmp_path)
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-6)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, synth_dir):
    from transformers import ViTConfig
    cfg = ViTConfig(image_size=64, patch_size=16, hidden_size=32, num_hidden_layers=2,
                    num_attention_heads=2, intermediate_size=64)
    df = load_metadata(synth_dir, min_per_class=10)
    labels = LabelMaps(sorted(df["category"].unique()))
    model = MultiTaskClassifier.from_backbone_config(cfg, len(labels.categories), len(labels.colours))
    out = tmp_path_factory.mktemp("run")
    metrics = train(df, out, model=model, epochs=25, freeze_epochs=0, batch_size=16,
                    lr_head=3e-3, lr_backbone=1e-3, image_size=64, num_workers=0,
                    device="cpu", log=lambda *_: None)
    return out, metrics


def test_training_writes_artifacts_and_learns(trained_run):
    out, metrics = trained_run
    for f in ["model.pt", "labels.json", "metrics.json", "confusion_category.csv",
              "confusion_colour.csv"]:
        assert (out / f).is_file()
    # Synthetic shapes/colours are easy; a working pipeline beats chance (1/3) clearly.
    assert metrics["test"]["category_acc"] > 0.5


def test_analyser_predicts_and_flags(trained_run, synth_dir):
    out, _ = trained_run
    an = Analyser.from_checkpoint(out)
    an.image_size = 64
    img = next((synth_dir / "images").glob("*.jpg"))
    p = an.predict(img)
    assert p.category in an.labels.categories and p.colour in an.labels.colours
    assert 0 <= p.category_confidence <= 1 and len(p.top_categories) == 3
    tiny = an.predict(np.full((10, 10, 3), 128, np.uint8))
    assert any("very small" in w for w in tiny.warnings)


def test_pipeline_without_llm(trained_run, synth_dir):
    out, _ = trained_run
    an = Analyser.from_checkpoint(out)
    an.image_size = 64
    img = next((synth_dir / "images").glob("*.jpg"))
    rec = analyse_product(an, img, "Brand Men Purple Watches")
    assert rec["gender"] == "men"
    assert rec["text_attributes"]["colours"] == ["purple"]
    assert any("colour" in c for c in rec["conflicts"])  # no purple in the synthetic set


class ToolCallingFake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_uses_tools(trained_run, synth_dir):
    out, _ = trained_run
    an = Analyser.from_checkpoint(out)
    an.image_size = 64
    img = str(next((synth_dir / "images").glob("*.jpg")))
    script = iter([
        AIMessage(content="", tool_calls=[{"name": "analyse_product", "id": "1",
                  "args": {"image_path": img, "product_text": "Brand Men Red Shoes"}}]),
        AIMessage(content="", tool_calls=[{"name": "draft_design_prompt", "id": "2",
                  "args": {"category": "Shoes", "colour": "red", "variant": "olive green, suede"}}]),
        AIMessage(content="Done."),
    ])
    agent = build_agent(an, model=ToolCallingFake(messages=script), allowed_dir=synth_dir)
    result = agent.invoke({"messages": [{"role": "user", "content": "analyse and make a variant"}]})
    tool_out = [json.loads(m.content) for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_out[0]["category"] in an.labels.categories
    assert "olive green, suede" in tool_out[1]["prompt"]
    assert result["messages"][-1].content == "Done."


def test_agent_tool_errors_are_reported_not_raised(trained_run, synth_dir, tmp_path):
    out, _ = trained_run
    an = Analyser.from_checkpoint(out)
    analyse, draft = make_tools(an, allowed_dir=synth_dir)
    outside = tmp_path / "x.jpg"
    cv2.imwrite(str(outside), np.zeros((40, 40, 3), np.uint8))
    assert "not allowed" in json.loads(analyse.invoke({"image_path": str(outside)}))["error"]
    bad = synth_dir / "broken.jpg"
    bad.write_bytes(b"nope")
    assert "ImageLoadError" in json.loads(analyse.invoke({"image_path": str(bad)}))["error"]
    bad.unlink()
    assert "error" in json.loads(draft.invoke({"category": "", "colour": "red", "variant": ""}))
