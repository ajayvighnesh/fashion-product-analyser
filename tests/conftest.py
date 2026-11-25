import sys
from pathlib import Path

import pytest
from transformers import ViTConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def tiny_vit_config():
    """A very small randomly initialised ViT: runs on CPU in seconds, no download."""
    return ViTConfig(image_size=64, patch_size=16, hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    from make_synthetic_data import make
    return make(tmp_path_factory.mktemp("synth"), per_class=40)
