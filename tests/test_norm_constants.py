import json
import numpy as np
from vis_utils import load_norm_constants, normalize_reward


def test_load_norm_constants_reads_json(tmp_path, monkeypatch):
    p = tmp_path / "norm_constants.json"
    p.write_text(json.dumps({"CliffWalking": {"x_random": -7700.0, "x_expert": -13.0}}))
    monkeypatch.setattr("vis_utils.NORM_CONSTANTS_PATH", str(p))
    d = load_norm_constants()
    assert d["CliffWalking"]["x_random"] == -7700.0


def test_random_policy_normalizes_to_zero():
    # With x_random set to the true random return, a random-level score maps to ~0.
    d = {"CliffWalking": {"x_random": -7700.0, "x_expert": -13.0}}
    assert abs(float(normalize_reward(-7700.0, "CliffWalking", d))) < 1e-9
