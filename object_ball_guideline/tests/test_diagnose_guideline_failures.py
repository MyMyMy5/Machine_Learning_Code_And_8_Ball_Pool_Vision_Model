from __future__ import annotations

from tools.diagnose_guideline_failures import _normalize_rescue_config


def test_normalize_rescue_config_accepts_multiple_image_veto_configs() -> None:
    manifest = {
        "image_final_veto": [
            {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\a.pt"},
            {"checkpoint_path": r"C:\My_Project\Test_New_Approach\runs\b.pt"},
        ]
    }

    normalized = _normalize_rescue_config(manifest, "image_final_veto")

    assert isinstance(normalized, list)
    assert len(normalized) == 2
    assert normalized[0]["checkpoint"]
    assert normalized[1]["checkpoint_path"]
