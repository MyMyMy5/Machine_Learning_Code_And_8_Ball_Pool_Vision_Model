from __future__ import annotations

import json

from src.supervised.manifest_inference import load_manifest


def test_load_manifest_normalizes_multiple_image_final_veto_configs(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": r"C:\My_Project\Test_New_Approach\runs\primary.pt",
                "image_final_veto": [
                    {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\a.pt"},
                    {"checkpoint_path": r"C:\My_Project\Test_New_Approach\runs\b.pt"},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert isinstance(manifest["image_final_veto"], list)
    assert len(manifest["image_final_veto"]) == 2
    assert str(manifest["image_final_veto"][0]["checkpoint"])
    assert str(manifest["image_final_veto"][1]["checkpoint_path"])


def test_load_manifest_normalizes_multiple_colored_blob_rescue_configs(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": r"C:\My_Project\Test_New_Approach\runs\primary.pt",
                "colored_blob_rescue": [
                    {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\a.pt"},
                    {"reranker_checkpoint": r"C:\My_Project\Test_New_Approach\runs\b.pt"},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert isinstance(manifest["colored_blob_rescue"], list)
    assert len(manifest["colored_blob_rescue"]) == 2
    assert str(manifest["colored_blob_rescue"][0]["checkpoint"])
    assert str(manifest["colored_blob_rescue"][1]["reranker_checkpoint"])


def test_load_manifest_normalizes_selector_candidate_pool_rescue_configs(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": r"C:\My_Project\Test_New_Approach\runs\primary.pt",
                "selector_candidate_pool_rescue": [
                    {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\a.pt"},
                    {"reranker_checkpoint": r"C:\My_Project\Test_New_Approach\runs\b.pt"},
                ],
                "post_external_selector_candidate_pool_rescue": [
                    {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\c.pt"},
                    {"reranker_checkpoint": r"C:\My_Project\Test_New_Approach\runs\d.pt"},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert isinstance(manifest["selector_candidate_pool_rescue"], list)
    assert len(manifest["selector_candidate_pool_rescue"]) == 2
    assert str(manifest["selector_candidate_pool_rescue"][0]["checkpoint"])
    assert str(manifest["selector_candidate_pool_rescue"][1]["reranker_checkpoint"])
    assert isinstance(manifest["post_external_selector_candidate_pool_rescue"], list)
    assert len(manifest["post_external_selector_candidate_pool_rescue"]) == 2
    assert str(manifest["post_external_selector_candidate_pool_rescue"][0]["checkpoint"])
    assert str(manifest["post_external_selector_candidate_pool_rescue"][1]["reranker_checkpoint"])


def test_load_manifest_normalizes_rescue_arbiter_configs(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "primary_checkpoint": r"C:\My_Project\Test_New_Approach\runs\primary.pt",
                "rescue_arbiter": {"checkpoint": r"C:\My_Project\Test_New_Approach\runs\arbiter.pt"},
                "post_external_rescue_arbiter": [
                    {"checkpoint_path": r"C:\My_Project\Test_New_Approach\runs\post.pt"},
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path)

    assert str(manifest["rescue_arbiter"]["checkpoint"])
    assert isinstance(manifest["post_external_rescue_arbiter"], list)
    assert str(manifest["post_external_rescue_arbiter"][0]["checkpoint_path"])
