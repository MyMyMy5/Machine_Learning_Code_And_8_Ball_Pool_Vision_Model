from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools.build_image_final_veto_dataset import build_dataset


def _row(
    *,
    item_id: str,
    gt_pixels: int,
    pred_pixels: int,
    iou: float,
    winner_source: str = "reticle_global_0",
) -> dict[str, object]:
    output_dir = Path("runs/test") / item_id
    return {
        "id": item_id,
        "split": "focused",
        "kind": "positive" if gt_pixels > 0 else "negative",
        "source": "unit",
        "image_path": "image.png",
        "output_dir": str(output_dir),
        "winner_source": winner_source,
        "gt_pixels": gt_pixels,
        "pred_pixels": pred_pixels,
        "iou": iou,
    }


def test_build_dataset_filters_keep_examples_by_min_iou(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rows = [
        _row(item_id="good_positive", gt_pixels=100, pred_pixels=80, iou=0.75),
        _row(item_id="bad_positive", gt_pixels=100, pred_pixels=80, iou=0.25),
        _row(item_id="negative_fp", gt_pixels=0, pred_pixels=80, iou=0.0),
    ]
    for row in rows:
        mask_path = Path(str(row["output_dir"])) / "mask_final.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (2, 2), color=255).save(mask_path)

    dataset = build_dataset(
        summary={"rows": rows},
        sources={"reticle_global_0"},
        min_keep_iou=0.5,
    )

    assert dataset["count"] == 2
    assert dataset["target_keep_count"] == 1
    assert dataset["target_veto_count"] == 1
    assert [item["id"] for item in dataset["items"]] == ["good_positive", "negative_fp"]
