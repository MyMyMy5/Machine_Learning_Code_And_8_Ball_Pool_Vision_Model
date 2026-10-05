from __future__ import annotations

import json
from pathlib import Path

import tools.build_gold_board as gold_board_module
from tools.build_gold_board import build_gold_board, frame_group_key


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_frame_group_key_buckets_neighboring_video_frames() -> None:
    first = Path(
        "C:/frames/video_a/YTDown.com_Example_f00000933.png"
    )
    nearby = Path(
        "C:/frames/video_a/YTDown.com_Example_f00000934.png"
    )
    far = Path(
        "C:/frames/video_a/YTDown.com_Example_f00002000.png"
    )

    assert frame_group_key(first, bucket_size=25) == frame_group_key(nearby, bucket_size=25)
    assert frame_group_key(first, bucket_size=25) != frame_group_key(far, bucket_size=25)


def test_build_gold_board_writes_holdout_and_training_exclusions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        gold_board_module,
        "_stable_fraction",
        lambda key: 0.1 if ("Example" in key or "holdout" in key) else 0.9,
    )
    old_index = {
        "val": [
            {
                "id": "old_val",
                "image_path": str(tmp_path / "old_val.png"),
                "mask_path": str(tmp_path / "old_val_mask.png"),
            }
        ],
        "hard_val": [
            {
                "id": "old_hard",
                "image_path": str(tmp_path / "old_hard.png"),
                "mask_path": str(tmp_path / "old_hard_mask.png"),
            }
        ],
    }
    index_path = tmp_path / "old_index.json"
    index_path.write_text(json.dumps(old_index), encoding="utf-8")

    data_root = tmp_path / "data_zoomprobe"
    harvest_rows = [
        {
            "id": "harvest_a",
            "image_path": str(data_root / "images" / "harvest_a.png"),
            "mask_path": str(data_root / "masks" / "harvest_a.png"),
            "source_image_path": "C:/frames/video_a/YTDown.com_Example_f00000025.png",
            "created_at": "2026-04-24T10:00:00Z",
        },
        {
            "id": "harvest_b",
            "image_path": str(data_root / "images" / "harvest_b.png"),
            "mask_path": str(data_root / "masks" / "harvest_b.png"),
            "source_image_path": "C:/frames/video_a/YTDown.com_Example_f00000026.png",
            "created_at": "2026-04-24T10:00:01Z",
        },
        {
            "id": "old_train",
            "image_path": str(data_root / "images" / "old_train.png"),
            "mask_path": str(data_root / "masks" / "old_train.png"),
            "source_image_path": "C:/images/old_train.png",
            "created_at": "2026-02-18T10:00:00Z",
        },
    ]
    _write_jsonl(data_root / "annotations.jsonl", harvest_rows)

    rejected_root = data_root / "Rejected"
    _write_jsonl(
        rejected_root / "annotations.jsonl",
        [
            {
                "id": "reject_holdout",
                "image_path": str(rejected_root / "images" / "reject_holdout.png"),
                "mask_path": str(rejected_root / "masks" / "reject_holdout.png"),
                "source_image_path": "C:/frames/video_b/holdout_f00000010.png",
                "created_at": "2026-04-24T11:00:00Z",
            },
            {
                "id": "reject_train",
                "image_path": str(rejected_root / "images" / "reject_train.png"),
                "mask_path": str(rejected_root / "masks" / "reject_train.png"),
                "source_image_path": "C:/frames/video_b/train_f00000010.png",
                "created_at": "2026-04-24T11:00:01Z",
            },
        ],
    )

    negative_root = tmp_path / "negative_selected"
    negative_root.mkdir()
    (negative_root / "flat_holdout.png").write_bytes(b"fake")
    (negative_root / "flat_train.png").write_bytes(b"fake")

    output_root = tmp_path / "gold"
    board = build_gold_board(
        index_json=index_path,
        data_root=data_root,
        rejected_root=rejected_root,
        negative_root=negative_root,
        output_root=output_root,
        holdout_fraction=0.5,
        frame_bucket_size=25,
    )

    assert [item["id"] for item in board["splits"]["old_val"]] == ["old_val"]
    assert [item["id"] for item in board["splits"]["old_hard_val"]] == ["old_hard"]
    assert {item["id"] for item in board["splits"]["harvest_holdout"]} == {"harvest_a", "harvest_b"}
    assert [item["id"] for item in board["splits"]["rejected_negatives"]] == ["reject_holdout"]
    assert [item["id"] for item in board["splits"]["flat_negatives"]] == ["flat_holdout"]
    assert "object-ball outgoing guideline" in board["policy"]["positive_rule"]
    assert "image-level zero-mask negatives" in board["policy"]["negative_rule"]
    assert "touch or belong to the cue-ball aiming line" not in board["policy"]["positive_rule"]

    excluded = json.loads((output_root / "train_exclude_ids.json").read_text(encoding="utf-8"))
    assert set(excluded) == {"old_val", "old_hard", "harvest_a", "harvest_b", "reject_holdout", "flat_holdout"}
    assert "reject_train" not in excluded
    assert "flat_train" not in excluded
