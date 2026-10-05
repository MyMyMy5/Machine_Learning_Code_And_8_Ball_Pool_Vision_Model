from __future__ import annotations

from tools.build_reranker_dataset_from_checkpoint import (
    _negative_frames_from_gold_board,
    _stable_fraction,
    _unique_positive_frames,
)


def test_negative_frames_from_gold_board_keeps_only_zero_mask_negative_kinds() -> None:
    gold_board = {
        "splits": {
            "supplemental_rejected_negatives": [
                {
                    "id": "reject_a",
                    "kind": "image_negative",
                    "image_path": "C:/images/reject_a.png",
                },
                {
                    "id": "positive_a",
                    "kind": "positive",
                    "image_path": "C:/images/positive_a.png",
                },
            ],
            "supplemental_flat_negatives": [
                {
                    "id": "flat_a",
                    "kind": "flat_negative",
                    "image_path": "C:/images/flat_a.png",
                },
                {
                    "id": "ignored_unknown",
                    "kind": "other",
                    "image_path": "C:/images/ignored_unknown.png",
                },
            ],
        }
    }

    frames = _negative_frames_from_gold_board(
        gold_board,
        ["supplemental_rejected_negatives", "supplemental_flat_negatives"],
    )

    assert frames == [
        {
            "frame_id": "flat_a",
            "image_path": "C:/images/flat_a.png",
            "mask_path": "",
        },
        {
            "frame_id": "reject_a",
            "image_path": "C:/images/reject_a.png",
            "mask_path": "",
        },
    ]


def test_stable_fraction_is_deterministic_unit_interval() -> None:
    first = _stable_fraction("frame_a")
    second = _stable_fraction("frame_a")

    assert first == second
    assert 0.0 <= first < 1.0


def test_unique_positive_frames_excludes_zero_mask_entries() -> None:
    index = {
        "train": [
            {
                "kind": "candidate_negative",
                "id": "frame_a",
                "image_path": "C:/images/frame_a.png",
                "mask_path": "C:/masks/frame_a.png",
                "zero_mask": True,
            },
            {
                "kind": "external_negative",
                "id": "negative_a_0",
                "image_path": "C:/images/negative_a.png",
                "mask_path": None,
                "zero_mask": True,
            },
            {
                "kind": "candidate_positive",
                "id": "frame_a",
                "image_path": "C:/images/frame_a.png",
                "mask_path": "C:/masks/frame_a.png",
            },
            {
                "kind": "view_positive",
                "id": "frame_b",
                "image_path": "C:/images/frame_b.png",
                "mask_path": "C:/masks/frame_b.png",
            },
        ]
    }

    frames = _unique_positive_frames(index, "train")

    assert frames == [
        {
            "frame_id": "frame_a",
            "image_path": "C:/images/frame_a.png",
            "mask_path": "C:/masks/frame_a.png",
        },
        {
            "frame_id": "frame_b",
            "image_path": "C:/images/frame_b.png",
            "mask_path": "C:/masks/frame_b.png",
        },
    ]
