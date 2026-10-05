import argparse
from pathlib import Path

import cv2 as cv


def extract_frames(video_path: Path, output_dir: Path, stride: int, start: int, max_frames: int) -> int:
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx < start or ((frame_idx - start) % stride) != 0:
            frame_idx += 1
            continue

        frame_name = f"{video_path.stem}_{frame_idx:06d}.png"
        frame_path = output_dir / frame_name
        cv.imwrite(str(frame_path), frame)
        saved += 1

        if max_frames > 0 and saved >= max_frames:
            break

        frame_idx += 1

    cap.release()
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("output", type=Path, help="Directory to store extracted frames")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (default: 1)")
    parser.add_argument("--start", type=int, default=0, help="Index of the first frame to keep")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to save (0 = no limit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = extract_frames(args.video, args.output, max(1, args.stride), max(0, args.start), max(0, args.max_frames))
    stem = args.video.stem
    print(f"Saved {saved} frames from {stem} into {args.output}")


if __name__ == "__main__":
    main()
