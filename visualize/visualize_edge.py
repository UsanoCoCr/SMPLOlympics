#!/usr/bin/env python3
"""Render G1->SMPL npz outputs through legacy vis.py pipeline.

This wrapper keeps vis.py untouched and reuses:
  - vis.SMPLSkeleton
  - vis.skeleton_render

It supports exporting gif directly (native vis.py behavior), or mp4 via a
post-conversion step from the generated gif.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from vis import SMPLSkeleton, skeleton_render


def _load_pose_and_trans(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    if "smpl_poses" in data:
        poses = data["smpl_poses"]  # (T,72) expected
    elif "global_orient" in data and "body_pose" in data:
        poses = np.concatenate([data["global_orient"], data["body_pose"]], axis=1)
    else:
        raise KeyError("Missing pose keys. Need smpl_poses or (global_orient + body_pose).")

    if "smpl_trans" in data:
        trans = data["smpl_trans"]
    elif "transl" in data:
        trans = data["transl"]
    else:
        raise KeyError("Missing translation keys. Need smpl_trans or transl.")

    if poses.ndim == 2 and poses.shape[1] == 72:
        poses = poses.reshape(-1, 24, 3)
    elif poses.ndim == 3 and poses.shape[1:] == (24, 3):
        pass
    else:
        raise ValueError(f"Unsupported pose shape: {poses.shape}, expected (T,72) or (T,24,3).")

    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Unsupported translation shape: {trans.shape}, expected (T,3).")

    if poses.shape[0] != trans.shape[0]:
        raise ValueError(f"Length mismatch: poses T={poses.shape[0]} vs trans T={trans.shape[0]}.")

    return poses.astype(np.float32), trans.astype(np.float32)


def _poses_to_xyz(poses_aa: np.ndarray, trans: np.ndarray, device: torch.device) -> np.ndarray:
    skel = SMPLSkeleton(device=device)

    rotations = torch.from_numpy(poses_aa).unsqueeze(0).to(device)  # (1,T,24,3)
    root_pos = torch.from_numpy(trans).unsqueeze(0).to(device)      # (1,T,3)

    with torch.no_grad():
        xyz = skel.forward(rotations, root_pos)[0].cpu().numpy()  # (T,24,3)
    return xyz


def _convert_gif_to_mp4(gif_path: Path, mp4_path: Path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot convert gif to mp4.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(gif_path),
        "-movflags",
        "faststart",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Render SMPL npz using legacy vis.py.")
    parser.add_argument("--npz", type=str, required=True, help="Path to generated SMPL .npz")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .gif/.mp4 path. If omitted, writes <npz_stem>.gif next to the npz.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--aist", action="store_true", default=True, help="Use AIST camera mode")
    parser.add_argument("--no-aist", dest="aist", action="store_false", help="Disable AIST camera mode")
    parser.add_argument("--keep-gif", action="store_true", help="Keep intermediate gif when exporting mp4")
    args = parser.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"npz not found: {npz_path}")

    if args.output is None:
        target_path = npz_path.with_suffix(".gif")
    else:
        target_path = Path(args.output).expanduser().resolve()

    if target_path.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("Output extension must be .gif or .mp4")

    out_dir = target_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_aa, trans = _load_pose_and_trans(npz_path)
    xyz = _poses_to_xyz(poses_aa, trans, torch.device(args.device))

    # vis.skeleton_render(sound=False) always writes a GIF:
    # gif path = out / (name_without_last_4_chars + ".gif")
    render_stub = target_path.stem + ".wav"
    expected_gif = out_dir / f"{target_path.stem}.gif"

    skeleton_render(
        xyz,
        epoch=0,
        out=str(out_dir),
        name=render_stub,
        sound=False,
        contact=None,
        aist=args.aist,
    )

    if not expected_gif.exists():
        raise RuntimeError(f"Expected gif not found after render: {expected_gif}")

    if target_path.suffix.lower() == ".gif":
        print(f"[visualize_edge] Saved gif: {expected_gif}")
        return

    _convert_gif_to_mp4(expected_gif, target_path)
    print(f"[visualize_edge] Saved mp4: {target_path}")

    if not args.keep_gif:
        expected_gif.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

