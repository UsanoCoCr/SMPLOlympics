import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/codex-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import imageio.v2 as imageio
import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from smpl_sim.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree


DEFAULT_INPUT = "sample_data/video_boxing_afterproc_upright.pkl"
DEFAULT_MJCF = "phc/data/assets/mjcf/smpl_humanoid_1.xml"
FIGURE_BG = "#f3efe6"
GROUND_COLOR = "#ded6c7"
BONE_COLOR = "#123a63"
JOINT_COLOR = "#f8fafc"
HAND_COLOR = "#cc4b37"
FOOT_COLOR = "#2f7d4a"
TRAIL_COLOR = "#7f8c8d"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render SMPL sample_data sequences into skeleton GIFs."
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="Input .pkl path.")
    parser.add_argument(
        "--mjcf-path",
        type=str,
        default=DEFAULT_MJCF,
        help="MJCF humanoid used to reconstruct the SMPL skeleton tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for exported GIFs. Defaults to output/visualizations/<input_stem>/",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="",
        help="Only export one motion key, such as data0.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Only export the first N motions after filtering. -1 means all.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Frame subsampling step. 1 keeps every frame.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output GIF fps. 0 uses source fps / every_n.",
    )
    parser.add_argument("--width", type=float, default=5.6, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.6, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=110, help="Figure dpi.")
    parser.add_argument(
        "--list-keys",
        action="store_true",
        help="Print available sequence keys and exit.",
    )
    return parser.parse_args()


def to_float_tensor(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().float()
    return torch.as_tensor(np.asarray(array_like), dtype=torch.float32)


def to_numpy(array_like):
    if isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def load_motion_dict(path):
    motion_data = joblib.load(path)
    if not isinstance(motion_data, dict):
        raise TypeError(f"{path} should contain a dict, got {type(motion_data)}")
    return motion_data


def resolve_motion_items(motion_data, key_filter="", limit=-1):
    items = list(motion_data.items())
    if key_filter:
        if key_filter not in motion_data:
            keys = ", ".join(motion_data.keys())
            raise KeyError(f"Unknown key '{key_filter}'. Available keys: {keys}")
        items = [(key_filter, motion_data[key_filter])]
    if limit > 0:
        items = items[:limit]
    return items


def reconstruct_joint_positions(sequence, skeleton_tree):
    if "body_pos" in sequence:
        return to_numpy(sequence["body_pos"]).astype(np.float32)

    if "pose_quat_global" in sequence and "root_trans_offset" in sequence:
        pose = to_float_tensor(sequence["pose_quat_global"])
        root = to_float_tensor(sequence["root_trans_offset"])
        state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            pose,
            root,
            is_local=False,
        )
        return state.global_translation.detach().cpu().numpy()

    if "pose_quat" in sequence and "root_trans_offset" in sequence:
        pose = to_float_tensor(sequence["pose_quat"])
        root = to_float_tensor(sequence["root_trans_offset"])
        state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            pose,
            root,
            is_local=True,
        )
        return state.global_translation.detach().cpu().numpy()

    raise KeyError(
        "Sequence does not contain a supported skeleton representation. "
        "Expected body_pos or pose_quat(_global) with root_trans_offset."
    )


def get_bones(skeleton_tree):
    parents = skeleton_tree.parent_indices.tolist()
    return [(parent_idx, joint_idx) for joint_idx, parent_idx in enumerate(parents) if parent_idx != -1]


def compute_view_bounds(joint_positions):
    flat = joint_positions.reshape(-1, 3)
    mins = flat.min(axis=0)
    maxs = flat.max(axis=0)

    xy_center = (mins[:2] + maxs[:2]) * 0.5
    z_min = min(0.0, float(mins[2]) - 0.05)
    z_max = float(maxs[2]) + 0.15
    span = max(
        float(maxs[0] - mins[0]),
        float(maxs[1] - mins[1]),
        z_max - z_min,
    )
    radius = max(0.8, span * 0.6)

    return {
        "xlim": (xy_center[0] - radius, xy_center[0] + radius),
        "ylim": (xy_center[1] - radius, xy_center[1] + radius),
        "zlim": (z_min, z_min + radius * 2.0),
        "ground": (
            xy_center[0] - radius,
            xy_center[0] + radius,
            xy_center[1] - radius,
            xy_center[1] + radius,
        ),
        "aspect": (1.0, 1.0, max(0.75, (z_max - z_min) / (radius * 2.0))),
    }


def create_canvas(bounds, width, height, dpi):
    fig = plt.figure(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(FIGURE_BG)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(FIGURE_BG)
    ax.view_init(elev=17, azim=-108)
    ax.set_xlim(*bounds["xlim"])
    ax.set_ylim(*bounds["ylim"])
    ax.set_zlim(*bounds["zlim"])
    ax.set_box_aspect(bounds["aspect"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    x0, x1, y0, y1 = bounds["ground"]
    xx, yy = np.meshgrid(np.linspace(x0, x1, 2), np.linspace(y0, y1, 2))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, color=GROUND_COLOR, alpha=0.18, shade=False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.line.set_color((0, 0, 0, 0))
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    return fig, ax


def update_scatter(scatter, points):
    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])


def render_sequence_gif(
    joint_positions,
    bones,
    output_path,
    motion_key,
    fps,
    every_n,
    width,
    height,
    dpi,
):
    bounds = compute_view_bounds(joint_positions)
    fig, ax = create_canvas(bounds, width, height, dpi)

    line_artists = [
        ax.plot([], [], [], color=BONE_COLOR, linewidth=2.6, solid_capstyle="round")[0]
        for _ in bones
    ]
    joint_scatter = ax.scatter([], [], [], s=24, c=JOINT_COLOR, edgecolors="none", depthshade=False)
    hand_indices = [idx for idx in [20, 21, 22, 23] if idx < joint_positions.shape[1]]
    foot_indices = [idx for idx in [7, 8, 10, 11] if idx < joint_positions.shape[1]]
    hand_scatter = ax.scatter([], [], [], s=44, c=HAND_COLOR, edgecolors="none", depthshade=False)
    foot_scatter = ax.scatter([], [], [], s=34, c=FOOT_COLOR, edgecolors="none", depthshade=False)
    trail_artist = ax.plot([], [], [], color=TRAIL_COLOR, linewidth=1.6, alpha=0.8)[0]
    title_artist = ax.text2D(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color="#162029",
    )

    frames = []
    sampled_positions = joint_positions[::every_n]
    total_frames = sampled_positions.shape[0]

    for frame_idx, frame_points in enumerate(sampled_positions):
        for artist, (parent_idx, joint_idx) in zip(line_artists, bones):
            segment = frame_points[[parent_idx, joint_idx]]
            artist.set_data(segment[:, 0], segment[:, 1])
            artist.set_3d_properties(segment[:, 2])

        update_scatter(joint_scatter, frame_points)
        if hand_indices:
            update_scatter(hand_scatter, frame_points[hand_indices])
        if foot_indices:
            update_scatter(foot_scatter, frame_points[foot_indices])

        root_trail = sampled_positions[: frame_idx + 1, 0]
        trail_artist.set_data(root_trail[:, 0], root_trail[:, 1])
        trail_artist.set_3d_properties(root_trail[:, 2])

        title_artist.set_text(f"{motion_key}  frame {frame_idx + 1}/{total_frames}")

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)

    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def derive_output_dir(input_path, output_dir):
    if output_dir:
        return Path(output_dir)
    input_stem = Path(input_path).stem
    return Path("output") / "visualizations" / input_stem


def main():
    args = parse_args()
    motion_data = load_motion_dict(args.input)

    if args.list_keys:
        for key in motion_data.keys():
            print(key)
        return

    if args.every_n < 1:
        raise ValueError("--every-n must be >= 1")

    skeleton_tree = SkeletonTree.from_mjcf(args.mjcf_path)
    bones = get_bones(skeleton_tree)
    output_dir = derive_output_dir(args.input, args.output_dir)
    motion_items = resolve_motion_items(motion_data, args.key, args.limit)

    print(f"Loaded {len(motion_items)} sequence(s) from {args.input}")
    print(f"Using skeleton topology from {args.mjcf_path}")
    print(f"Export directory: {output_dir}")

    for motion_key, sequence in motion_items:
        joint_positions = reconstruct_joint_positions(sequence, skeleton_tree)
        source_fps = float(sequence.get("fps", 30))
        output_fps = args.fps if args.fps > 0 else max(1.0, source_fps / args.every_n)
        output_path = output_dir / f"{motion_key}.gif"

        print(
            f"[{motion_key}] joints={joint_positions.shape[1]} "
            f"frames={joint_positions.shape[0]} every_n={args.every_n} fps={output_fps:.2f}"
        )
        render_sequence_gif(
            joint_positions=joint_positions,
            bones=bones,
            output_path=output_path,
            motion_key=motion_key,
            fps=output_fps,
            every_n=args.every_n,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
