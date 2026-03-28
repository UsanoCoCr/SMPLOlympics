"""
Visualize paired boxing rollout data (from rollout_boxing_pairs.py).

Renders two SMPL skeletons (player1=blue, player2=red) with per-step
reward overlaid in the corner.  Supports interactive playback and
headless mp4 export.

Usage (interactive, needs display or X-forwarding):
    python scripts/vis/vis_boxing_pairs.py --pkl rollout_data/boxing/boxing_pairs_XXXX.pkl

Usage (headless mp4 export, no GUI needed):
    python scripts/vis/vis_boxing_pairs.py --pkl rollout_data/boxing/boxing_pairs_XXXX.pkl \
        --headless --output output/renderings/boxing_vis.mp4

Controls (interactive mode):
    Space  : pause / resume
    R      : reset to frame 0
    L      : toggle mp4 recording
    Z      : zoom in
    Right  : next sequence
    Left   : previous sequence
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.append(os.getcwd())

import open3d as o3d
import imageio
from datetime import datetime

# ─────────────────────── SMPL skeleton definition ────────────────────────
# 24 SMPL joints.  Bone connectivity for drawing lines.
JOINT_NAMES = [
    'Pelvis',     # 0
    'L_Hip',      # 1
    'R_Hip',      # 2
    'Spine',      # 3  (actually Torso in some conventions)
    'L_Knee',     # 4
    'R_Knee',     # 5
    'Spine1',     # 6  (Spine)
    'L_Ankle',    # 7
    'R_Ankle',    # 8
    'Spine2',     # 9  (Chest)
    'L_Toe',      # 10
    'R_Toe',      # 11
    'Neck',       # 12
    'L_Thorax',   # 13 (L_Collar)
    'R_Thorax',   # 14 (R_Collar)
    'Head',       # 15
    'L_Shoulder',  # 16
    'R_Shoulder',  # 17
    'L_Elbow',    # 18
    'R_Elbow',    # 19
    'L_Wrist',    # 20
    'R_Wrist',    # 21
    'L_Hand',     # 22
    'R_Hand',     # 23
]

BONES = [
    (0, 1), (0, 2), (0, 3),      # pelvis → hips, spine
    (1, 4), (2, 5),               # hip → knee
    (4, 7), (5, 8),               # knee → ankle
    (7, 10), (8, 11),             # ankle → toe
    (3, 6), (6, 9), (9, 12),     # spine chain
    (12, 15),                      # neck → head
    (9, 13), (9, 14),             # chest → collar/thorax
    (13, 16), (14, 17),           # collar → shoulder
    (16, 18), (17, 19),           # shoulder → elbow
    (18, 20), (19, 21),           # elbow → wrist
    (20, 22), (21, 23),           # wrist → hand
]

# ─────────────────────── globals ─────────────────────────────────────────
paused = False
reset = False
recording = False
writer = None
curr_zoom = 0.5


def pause_func(vis):
    global paused
    paused = not paused
    print(f"Paused: {paused}")
    return True

def reset_func(vis):
    global reset
    reset = True
    return True

def record_func(vis):
    global recording, writer
    if not recording:
        os.makedirs("output/renderings/o3d", exist_ok=True)
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = f"output/renderings/o3d/{ts}-boxing-pairs.mp4"
        writer = imageio.get_writer(path, fps=30, macro_block_size=None)
        print(f"Recording → {path}")
    else:
        if writer is not None:
            writer.close()
            writer = None
        print("Recording stopped.")
    recording = not recording
    return True

def zoom_func(vis):
    global curr_zoom
    curr_zoom *= 0.9
    ctr = vis.get_view_control()
    ctr.set_zoom(curr_zoom)
    return True


# ─────────────────────── skeleton drawing helpers ────────────────────────

def create_skeleton_geometries(color, num_joints=24):
    """Create spheres (joints) and a LineSet (bones) for one player."""
    spheres = []
    for _ in range(num_joints):
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        s.compute_vertex_normals()
        s.paint_uniform_color(color)
        spheres.append(s)

    # hand spheres are slightly larger to highlight striking
    for idx in [20, 21, 22, 23]:  # wrists + hands
        spheres[idx] = o3d.geometry.TriangleMesh.create_sphere(radius=0.045)
        spheres[idx].compute_vertex_normals()
        # slightly brighter for hands
        hand_color = np.clip(np.array(color) + 0.2, 0, 1).tolist()
        spheres[idx].paint_uniform_color(hand_color)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(np.zeros((num_joints, 3)))
    lines.lines = o3d.utility.Vector2iVector(BONES)
    lines.colors = o3d.utility.Vector3dVector([color] * len(BONES))

    return spheres, lines


def update_skeleton(spheres, lines, body_pos, prev_positions):
    """Move skeleton geometry to new body_pos (24, 3)."""
    for j in range(len(spheres)):
        delta = body_pos[j] - prev_positions[j]
        spheres[j].translate(delta)
    prev_positions[:] = body_pos

    lines.points = o3d.utility.Vector3dVector(body_pos)
    return prev_positions


# ─────────────────────── headless renderer ───────────────────────────────

def render_headless(sequences, seq_idx, output_path, fps=30):
    """Render a sequence to mp4 without any GUI using OffscreenRenderer."""
    seq = sequences[seq_idx]
    p1_body = seq['player1']['body_pos']  # (T, 24, 3)
    p2_body = seq['player2']['body_pos']
    p1_reward = seq['player1'].get('reward', np.zeros(p1_body.shape[0]))
    p2_reward = seq['player2'].get('reward', np.zeros(p2_body.shape[0]))
    T = p1_body.shape[0]
    dt = seq.get('dt', 1.0 / 30)

    W, H = 1280, 720
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)

    # materials
    mat_p1 = o3d.visualization.rendering.MaterialRecord()
    mat_p1.base_color = [0.2, 0.4, 1.0, 1.0]
    mat_p1.shader = "defaultLit"

    mat_p2 = o3d.visualization.rendering.MaterialRecord()
    mat_p2.base_color = [1.0, 0.3, 0.2, 1.0]
    mat_p2.shader = "defaultLit"

    mat_bone1 = o3d.visualization.rendering.MaterialRecord()
    mat_bone1.base_color = [0.2, 0.4, 1.0, 1.0]
    mat_bone1.shader = "unlitLine"
    mat_bone1.line_width = 3.0

    mat_bone2 = o3d.visualization.rendering.MaterialRecord()
    mat_bone2.base_color = [1.0, 0.3, 0.2, 1.0]
    mat_bone2.shader = "unlitLine"
    mat_bone2.line_width = 3.0

    # ground plane
    mat_ground = o3d.visualization.rendering.MaterialRecord()
    mat_ground.base_color = [0.85, 0.85, 0.85, 1.0]
    ground = o3d.geometry.TriangleMesh.create_box(10, 0.01, 10)
    ground.translate([-5, -0.01, -5])
    ground.compute_vertex_normals()
    render.scene.add_geometry("ground", ground, mat_ground)

    # camera: look at midpoint of the two players at frame 0
    mid = (p1_body[0, 0] + p2_body[0, 0]) / 2
    eye_offset = np.array([5.0, 3.0, 2.0])
    render.setup_camera(45.0,
                        mid.tolist(),
                        (mid + eye_offset).tolist(),
                        [0, 0, 1])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    vid_writer = imageio.get_writer(output_path, fps=fps, macro_block_size=None)

    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False

    print(f"Headless rendering {T} frames → {output_path}")
    for t in range(T):
        # remove old geometries
        for name in [n for n in ["p1_bones", "p2_bones"] ]:
            render.scene.remove_geometry(name)
        for j in range(24):
            render.scene.remove_geometry(f"p1_j{j}")
            render.scene.remove_geometry(f"p2_j{j}")

        # player 1 joints
        for j in range(24):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            s.translate(p1_body[t, j])
            s.compute_vertex_normals()
            render.scene.add_geometry(f"p1_j{j}", s, mat_p1)

        # player 2 joints
        for j in range(24):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            s.translate(p2_body[t, j])
            s.compute_vertex_normals()
            render.scene.add_geometry(f"p2_j{j}", s, mat_p2)

        # bones
        ls1 = o3d.geometry.LineSet()
        ls1.points = o3d.utility.Vector3dVector(p1_body[t])
        ls1.lines = o3d.utility.Vector2iVector(BONES)
        render.scene.add_geometry("p1_bones", ls1, mat_bone1)

        ls2 = o3d.geometry.LineSet()
        ls2.points = o3d.utility.Vector3dVector(p2_body[t])
        ls2.lines = o3d.utility.Vector2iVector(BONES)
        render.scene.add_geometry("p2_bones", ls2, mat_bone2)

        img = np.asarray(render.render_to_image())

        # overlay reward text
        if has_cv2:
            r1 = p1_reward[t] if t < len(p1_reward) else 0
            r2 = p2_reward[t] if t < len(p2_reward) else 0
            img = img.copy()
            cv2.putText(img, f"Frame {t}/{T}  dt={dt:.4f}s",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
            cv2.putText(img, f"P1 reward: {r1:+.3f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 255), 2)
            cv2.putText(img, f"P2 reward: {r2:+.3f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 50), 2)

        vid_writer.append_data(img)

        if (t + 1) % 50 == 0:
            print(f"  rendered {t+1}/{T}")

    vid_writer.close()
    print(f"Saved: {output_path}  ({T} frames, {T*dt:.1f}s)")


# ─────────────────────── interactive viewer ──────────────────────────────

def run_interactive(sequences, start_seq=0):
    """Open3D interactive viewer with keyboard controls."""
    global paused, reset, recording, writer, curr_zoom

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Boxing Pairs Viewer", width=1280, height=720)

    vis.register_key_callback(32, pause_func)    # Space
    vis.register_key_callback(82, reset_func)    # R
    vis.register_key_callback(76, record_func)   # L
    vis.register_key_callback(90, zoom_func)     # Z

    seq_idx = start_seq
    seq = sequences[seq_idx]
    p1_body = seq['player1']['body_pos']
    p2_body = seq['player2']['body_pos']
    p1_reward = seq['player1'].get('reward', np.zeros(p1_body.shape[0]))
    p2_reward = seq['player2'].get('reward', np.zeros(p2_body.shape[0]))
    T = p1_body.shape[0]
    dt = seq.get('dt', 1.0/30)

    # player 1 = blue, player 2 = red
    color_p1 = [0.2, 0.4, 1.0]
    color_p2 = [1.0, 0.3, 0.2]

    spheres_p1, lines_p1 = create_skeleton_geometries(color_p1)
    spheres_p2, lines_p2 = create_skeleton_geometries(color_p2)

    prev_pos_p1 = np.zeros((24, 3))
    prev_pos_p2 = np.zeros((24, 3))

    # initialise to frame 0
    update_skeleton(spheres_p1, lines_p1, p1_body[0], prev_pos_p1)
    update_skeleton(spheres_p2, lines_p2, p2_body[0], prev_pos_p2)

    # ground plane
    ground = o3d.geometry.TriangleMesh.create_box(10, 0.01, 10)
    ground.translate([-5, -0.01, -5])
    ground.compute_vertex_normals()
    ground.paint_uniform_color([0.85, 0.85, 0.85])

    for s in spheres_p1:
        vis.add_geometry(s)
    for s in spheres_p2:
        vis.add_geometry(s)
    vis.add_geometry(lines_p1)
    vis.add_geometry(lines_p2)
    vis.add_geometry(ground)

    # camera
    ctr = vis.get_view_control()
    ctr.set_up([0, 0, 1])
    ctr.set_front([1, 0, 0.3])
    mid = (p1_body[0, 0] + p2_body[0, 0]) / 2
    ctr.set_lookat(mid)
    ctr.set_zoom(curr_zoom)

    frame = 0
    print(f"Sequence {seq_idx}/{len(sequences)-1}  |  {T} frames  |  "
          f"Press Space=pause, R=reset, L=record, Left/Right=prev/next seq")

    # next/prev sequence via closures
    def next_seq(v):
        nonlocal seq_idx
        seq_idx = min(seq_idx + 1, len(sequences) - 1)
        load_sequence(seq_idx)
        return True

    def prev_seq(v):
        nonlocal seq_idx
        seq_idx = max(seq_idx - 1, 0)
        load_sequence(seq_idx)
        return True

    def load_sequence(idx):
        nonlocal seq, p1_body, p2_body, p1_reward, p2_reward, T, dt, frame
        seq = sequences[idx]
        p1_body = seq['player1']['body_pos']
        p2_body = seq['player2']['body_pos']
        p1_reward = seq['player1'].get('reward', np.zeros(p1_body.shape[0]))
        p2_reward = seq['player2'].get('reward', np.zeros(p2_body.shape[0]))
        T = p1_body.shape[0]
        dt = seq.get('dt', 1.0/30)
        frame = 0
        print(f"Loaded sequence {idx}/{len(sequences)-1}  |  {T} frames")

    vis.register_key_callback(262, next_seq)   # Right arrow
    vis.register_key_callback(263, prev_seq)   # Left arrow

    while True:
        vis.poll_events()

        if reset:
            frame = 0
            reset = False

        # update skeleton positions
        prev_pos_p1 = update_skeleton(spheres_p1, lines_p1, p1_body[frame % T], prev_pos_p1)
        prev_pos_p2 = update_skeleton(spheres_p2, lines_p2, p2_body[frame % T], prev_pos_p2)

        for s in spheres_p1:
            vis.update_geometry(s)
        for s in spheres_p2:
            vis.update_geometry(s)
        vis.update_geometry(lines_p1)
        vis.update_geometry(lines_p2)

        if not paused:
            frame += 1

        # print reward to terminal every N frames
        if not paused and frame % 30 == 0:
            t = frame % T
            r1 = p1_reward[t] if t < len(p1_reward) else 0
            r2 = p2_reward[t] if t < len(p2_reward) else 0
            print(f"\rSeq {seq_idx} | Frame {t:4d}/{T} | "
                  f"P1 reward: {r1:+.3f} | P2 reward: {r2:+.3f}   ", end="")

        if recording and writer is not None:
            rgb = vis.capture_screen_float_buffer()
            rgb = (np.asarray(rgb) * 255).astype(np.uint8)

            # overlay reward text on captured frame
            try:
                import cv2
                t = frame % T
                r1 = p1_reward[t] if t < len(p1_reward) else 0
                r2 = p2_reward[t] if t < len(p2_reward) else 0
                cv2.putText(rgb, f"Frame {t}/{T}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
                cv2.putText(rgb, f"P1 reward: {r1:+.3f}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 255), 2)
                cv2.putText(rgb, f"P2 reward: {r2:+.3f}",
                            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 50), 2)
            except ImportError:
                pass

            writer.append_data(rgb)

        vis.update_renderer()


# ─────────────────────── entry point ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize boxing paired rollout data")
    parser.add_argument("--pkl", type=str, required=True,
                        help="Path to the .pkl file from rollout_boxing_pairs.py")
    parser.add_argument("--seq", type=int, default=0,
                        help="Which sequence index to start from (default: 0)")
    parser.add_argument("--headless", action="store_true",
                        help="Headless mode: render to mp4 without GUI")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mp4 path (headless mode). Default: auto-generated")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for headless video (default: 30)")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    sequences = data['sequences']
    meta = data.get('metadata', {})
    print(f"Loaded {len(sequences)} sequences.  "
          f"Total frames: {meta.get('total_frames', '?')}  "
          f"FPS: {meta.get('fps', '?')}")

    if args.seq >= len(sequences):
        print(f"Sequence index {args.seq} out of range [0, {len(sequences)-1}]")
        return

    if args.headless:
        if args.output is None:
            os.makedirs("output/renderings", exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f"output/renderings/boxing_pairs_seq{args.seq}_{ts}.mp4"
        render_headless(sequences, args.seq, args.output, args.fps)
    else:
        run_interactive(sequences, args.seq)


if __name__ == "__main__":
    main()
