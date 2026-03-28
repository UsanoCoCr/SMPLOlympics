import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm
import imageio
from g1_to_smpl import EDGE_SMPL_JOINT_NAMES, EDGE_SMPL_PARENTS, EDGE_SMPL_OFFSETS

smpl_joints = EDGE_SMPL_JOINT_NAMES
smpl_parents = EDGE_SMPL_PARENTS
smpl_offsets = EDGE_SMPL_OFFSETS

def visualize_2d(coco_2d, save_path_2d=None, fps=30):
    """
    可视化 COCO 2D 关键点序列并生成 GIF
    coco_2d: (T,17,2)
    """

    if hasattr(coco_2d, "detach"):
        coco_2d = coco_2d.detach().cpu().numpy()
    coco_2d = np.asarray(coco_2d)
    if coco_2d.ndim == 2:
        coco_2d = coco_2d[None, ...]  # (1,17,2)

    T = coco_2d.shape[0]
    skeleton = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 5), (4, 6)
    ]

    os.makedirs(os.path.dirname(save_path_2d), exist_ok=True)

    x_min, x_max = coco_2d[:, :, 0].min(), coco_2d[:, :, 0].max()
    y_min, y_max = coco_2d[:, :, 1].min(), coco_2d[:, :, 1].max()
    pad = 0.05 * max(x_max - x_min, y_max - y_min)
    x_min -= pad; x_max += pad; y_min -= pad; y_max += pad

    frames = []
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    for i in range(T):
        ax.clear()
        pts = coco_2d[i]
        ax.scatter(pts[:, 0], pts[:, 1], c='red', s=15)

        for a, b in skeleton:
            if a < pts.shape[0] and b < pts.shape[0]:
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], c='blue', lw=2)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"Frame {i}")
        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # 修复 reshape: get_width_height() 返回 (width, height)，但 reshape 需要 (height, width, 3)
        width, height = fig.canvas.get_width_height()
        frame = frame.reshape(height, width, 3)
        frames.append(frame)

    plt.close(fig)
    imageio.mimsave(save_path_2d, frames, fps=fps)
    # print(f"Saved GIF: {save_path_2d}")
    return save_path_2d

def load_norm_stats(cache_dir):
    obj = torch.load(os.path.join(cache_dir, "norm_stats.pt"))
    return obj["mean"], obj["std"]  # [D], [D]

def denorm(x_norm, mean, std):
    """
    x_norm: [B, T, D] 或 [T, D] 的标准化张量
    mean, std: [D]
    return: 与 x_norm 同形状
    """
    return x_norm * std + mean

def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def plot_single_pose(num, poses, lines, ax, axrange, scat, contact, aist=False):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        # don't plot root
        if i == 0:
            continue
        # stack to create a line
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    if num == 0:
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        if aist:
            xcenter, ycenter, zcenter = 0, 0, 1.5
        else:
            xcenter, ycenter, zcenter = 1.5, -1.5, 2.5
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
    aist=False,
):
    if render:
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]
        # print("pose shape: ", poses.shape)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        if aist:
            xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 2), np.linspace(-1.5, 1.5, 2))
        else:
            xx, yy = np.meshgrid(np.linspace(0.0, 3.0, 2), np.linspace(-3.0, 0.0, 2))
        z = (-normal[0] * xx - normal[1] * yy - d) * 0.0 / normal[2]

        # plot the plane
        ax.plot_surface(xx, yy, z, zorder=-11, cmap=cm.twilight)
        # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in smpl_parents
        ]
        scat = [
            # ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            ax.scatter([], [], [], zorder=10, s=0)
            for _ in range(4)
        ]
        axrange = 3

        # create contact labels
        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        if contact is None:
            contact = feetv < 0.01
        else:
            contact = contact > 0.95

        # Creating the Animation object
        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(poses, lines, ax, axrange, scat, contact, aist),
            interval=1000 // 25,
        )
    if sound:
        # make a temporary directory to save the intermediate gif in
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            anim.save(gifname)

        # stitch wavs
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # save a dummy spliced audio
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4',
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            if not os.path.exists(audioname):
                audioname = os.path.join(out, audioname)
            if epoch is not None:
                outname = os.path.join(
                    out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4"
                )
            else:
                outname = os.path.join(
                    out, f"{os.path.splitext(os.path.basename(name))[0]}.mp4"
                )
        if render:
            print("gifname, audioname, outname for render: ", gifname, audioname, outname)
            out = os.system(
                # f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -crf 26 -c:a aac -q:a 4 {outname}"
                # f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -c:a aac -q:a 4 {outname}"
                f"ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v mpeg4 -c:a aac -q:a 4 {outname}"
            )
    else:
        if render:
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"},)
    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
