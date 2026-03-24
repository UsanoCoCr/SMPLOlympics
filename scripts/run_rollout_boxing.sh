#!/bin/bash
# =============================================================================
# Rollout boxing self-play policy to collect paired SMPL motion sequences.
#
# Usage:
#   bash scripts/run_rollout_boxing.sh
#
# You can override any parameter by editing the variables below or by
# appending hydra overrides, e.g.:
#   bash scripts/run_rollout_boxing.sh env.num_envs=512
#
# This script does NOT modify any original code files.
# =============================================================================

export OMP_NUM_THREADS=1

# ---------- User-configurable parameters ----------

# Path to your trained checkpoint (without _op suffix; the opponent checkpoint
# will be auto-discovered as <name>_op.pth)
CHECKPOINT="output/SMPLOlympics/boxing_amp/Humanoid.pth"

# Number of parallel environments for rollout.
# More envs = faster collection but more GPU memory.
NUM_ENVS=256

# Episode length in control steps.  Default training uses 300.
EPISODE_LENGTH=300

# Number of *valid* paired episodes to collect (episodes where neither
# agent fell will be kept; early-terminated episodes are discarded).
NUM_EPISODES=200

# Minimum sequence length to keep (in steps).  Sequences shorter than
# this are discarded.  Set to episode_length to keep only full episodes.
MIN_SEQ_LEN=60

# Use deterministic (mean) actions instead of sampling.
DETERMINISTIC=True

# Where to save the output .pkl file.
OUTPUT_DIR="rollout_data/boxing"

# Filename prefix for the saved .pkl file.
SAVE_PREFIX="boxing_pairs"

# ---------- End of user-configurable parameters ----------

CUDA_VISIBLE_DEVICES=0 python scripts/rollout_boxing_pairs.py \
    project_name=SMPLOlympics \
    num_agents=2 \
    learning=amp_z_self_play \
    exp_name=boxing_amp \
    env=env_amp_z \
    env.num_envs=${NUM_ENVS} \
    env.task=HumanoidBoxingZ \
    env.enableTaskObs=True \
    env.stateInit=Start \
    robot=smpl_humanoid_boxing \
    +env.models='["output/HumanoidIm/pulse_vae_iclr/Humanoid.pth"]' \
    env.motion_file=./sample_data/video_boxing_afterproc_upright.pkl \
    headless=True \
    env.episode_length=${EPISODE_LENGTH} \
    test=True \
    epoch=-1 \
    no_log=True \
    +rollout_checkpoint="${CHECKPOINT}" \
    +rollout_num_episodes=${NUM_EPISODES} \
    +rollout_min_seq_len=${MIN_SEQ_LEN} \
    +rollout_deterministic=${DETERMINISTIC} \
    +rollout_output_dir="${OUTPUT_DIR}" \
    +rollout_save_prefix="${SAVE_PREFIX}" \
    "$@"
