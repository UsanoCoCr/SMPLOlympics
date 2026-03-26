#!/bin/bash
# =============================================================================
# Train boxing policies with different fighter styles.
#
# Each style is a different reward weight preset injected via Hydra overrides.
# The original default behavior is fully preserved when no env.reward_*
# overrides are passed — the code falls back to the hardcoded defaults.
#
# Usage:
#   bash scripts/run_boxing_styles.sh <style>
#   e.g.:  bash scripts/run_boxing_styles.sh aggressive
#          bash scripts/run_boxing_styles.sh default
#
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                    FIGHTER STYLE PARAMETER REFERENCE                        │
# ├──────────────┬────────┬────────┬────────┬────────┬────────┬────────────────┤
# │              │ rwd_f  │ rwd_v  │ rwd_s  │ rwd_t  │ rwd_h  │  notes        │
# │              │(facing)│ (vel)  │(strike)│  (KO)  │ (hit)  │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ default      │  0.3   │  0.2   │  1.0   │  0.5   │  0.3   │ original paper│
# │ aggressive   │  0.2   │  0.5   │  1.5   │  1.0   │  0.5   │ rush & KO     │
# │ defensive    │  0.5   │  0.0   │  0.3   │  0.0   │  0.1   │ face & survive│
# │ counter      │  0.5   │  0.1   │  1.2   │  0.8   │  0.2   │ wait → punish │
# │ brawler      │  0.1   │  0.3   │  2.0   │  0.5   │  0.8   │ max damage    │
# │ outfighter   │  0.4   │  0.1   │  0.6   │  0.2   │  0.6   │ poke from far │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │tar_spd │vel_err │fac_err │hd_mul  │str_div │               │
# │              │ (m/s)  │(scale) │(scale) │(head×) │(÷norm) │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ default      │  1.0   │  4.0   │  2.0   │  2.0   │  10.0  │               │
# │ aggressive   │  1.5   │  4.0   │  2.0   │  3.0   │   8.0  │ faster close  │
# │ defensive    │  0.3   │  2.0   │  3.0   │  1.0   │  15.0  │ slow, cautious│
# │ counter      │  0.5   │  3.0   │  3.0   │  3.0   │   8.0  │ wait for head │
# │ brawler      │  1.2   │  4.0   │  1.5   │  1.5   │   6.0  │ raw force     │
# │ outfighter   │  0.6   │  3.0   │  2.5   │  2.0   │  12.0  │ controlled    │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │task_rw │disc_rw │sw_freq │               extra notes        │
# │              │(task%) │(AMP %) │(epoch) │                                  │
# ├──────────────┼────────┼────────┼────────┼──────────────────────────────────┤
# │ default      │  0.5   │  0.5   │  250   │ balanced task + motion quality   │
# │ aggressive   │  0.7   │  0.3   │  200   │ favor task reward → exploit more │
# │ defensive    │  0.3   │  0.7   │  300   │ favor AMP → natural, safe motion │
# │ counter      │  0.5   │  0.5   │  250   │ balanced                         │
# │ brawler      │  0.8   │  0.2   │  200   │ max task signal, less natural    │
# │ outfighter   │  0.4   │  0.6   │  300   │ more natural, controlled style   │
# └──────────────┴────────┴────────┴────────┴──────────────────────────────────┘
#
# Design rationale per style:
#
#   default     — Original SMPLOlympics paper setting. Balanced all-rounder.
#
#   aggressive  — "Pressure fighter". High velocity weight drives the agent to
#                 close distance fast (tar_speed=1.5). Elevated strike & KO
#                 rewards incentivise finishing fights. Higher task_reward_w
#                 lets it sacrifice motion naturalness for aggression.
#
#   defensive   — "Evasive / survival". Zero velocity reward means no
#                 incentive to approach. High facing reward keeps awareness.
#                 Low strike + zero KO reward removes attack motivation.
#                 High disc_reward_w keeps motion natural and upright.
#                 strike_divisor=15 further dampens any residual strike signal.
#
#   counter     — "Counter-puncher". Low velocity → patient, doesn't rush.
#                 High facing → always watching. When opponent comes in,
#                 strong strike + KO + head_strike_mul=3 punish hard.
#
#   brawler     — "Heavy hitter / slugger". Massive strike weight (2.0) and
#                 high hit proximity (0.8) reward. Small strike_divisor=6
#                 amplifies raw force difference. Less concerned with facing
#                 (0.1). Trades style for damage — low disc_reward_w.
#
#   outfighter  — "Technical / jab-and-move". Moderate velocity to maintain
#                 distance. High facing + hit rewards (poke from distance).
#                 Lower strike weight — fights on points, not power.
#                 High disc_reward_w keeps movement fluid and natural.
#
# =============================================================================

export OMP_NUM_THREADS=1

STYLE="${1:-default}"

# ---------- Common base command ----------
BASE_CMD="python phc/run_hydra.py project_name=SMPLOlympics num_agents=2 \
    learning=amp_z_self_play \
    env=env_amp_z env.num_envs=2048 env.task=HumanoidBoxingZ env.enableTaskObs=True \
    env.stateInit=Start robot=smpl_humanoid_boxing \
    +env.models=[\"output/HumanoidIm/pulse_vae_iclr/Humanoid.pth\"] \
    env.motion_file=./sample_data/video_boxing_afterproc_upright.pkl \
    headless=True env.episode_length=300"

case "${STYLE}" in

  default)
    echo "=== Training DEFAULT (balanced) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_default \
      learning.params.config.switch_frequency=250
    ;;

  aggressive)
    echo "=== Training AGGRESSIVE (pressure) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_aggressive \
      env.reward_f=0.2 env.reward_v=0.5 env.reward_s=1.5 env.reward_t=1.0 env.reward_h=0.5 \
      env.tar_speed=1.5 env.head_strike_mul=3.0 env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.7 \
      learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200
    ;;

  defensive)
    echo "=== Training DEFENSIVE (evasive) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_defensive \
      env.reward_f=0.5 env.reward_v=0.0 env.reward_s=0.3 env.reward_t=0.0 env.reward_h=0.1 \
      env.tar_speed=0.3 env.vel_err_scale=2.0 env.facing_err_scale=3.0 \
      env.head_strike_mul=1.0 env.strike_divisor=15.0 \
      learning.params.config.task_reward_w=0.3 \
      learning.params.config.disc_reward_w=0.7 \
      learning.params.config.switch_frequency=300
    ;;

  counter)
    echo "=== Training COUNTER-PUNCHER fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_counter \
      env.reward_f=0.5 env.reward_v=0.1 env.reward_s=1.2 env.reward_t=0.8 env.reward_h=0.2 \
      env.tar_speed=0.5 env.vel_err_scale=3.0 env.facing_err_scale=3.0 \
      env.head_strike_mul=3.0 env.strike_divisor=8.0 \
      learning.params.config.switch_frequency=250
    ;;

  brawler)
    echo "=== Training BRAWLER (heavy hitter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_brawler \
      env.reward_f=0.1 env.reward_v=0.3 env.reward_s=2.0 env.reward_t=0.5 env.reward_h=0.8 \
      env.tar_speed=1.2 env.facing_err_scale=1.5 \
      env.head_strike_mul=1.5 env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.8 \
      learning.params.config.disc_reward_w=0.2 \
      learning.params.config.switch_frequency=200
    ;;

  outfighter)
    echo "=== Training OUTFIGHTER (technical) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_outfighter \
      env.reward_f=0.4 env.reward_v=0.1 env.reward_s=0.6 env.reward_t=0.2 env.reward_h=0.6 \
      env.tar_speed=0.6 env.vel_err_scale=3.0 env.facing_err_scale=2.5 \
      env.strike_divisor=12.0 \
      learning.params.config.task_reward_w=0.4 \
      learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300
    ;;

  all)
    echo "=== Training ALL styles in parallel (6 GPUs) ==="
    echo "Make sure you have 6 GPUs available!"
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} exp_name=boxing_default \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=1 ${BASE_CMD} exp_name=boxing_aggressive \
      env.reward_f=0.2 env.reward_v=0.5 env.reward_s=1.5 env.reward_t=1.0 env.reward_h=0.5 \
      env.tar_speed=1.5 env.head_strike_mul=3.0 env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.7 learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=2 ${BASE_CMD} exp_name=boxing_defensive \
      env.reward_f=0.5 env.reward_v=0.0 env.reward_s=0.3 env.reward_t=0.0 env.reward_h=0.1 \
      env.tar_speed=0.3 env.vel_err_scale=2.0 env.facing_err_scale=3.0 \
      env.head_strike_mul=1.0 env.strike_divisor=15.0 \
      learning.params.config.task_reward_w=0.3 learning.params.config.disc_reward_w=0.7 \
      learning.params.config.switch_frequency=300 &

    CUDA_VISIBLE_DEVICES=3 ${BASE_CMD} exp_name=boxing_counter \
      env.reward_f=0.5 env.reward_v=0.1 env.reward_s=1.2 env.reward_t=0.8 env.reward_h=0.2 \
      env.tar_speed=0.5 env.vel_err_scale=3.0 env.facing_err_scale=3.0 \
      env.head_strike_mul=3.0 env.strike_divisor=8.0 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=4 ${BASE_CMD} exp_name=boxing_brawler \
      env.reward_f=0.1 env.reward_v=0.3 env.reward_s=2.0 env.reward_t=0.5 env.reward_h=0.8 \
      env.tar_speed=1.2 env.facing_err_scale=1.5 \
      env.head_strike_mul=1.5 env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.8 learning.params.config.disc_reward_w=0.2 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=5 ${BASE_CMD} exp_name=boxing_outfighter \
      env.reward_f=0.4 env.reward_v=0.1 env.reward_s=0.6 env.reward_t=0.2 env.reward_h=0.6 \
      env.tar_speed=0.6 env.vel_err_scale=3.0 env.facing_err_scale=2.5 \
      env.strike_divisor=12.0 \
      learning.params.config.task_reward_w=0.4 learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300 &

    wait
    echo "=== All training jobs finished ==="
    ;;

  *)
    echo "Unknown style: ${STYLE}"
    echo "Available: default | aggressive | defensive | counter | brawler | outfighter | all"
    exit 1
    ;;
esac
