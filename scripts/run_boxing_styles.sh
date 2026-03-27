#!/bin/bash
# =============================================================================
# Train boxing policies with different fighter styles.
#
# Each style is a different reward weight preset injected via Hydra overrides.
# The original default behavior is fully preserved when no env.reward_*
# overrides are passed — the code falls back to the hardcoded defaults.
#
# v2 — Redesigned to address the "lazy equilibrium" problem observed at ~40k
#       steps where both agents learn to approach but never strike.
#
# Root cause:  facing (0.3) + vel (0.2) + hit (0.3) = 0.8 dense easy reward
#              vs. strike (1.0) + terminate (0.5) = 1.5 sparse hard reward.
#              Combined with disc_reward_w=0.5 penalising "unnatural" swings,
#              agents converge on "walk close and stand" — a degenerate Nash
#              equilibrium. The AMP discriminator acts as a regulariser that
#              suppresses the exploratory punching needed to discover strike.
#
# Fixes applied across ALL styles:
#   1. Halve (or more) the approach-only reward weights (facing, vel).
#   2. Boost strike-related signals (strike, terminate, hit proximity).
#   3. Lower strike_divisor so the same contact force produces bigger reward.
#   4. Lower disc_reward_w to let agents explore aggressive/unnatural motions.
#   5. Use hit_reward as the "bridge" — it rewards hand proximity to targets,
#      guiding the agent from "close range" to "actual contact".
#
# Usage:
#   bash scripts/run_boxing_styles.sh <style>
#   e.g.:  bash scripts/run_boxing_styles.sh aggressive
#          bash scripts/run_boxing_styles.sh default
#          bash scripts/run_boxing_styles.sh all
#
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                    FIGHTER STYLE PARAMETER REFERENCE  (v2)                  │
# ├──────────────┬────────┬────────┬────────┬────────┬────────┬────────────────┤
# │              │ rwd_f  │ rwd_v  │ rwd_s  │ rwd_t  │ rwd_h  │  notes        │
# │              │(facing)│ (vel)  │(strike)│  (KO)  │ (hit)  │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ orig paper   │  0.3   │  0.2   │  1.0   │  0.5   │  0.3   │ ← has lazy eq│
# │ default (v2) │  0.15  │  0.1   │  2.0   │  1.0   │  0.5   │ balanced fix  │
# │ aggressive   │  0.1   │  0.15  │  2.5   │  1.5   │  0.6   │ rush & KO     │
# │ defensive    │  0.3   │  0.0   │  0.8   │  0.3   │  0.2   │ survive+counter│
# │ counter      │  0.2   │  0.05  │  2.5   │  1.5   │  0.3   │ wait → punish │
# │ brawler      │  0.05  │  0.1   │  3.0   │  1.0   │  0.8   │ max damage    │
# │ outfighter   │  0.2   │  0.05  │  1.2   │  0.3   │  0.8   │ poke from far │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │tar_spd │vel_err │fac_err │hd_mul  │str_div │               │
# │              │ (m/s)  │(scale) │(scale) │(head×) │(÷norm) │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ orig paper   │  1.0   │  4.0   │  2.0   │  2.0   │  10.0  │               │
# │ default (v2) │  1.0   │  4.0   │  2.0   │  2.0   │   5.0  │ 2× force amp  │
# │ aggressive   │  1.5   │  4.0   │  2.0   │  3.0   │   4.0  │ fast + headhnt│
# │ defensive    │  0.3   │  2.0   │  3.0   │  2.0   │   8.0  │ cautious      │
# │ counter      │  0.4   │  3.0   │  3.0   │  3.0   │   4.0  │ head hunter   │
# │ brawler      │  1.2   │  4.0   │  1.5   │  1.5   │   3.0  │ raw force 3×  │
# │ outfighter   │  0.5   │  3.0   │  2.5   │  2.0   │   6.0  │ controlled    │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │task_rw │disc_rw │sw_freq │               extra notes        │
# │              │(task%) │(AMP %) │(epoch) │                                  │
# ├──────────────┼────────┼────────┼────────┼──────────────────────────────────┤
# │ orig paper   │  0.5   │  0.5   │  250   │ ← AMP too strong for early expl │
# │ default (v2) │  0.6   │  0.4   │  250   │ loosen AMP for exploration       │
# │ aggressive   │  0.75  │  0.25  │  200   │ max task → sacrifice naturalness │
# │ defensive    │  0.4   │  0.6   │  300   │ more AMP → natural evasion       │
# │ counter      │  0.6   │  0.4   │  250   │ balanced, explosive on contact   │
# │ brawler      │  0.8   │  0.2   │  200   │ minimal AMP → wild swings ok     │
# │ outfighter   │  0.5   │  0.5   │  300   │ balanced AMP → technical style   │
# └──────────────┴────────┴────────┴────────┴──────────────────────────────────┘
#
# Design rationale per style (v2):
#
#   default (v2) — Fixed version of original. Approach rewards cut by half,
#                  strike doubled, terminate doubled, strike_divisor halved
#                  (5→same force gives 2× reward). disc_reward_w reduced to
#                  0.4 so agent can explore striking motions more freely.
#                  This should break the "walk close and stand" equilibrium.
#
#   aggressive   — "Pressure fighter". Highest strike (2.5) and KO (1.5)
#                  rewards combined with fast approach (tar_speed=1.5).
#                  Very low disc_reward_w=0.25 allows wild aggressive motions.
#                  head_strike_mul=3.0 rewards head targeting. Fastest
#                  switch_frequency=200 keeps competitive pressure high.
#                  The small vel reward (0.15) just enough to close distance.
#
#   defensive    — "Evasive / counter-survival". Zero velocity reward means
#                  no incentive to chase. Moderate facing (0.3) for awareness.
#                  Still has meaningful strike (0.8) and terminate (0.3) so
#                  the agent learns to counter-strike when opponent rushes in.
#                  Previous v1 had strike=0.3/terminate=0.0 which made the
#                  agent truly passive — that's unrealistic. Even a defensive
#                  boxer needs to hit back. Higher disc_reward_w=0.6 keeps
#                  motion fluid and evasive. strike_divisor=8 (not 15).
#
#   counter      — "Counter-puncher". Very low approach motivation (vel=0.05).
#                  Patient, waits for opponent. But when contact happens:
#                  strike=2.5, terminate=1.5, head_strike_mul=3.0 creates
#                  explosive punishment. strike_divisor=4.0 means even moderate
#                  contact produces large reward — encouraging quick decisive
#                  counter-strikes rather than sustained brawling.
#
#   brawler      — "Heavy hitter / slugger". Maximum strike emphasis (3.0)
#                  with the lowest strike_divisor (3.0) — same force gives
#                  3.3× the reward vs original. High hit proximity (0.8)
#                  strongly pulls hands toward target. Lowest disc_reward_w
#                  (0.2) accepts messy/unnatural motions for pure aggression.
#                  Minimal facing reward (0.05) — doesn't care about stance,
#                  just wants to hit.
#
#   outfighter   — "Technical / jab-and-move". Moderate strike (1.2) but high
#                  hit proximity (0.8) — rewards getting hands near targets
#                  without necessarily needing huge force (light jabs). Higher
#                  disc_reward_w=0.5 keeps movement fluid and natural. Low
#                  vel (0.05) means it doesn't chase aggressively. The
#                  combination of high hit + moderate strike + balanced AMP
#                  should produce a "poke and move" style.
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
    echo "=== Training DEFAULT v2 (balanced, lazy-eq fix) ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_default_v2 \
      +env.reward_f=0.15 +env.reward_v=0.1 +env.reward_s=2.0 +env.reward_t=1.0 +env.reward_h=0.5 \
      +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  aggressive)
    echo "=== Training AGGRESSIVE (pressure) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_aggressive_v2 \
      +env.reward_f=0.1 +env.reward_v=0.15 +env.reward_s=2.5 +env.reward_t=1.5 +env.reward_h=0.6 \
      +env.tar_speed=1.5 +env.head_strike_mul=3.0 +env.strike_divisor=4.0 \
      learning.params.config.task_reward_w=0.75 \
      learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200
    ;;

  defensive)
    echo "=== Training DEFENSIVE (evasive + counter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_defensive_v2 \
      +env.reward_f=0.3 +env.reward_v=0.0 +env.reward_s=0.8 +env.reward_t=0.3 +env.reward_h=0.2 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=2.0 +env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.4 \
      learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300
    ;;

  counter)
    echo "=== Training COUNTER-PUNCHER fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_counter_v2 \
      +env.reward_f=0.2 +env.reward_v=0.05 +env.reward_s=2.5 +env.reward_t=1.5 +env.reward_h=0.3 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=3.0 +env.strike_divisor=4.0 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  brawler)
    echo "=== Training BRAWLER (heavy hitter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_brawler_v2 \
      +env.reward_f=0.05 +env.reward_v=0.1 +env.reward_s=3.0 +env.reward_t=1.0 +env.reward_h=0.8 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      +env.head_strike_mul=1.5 +env.strike_divisor=3.0 \
      learning.params.config.task_reward_w=0.8 \
      learning.params.config.disc_reward_w=0.2 \
      learning.params.config.switch_frequency=200
    ;;

  outfighter)
    echo "=== Training OUTFIGHTER (technical) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_outfighter_v2 \
      +env.reward_f=0.2 +env.reward_v=0.05 +env.reward_s=1.2 +env.reward_t=0.3 +env.reward_h=0.8 \
      +env.tar_speed=0.5 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
      +env.head_strike_mul=2.0 +env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.5 \
      learning.params.config.disc_reward_w=0.5 \
      learning.params.config.switch_frequency=300
    ;;

  all)
    echo "=== Training ALL styles in parallel (6 GPUs) ==="
    echo "Make sure you have 6 GPUs available!"

    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} exp_name=boxing_default_v2 \
      +env.reward_f=0.15 +env.reward_v=0.1 +env.reward_s=2.0 +env.reward_t=1.0 +env.reward_h=0.5 \
      +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=1 ${BASE_CMD} exp_name=boxing_aggressive_v2 \
      +env.reward_f=0.1 +env.reward_v=0.15 +env.reward_s=2.5 +env.reward_t=1.5 +env.reward_h=0.6 \
      +env.tar_speed=1.5 +env.head_strike_mul=3.0 +env.strike_divisor=4.0 \
      learning.params.config.task_reward_w=0.75 learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=2 ${BASE_CMD} exp_name=boxing_defensive_v2 \
      +env.reward_f=0.3 +env.reward_v=0.0 +env.reward_s=0.8 +env.reward_t=0.3 +env.reward_h=0.2 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=2.0 +env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.4 learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300 &

    CUDA_VISIBLE_DEVICES=3 ${BASE_CMD} exp_name=boxing_counter_v2 \
      +env.reward_f=0.2 +env.reward_v=0.05 +env.reward_s=2.5 +env.reward_t=1.5 +env.reward_h=0.3 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=3.0 +env.strike_divisor=4.0 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=4 ${BASE_CMD} exp_name=boxing_brawler_v2 \
      +env.reward_f=0.05 +env.reward_v=0.1 +env.reward_s=3.0 +env.reward_t=1.0 +env.reward_h=0.8 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      +env.head_strike_mul=1.5 +env.strike_divisor=3.0 \
      learning.params.config.task_reward_w=0.8 learning.params.config.disc_reward_w=0.2 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=5 ${BASE_CMD} exp_name=boxing_outfighter_v2 \
      +env.reward_f=0.2 +env.reward_v=0.05 +env.reward_s=1.2 +env.reward_t=0.3 +env.reward_h=0.8 \
      +env.tar_speed=0.5 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
      +env.head_strike_mul=2.0 +env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.5 learning.params.config.disc_reward_w=0.5 \
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
