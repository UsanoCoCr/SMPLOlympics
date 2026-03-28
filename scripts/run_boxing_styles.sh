#!/bin/bash
# =============================================================================
# Train boxing policies with different fighter styles.
#
# Each style is a different reward weight preset injected via Hydra overrides.
# The original default behavior is fully preserved when no env.reward_*
# overrides are passed — the code falls back to the hardcoded defaults.
#
# v3 — Fix reward magnitude explosion from v2 that caused training instability.
#
# v2 mistake: inflated total reward weights (2.3 → 4.5+) AND reduced
#   strike_divisor aggressively (10 → 3~4). The combined effect made the
#   effective strike signal 6× larger, causing PPO's value function to diverge,
#   huge advantage estimates, wild policy updates, and eventual NaN crash.
#
# v3 fix: keep total weight sum ≈ 2.3 (same as original) for ALL styles.
#   Only redistribute the budget — shift proportion from approach (facing, vel)
#   to combat (strike, terminate, hit). strike_divisor reduced modestly
#   (10 → 6~8, never below 5). disc_reward_w still lowered to allow
#   exploration, which is the safest lever to pull.
#
# Usage:
#   bash scripts/run_boxing_styles.sh <style>
#   e.g.:  bash scripts/run_boxing_styles.sh aggressive
#          bash scripts/run_boxing_styles.sh default
#          bash scripts/run_boxing_styles.sh all
#          bash scripts/run_boxing_styles.sh all
#
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                    FIGHTER STYLE PARAMETER REFERENCE  (v3)                  │
# │                    total reward weight sum ≈ 2.3 for all                    │
# ├──────────────┬────────┬────────┬────────┬────────┬────────┬────────────────┤
# │              │ rwd_f  │ rwd_v  │ rwd_s  │ rwd_t  │ rwd_h  │ sum   notes   │
# │              │(facing)│ (vel)  │(strike)│  (KO)  │ (hit)  │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ orig paper   │  0.3   │  0.2   │  1.0   │  0.5   │  0.3   │ 2.3  lazy eq  │
# │ default (v3) │  0.15  │  0.10  │  1.20  │  0.50  │  0.35  │ 2.3  balanced │
# │ aggressive   │  0.05  │  0.10  │  1.30  │  0.50  │  0.35  │ 2.3  rush&KO  │
# │ defensive    │  0.40  │  0.00  │  1.00  │  0.30  │  0.60  │ 2.3  survive  │
# │ counter      │  0.20  │  0.05  │  1.30  │  0.50  │  0.25  │ 2.3  punish   │
# │ brawler      │  0.05  │  0.05  │  1.40  │  0.40  │  0.40  │ 2.3  damage   │
# │ outfighter   │  0.20  │  0.05  │  1.00  │  0.25  │  0.80  │ 2.3  poke     │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │tar_spd │vel_err │fac_err │hd_mul  │str_div │               │
# │              │ (m/s)  │(scale) │(scale) │(head×) │(÷norm) │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ orig paper   │  1.0   │  4.0   │  2.0   │  2.0   │  10.0  │               │
# │ default (v3) │  1.0   │  4.0   │  2.0   │  2.0   │   6.0  │ modest amp    │
# │ aggressive   │  1.5   │  4.0   │  2.0   │  3.0   │   5.0  │ fast+headhunt │
# │ defensive    │  0.3   │  2.0   │  3.0   │  2.0   │   8.0  │ cautious      │
# │ counter      │  0.4   │  3.0   │  3.0   │  3.0   │   5.0  │ head hunter   │
# │ brawler      │  1.2   │  4.0   │  1.5   │  1.5   │   5.0  │ raw force     │
# │ outfighter   │  0.5   │  3.0   │  2.5   │  2.0   │   7.0  │ controlled    │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │task_rw │disc_rw │sw_freq │               extra notes        │
# │              │(task%) │(AMP %) │(epoch) │                                  │
# ├──────────────┼────────┼────────┼────────┼──────────────────────────────────┤
# │ orig paper   │  0.5   │  0.5   │  250   │ AMP too strong for early expl    │
# │ default (v3) │  0.6   │  0.4   │  250   │ loosen AMP for exploration       │
# │ aggressive   │  0.7   │  0.3   │  200   │ more task → sacrifice style      │
# │ defensive    │  0.4   │  0.6   │  300   │ more AMP → natural evasion       │
# │ counter      │  0.6   │  0.4   │  250   │ balanced, explosive on contact   │
# │ brawler      │  0.75  │  0.25  │  200   │ low AMP → messy swings ok        │
# │ outfighter   │  0.5   │  0.5   │  300   │ balanced AMP → fluid movement    │
# └──────────────┴────────┴────────┴────────┴──────────────────────────────────┘
#
# Key differences vs original (all styles share these):
#
#   1. Approach rewards (facing + vel) reduced from 0.5 to 0.05~0.25.
#      This removes the dense "walk close and stand" local optimum.
#
#   2. The freed budget is redistributed to strike/terminate/hit.
#      strike is now 1.0~1.4 (was 1.0), but the real amplification
#      comes from strike_divisor reduction: 6~8 (was 10), giving
#      1.25~1.67× effective strike signal WITHOUT inflating totals.
#
#   3. disc_reward_w lowered (0.5 → 0.25~0.5) — the single safest
#      lever. This doesn't affect reward scale at all, but lets the
#      agent explore motions the AMP discriminator considers "unnatural"
#      (e.g. punching), which is crucial for breaking the lazy equilibrium.
#
# Per-style rationale:
#
#   default (v3) — Conservative fix. Approach budget halved (0.5→0.25),
#                  redistributed to strike (+0.2) and hit (+0.05).
#                  strike_divisor 10→6 gives 1.67× force amplification.
#                  disc_reward_w 0.5→0.4. Should break lazy eq safely.
#
#   aggressive   — Minimal approach (0.15), most budget on strike (1.3)
#                  and KO (0.5). tar_speed=1.5 closes distance fast.
#                  head_strike_mul=3.0 hunts headshots. disc_rw=0.3
#                  permits aggressive motions. strike_divisor=5.
#
#   defensive    — Zero velocity, high facing (0.4) for awareness.
#                  High hit proximity (0.6) — hands stay near guard.
#                  strike=1.0 and terminate=0.3 still allow counter-
#                  strikes. disc_rw=0.6 keeps evasive motion natural.
#
#   counter      — Very patient (vel=0.05). When opponent comes in:
#                  strike=1.3, terminate=0.5, head_mul=3.0. The high
#                  head multiplier + low divisor (5.0) means a single
#                  clean headshot produces a big reward spike, teaching
#                  the agent to wait and punish.
#
#   brawler      — Maximum strike proportion (1.4/2.3 = 61%). Doesn't
#                  care about stance (facing=0.05) or approach (vel=0.05).
#                  Lowest disc_rw (0.25) — messy brawling is fine.
#                  hit=0.4 pulls hands toward targets constantly.
#
#   outfighter   — Highest hit proximity (0.8/2.3 = 35%) — rewards
#                  keeping hands near targets (jab range) more than
#                  raw force. Moderate strike (1.0), balanced disc_rw
#                  (0.5) for fluid technical movement.
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
    echo "=== Training DEFAULT v3 (balanced, lazy-eq fix) ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_default_v3 \
      +env.reward_f=0.15 +env.reward_v=0.10 +env.reward_s=1.20 +env.reward_t=0.50 +env.reward_h=0.35 \
      +env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  aggressive)
    echo "=== Training AGGRESSIVE (pressure) fighter ==="
    CUDA_VISIBLE_DEVICES=2 ${BASE_CMD} \
      exp_name=boxing_aggressive_v3 \
      +env.reward_f=0.05 +env.reward_v=0.10 +env.reward_s=1.30 +env.reward_t=0.50 +env.reward_h=0.35 \
      +env.tar_speed=1.5 +env.head_strike_mul=3.0 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.7 \
      learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200
    ;;

  defensive)
    echo "=== Training DEFENSIVE (evasive + counter) fighter ==="
    echo "=== Training DEFENSIVE (evasive + counter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_defensive_v3 \
      +env.reward_f=0.40 +env.reward_v=0.00 +env.reward_s=1.00 +env.reward_t=0.30 +env.reward_h=0.60 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=2.0 +env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.4 \
      learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300
    ;;

  counter)
    echo "=== Training COUNTER-PUNCHER fighter ==="
    CUDA_VISIBLE_DEVICES=3 ${BASE_CMD} \
      exp_name=boxing_counter_v3 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_s=1.30 +env.reward_t=0.50 +env.reward_h=0.25 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=3.0 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  brawler)
    echo "=== Training BRAWLER (heavy hitter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_brawler_v3 \
      +env.reward_f=0.05 +env.reward_v=0.05 +env.reward_s=1.40 +env.reward_t=0.40 +env.reward_h=0.40 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      +env.head_strike_mul=1.5 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.75 \
      learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200
    ;;

  outfighter)
    echo "=== Training OUTFIGHTER (technical) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_outfighter_v3 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_s=1.00 +env.reward_t=0.25 +env.reward_h=0.80 \
      +env.tar_speed=0.5 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
      +env.head_strike_mul=2.0 +env.strike_divisor=7.0 \
      learning.params.config.task_reward_w=0.5 \
      learning.params.config.disc_reward_w=0.5 \
      learning.params.config.switch_frequency=300
    ;;

  all)
    echo "=== Training ALL styles in parallel (6 GPUs) ==="
    echo "Make sure you have 6 GPUs available!"

    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} exp_name=boxing_default_v3 \
      +env.reward_f=0.15 +env.reward_v=0.10 +env.reward_s=1.20 +env.reward_t=0.50 +env.reward_h=0.35 \
      +env.strike_divisor=6.0 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=1 ${BASE_CMD} exp_name=boxing_aggressive_v3 \
      +env.reward_f=0.05 +env.reward_v=0.10 +env.reward_s=1.30 +env.reward_t=0.50 +env.reward_h=0.35 \
      +env.tar_speed=1.5 +env.head_strike_mul=3.0 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.7 learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=2 ${BASE_CMD} exp_name=boxing_defensive_v3 \
      +env.reward_f=0.40 +env.reward_v=0.00 +env.reward_s=1.00 +env.reward_t=0.30 +env.reward_h=0.60 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=2.0 +env.strike_divisor=8.0 \
      learning.params.config.task_reward_w=0.4 learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300 &

    CUDA_VISIBLE_DEVICES=3 ${BASE_CMD} exp_name=boxing_counter_v3 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_s=1.30 +env.reward_t=0.50 +env.reward_h=0.25 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      +env.head_strike_mul=3.0 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=4 ${BASE_CMD} exp_name=boxing_brawler_v3 \
      +env.reward_f=0.05 +env.reward_v=0.05 +env.reward_s=1.40 +env.reward_t=0.40 +env.reward_h=0.40 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      +env.head_strike_mul=1.5 +env.strike_divisor=5.0 \
      learning.params.config.task_reward_w=0.75 learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=5 ${BASE_CMD} exp_name=boxing_outfighter_v3 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_s=1.00 +env.reward_t=0.25 +env.reward_h=0.80 \
      +env.tar_speed=0.5 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
      +env.head_strike_mul=2.0 +env.strike_divisor=7.0 \
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
