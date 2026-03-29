#!/bin/bash
# =============================================================================
# Train boxing policies with different fighter styles.
#
# v4 — Fix reward variance explosion from v2/v3.
#
# Root cause of v2/v3 instability:
#   strike_reward is the ONLY unbounded component (range ≈ ±140 at default).
#   Reducing strike_divisor (10→5) and raising head_strike_mul (2→3) doubled
#   the raw value to ±320. Combined with increased strike weight, the effective
#   variance grew ~9×, causing PPO value function divergence and crash.
#
# v4 rule: NEVER modify strike_divisor, head_strike_mul, reward_s, or reward_t.
#   These control the only unbounded/spiky components. Keep them at original
#   values to preserve training stability.
#
# Safe levers for breaking lazy equilibrium:
#   1. Reduce facing + vel weights (bounded [0,1]) → less "approach" incentive
#   2. Increase hit weight (bounded [0,1]) → bridge from approach to striking
#   3. Lower disc_reward_w → let agent explore "unnatural" punching motions
#   4. Tune tar_speed, vel_err_scale, facing_err_scale → shape bounded rewards
#   5. Tune switch_frequency → change self-play competitive pressure
#
# Usage:
#   bash scripts/run_boxing_styles.sh <style>
#   bash scripts/run_boxing_styles.sh all
#
# =============================================================================
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │                    FIGHTER STYLE PARAMETER REFERENCE  (v4)                  │
# │         total weight sum ≈ 2.3 | strike params LOCKED to original          │
# ├──────────────┬────────┬────────┬────────┬────────┬────────┬────────────────┤
# │              │ rwd_f  │ rwd_v  │ rwd_s  │ rwd_t  │ rwd_h  │ sum   notes   │
# │              │ rwd_f  │ rwd_v  │ rwd_s  │ rwd_t  │ rwd_h  │ sum   notes   │
# │              │(facing)│ (vel)  │(strike)│  (KO)  │ (hit)  │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ orig paper   │  0.3   │  0.2   │  1.0   │  0.5   │  0.3   │ 2.3  lazy eq  │
# │ default (v4) │  0.15  │  0.10  │ *1.0*  │ *0.5*  │  0.55  │ 2.3  fix lazy │
# │ aggressive   │  0.05  │  0.15  │ *1.0*  │ *0.5*  │  0.60  │ 2.3  rush&KO  │
# │ defensive    │  0.40  │  0.00  │ *1.0*  │ *0.5*  │  0.40  │ 2.3  survive  │
# │ counter      │  0.20  │  0.05  │ *1.0*  │ *0.5*  │  0.55  │ 2.3  punish   │
# │ brawler      │  0.05  │  0.10  │ *1.0*  │ *0.5*  │  0.65  │ 2.3  damage   │
# │ outfighter   │  0.15  │  0.05  │ *1.0*  │ *0.5*  │  0.60  │ 2.3  poke     │
# │              │        │        │ LOCKED │ LOCKED │        │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │tar_spd │vel_err │fac_err │hd_mul  │str_div │               │
# │              │ (m/s)  │(scale) │(scale) │(head×) │(÷norm) │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ ALL STYLES   │  vary  │  vary  │  vary  │ *2.0*  │ *10.0* │ LOCKED cols   │
# │              │        │        │        │ LOCKED │ LOCKED │               │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │ default (v4) │  1.0   │  4.0   │  2.0   │  2.0   │  10.0  │ = original    │
# │ aggressive   │  1.5   │  4.0   │  2.0   │  2.0   │  10.0  │ faster close  │
# │ defensive    │  0.3   │  2.0   │  3.0   │  2.0   │  10.0  │ slow+aware    │
# │ counter      │  0.4   │  3.0   │  3.0   │  2.0   │  10.0  │ patient       │
# │ brawler      │  1.2   │  4.0   │  1.5   │  2.0   │  10.0  │ charge in     │
# │ outfighter   │  0.6   │  3.0   │  2.5   │  2.0   │  10.0  │ controlled    │
# ├──────────────┼────────┼────────┼────────┼────────┼────────┼────────────────┤
# │              │task_rw │disc_rw │sw_freq │               extra notes        │
# │              │(task%) │(AMP %) │(epoch) │                                  │
# ├──────────────┼────────┼────────┼────────┼──────────────────────────────────┤
# │ orig paper   │  0.5   │  0.5   │  250   │ AMP suppresses punch exploration│
# │ default (v4) │  0.6   │  0.4   │  250   │ main lever: loosen AMP          │
# │ aggressive   │  0.7   │  0.3   │  200   │ most task freedom               │
# │ defensive    │  0.4   │  0.6   │  300   │ natural evasive motion           │
# │ counter      │  0.6   │  0.4   │  250   │ balanced                         │
# │ brawler      │  0.75  │  0.25  │  200   │ max task freedom, messy ok       │
# │ outfighter   │  0.5   │  0.5   │  300   │ fluid technical movement         │
# └──────────────┴────────┴────────┴────────┴──────────────────────────────────┘
#
# Style differentiation strategy (variance-safe):
#
#   The 5 reward components split into two groups:
#     BOUNDED [0,1]: facing, vel, hit  — safe to redistribute freely
#     UNBOUNDED:     strike (±hundreds), terminate (±1 but spiky)
#
#   Styles are differentiated ONLY through:
#     a) Redistribution within bounded group (facing/vel ↔ hit)
#     b) disc_reward_w (doesn't touch task reward variance at all)
#     c) tar_speed (changes approach behavior, not reward scale)
#     d) facing_err_scale / vel_err_scale (shape of bounded rewards)
#     e) switch_frequency (self-play dynamics)
#
#   NEVER through strike_divisor, head_strike_mul, reward_s, reward_t.
#
# Per-style rationale:
#
#   default (v4) — Approach budget halved (0.5→0.25), redirected to hit
#                  (0.3→0.55). disc_rw=0.4 loosens AMP exploration.
#                  All strike params = original. Training variance ≈ original.
#
#   aggressive   — Minimal approach budget (0.20). tar_speed=1.5 rushes in.
#                  Highest hit (0.60) → hands always seeking targets.
#                  Lowest disc_rw (0.3) → wild motion allowed.
#                  Fast switch (200) → high competitive pressure.
#
#   defensive    — Zero vel reward, high facing (0.40) → watches but doesn't
#                  chase. tar_speed=0.3, vel_err_scale=2.0 → very slow.
#                  facing_err_scale=3.0 → sharp penalty for looking away.
#                  Highest disc_rw (0.6) → natural upright evasive motion.
#
#   counter      — Patient: vel=0.05, tar_speed=0.4. Waits for opponent.
#                  facing=0.20 + facing_err_scale=3.0 → always watching.
#                  hit=0.55 as bridge. Strike/terminate at original values
#                  still reward hard when contact happens — just doesn't
#                  need weight inflation because divisor=10 already gives
#                  big spikes on clean hits.
#
#   brawler      — Minimal facing (0.05) — doesn't care about stance.
#                  Highest hit (0.65) → hands constantly near targets.
#                  facing_err_scale=1.5 → loose facing tolerance.
#                  Lowest disc_rw (0.25) → messy brawling motion ok.
#
#   outfighter   — Moderate approach (0.20), high hit (0.60) → poke range.
#                  tar_speed=0.6 → doesn't rush but doesn't retreat.
#                  Balanced disc_rw (0.5) → fluid controlled movement.
#                  vel_err_scale=3.0 → precise speed control.
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
    echo "=== Training DEFAULT v4 (lazy-eq fix, stable variance) ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_default_v4 \
      +env.reward_f=0.15 +env.reward_v=0.10 +env.reward_h=0.55 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  aggressive)
    echo "=== Training AGGRESSIVE (pressure) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_aggressive_v4 \
      +env.reward_f=0.05 +env.reward_v=0.15 +env.reward_h=0.60 \
      +env.tar_speed=1.5 \
      learning.params.config.task_reward_w=0.7 \
      learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200
    ;;

  defensive)
    echo "=== Training DEFENSIVE (evasive + counter) fighter ==="
    echo "=== Training DEFENSIVE (evasive + counter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_defensive_v4 \
      +env.reward_f=0.40 +env.reward_v=0.00 +env.reward_h=0.40 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      learning.params.config.task_reward_w=0.4 \
      learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300
    ;;

  counter)
    echo "=== Training COUNTER-PUNCHER fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_counter_v4 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_h=0.55 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      learning.params.config.task_reward_w=0.6 \
      learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250
    ;;

  brawler)
    echo "=== Training BRAWLER (heavy hitter) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_brawler_v4 \
      +env.reward_f=0.05 +env.reward_v=0.10 +env.reward_h=0.65 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      learning.params.config.task_reward_w=0.75 \
      learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200
    ;;

  outfighter)
    echo "=== Training OUTFIGHTER (technical) fighter ==="
    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} \
      exp_name=boxing_outfighter_v4 \
      +env.reward_f=0.15 +env.reward_v=0.05 +env.reward_h=0.60 \
      +env.tar_speed=0.6 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
      learning.params.config.task_reward_w=0.5 \
      learning.params.config.disc_reward_w=0.5 \
      learning.params.config.switch_frequency=300
    ;;

  all)
    echo "=== Training ALL styles in parallel (6 GPUs) ==="
    echo "Make sure you have 6 GPUs available!"

    CUDA_VISIBLE_DEVICES=0 ${BASE_CMD} exp_name=boxing_default_v4 \
      +env.reward_f=0.15 +env.reward_v=0.10 +env.reward_h=0.55 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=1 ${BASE_CMD} exp_name=boxing_aggressive_v4 \
      +env.reward_f=0.05 +env.reward_v=0.15 +env.reward_h=0.60 \
      +env.tar_speed=1.5 \
      learning.params.config.task_reward_w=0.7 learning.params.config.disc_reward_w=0.3 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=2 ${BASE_CMD} exp_name=boxing_defensive_v4 \
      +env.reward_f=0.40 +env.reward_v=0.00 +env.reward_h=0.40 \
      +env.tar_speed=0.3 +env.vel_err_scale=2.0 +env.facing_err_scale=3.0 \
      learning.params.config.task_reward_w=0.4 learning.params.config.disc_reward_w=0.6 \
      learning.params.config.switch_frequency=300 &

    CUDA_VISIBLE_DEVICES=3 ${BASE_CMD} exp_name=boxing_counter_v4 \
      +env.reward_f=0.20 +env.reward_v=0.05 +env.reward_h=0.55 \
      +env.tar_speed=0.4 +env.vel_err_scale=3.0 +env.facing_err_scale=3.0 \
      learning.params.config.task_reward_w=0.6 learning.params.config.disc_reward_w=0.4 \
      learning.params.config.switch_frequency=250 &

    CUDA_VISIBLE_DEVICES=4 ${BASE_CMD} exp_name=boxing_brawler_v4 \
      +env.reward_f=0.05 +env.reward_v=0.10 +env.reward_h=0.65 \
      +env.tar_speed=1.2 +env.facing_err_scale=1.5 \
      learning.params.config.task_reward_w=0.75 learning.params.config.disc_reward_w=0.25 \
      learning.params.config.switch_frequency=200 &

    CUDA_VISIBLE_DEVICES=5 ${BASE_CMD} exp_name=boxing_outfighter_v4 \
      +env.reward_f=0.15 +env.reward_v=0.05 +env.reward_h=0.60 \
      +env.tar_speed=0.6 +env.vel_err_scale=3.0 +env.facing_err_scale=2.5 \
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
