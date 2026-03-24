"""
Rollout boxing self-play policy and collect paired SMPL motion sequences.

This script launches the boxing environment in inference mode, runs the trained
policy for both players, and saves high-quality continuous paired SMPL data.

Usage example (run from the SMPLOlympics root directory):
    python scripts/rollout_boxing_pairs.py \
        --checkpoint output/SMPLOlympics/boxing_amp/Humanoid.pth \
        --num_envs 256 \
        --num_rollout_episodes 100 \
        --episode_length 300 \
        --output_dir rollout_data/boxing \
        --min_seq_len 60

Key design decisions:
  - Does NOT modify any original source files.
  - Filters out episodes where either agent falls (early termination), keeping
    only full-length, high-quality continuous sequences.
  - Saves player1 and player2 SMPL data in the same file with clear keys.
"""

import os
import sys
import argparse
import time
import glob
import copy

import numpy as np
import torch
import pickle

# ---------------------------------------------------------------------------
# Ensure correct import paths (same as run_hydra.py)
# ---------------------------------------------------------------------------
os.environ['OMP_NUM_THREADS'] = "1"
sys.path.insert(0, os.getcwd())
sys.path.insert(0, './SMPLSim')

from phc.utils.config import set_np_formatting, set_seed, SIM_TIMESTEP
from phc.utils.parse_task import parse_task
from phc.utils.flags import flags
from isaacgym import gymapi, gymutil
from rl_games.algos_torch import torch_ext
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import AlgoObserver
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import hydra

from learning import amp_agent, amp_self_play_agent
from learning import amp_players, amp_self_play_players
from learning import amp_models
from learning import amp_network_builder, amp_network_mcp_builder
from learning import amp_network_pnn_builder, amp_network_z_builder
from learning import im_amp, im_amp_players
from env.tasks import humanoid_amp_task


# ========================== helper functions ================================

def compute_duration_estimate(num_envs, num_rollout_episodes, episode_length,
                              dt, min_seq_len):
    """Print a human-readable estimate of total rollout data duration."""
    # dt = control_freq_inv * sim_timestep.  Default: 2 * (1/60) = 1/30 s
    total_episodes_upper_bound = num_rollout_episodes  # we collect this many *good* episodes
    # worst-case: every episode yields a valid sequence
    best_case_total_frames = total_episodes_upper_bound * episode_length
    best_case_seconds = best_case_total_frames * dt
    best_case_hours = best_case_seconds / 3600.0

    # With parallel envs, each rollout step produces num_envs frames in
    # parallel.  The total UNIQUE frames is still episodes * episode_length.
    print("=" * 65)
    print("  Rollout Data Estimate")
    print("=" * 65)
    print(f"  num_envs (parallel)        : {num_envs}")
    print(f"  num_rollout_episodes       : {num_rollout_episodes}")
    print(f"  episode_length (steps)     : {episode_length}")
    print(f"  min_seq_len (filter)       : {min_seq_len}")
    print(f"  control dt                 : {dt:.4f} s   (= {1.0/dt:.1f} Hz)")
    print(f"  episode duration           : {episode_length * dt:.2f} s")
    print("-" * 65)
    print(f"  max valid episodes         : {num_rollout_episodes}")
    print(f"  max total frames (paired)  : {best_case_total_frames}")
    print(f"  max total duration         : {best_case_seconds:.1f} s  "
          f"= {best_case_hours:.4f} hours")
    print("=" * 65)
    return best_case_hours


def extract_smpl_frame(task, agent_idx):
    """
    Extract a single-frame SMPL state dict for one agent from the live env.

    Returns a dict with numpy arrays, all shaped [num_envs, ...].
    """
    data = {}
    data['root_pos']         = task._humanoid_root_states_list[agent_idx][:, 0:3].cpu().numpy().copy()   # (N, 3)
    data['root_rot']         = task._humanoid_root_states_list[agent_idx][:, 3:7].cpu().numpy().copy()   # (N, 4)  quat
    data['root_vel']         = task._humanoid_root_states_list[agent_idx][:, 7:10].cpu().numpy().copy()  # (N, 3)
    data['root_ang_vel']     = task._humanoid_root_states_list[agent_idx][:, 10:13].cpu().numpy().copy() # (N, 3)
    data['dof_pos']          = task._dof_pos_list[agent_idx].cpu().numpy().copy()                        # (N, 69)
    data['dof_vel']          = task._dof_vel_list[agent_idx].cpu().numpy().copy()                        # (N, 69)
    data['body_pos']         = task._rigid_body_pos_list[agent_idx].cpu().numpy().copy()                 # (N, 24, 3)
    data['body_rot']         = task._rigid_body_rot_list[agent_idx].cpu().numpy().copy()                 # (N, 24, 4)
    data['body_vel']         = task._rigid_body_vel_list[agent_idx].cpu().numpy().copy()                 # (N, 24, 3)
    data['body_ang_vel']     = task._rigid_body_ang_vel_list[agent_idx].cpu().numpy().copy()             # (N, 24, 3)
    return data


# ========================== globals for hydra ===============================
cfg = None
cfg_train = None


def parse_sim_params_local(cfg):
    """Mirrors run_hydra.py parse_sim_params."""
    sim_params = gymapi.SimParams()
    sim_params.dt = SIM_TIMESTEP
    sim_params.num_client_threads = cfg.sim.slices

    if cfg.sim.use_flex:
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        sim_params.physx.max_gpu_contact_pairs = 16 * 1024 * 1024

    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]

    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads

    return sim_params


def create_rlgpu_env(**kwargs):
    global cfg, cfg_train
    sim_params = parse_sim_params_local(cfg)
    args_env = EasyDict({
        "task": cfg.env.task,
        "device_id": cfg.device_id,
        "rl_device": cfg.rl_device,
        "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
        "headless": cfg.headless,
        "device": cfg.device,
    })
    task, env = parse_task(args_env, cfg, cfg_train, sim_params)
    frames = kwargs.pop('frames', 1)
    return env


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)
        self.use_global_obs = (self.env.num_states > 0)
        self.full_state = {}
        self.full_state["obs"] = self.reset()
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()

    def step(self, action):
        next_obs, reward, is_done, info = self.env.step(action)
        self.full_state["obs"] = next_obs
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state, reward, is_done, info
        else:
            return self.full_state["obs"], reward, is_done, info

    def reset(self, env_ids=None):
        self.full_state["obs"] = self.env.reset(env_ids)
        if self.use_global_obs:
            self.full_state["states"] = self.env.get_state()
            return self.full_state
        else:
            return self.full_state["obs"]

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        info['amp_observation_space'] = self.env.amp_observation_space
        info['enc_amp_observation_space'] = self.env.enc_amp_observation_space
        if isinstance(self.env.task, humanoid_amp_task.HumanoidAMPTask):
            info['task_obs_size'] = self.env.task.get_task_obs_size()
        else:
            info['task_obs_size'] = 0
        if self.use_global_obs:
            info['state_space'] = self.env.state_space
        return info


class DummyAlgoObserver(AlgoObserver):
    def after_init(self, algo): pass
    def process_infos(self, infos, done_indices): pass
    def after_clear_stats(self): pass
    def after_print_stats(self, frame, epoch_num, total_time): pass


# ========================== main rollout ====================================

def rollout(player, args):
    """
    Core rollout loop.  Mimics CommonPlayer.run() but:
      - records per-env, per-step SMPL data for BOTH agents
      - discards episodes that terminated early (knockdown / fall)
      - keeps only clean, continuous sequences of length >= min_seq_len
    """
    env = player.env
    task = env.task
    num_envs = task.num_envs
    num_agents = task.num_agents
    episode_length = task.max_episode_length
    dt = task.dt

    # ---------- estimate data volume ----------
    compute_duration_estimate(
        num_envs=num_envs,
        num_rollout_episodes=args.num_rollout_episodes,
        episode_length=episode_length,
        dt=dt,
        min_seq_len=args.min_seq_len,
    )

    # ---------- per-env accumulation buffers ----------
    # Each element is a list of frame-dicts; index by env_id
    buffers_p1 = [[] for _ in range(num_envs)]  # player 1 (agent 0)
    buffers_p2 = [[] for _ in range(num_envs)]  # player 2 (agent 1)
    env_step_counters = np.zeros(num_envs, dtype=np.int64)  # steps since last reset in each env
    env_had_termination = np.zeros(num_envs, dtype=bool)    # flag: early-term happened

    saved_sequences = []
    total_valid_episodes = 0
    target_episodes = args.num_rollout_episodes

    is_deterministic = args.deterministic

    # ---------- initial reset ----------
    obs_dict = player.env_reset()
    if player.is_rnn:
        player.init_rnn()

    print(f"\nStarting rollout … target {target_episodes} valid episodes "
          f"(min_seq_len={args.min_seq_len})")
    t_start = time.time()

    step = 0
    done_indices = []

    while total_valid_episodes < target_episodes:
        # reset envs that were done on previous step
        obs_dict = player.env_reset(done_indices)

        # get action from policy (handles both agents internally)
        with torch.no_grad():
            action = player.get_action(obs_dict, is_deterministic)

        # step environment
        obs_dict, r, done, info = player.env_step(player.env, action)
        step += 1

        # ---- record SMPL state for every env ----
        frame_p1 = extract_smpl_frame(task, 0)
        frame_p2 = extract_smpl_frame(task, 1)

        for eid in range(num_envs):
            buffers_p1[eid].append({k: v[eid] for k, v in frame_p1.items()})
            buffers_p2[eid].append({k: v[eid] for k, v in frame_p2.items()})
            env_step_counters[eid] += 1

        # ---- check which envs are done ----
        # done is shaped [num_envs * num_agents] in multi-agent, but we use
        # the base done (same for all agents in the same env).
        # _terminate_buf tells us if the episode ended due to early termination
        # (fall), vs natural timeout.
        terminate_buf = task._terminate_buf.cpu().numpy()  # (num_envs,)
        done_np = done.cpu().numpy()
        # done is repeated for each agent; take every num_agents-th entry
        done_base = done_np[::num_agents] if done_np.shape[0] > num_envs else done_np

        # mark any env that had early termination this step
        for eid in range(num_envs):
            if terminate_buf[eid] > 0:
                env_had_termination[eid] = True

        # process done envs
        all_done_indices = done.clone().nonzero(as_tuple=False)
        done_indices_for_reset = all_done_indices[::num_agents]
        done_env_ids = done_indices_for_reset[:, 0].cpu().numpy() if len(done_indices_for_reset) > 0 else []

        for eid in done_env_ids:
            seq_len = env_step_counters[eid]

            # Only keep the sequence if:
            #   1) No early termination happened (clean continuous motion)
            #   2) Sequence length >= min_seq_len
            if (not env_had_termination[eid]) and (seq_len >= args.min_seq_len):
                # stack frames into arrays
                seq_p1 = {k: np.stack([f[k] for f in buffers_p1[eid]], axis=0) for k in buffers_p1[eid][0]}
                seq_p2 = {k: np.stack([f[k] for f in buffers_p2[eid]], axis=0) for k in buffers_p2[eid][0]}
                saved_sequences.append({
                    'player1': seq_p1,
                    'player2': seq_p2,
                    'seq_len': int(seq_len),
                    'dt': float(dt),
                    'fps': float(1.0 / dt),
                })
                total_valid_episodes += 1

                if total_valid_episodes % 50 == 0 or total_valid_episodes == target_episodes:
                    elapsed = time.time() - t_start
                    print(f"  collected {total_valid_episodes}/{target_episodes} valid episodes  "
                          f"({elapsed:.1f}s elapsed)")

            # clear buffer for this env regardless
            buffers_p1[eid] = []
            buffers_p2[eid] = []
            env_step_counters[eid] = 0
            env_had_termination[eid] = False

            if total_valid_episodes >= target_episodes:
                break

        # prepare done_indices for env_reset on next iteration
        done_indices = done_indices_for_reset[:, 0] if len(done_indices_for_reset) > 0 else []

    elapsed = time.time() - t_start
    total_frames = sum(s['seq_len'] for s in saved_sequences)
    total_seconds = total_frames * dt
    print(f"\nRollout complete.")
    print(f"  Total valid episodes : {len(saved_sequences)}")
    print(f"  Total paired frames  : {total_frames}")
    print(f"  Total motion duration: {total_seconds:.1f}s = {total_seconds/3600:.4f} hours")
    print(f"  Wall-clock time      : {elapsed:.1f}s")

    return saved_sequences


# ========================== hydra entry =====================================

# We register our own env creator, identical to run_hydra.py
vecenv.register('RLGPU',
    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'env_creator': lambda **kwargs: create_rlgpu_env(**kwargs),
    'vecenv_type': 'RLGPU',
})


def build_alg_runner(algo_observer):
    """Same as run_hydra.py – register all builders."""
    runner = Runner(algo_observer)
    runner.player_factory.register_builder(
        'amp_discrete', lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs))
    runner.algo_factory.register_builder(
        'amp', lambda **kwargs: amp_agent.AMPAgent(**kwargs))
    runner.algo_factory.register_builder(
        'amp_self_play', lambda **kwargs: amp_self_play_agent.AMPSelfPlayAgent(**kwargs))
    runner.player_factory.register_builder(
        'amp', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))
    runner.player_factory.register_builder(
        'amp_self_play', lambda **kwargs: amp_self_play_players.AMPSelfPlayPlayerContinuous(**kwargs))
    runner.model_builder.model_factory.register_builder(
        'amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder(
        'amp', lambda **kwargs: amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder(
        'amp_mcp', lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder())
    runner.model_builder.network_factory.register_builder(
        'amp_pnn', lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder())
    runner.model_builder.network_factory.register_builder(
        'amp_z', lambda **kwargs: amp_network_z_builder.AMPZBuilder())
    runner.algo_factory.register_builder(
        'im_amp', lambda **kwargs: im_amp.IMAmpAgent(**kwargs))
    runner.player_factory.register_builder(
        'im_amp', lambda **kwargs: im_amp_players.IMAMPPlayerContinuous(**kwargs))
    return runner


@hydra.main(
    version_base=None,
    config_path="../phc/data/cfg",
    config_name="config",
)
def main(cfg_hydra: DictConfig) -> None:
    global cfg, cfg_train

    # ---- parse the extra CLI args we added (hydra passes them through) ----
    # We sniff them out of sys.argv before hydra consumes them; alternatively,
    # they can be injected via the config.  Here we use hydra overrides for the
    # standard config and our own argparse for rollout-specific flags.

    cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))

    set_np_formatting()

    # Force test / inference mode
    flags.test = True
    cfg.test = True
    cfg.train = False
    cfg.no_log = True
    flags.debug = cfg.debug
    flags.follow = cfg.follow
    flags.fixed = False
    flags.divide_group = False
    flags.no_collision_check = False
    flags.fixed_path = False
    flags.real_path = False
    flags.show_traj = True
    flags.server_mode = cfg.server_mode
    flags.slow = False
    flags.real_traj = False
    flags.im_eval = cfg.im_eval
    flags.no_virtual_display = cfg.no_virtual_display
    flags.render_o3d = cfg.render_o3d
    flags.add_proj = cfg.add_proj
    flags.has_eval = cfg.has_eval
    flags.trigger_input = False

    set_seed(cfg.get("seed", -1), cfg.get("torch_deterministic", False))

    cfg_train = cfg.learning
    cfg_train['params']['config']['network_path'] = cfg.output_path
    cfg_train['params']['config']['train_dir'] = cfg.output_path
    cfg_train["params"]["config"]["num_actors"] = cfg.env.num_envs

    # Load checkpoint
    checkpoint_path = cfg.get("rollout_checkpoint", None)
    if checkpoint_path is None:
        # Fallback: use epoch=-1 convention (latest)
        path = os.path.join(cfg.output_path,
                            cfg_train["params"]["config"]['name'] + '.pth')
        if os.path.exists(path):
            checkpoint_path = path
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {path}. "
                "Specify rollout_checkpoint=<path> on the command line.")

    cfg_train["params"]["load_checkpoint"] = True
    cfg_train["params"]["load_path"] = checkpoint_path
    print(f"Loading checkpoint: {checkpoint_path}")

    # Build runner & create player via rl_games Runner
    algo_observer = DummyAlgoObserver()
    runner = build_alg_runner(algo_observer)
    runner.load(cfg_train)
    runner.reset()

    # runner.run() internally calls run_play when test=True.
    # We replicate its logic to get the player object:
    player = runner.create_player()
    player.restore(checkpoint_path)
    print(f"Player created. Model on device: {player.device}")

    # ---- Rollout-specific args from hydra overrides ----
    rollout_args = argparse.Namespace(
        num_rollout_episodes=cfg.get("rollout_num_episodes", 200),
        min_seq_len=cfg.get("rollout_min_seq_len", 60),
        deterministic=cfg.get("rollout_deterministic", True),
        output_dir=cfg.get("rollout_output_dir", "rollout_data/boxing"),
        save_prefix=cfg.get("rollout_save_prefix", "boxing_pairs"),
    )

    # ---- Run rollout ----
    sequences = rollout(player, rollout_args)

    # ---- Save ----
    os.makedirs(rollout_args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(
        rollout_args.output_dir,
        f"{rollout_args.save_prefix}_{timestamp}.pkl"
    )

    output = {
        'metadata': {
            'num_sequences': len(sequences),
            'total_frames': sum(s['seq_len'] for s in sequences),
            'dt': float(sequences[0]['dt']) if sequences else 0,
            'fps': float(sequences[0]['fps']) if sequences else 0,
            'checkpoint': checkpoint_path,
            'num_envs': cfg.env.num_envs,
            'episode_length': cfg.env.episode_length,
            'min_seq_len': rollout_args.min_seq_len,
            'deterministic': rollout_args.deterministic,
        },
        'sequences': sequences,
        # Description of data layout:
        # sequences[i]['player1'] and sequences[i]['player2'] each contain:
        #   'root_pos'     : (T, 3)       root position
        #   'root_rot'     : (T, 4)       root quaternion (w, x, y, z)
        #   'root_vel'     : (T, 3)       root linear velocity
        #   'root_ang_vel' : (T, 3)       root angular velocity
        #   'dof_pos'      : (T, 69)      joint angles
        #   'dof_vel'      : (T, 69)      joint angular velocities
        #   'body_pos'     : (T, 24, 3)   per-body world positions
        #   'body_rot'     : (T, 24, 4)   per-body world rotations (quat)
        #   'body_vel'     : (T, 24, 3)   per-body linear velocities
        #   'body_ang_vel' : (T, 24, 3)   per-body angular velocities
    }

    with open(save_path, 'wb') as f:
        pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_seconds = output['metadata']['total_frames'] * output['metadata']['dt']
    print(f"\nSaved {len(sequences)} paired sequences to:\n  {save_path}")
    print(f"  Total motion duration: {total_seconds:.1f}s = {total_seconds/3600:.4f} hours")
    print(f"  File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    main()
