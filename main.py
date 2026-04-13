"""
main.py — 完整复现论文训练流程

修改对照:
──────────────────────────────────────────────────────────────
[1]  新增模型: MapEncoder, MapTransitionModel, ObstacleForecaster,
     ContinuationModel, DifferentiableMapUpdater
[2]  TransitionModel 接收 map_embeddings (公式 14)
[3]  Actor/Value/Reward 接收 map_embedding (公式 22, 29, 30)
[4]  ObservationModel 接收 map_embedding (公式 21)
[5]  完整损失函数 L_WM (公式 35-43, 9 项)
[6]  imagine_ahead 传播 map_embedding (公式 17-20)
[7]  lambda_return 使用 continuation (公式 31-32)
[8]  observation_loss 权重修正 (从 0.001 → λ_img)
──────────────────────────────────────────────────────────────
"""

import argparse
import os
import gymnasium as gym
import numpy as np
import torch
try:
    from torch.amp import autocast as _autocast_new, GradScaler
    def autocast(enabled=True):
        return _autocast_new('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from env import CONTROL_SUITE_ENVS, GYM_ENVS, Env, EnvBatcher
from memory import ExperienceReplay
from models import (
    ActorModel, Encoder, ObservationModel, RewardModel, TransitionModel,
    ValueModel, bottle, UAVHybridEncoder, SemanticFeatureExtractor,
    MapEncoder, MapTransitionModel, ObstacleForecaster, ContinuationModel,
    DifferentiableMapUpdater,
)
from utils import FreezeParameters, imagine_ahead, lambda_return, lineplot, write_video

# ============================================================================
#  Hyperparameters
# ============================================================================
parser = argparse.ArgumentParser(description='Semantic World Model for UAV Navigation')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument(
    '--env', type=str, default='UAV-v0',
    choices=GYM_ENVS + CONTROL_SUITE_ENVS + ['UAV-v0'],
    help='Environment',
)
parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T')
parser.add_argument('--experience-size', type=int, default=500000, metavar='D',
                    help='Experience replay size')
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F))
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F))
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E')
parser.add_argument('--hidden-size', type=int, default=200, metavar='H')
parser.add_argument('--belief-size', type=int, default=200, metavar='H')
parser.add_argument('--state-size', type=int, default=30, metavar='Z')
parser.add_argument('--semantic-size', type=int, default=512, metavar='S', help='Semantic feature size (gθ output)')
parser.add_argument('--map-embedding-size', type=int, default=256, metavar='M', help='Map embedding size (E_m output)')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R')
parser.add_argument('--action-noise', type=float, default=0.15, metavar='ε')
parser.add_argument('--episodes', type=int, default=1000, metavar='E')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C')
parser.add_argument('--batch-size', type=int, default=16, metavar='B',
                    help='Batch size per gradient step')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L',
                    help='Chunk length')
parser.add_argument('--grad-accumulate', type=int, default=1, metavar='GA',
                    help='Gradient accumulation steps (effective batch = batch-size * grad-accumulate)')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg')
parser.add_argument('--free-nats', type=float, default=3, metavar='F')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B')
parser.add_argument('--model-learning-rate', type=float, default=5e-4, metavar='α')
parser.add_argument('--actor-learning-rate', type=float, default=8e-5, metavar='α')
parser.add_argument('--value-learning-rate', type=float, default=8e-5, metavar='α')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS')
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε')
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C')
parser.add_argument('--planning-horizon', type=int, default=10, metavar='H',
                    help='Planning horizon for imagination (default 10 for 6GB GPU)')
parser.add_argument('--discount', type=float, default=0.99, metavar='H')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=25, metavar='I')
parser.add_argument('--test-episodes', type=int, default=10, metavar='E')
parser.add_argument('--checkpoint-interval', type=int, default=25, metavar='I')
parser.add_argument('--checkpoint-experience', action='store_true')
parser.add_argument('--models', type=str, default='', metavar='M')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER')
parser.add_argument('--render', action='store_true')
parser.add_argument('--semantic-loss-scale', type=float, default=10.0)
parser.add_argument('--contrastive-margin', type=float, default=10.0)
parser.add_argument('--similarity-dist-thresh', type=float, default=300.0)
# [论文公式 39] KL 损失权重
parser.add_argument('--dyn-scale', type=float, default=0.5, help='Weight for dynamics loss (公式39 L_dyn)')
parser.add_argument('--rep-scale', type=float, default=0.1, help='Weight for representation loss (公式39 L_rep)')
# [论文公式 35] 完整损失权重
parser.add_argument('--lambda-img', type=float, default=1.0, help='Weight for image loss (公式36)')
parser.add_argument('--lambda-r', type=float, default=1.0, help='Weight for reward loss (公式37)')
parser.add_argument('--lambda-c', type=float, default=1.0, help='Weight for continuation loss (公式38)')
parser.add_argument('--lambda-map', type=float, default=10.0, help='Weight for map prediction loss (公式40)')
parser.add_argument('--lambda-occ', type=float, default=5.0, help='Weight for occupancy loss (公式41)')
parser.add_argument('--lambda-flow', type=float, default=2.0, help='Weight for motion flow loss (公式42)')
parser.add_argument('--lambda-trk', type=float, default=1.0, help='Weight for trajectory loss (公式43)')
parser.add_argument('--forecast-horizon', type=int, default=5, help='Obstacle forecasting K steps')
parser.add_argument('--encode-batch', type=int, default=100, metavar='EB',
                    help='Mini-batch size for encoder forward')
parser.add_argument('--eval-steps', type=int, default=1000, metavar='ES')
args = parser.parse_args()

args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)

print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))

# ============================================================================
#  Setup
# ============================================================================
results_dir = os.path.join('results', '{}_{}'.format(args.env, args.id))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
    print("Using CUDA")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)

    # --- CUDA/cuDNN 诊断 ---
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}, PyTorch: {torch.__version__}")
    _cc = torch.cuda.get_device_capability(0)
    print(f"  Compute capability: {_cc[0]}.{_cc[1]}")

    # cuDNN 设置: benchmark=False 节省显存, enabled=True 使用 cuDNN 加速
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
else:
    print("Using CPU")
    args.device = torch.device('cpu')

metrics = {
    'steps': [], 'episodes': [], 'train_rewards': [],
    'test_episodes': [], 'test_rewards': [], 'test_avg_rewards': [],
    'observation_loss': [], 'reward_loss': [], 'kl_loss': [],
    'continuation_loss': [], 'map_loss': [], 'occ_loss': [], 'flow_loss': [],
    'actor_loss': [], 'value_loss': [],
}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))
print("Writer is ready.")

# Initialise environment
env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
if env is None:
    raise ValueError(f"Environment '{args.env}' not found.")
print("Environment is loaded.")

# Experience Replay
if args.experience_replay != '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay, weights_only=False)  # ExperienceReplay 自定义对象, 需要 pickle
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
elif not args.test:
    D = ExperienceReplay(
        args.experience_size, args.symbolic_env, env.observation_size,
        env.action_size, args.bit_depth, args.device
    )
    for s in range(1, args.seed_episodes + 1):
        observation, done, t = env.reset(), False, 0
        while not done:
            action = env.sample_random_action()
            next_observation, reward, done = env.step(action)
            D.append(observation, action, reward, done)
            observation = next_observation
            t += 1
        metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
        metrics['episodes'].append(s)
print("Experience replay buffer is ready.")

#  Model Initialization
# --- TransitionModel (公式 14-18) ---
transition_model = TransitionModel(
    belief_size=args.belief_size,
    state_size=args.state_size,
    action_size=env.action_size,
    hidden_size=args.hidden_size,
    embedding_size=args.embedding_size,
    semantic_size=args.semantic_size,
    semantic_state_size=args.semantic_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)

# --- ObservationModel (公式 21) ---
observation_model = ObservationModel(
    args.symbolic_env, env.observation_size,
    args.belief_size, args.state_size, args.embedding_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.cnn_activation_function,
).to(device=args.device)

# --- RewardModel (公式 22) ---
reward_model = RewardModel(
    args.belief_size, args.state_size, args.hidden_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)

# --- ContinuationModel (公式 22, 38) ---
continuation_model = ContinuationModel(
    args.belief_size, args.state_size, args.map_embedding_size, args.hidden_size,
).to(device=args.device)

# --- MapEncoder (公式 12) ---
map_encoder = MapEncoder(
    n_channels=6, grid_size=30, map_embedding_size=args.map_embedding_size,
).to(device=args.device)

# --- MapTransitionModel (公式 19, 23) ---
map_transition_model = MapTransitionModel(
    belief_size=args.belief_size, state_size=args.state_size,
    action_size=env.action_size, map_embedding_size=args.map_embedding_size,
    n_channels=6, grid_size=30,
).to(device=args.device)

# 为 imagine_ahead 添加 map_embedding 投影层
map_transition_model.imagine_proj = nn.Linear(512, args.map_embedding_size).to(device=args.device)

# --- ObstacleForecaster (公式 24-25) ---
obstacle_forecaster = ObstacleForecaster(
    belief_size=args.belief_size, state_size=args.state_size,
    map_embedding_size=args.map_embedding_size,
    forecast_horizon=args.forecast_horizon, grid_size=30,
).to(device=args.device)

# --- DifferentiableMapUpdater (公式 11) ---
map_updater = DifferentiableMapUpdater(
    embedding_size=args.embedding_size, n_channels=6, grid_size=30,
).to(device=args.device)

# --- Encoder ---
if args.env == 'UAV-v0':
    print("Using UAVHybridEncoder.")
    encoder = UAVHybridEncoder(args.embedding_size, args.cnn_activation_function).to(device=args.device)
    semantic_extractor = SemanticFeatureExtractor(
        semantic_size=args.semantic_size,
        activation_function=args.cnn_activation_function
    ).to(device=args.device)
    print("Using independent SemanticFeatureExtractor (gθ).")
else:
    encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size,
                      args.cnn_activation_function).to(device=args.device)
    semantic_extractor = None

# --- ActorModel (公式 29) ---
actor_model = ActorModel(
    args.belief_size, args.state_size, args.hidden_size, env.action_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)

# --- ValueModel (公式 30) ---
value_model = ValueModel(
    args.belief_size, args.state_size, args.hidden_size,
    map_embedding_size=args.map_embedding_size,
    activation_function=args.dense_activation_function,
).to(device=args.device)

# ============================================================================
#  Optimizers
# ============================================================================

# 世界模型参数 (transition + observation + reward + continuation + encoder
#              + semantic_extractor + map_encoder + map_transition + map_updater
#              + obstacle_forecaster)
world_model_modules = [
    transition_model, observation_model, reward_model, continuation_model,
    encoder, map_encoder, map_transition_model, map_updater, obstacle_forecaster,
]
if semantic_extractor is not None:
    world_model_modules.append(semantic_extractor)

param_list = []
for m in world_model_modules:
    param_list.extend(list(m.parameters()))
# 注: imagine_proj 是 map_transition_model 的动态子模块,
# 已被 map_transition_model.parameters() 自动收集, 不需要额外添加.

model_optimizer = optim.Adam(
    param_list,
    lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate,
    eps=args.adam_epsilon,
)
actor_optimizer = optim.Adam(
    actor_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate,
    eps=args.adam_epsilon,
)
value_optimizer = optim.Adam(
    value_model.parameters(),
    lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate,
    eps=args.adam_epsilon,
)

# Load pretrained models
if args.models != '' and os.path.exists(args.models):
    print("loading pre-trained models")
    model_dicts = torch.load(args.models, weights_only=True)
    transition_model.load_state_dict(model_dicts['transition_model'])
    observation_model.load_state_dict(model_dicts['observation_model'])
    reward_model.load_state_dict(model_dicts['reward_model'])
    encoder.load_state_dict(model_dicts['encoder'])
    actor_model.load_state_dict(model_dicts['actor_model'])
    value_model.load_state_dict(model_dicts['value_model'])
    model_optimizer.load_state_dict(model_dicts['model_optimizer'])
    if 'continuation_model' in model_dicts:
        continuation_model.load_state_dict(model_dicts['continuation_model'])
    if 'map_encoder' in model_dicts:
        map_encoder.load_state_dict(model_dicts['map_encoder'])
    if 'map_transition_model' in model_dicts:
        map_transition_model.load_state_dict(model_dicts['map_transition_model'])
    if 'obstacle_forecaster' in model_dicts:
        obstacle_forecaster.load_state_dict(model_dicts['obstacle_forecaster'])
    if 'map_updater' in model_dicts:
        map_updater.load_state_dict(model_dicts['map_updater'])
    if semantic_extractor is not None and 'semantic_extractor' in model_dicts:
        semantic_extractor.load_state_dict(model_dicts['semantic_extractor'])

planner = actor_model
global_prior = Normal(
    torch.zeros(args.batch_size, args.state_size, device=args.device),
    torch.ones(args.batch_size, args.state_size, device=args.device),
)
free_nats = torch.full((1,), args.free_nats, device=args.device)

# [NaN 修复] 完全禁用 AMP — FP32 全程运行
# AMP 的 GradScaler 反复导致 encoder/TransitionModel 权重 NaN 爆炸.
# 6GB GPU + batch_size=4 + chunk_size=15 在 FP32 下可以跑.
# 稳定性比速度更重要.
use_amp = False
print("AMP: OFF (forced FP32 for stability)")

model_scaler = GradScaler(enabled=False)
actor_scaler = GradScaler(enabled=False)
value_scaler = GradScaler(enabled=False)
print("Semantic World Model is ready.")


# ============================================================================
#  Collect component_modules for FreezeParameters
# ============================================================================

def get_all_model_modules():
    """收集所有世界模型的 component_modules 用于 FreezeParameters."""
    modules = []
    for m in world_model_modules:
        if hasattr(m, 'component_modules'):
            modules.extend(m.component_modules)
        else:
            modules.append(m)
    # imagine_proj 是动态挂载到 map_transition_model 的子模块,
    # 不在其 component_modules 列表中, 需要单独加入以确保 FreezeParameters 能冻结它.
    if hasattr(map_transition_model, 'imagine_proj'):
        modules.append(map_transition_model.imagine_proj)
    return modules


# ============================================================================
#  update_belief_and_act
# ============================================================================

def update_belief_and_act(
    args, env, planner, transition_model, encoder,
    belief, posterior_state, semantic_state, action, observation,
    map_encoder, current_map_embedding=None,
    semantic_extractor=None, explore=False
):
    """
    一步: 编码观测 → 更新 RSSM → 选动作 → 执行.

    修改:
    - 传入 map_encoder 和 current_map_embedding
    - Actor 接收 map_embedding
    """
    if isinstance(observation, dict):
        obs_img = observation['image'].to(args.device)
        obs_tgt_img = observation['target'].to(args.device)
        pos = observation['position'].to(args.device)
        if pos.dim() == 1:
            pos = pos.unsqueeze(0)
        obs_vec = pos

        # Encoder: e_t = E_φ(I_t, I^g, p_t)
        full_output = encoder(obs_img, obs_tgt_img, obs_vec)
        embed = full_output[0] if isinstance(full_output, tuple) else full_output

        # gθ: 语义特征
        if semantic_extractor is not None:
            obs_sem_map = observation['semantic_map'].to(args.device)
            if obs_sem_map.dim() == 3:
                obs_sem_map = obs_sem_map.unsqueeze(0)
            sem_feat = semantic_extractor(obs_img, obs_tgt_img, obs_vec, obs_sem_map)
        else:
            sem_feat = None

        # MapEncoder: m_t = E_m(M_t)
        if map_encoder is not None and 'semantic_map' in observation:
            obs_sem_map = observation['semantic_map'].to(args.device)
            if obs_sem_map.dim() == 3:
                obs_sem_map = obs_sem_map.unsqueeze(0)
            current_map_embedding = map_encoder(obs_sem_map)
        elif current_map_embedding is None:
            current_map_embedding = torch.zeros(1, args.map_embedding_size, device=args.device)
    else:
        obs_input = observation.to(args.device)
        full_output = encoder(obs_input)
        embed = full_output[0] if isinstance(full_output, tuple) else full_output
        sem_feat = None
        if current_map_embedding is None:
            current_map_embedding = torch.zeros(1, args.map_embedding_size, device=args.device)

    # TransitionModel
    belief, _, _, _, posterior_state, _, _, next_semantic_state = transition_model(
        posterior_state,
        action.unsqueeze(dim=0),
        belief,
        semantic_state,
        embed.unsqueeze(dim=0),
        semantic_features=sem_feat.unsqueeze(dim=0) if sem_feat is not None else None,
        map_embeddings=current_map_embedding.unsqueeze(dim=0),
    )

    belief = belief.squeeze(dim=0)
    posterior_state = posterior_state.squeeze(dim=0)
    next_semantic_state = next_semantic_state.squeeze(dim=0)

    # Actor: π_η(a | h, z, m)
    action = planner.get_action(belief, posterior_state, current_map_embedding, det=not explore)

    if explore:
        B = action.shape[0]
        action_size = action.shape[-1]
        rand_mask = (torch.rand(B, device=action.device) < args.action_noise)
        if rand_mask.any():
            rand_idx = torch.randint(0, action_size, (B,), device=action.device)
            rand_onehot = F.one_hot(rand_idx, num_classes=action_size).float()
            action = torch.where(rand_mask.unsqueeze(-1), rand_onehot, action)

    next_observation, reward, done = env.step(
        action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu()
    )

    return belief, posterior_state, next_semantic_state, current_map_embedding, action, next_observation, reward, done


# ============================================================================
#  Training Loop
# ============================================================================
for episode in tqdm(
    range(metrics['episodes'][-1] + 1, args.episodes + 1),
    total=args.episodes, initial=metrics['episodes'][-1] + 1
):
    model_modules = get_all_model_modules()
    print(f"Training loop EP:{episode}")

    losses = []
    for s in tqdm(range(args.collect_interval)):
        observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)

        # ===============================================================
        #  准备数据 — [OOM 修复] 最小化 GPU 常驻张量
        # ===============================================================
        _dev = args.device

        actions = actions.to(_dev)
        rewards = rewards.to(_dev)
        nonterminals = nonterminals.to(_dev)

        if isinstance(observations, dict):
            obs_img = observations['image']              # (L, B, 3, 64, 64) CPU — 保持 CPU!
            obs_tgt_img = observations['target']         # CPU
            obs_pos = observations['position']               # CPU — 延迟搬运
            obs_vec = obs_pos
            obs_sem_map = observations['semantic_map']      # CPU — 延迟搬运!

            if 'target_position' in observations:
                obs_tgt_pos = observations['target_position']  # CPU
            else:
                obs_tgt_pos = None

            T, B = obs_img.shape[:2]
            TB = T * B
            ENCODE_BATCH = min(TB, args.encode_batch)

            flat_img_cpu = obs_img.reshape(TB, *obs_img.shape[2:])
            flat_tgt_cpu = obs_tgt_img.reshape(TB, *obs_tgt_img.shape[2:])
            flat_vec_cpu = obs_vec.reshape(TB, -1)
            flat_sem_map_cpu = obs_sem_map.reshape(TB, *obs_sem_map.shape[2:])

            # === Encoder (mini-batch) → 只保留 embed, 对比特征立即 detach ===
            embed_chunks = []
            curr_feat_chunks = []
            targ_feat_chunks = []
            for i in range(0, TB, ENCODE_BATCH):
                j = min(i + ENCODE_BATCH, TB)
                _img = flat_img_cpu[i:j].contiguous().float().to(_dev)
                _tgt = flat_tgt_cpu[i:j].contiguous().float().to(_dev)
                _vec = flat_vec_cpu[i:j].contiguous().float().to(_dev)
                with autocast(enabled=use_amp):
                    out = encoder(_img, _tgt, _vec)
                if isinstance(out, tuple):
                    embed_chunks.append(out[0])
                    curr_feat_chunks.append(out[1].detach())
                    targ_feat_chunks.append(out[2].detach())
                else:
                    embed_chunks.append(out)
                del _img, _tgt, _vec, out
            torch.cuda.empty_cache()

            flat_embed = torch.cat(embed_chunks, dim=0)
            del embed_chunks

            if curr_feat_chunks:
                flat_curr_feat = torch.cat(curr_feat_chunks, dim=0)  # CPU-like (detached)
                flat_targ_feat = torch.cat(targ_feat_chunks, dim=0)
            else:
                flat_curr_feat, flat_targ_feat = None, None
            del curr_feat_chunks, targ_feat_chunks

            full_embed = flat_embed.float().view(T, B, -1)  # 确保 FP32 for TransitionModel
            embed = full_embed[1:]

            # === gθ (mini-batch) ===
            sem_feat_chunks = []
            for i in range(0, TB, ENCODE_BATCH):
                j = min(i + ENCODE_BATCH, TB)
                _img = flat_img_cpu[i:j].contiguous().float().to(_dev)
                _tgt = flat_tgt_cpu[i:j].contiguous().float().to(_dev)
                _vec = flat_vec_cpu[i:j].contiguous().float().to(_dev)
                _sem = flat_sem_map_cpu[i:j].contiguous().float().to(_dev)
                with autocast(enabled=use_amp):
                    sem_feat_chunks.append(semantic_extractor(_img, _tgt, _vec, _sem))
                del _img, _tgt, _vec, _sem
            torch.cuda.empty_cache()

            flat_sem_feat = torch.cat(sem_feat_chunks, dim=0).float()  # FP32 for TransitionModel
            del sem_feat_chunks
            full_sem_feat = flat_sem_feat.view(T, B, -1)
            del flat_sem_feat
            sem_feat_for_tm = full_sem_feat[1:]

            # === MapEncoder ===
            # 分批处理 MapEncoder 也有帮助
            map_emb_chunks = []
            for i in range(0, TB, ENCODE_BATCH):
                j = min(i + ENCODE_BATCH, TB)
                _sem = flat_sem_map_cpu[i:j].contiguous().float().to(_dev)
                with autocast(enabled=use_amp):
                    map_emb_chunks.append(map_encoder(_sem))
                del _sem
            torch.cuda.empty_cache()
            flat_map_emb = torch.cat(map_emb_chunks, dim=0).float()  # FP32 for TransitionModel
            del map_emb_chunks
            full_map_emb = flat_map_emb.view(T, B, -1)
            del flat_map_emb
            map_emb_for_tm = full_map_emb[:-1]
            map_emb_for_loss = full_map_emb[1:]

            # DifferentiableMapUpdater 输入 — 在 CPU, loss 阶段再按块搬
            updater_embed = full_embed[1:]  # GPU (encoder output)
            source_maps_cpu = obs_sem_map[:-1]   # CPU — M_{t-1}
            updater_pos_cpu = obs_pos[1:]        # CPU — p_t (当前步位置, 与 e_t / M_t 对齐)

            # 对比损失数据
            if flat_curr_feat is not None:
                curr_feat = flat_curr_feat.view(T, B, -1)
                targ_feat = flat_targ_feat.view(T, B, -1)
                loss_curr_feat = curr_feat[1:]
                loss_targ_feat = targ_feat[1:]
                loss_pos = obs_pos[1:].to(_dev) if obs_tgt_pos is not None else None
                loss_tgt_pos = obs_tgt_pos[1:].to(_dev) if obs_tgt_pos is not None else None
            else:
                loss_curr_feat = None

            target_obs_cpu = obs_img[1:]       # CPU
            target_maps_cpu = obs_sem_map[1:]  # CPU — 不再是 GPU!

            del flat_img_cpu, flat_tgt_cpu, obs_img, obs_tgt_img
            del flat_vec_cpu, flat_sem_map_cpu

        else:
            obs_tensor = observations.to(_dev)
            embed = bottle(encoder, (obs_tensor[1:],))
            target_obs_cpu = obs_tensor[1:]  # 已在 GPU
            loss_curr_feat = None
            sem_feat_for_tm = None
            map_emb_for_tm = None
            map_emb_for_loss = None
            target_maps_cpu = None
            source_maps_cpu = None
            updater_embed = None
            updater_pos_cpu = None
            full_map_emb = None

        # ===============================================================
        #  TransitionModel Forward
        # ===============================================================
        init_belief = torch.zeros(args.batch_size, args.belief_size, device=args.device)
        init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
        init_semantic = torch.zeros(args.batch_size, args.semantic_size, device=args.device)

        # [重要] TransitionModel 不使用 autocast!
        # GRU 在 FP16 下数值不稳定: 多步展开后 hidden state 溢出 ±65504 范围 → NaN.
        (
            beliefs, prior_states, prior_means, prior_std_devs,
            posterior_states, posterior_means, posterior_std_devs,
            semantic_states,
        ) = transition_model(
            init_state, actions[:-1], init_belief, init_semantic, embed, nonterminals[:-1],
            semantic_features=sem_feat_for_tm,
            map_embeddings=map_emb_for_tm,
        )

        # [NaN 安全] 检测 TransitionModel 输出是否包含 NaN
        if torch.isnan(beliefs).any() or torch.isnan(posterior_states).any():
            print(f"  [WARN] NaN detected in TransitionModel output at step {s}, skipping this batch.")
            # 回退: 清零梯度, 跳过此 batch
            model_optimizer.zero_grad()
            losses.append([0.0] * 10)
            torch.cuda.empty_cache()
            continue

        # ===============================================================
        #  [公式 35] 完整损失 L_WM
        # ===============================================================

        # 准备 flattened tensors for loss computation
        T_loss, B_loss = beliefs.shape[:2]
        flat_beliefs = beliefs.view(T_loss * B_loss, -1)
        flat_post_states = posterior_states.view(T_loss * B_loss, -1)

        # [关键] decoder/reward/continuation 需要 m_t (与 beliefs 对齐),
        # 而 TransitionModel 输入用 m_{t-1}. 二者不同！
        if map_emb_for_loss is not None:
            flat_map_emb_loss = map_emb_for_loss.view(T_loss * B_loss, -1)
        elif map_emb_for_tm is not None:
            # fallback for non-UAV (不应触发)
            flat_map_emb_loss = map_emb_for_tm.view(T_loss * B_loss, -1)
        else:
            flat_map_emb_loss = None

        # --- [公式 36] L_img: 图像重建损失 ---
        # 使用 MSE (等价于 Normal(mu,1) 的 -log_prob 去掉常数项, 且用 mean reduction)
        if flat_map_emb_loss is not None:
            obs_loss_accum = torch.tensor(0.0, device=_dev)
            obs_count = 0
            OBS_CHUNK = min(T_loss * B_loss, args.encode_batch)
            for oi in range(0, T_loss * B_loss, OBS_CHUNK):
                oj = min(oi + OBS_CHUNK, T_loss * B_loss)
                with autocast(enabled=use_amp):
                    _obs_mean = observation_model(
                        flat_beliefs[oi:oj], flat_post_states[oi:oj], flat_map_emb_loss[oi:oj]
                    )
                _target = target_obs_cpu.reshape(T_loss * B_loss, *target_obs_cpu.shape[2:])[oi:oj]
                _target = _target.contiguous().float().to(_dev)
                _obs_mean_f = _obs_mean.float()
                obs_loss_accum = obs_loss_accum + F.mse_loss(_obs_mean_f, _target, reduction='sum')
                obs_count += _obs_mean_f.numel()
                del _obs_mean, _obs_mean_f, _target
            observation_loss = obs_loss_accum / max(obs_count, 1)
            del obs_loss_accum
        else:
            with autocast(enabled=use_amp):
                obs_mean = bottle(observation_model, (beliefs, posterior_states))
            if isinstance(obs_mean, tuple):
                obs_mean = obs_mean[0]
            _target_gpu = target_obs_cpu.to(_dev) if not target_obs_cpu.is_cuda else target_obs_cpu
            observation_loss = F.mse_loss(obs_mean.float(), _target_gpu, reduction='mean')
            del obs_mean, _target_gpu
        torch.cuda.empty_cache()

        # --- [公式 37] L_r: 奖励回归损失 ---
        with autocast(enabled=use_amp):
            if flat_map_emb_loss is not None:
                reward_pred = reward_model(flat_beliefs, flat_post_states, flat_map_emb_loss)
                reward_pred = reward_pred.view(T_loss, B_loss)
            else:
                _flat_b = beliefs.view(T_loss * B_loss, -1)
                _flat_s = posterior_states.view(T_loss * B_loss, -1)
                reward_pred = reward_model(_flat_b, _flat_s).view(T_loss, B_loss)
        reward_loss = F.mse_loss(reward_pred.float(), rewards[:-1], reduction='mean')

        # --- [公式 38] L_c: Continuation 损失 (BCE) ---
        with autocast(enabled=use_amp):
            if flat_map_emb_loss is not None:
                cont_pred = continuation_model(flat_beliefs, flat_post_states, flat_map_emb_loss)
                cont_pred = cont_pred.view(T_loss, B_loss)
            else:
                _flat_b = beliefs.view(T_loss * B_loss, -1)
                _flat_s = posterior_states.view(T_loss * B_loss, -1)
                cont_pred = continuation_model(_flat_b, _flat_s).view(T_loss, B_loss)
        cont_target = nonterminals[:-1].squeeze(-1)
        continuation_loss = F.binary_cross_entropy(cont_pred.float(), cont_target, reduction='mean')

        # --- [公式 39] L_dyn + L_rep: 非对称 KL 损失 ---
        dist_post = Normal(posterior_means, posterior_std_devs)
        dist_prior = Normal(prior_means, prior_std_devs)

        dist_post_detached = Normal(posterior_means.detach(), posterior_std_devs.detach())
        kl_loss_dyn = kl_divergence(dist_post_detached, dist_prior).sum(dim=2).mean(dim=(0, 1))

        dist_prior_detached = Normal(prior_means.detach(), prior_std_devs.detach())
        kl_loss_rep = kl_divergence(dist_post, dist_prior_detached).sum(dim=2).mean(dim=(0, 1))

        kl_loss = args.dyn_scale * kl_loss_dyn + args.rep_scale * kl_loss_rep

        # --- [公式 11] L_map_updater: 可微地图更新损失 ---
        if target_maps_cpu is not None:
            T_u, B_u = source_maps_cpu.shape[:2]
            N_u = T_u * B_u
            _flat_src_cpu = source_maps_cpu.reshape(N_u, *source_maps_cpu.shape[2:])
            _flat_emb = updater_embed.reshape(N_u, -1)  # 已在 GPU
            _flat_pos_cpu = updater_pos_cpu.reshape(N_u, -1)
            _flat_tgt_cpu = target_maps_cpu.reshape(N_u, *target_maps_cpu.shape[2:])

            _upd_loss = torch.tensor(0.0, device=_dev)
            _upd_count = 0
            for ci in range(0, N_u, args.encode_batch):
                cj = min(ci + args.encode_batch, N_u)
                _src = _flat_src_cpu[ci:cj].float().to(_dev)
                _pos = _flat_pos_cpu[ci:cj].float().to(_dev)
                _tgt = _flat_tgt_cpu[ci:cj].float().to(_dev)
                with autocast(enabled=use_amp):
                    _pred, _ = map_updater(_src, _flat_emb[ci:cj], _pos)
                _upd_loss = _upd_loss + F.l1_loss(_pred.float(), _tgt, reduction='sum')
                _upd_count += _tgt.numel()
                del _pred, _src, _pos, _tgt
            map_updater_loss = _upd_loss / max(_upd_count, 1)
            del _flat_src_cpu, _flat_emb, _flat_pos_cpu, _flat_tgt_cpu, _upd_loss
        else:
            map_updater_loss = torch.tensor(0.0, device=_dev)

        # --- [公式 40] L_map: MapTransitionModel 损失 ---
        # 论文公式23: D_M(h_t, z_t, a_t, m_t) → M̂_{t+1}
        # beliefs = h_1..h_{T-1}, posterior_states = z_1..z_{T-1}
        # 对齐: a_t = actions[1:] (a_1..a_{T-1}), m_t = full_map_emb[1:] (m_1..m_{T-1})
        # source = M_t = obs_sem_map[1:-1] (M_1..M_{T-2}), target = M_{t+1} = obs_sem_map[2:] (M_2..M_{T-1})
        # 注意: 因为 beliefs 有 T-1 步但 actions[1:] 只有 T-1 步, obs_sem_map[1:-1] 有 T-2 步,
        #       所以需要截断 beliefs 和 posterior_states 到前 T-2 步
        if target_maps_cpu is not None and map_emb_for_tm is not None and full_map_emb is not None:
            # 源地图 M_t (t=1..T-2), 目标 M_{t+1} (t=2..T-1)
            src_maps_cpu = obs_sem_map[1:-1]   # (T-2, B, 6, 30, 30) CPU
            tgt_maps_cpu = obs_sem_map[2:]     # (T-2, B, 6, 30, 30) CPU
            T_m, B_m = src_maps_cpu.shape[:2]
            if T_m > 0:
                N_m = T_m * B_m
                _fp_cpu = src_maps_cpu.reshape(N_m, *src_maps_cpu.shape[2:])
                _fb = beliefs[:T_m].reshape(N_m, -1)               # h_1..h_{T-2}
                _fs = posterior_states[:T_m].reshape(N_m, -1)       # z_1..z_{T-2}
                _fa = actions[1:1+T_m].reshape(N_m, -1)            # a_1..a_{T-2}
                _fm = full_map_emb[1:1+T_m].reshape(N_m, -1)       # m_1..m_{T-2}
                _ft_cpu = tgt_maps_cpu.reshape(N_m, *tgt_maps_cpu.shape[2:])

                _map_loss = torch.tensor(0.0, device=_dev)
                _map_count = 0
                for ci in range(0, N_m, args.encode_batch):
                    cj = min(ci + args.encode_batch, N_m)
                    _fp_g = _fp_cpu[ci:cj].float().to(_dev)
                    _ft_g = _ft_cpu[ci:cj].float().to(_dev)
                    with autocast(enabled=use_amp):
                        _pred = map_transition_model(_fp_g, _fb[ci:cj], _fs[ci:cj], _fa[ci:cj], _fm[ci:cj])
                    _map_loss = _map_loss + F.l1_loss(_pred.float(), _ft_g, reduction='sum')
                    _map_count += _ft_g.numel()
                    del _pred, _fp_g, _ft_g
                map_loss = _map_loss / max(_map_count, 1)
                del _fp_cpu, _fb, _fs, _fa, _fm, _ft_cpu, _map_loss
            else:
                map_loss = torch.tensor(0.0, device=_dev)
        else:
            map_loss = torch.tensor(0.0, device=_dev)
        torch.cuda.empty_cache()

        # --- [公式 41-42] L_occ + L_flow ---
        if target_maps_cpu is not None and flat_map_emb_loss is not None:
            K = args.forecast_horizon

            # 构建 K 步 GT
            gt_occ_list = []
            gt_flow_list = []
            for k in range(1, K + 1):
                shifted = torch.roll(target_maps_cpu, shifts=-k, dims=0)
                if k < T_loss:
                    shifted[-k:] = target_maps_cpu[-1:]
                else:
                    shifted = target_maps_cpu[-1:].expand_as(target_maps_cpu)
                gt_occ_list.append(shifted[:, :, 3, :, :])
                gt_flow_list.append(torch.stack([shifted[:, :, 4, :, :], shifted[:, :, 5, :, :]], dim=2))

            gt_occ_k = torch.stack(gt_occ_list, dim=2).reshape(T_loss * B_loss, K, 30, 30)
            gt_flow_k = torch.stack(gt_flow_list, dim=2).reshape(T_loss * B_loss, K, 2, 30, 30)
            del gt_occ_list, gt_flow_list

            # 分块预测
            _occ_loss = torch.tensor(0.0, device=_dev)
            _flow_loss = torch.tensor(0.0, device=_dev)
            _occ_count = 0
            _flow_count = 0
            N_of = T_loss * B_loss
            for ci in range(0, N_of, args.encode_batch):
                cj = min(ci + args.encode_batch, N_of)
                with autocast(enabled=use_amp):
                    _occ_p, _flow_p = obstacle_forecaster(
                        flat_beliefs[ci:cj], flat_post_states[ci:cj], flat_map_emb_loss[ci:cj]
                    )
                _gt_occ = gt_occ_k[ci:cj].float().to(_dev)
                _gt_flow = gt_flow_k[ci:cj].float().to(_dev)
                _occ_loss = _occ_loss + F.binary_cross_entropy(
                    _occ_p.float(), _gt_occ.clamp(0, 1), reduction='sum'
                )
                _flow_loss = _flow_loss + F.l1_loss(_flow_p.float(), _gt_flow, reduction='sum')
                _occ_count += _gt_occ.numel()
                _flow_count += _gt_flow.numel()
                del _occ_p, _flow_p, _gt_occ, _gt_flow
            occ_loss = _occ_loss / max(_occ_count, 1)
            flow_loss = _flow_loss / max(_flow_count, 1)
            del gt_occ_k, gt_flow_k, _occ_loss, _flow_loss
        else:
            occ_loss = torch.tensor(0.0, device=_dev)
            flow_loss = torch.tensor(0.0, device=_dev)
        torch.cuda.empty_cache()

        # --- 对比语义损失 (Siamese contrastive loss) ---
        if isinstance(observations, dict) and loss_curr_feat is not None and loss_pos is not None:
            curr_emb = loss_curr_feat.view(-1, args.embedding_size)
            targ_emb = loss_targ_feat.view(-1, args.embedding_size)
            d_f = F.pairwise_distance(curr_emb, targ_emb)

            curr_pos_flat = loss_pos.view(-1, 2)
            tgt_pos_flat = loss_tgt_pos.view(-1, 2)
            d_phys = torch.norm(curr_pos_flat - tgt_pos_flat, dim=1)

            y = (d_phys < args.similarity_dist_thresh).float()
            loss_sim = 0.5 * y * (d_f ** 2)
            loss_dissim = 0.5 * (1 - y) * torch.clamp(args.contrastive_margin - d_f, min=0.0) ** 2
            semantic_loss = torch.mean(loss_sim + loss_dissim) * args.semantic_loss_scale
        else:
            semantic_loss = torch.tensor(0.0, device=args.device)

        # --- [公式 35] 总损失 ---
        model_loss = (
            args.lambda_img * observation_loss
            + args.lambda_r * reward_loss
            + args.lambda_c * continuation_loss
            + kl_loss
            + args.lambda_map * (map_loss + map_updater_loss)
            + args.lambda_occ * occ_loss
            + args.lambda_flow * flow_loss
            + semantic_loss
        )

        # [梯度累积] 缩放损失, 累积多步后再 step
        model_loss = model_loss / args.grad_accumulate

        if args.learning_rate_schedule != 0:
            for group in model_optimizer.param_groups:
                group['lr'] = min(
                    group['lr'] + args.model_learning_rate / args.learning_rate_schedule,
                    args.model_learning_rate
                )

        # 仅在累积周期首步清零梯度
        if s % args.grad_accumulate == 0:
            model_optimizer.zero_grad()

        model_scaler.scale(model_loss).backward()

        # 在累积周期末步执行 step
        if (s + 1) % args.grad_accumulate == 0 or (s + 1) == args.collect_interval:
            model_scaler.unscale_(model_optimizer)
            nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
            model_scaler.step(model_optimizer)
            model_scaler.update()

        # [OOM 修复] 释放世界模型计算图, 回收显存给 actor/value 阶段
        _loss_items = [
            observation_loss.item(), reward_loss.item(), kl_loss.item(),
            continuation_loss.item(), map_loss.item(), map_updater_loss.item(),
            occ_loss.item(), flow_loss.item(),
        ]
        del model_loss, observation_loss, reward_loss, continuation_loss
        del kl_loss, map_loss, map_updater_loss, occ_loss, flow_loss, semantic_loss
        try:
            del reward_pred, cont_pred
        except NameError:
            pass
        torch.cuda.empty_cache()

        # ===============================================================
        #  Actor Training (公式 33)
        # ===============================================================
        with torch.no_grad():
            actor_states = posterior_states.detach()
            actor_beliefs = beliefs.detach()
            actor_semantics = semantic_states.detach()
            if map_emb_for_loss is not None:
                actor_map_emb = map_emb_for_loss.detach()
            elif map_emb_for_tm is not None:
                actor_map_emb = map_emb_for_tm.detach()
            else:
                actor_map_emb = torch.zeros(
                    *actor_beliefs.shape[:2], args.map_embedding_size, device=args.device
                )

        # [OOM 修复] 释放世界模型前向中间变量
        del beliefs, prior_states, prior_means, prior_std_devs
        del posterior_states, posterior_means, posterior_std_devs, semantic_states
        del embed, sem_feat_for_tm, map_emb_for_tm, map_emb_for_loss
        torch.cuda.empty_cache()

        with FreezeParameters(model_modules):
            # imagine_ahead 内部调用 TransitionModel (GRU), 不能用 autocast
            imagination_traj = imagine_ahead(
                actor_states, actor_beliefs, actor_semantics, actor_map_emb,
                actor_model, transition_model, map_encoder, map_transition_model,
                args.planning_horizon,
            )

        (imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs,
         imged_semantics, imged_map_emb, _) = imagination_traj

        with FreezeParameters(model_modules + value_model.component_modules):
            with autocast(enabled=use_amp):
                H, N = imged_beliefs.shape[:2]
                flat_ib = imged_beliefs.view(H * N, -1)
                flat_ips = imged_prior_states.view(H * N, -1)
                flat_ime = imged_map_emb.view(H * N, -1)

                imged_reward = reward_model(flat_ib, flat_ips, flat_ime).view(H, N)
                value_pred = value_model(flat_ib, flat_ips, flat_ime).view(H, N)
                imged_cont = continuation_model(flat_ib, flat_ips, flat_ime).view(H, N)

        returns = lambda_return(
            imged_reward, value_pred, bootstrap=value_pred[-1],
            cont_pred=imged_cont, discount=args.discount, lambda_=args.disclam,
        )

        actor_loss = -torch.mean(returns)
        _actor_loss_val = actor_loss.item()
        actor_optimizer.zero_grad()
        actor_scaler.scale(actor_loss).backward()
        actor_scaler.unscale_(actor_optimizer)
        nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
        actor_scaler.step(actor_optimizer)
        actor_scaler.update()

        # ===============================================================
        #  Value Training (公式 34)
        # ===============================================================
        with torch.no_grad():
            value_beliefs = imged_beliefs.detach()
            value_prior_states = imged_prior_states.detach()
            value_map_emb = imged_map_emb.detach()
            target_return = returns.detach()

        del imagination_traj, imged_beliefs, imged_prior_states, imged_prior_means
        del imged_prior_std_devs, imged_semantics, imged_map_emb, returns
        del actor_loss, imged_reward, imged_cont
        torch.cuda.empty_cache()

        flat_vb = value_beliefs.view(H * N, -1)
        flat_vps = value_prior_states.view(H * N, -1)
        flat_vme = value_map_emb.view(H * N, -1)
        with autocast(enabled=use_amp):
            value_pred_train = value_model(flat_vb, flat_vps, flat_vme).view(H, N)
            value_loss = 0.5 * F.mse_loss(value_pred_train, target_return, reduction='mean')

        value_optimizer.zero_grad()
        value_scaler.scale(value_loss).backward()
        value_scaler.unscale_(value_optimizer)
        nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
        value_scaler.step(value_optimizer)
        value_scaler.update()

        _loss_items.extend([_actor_loss_val, value_loss.item()])
        losses.append(_loss_items)

        # [OOM 修复] 彻底清理本轮迭代
        del value_loss, value_beliefs, value_prior_states, value_map_emb, target_return
        torch.cuda.empty_cache()

    # ===============================================================
    #  Log Metrics
    # ===============================================================
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(np.mean(losses[0]))
    metrics['reward_loss'].append(np.mean(losses[1]))
    metrics['kl_loss'].append(np.mean(losses[2]))
    metrics['continuation_loss'].append(np.mean(losses[3]))
    metrics['map_loss'].append(np.mean(losses[4]))
    metrics['map_updater_loss'] = metrics.get('map_updater_loss', [])
    metrics['map_updater_loss'].append(np.mean(losses[5]))
    metrics['occ_loss'].append(np.mean(losses[6]))
    metrics['flow_loss'].append(np.mean(losses[7]))
    metrics['actor_loss'].append(np.mean(losses[8]))
    metrics['value_loss'].append(np.mean(losses[9]))

    lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'Observation Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'Reward Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'KL Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['continuation_loss']):], metrics['continuation_loss'], 'Continuation Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['map_loss']):], metrics['map_loss'], 'Map Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['map_updater_loss']):], metrics['map_updater_loss'], 'MapUpdater Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['occ_loss']):], metrics['occ_loss'], 'Occ Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'Actor Loss', results_dir)
    lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'Value Loss', results_dir)

    writer.add_scalar('Loss/observation', metrics['observation_loss'][-1], episode)
    writer.add_scalar('Loss/reward', metrics['reward_loss'][-1], episode)
    writer.add_scalar('Loss/kl', metrics['kl_loss'][-1], episode)
    writer.add_scalar('Loss/continuation', metrics['continuation_loss'][-1], episode)
    writer.add_scalar('Loss/map', metrics['map_loss'][-1], episode)
    writer.add_scalar('Loss/map_updater', metrics['map_updater_loss'][-1], episode)
    writer.add_scalar('Loss/occ', metrics['occ_loss'][-1], episode)
    writer.add_scalar('Loss/flow', metrics['flow_loss'][-1], episode)
    writer.add_scalar('Loss/actor', metrics['actor_loss'][-1], episode)
    writer.add_scalar('Loss/value', metrics['value_loss'][-1], episode)

    # ===============================================================
    #  Evaluation — [论文 Section V-D] 完整评估指标
    # ===============================================================
    if episode % args.test_interval == 0:
        for m in world_model_modules:
            m.eval()
        actor_model.eval()
        value_model.eval()

        # 两个评估域: in-domain (训练地图的新配置) 和 out-of-domain (held-out 地图)
        eval_results = {}
        for eval_mode, eval_seed, is_ood in [('in_domain', 42, False), ('ood', 9999, True)]:
            ep_successes = 0
            ep_collisions = 0
            ep_total = 0
            ep_path_ratios = []
            ep_min_clearances = []
            ep_rewards = []

            with torch.no_grad():
                for test_ep in range(args.test_episodes):
                    # [论文 V-B] ood=True 时使用 held-out 测试地图集
                    observation = env.reset(map_seed=eval_seed + test_ep, ood=is_ood)
                    belief = torch.zeros(1, args.belief_size, device=args.device)
                    posterior_state = torch.zeros(1, args.state_size, device=args.device)
                    semantic_state = torch.zeros(1, args.semantic_size, device=args.device)
                    map_emb = torch.zeros(1, args.map_embedding_size, device=args.device)
                    action = torch.zeros(1, env.action_size, device=args.device)
                    done = False

                    ep_total += 1
                    path_length = 0.0
                    min_clearance = float('inf')
                    ep_reward = 0.0
                    reached = False
                    collided = False

                    # 记录起点位置, 用于 APLR 计算
                    _inner_env = env._env if hasattr(env, '_env') else env
                    start_pos = _inner_env.agent_pos.copy() if hasattr(_inner_env, 'agent_pos') else None

                    while not done:
                        (belief, posterior_state, semantic_state, map_emb,
                         action, next_observation, reward, done) = update_belief_and_act(
                            args, env, planner, transition_model, encoder,
                            belief, posterior_state, semantic_state, action, observation,
                            map_encoder=map_encoder,
                            current_map_embedding=map_emb,
                            semantic_extractor=semantic_extractor,
                            explore=False,
                        )

                        r_val = reward.item() if torch.is_tensor(reward) else reward
                        ep_reward += r_val
                        path_length += 1.0

                        # 最小间距: 当前位置到所有障碍物的最小距离
                        if hasattr(_inner_env, 'obstacles') and hasattr(_inner_env, 'agent_pos'):
                            for obs_j in _inner_env.obstacles:
                                d = float(np.linalg.norm(_inner_env.agent_pos - obs_j.q))
                                if d < min_clearance:
                                    min_clearance = d

                        # 到达/碰撞: 优先从 env.info dict 读取, 回退到 reward 阈值
                        _info = getattr(_inner_env, '_last_info', {})
                        if not _info:
                            # Env wrapper 可能不暴露 info, 用 reward 阈值判断
                            _reach_thresh = getattr(_inner_env, 'reward_reach', 100.0)
                            if r_val >= _reach_thresh * 0.9:
                                _info = {'reach': True}
                            _col_thresh = getattr(_inner_env, 'reward_collision', -10.0)
                            if r_val <= _col_thresh * 0.9:
                                _info['collision'] = True

                        if _info.get('reach', False):
                            reached = True
                        if _info.get('collision', False):
                            collided = True

                        observation = next_observation

                    # Episode 结束统计
                    if reached:
                        ep_successes += 1
                    if collided:
                        ep_collisions += 1

                    # APLR: 实际路径步数 / 最短路径 (切比雪夫网格距离)
                    if reached and start_pos is not None and hasattr(_inner_env, 'target_pos'):
                        _target = _inner_env.target_pos
                        shortest = max(abs(start_pos[0] - _target[0]),
                                       abs(start_pos[1] - _target[1])) / _inner_env.cell_size
                        if shortest > 0:
                            ep_path_ratios.append(path_length / shortest)
                    if min_clearance < float('inf'):
                        ep_min_clearances.append(min_clearance)
                    ep_rewards.append(ep_reward)

            # 汇总指标 (论文 Table I)
            sr = ep_successes / max(ep_total, 1) * 100
            cr = ep_collisions / max(ep_total, 1) * 100
            aplr = float(np.mean(ep_path_ratios)) if ep_path_ratios else 0.0
            min_clr = float(np.mean(ep_min_clearances)) if ep_min_clearances else 0.0
            avg_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

            eval_results[eval_mode] = {
                'SR': sr, 'CR': cr, 'APLR': aplr, 'MinClr': min_clr, 'Reward': avg_reward
            }

            current_training_steps = metrics['steps'][-1] if len(metrics['steps']) > 0 else 0
            writer.add_scalar(f'Eval/{eval_mode}_SR', sr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_CR', cr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_APLR', aplr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_MinClr', min_clr, current_training_steps)
            writer.add_scalar(f'Eval/{eval_mode}_Reward', avg_reward, current_training_steps)

        # 打印
        for mode, res in eval_results.items():
            print(f"  [{mode}] SR={res['SR']:.1f}% CR={res['CR']:.1f}% "
                  f"APLR={res['APLR']:.2f} MinClr={res['MinClr']:.0f} Reward={res['Reward']:.1f}")

        # 保存到 metrics
        metrics['test_episodes'].append(episode)
        for mode in ['in_domain', 'ood']:
            for key in ['SR', 'CR', 'APLR', 'MinClr', 'Reward']:
                mkey = f'test_{mode}_{key}'
                if mkey not in metrics:
                    metrics[mkey] = []
                metrics[mkey].append(eval_results[mode][key])

        # 兼容旧接口
        metrics['test_rewards'].append(eval_results['in_domain']['SR'])
        metrics['test_avg_rewards'].append(eval_results['in_domain']['Reward'])

        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_SR', []), 'SR_InDomain', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_ood_SR', []), 'SR_OOD', results_dir)
        lineplot(metrics['test_episodes'], metrics.get('test_in_domain_CR', []), 'CR_InDomain', results_dir)
        torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

        # ---- Occ-IoU / ADE / FDE: 障碍物预测质量 ----
        # [论文 Section V-D] 这些指标需要在 world model 想象中计算,
        # 与环境的实际未来状态对比. 在此记录框架, 训练后可单独跑 eval 脚本.
        # 占位: 在 checkpoint 时保存模型, 用独立脚本 eval_forecast.py 计算.
        if episode % args.checkpoint_interval == 0:
            print("  [NOTE] Occ-IoU/ADE/FDE 需要独立评估脚本 (eval_forecast.py), "
                  "模型已保存, 可离线计算.")

        for m in world_model_modules:
            m.train()
        actor_model.train()
        value_model.train()

    # ===============================================================
    #  Collect new episode
    # ===============================================================
    with torch.no_grad():
        observation, total_reward = env.reset(), 0
        belief = torch.zeros(1, args.belief_size, device=args.device)
        posterior_state = torch.zeros(1, args.state_size, device=args.device)
        semantic_state = torch.zeros(1, args.semantic_size, device=args.device)
        map_emb = torch.zeros(1, args.map_embedding_size, device=args.device)
        action = torch.zeros(1, env.action_size, device=args.device)

        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
            (belief, posterior_state, semantic_state, map_emb,
             action, next_observation, reward, done) = update_belief_and_act(
                args, env, planner, transition_model, encoder,
                belief, posterior_state, semantic_state, action, observation,
                map_encoder=map_encoder,
                current_map_embedding=map_emb,
                semantic_extractor=semantic_extractor,
                explore=True,
            )
            D.append(observation, action.cpu(), reward, done)
            total_reward += reward
            observation = next_observation
            if args.render:
                env.render()
            if done:
                pbar.close()
                break

        metrics['steps'].append(t + metrics['steps'][-1])
        metrics['episodes'].append(episode)
        metrics['train_rewards'].append(total_reward)
        lineplot(
            metrics['episodes'][-len(metrics['train_rewards']):],
            metrics['train_rewards'], 'Train Rewards', results_dir,
        )

    # ===============================================================
    #  Checkpoint
    # ===============================================================
    if episode % args.checkpoint_interval == 0:
        save_dict = {
            'transition_model': transition_model.state_dict(),
            'observation_model': observation_model.state_dict(),
            'reward_model': reward_model.state_dict(),
            'continuation_model': continuation_model.state_dict(),
            'encoder': encoder.state_dict(),
            'actor_model': actor_model.state_dict(),
            'value_model': value_model.state_dict(),
            'map_encoder': map_encoder.state_dict(),
            'map_transition_model': map_transition_model.state_dict(),
            'obstacle_forecaster': obstacle_forecaster.state_dict(),
            'map_updater': map_updater.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(),
            'value_optimizer': value_optimizer.state_dict(),
        }
        if semantic_extractor is not None:
            save_dict['semantic_extractor'] = semantic_extractor.state_dict()
        torch.save(save_dict, os.path.join(results_dir, 'models_%d.pth' % episode))
        if args.checkpoint_experience:
            torch.save(D, os.path.join(results_dir, 'experience.pth'))

env.close()