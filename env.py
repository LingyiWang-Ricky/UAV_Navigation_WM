import cv2
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import mysql.connector
import pickle
import os
import time

# Constants
GYM_ENVS = ['Pendulum-v1', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['acrobot-swingup', 'cartpole-balance', 'cartpole-balance_sparse', 'cartpole-swingup',
                      'cartpole-swingup_sparse', 'ball_in_cup-catch', 'finger-spin', 'finger-turn_easy',
                      'finger-turn_hard', 'hopper-hop', 'hopper-stand', 'pendulum-swingup', 'quadruped-run',
                      'quadruped-walk', 'reacher-easy', 'reacher-hard', 'walker-run', 'walker-stand', 'walker-walk']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2,
                                'humanoid': 2, 'fish': 2, 'acrobot': 4}


# ============================================================================
#  Helper Functions
# ============================================================================

def preprocess_observation_(observation, bit_depth):
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(
        np.uint8)


def _images_to_observation(images, bit_depth):
    if images is None: raise ValueError("Render returned None.")
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                          dtype=torch.float32)
    preprocess_observation_(images, bit_depth)
    return images.unsqueeze(dim=0)


# ============================================================================
#  Siamese Network  (论文 Section III - SMM 语义认知子模块)
#
#  论文:
#    - 双分支 ResNet-18, 共享参数 φ, 每分支输出 128 维特征
#    - 特征距离  df = ||ft - fg||₂           (公式 4)
#    - 相似度    vs = max(0, 1 - df / df_max)
#    - 训练使用对比损失 Lc (公式 9), 需单独预训练
# ============================================================================

class SiameseNetwork(nn.Module):
    """论文中的 Siamese 网络: 双分支 ResNet-18, 共享参数, 128 维特征输出。"""

    def __init__(self, feature_dim=128):
        super(SiameseNetwork, self).__init__()
        try:
            import torchvision.models as models
            resnet = models.resnet18(weights=None)
        except ImportError:
            raise ImportError("需要安装 torchvision: pip install torchvision")
        # 移除最后的 FC 层, 保留到 AdaptiveAvgPool → 输出 512 维
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)  # 512 → 128 (论文: 128 outputs)

    def forward_one(self, x):
        """单分支: image (B,3,H,W) → 128 维特征"""
        x = self.backbone(x)           # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 512)
        x = self.fc(x)                 # (B, 128)
        return x

    def forward(self, x1, x2):
        """双分支 (共享参数)"""
        return self.forward_one(x1), self.forward_one(x2)


class SiameseNetworkLite(nn.Module):
    """轻量 Siamese backbone，与 train_siamese.py 的 lightweight 版本兼容。"""
    def __init__(self, feature_dim=128):
        super(SiameseNetworkLite, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward_one(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x1, x2):
        return self.forward_one(x1), self.forward_one(x2)


# ============================================================================
#  UAV Navigation Environment (Redesigned)
# ============================================================================

class DynamicObstacle:
    """
    动态障碍物。

    论文 Section III-A, 公式 (2)(3):
        状态 o^j_t = [q^j_t, v^j_t, ρ^j]
        运动: o^j_{t+1} ~ p(o^j_{t+1} | o^j_t, E)

    运动模式 (motion_type):
        'cv'       — 匀速直线 (constant-velocity), 碰壁后反射
        'random'   — 随机游走 (每步随机扰动速度方向)
        'road'     — 地图感知: 沿亮度较高区域 (道路/人行道) 行驶,
                     实现论文公式(3)的 p(o^j_{t+1} | o^j_t, E)
    """

    def __init__(self, q, v, rho, motion_type='cv', D=3000, cell_size=100.0,
                 satellite_map=None):
        self.q = np.array(q, dtype=np.float32)
        self.v = np.array(v, dtype=np.float32)
        self.rho = float(rho)
        self.motion_type = motion_type
        self.D = D
        self.cell_size = cell_size
        self.satellite_map = satellite_map
        # 步计数器: 用于确定性运动模式的相位计算
        self.step_count = 0

    def step(self, rng=None):
        """
        前进一步, 更新 q 和 v。

        [关键设计] 所有运动模式在 episode 内都是确定性的:
        给定初始状态 (q_0, v_0) 和地图 E, 轨迹完全可预测.
        这样世界模型才能从观测序列中学到障碍物动力学.
        随机性只存在于 episode 间的初始化 (_spawn_obstacles).
        """
        self.step_count += 1

        if self.motion_type == 'cv':
            # 匀速直线, 碰壁弹射 — 天然确定性
            self.q = self.q + self.v
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'random':
            # [修改] 确定性正弦蛇形运动:
            # 方向按固定正弦规律偏转, 振幅 ±15°, 周期 40 步.
            # 给定 (q_0, v_0), 任意时刻 t 的轨迹都可精确重算.
            speed = np.linalg.norm(self.v)
            if speed < 1e-6:
                speed = self.cell_size * 0.5
            base_angle = np.arctan2(self.v[1], self.v[0])
            # 确定性偏转: sin(2π * t / 40) * π/12
            drift = np.sin(2 * np.pi * self.step_count / 40.0) * (np.pi / 12)
            new_angle = base_angle + drift
            self.v = np.array([np.cos(new_angle) * speed,
                               np.sin(new_angle) * speed], dtype=np.float32)
            self.q = self.q + self.v
            # 碰壁弹射 (与 cv 一致)
            for i in range(2):
                if self.q[i] < self.rho:
                    self.q[i] = self.rho
                    self.v[i] = abs(self.v[i])
                elif self.q[i] > self.D - self.rho:
                    self.q[i] = self.D - self.rho
                    self.v[i] = -abs(self.v[i])

        elif self.motion_type == 'road':
            # [论文公式(3)] 确定性地图感知运动:
            # 评估 3 个候选方向的地图亮度, 选最亮的 (无随机扰动).
            # 给定 (q_0, v_0, E), 轨迹确定.
            speed = np.linalg.norm(self.v)
            if speed < 1e-6:
                speed = self.cell_size * 0.5
            angle = np.arctan2(self.v[1], self.v[0])

            if self.satellite_map is not None:
                candidates = [angle, angle + np.pi / 4, angle - np.pi / 4]
                best_score = -1.0
                best_angle = angle

                map_h, map_w = self.satellite_map.shape[:2]
                for a in candidates:
                    nq = self.q + np.array([np.cos(a), np.sin(a)], dtype=np.float32) * speed
                    nq = np.clip(nq, self.rho, self.D - self.rho)
                    px = int(np.clip(nq[0], 0, map_h - 1))
                    py = int(np.clip(nq[1], 0, map_w - 1))
                    # 纯确定性: 只看亮度, 不加随机扰动
                    brightness = float(np.mean(self.satellite_map[px, py]))
                    if brightness > best_score:
                        best_score = brightness
                        best_angle = a

                self.v = np.array([np.cos(best_angle) * speed,
                                   np.sin(best_angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)
            else:
                # 无地图: 退化为确定性缓慢偏转 (每 10 步右转 90°)
                if self.step_count % 10 == 0:
                    angle += np.pi / 2
                self.v = np.array([np.cos(angle) * speed,
                                   np.sin(angle) * speed], dtype=np.float32)
                self.q = self.q + self.v
                self.q = np.clip(self.q, self.rho, self.D - self.rho)

    @property
    def grid_pos(self):
        """返回网格坐标 (gx, gy)"""
        gx = int(np.clip(self.q[0] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        gy = int(np.clip(self.q[1] / self.cell_size, 0, int(self.D / self.cell_size) - 1))
        return gx, gy

    @property
    def grid_vel(self):
        """返回网格速度 (vx_grid, vy_grid) — 单位: 格/步"""
        return self.v / self.cell_size


class UAVNavigationEnv(gym.Env):
    """
    UAV视觉导航环境 —— 训练阶段随机地图 + 随机起终点 + 动态障碍物。

    严格参照论文修改的部分:
    ──────────────────────────────────────────────────────────────
    [1] 地图/网格 (Section III-A):
        D = 3km → map_size = 3000, Ng = 30, cell = 100px

    [2] 语义地图 Mt ∈ R^{6×Ng×Ng} (公式 8, 9):
        通道 0: Mexp — 探索置信度  (当前=1, 已访=0, 未探=-1)
        通道 1: Mrel — 目标相关性  (Siamese 相似度, 未探=-1)
        通道 2: Mtrv — 可通行度    (初始=1, 碰撞=-1)
        通道 3: Mocc — 动态占用概率 (障碍物预测, ∈[0,1])
        通道 4: Mu  — 障碍物水平运动流
        通道 5: Mv  — 障碍物垂直运动流

    [3] 奖励函数 (公式 26):
        rg·I[到达] − λ_step − λ_out·I[出界] − λ_col·I[碰撞]
        + λ_rel·Δs_t − λ_risk·χ_t

    [4] 到达判定 (论文: pt = pG):
        agent 所在网格 == 目标网格

    [5] 障碍物碰撞判定 (公式 4):
        ‖pt − q^j_t‖₂ ≤ ρ_u + ρ^j
    ──────────────────────────────────────────────────────────────
    """

    # 语义地图通道索引 (公式 9)
    CH_EXP = 0   # Mexp: 探索置信度
    CH_REL = 1   # Mrel: 目标相关性
    CH_TRV = 2   # Mtrv: 可通行度
    CH_OCC = 3   # Mocc: 动态占用概率
    CH_U   = 4   # Mu:   障碍物水平运动流
    CH_V   = 5   # Mv:   障碍物垂直运动流
    N_MAP_CHANNELS = 6

    def __init__(
        self,
        map_size=3000,             # [论文] D=3km → 3000 像素
        grid_size=30,              # [论文] Ng=30 → 30×30=900 格
        max_steps=500,             # [论文 V-A] Nm=500
        db_config=None,
        semantic_patch_size=30,
        pos_margin_cells=1,
        min_grid_distance=25,
        fixed_position_seed=12345,
        speed_scale=1.0,
        # --- [论文公式 26] 奖励参数 (已平衡) ---
        reward_reach=50.0,        # rg
        reward_boundary=-10.0,     # λ_out — 出界惩罚 (不需要太大, terminate 已是惩罚)
        reward_step_penalty=-0.005, # λ_step
        reward_collision=-10.0,    # λ_col — 碰撞=坠毁, 中等惩罚 (terminate 本身是大惩罚)
        reward_rel_scale=1.0,      # λ_rel — 提高语义进展权重, 提供密集引导
        reward_risk_scale=0.005,     # λ_risk — 降低, 避免过度保守 (0.2×1=0.2 max/step)
        reward_dist_scale=2.0,     # 新增: 距离势函数 shaping, 提高稳定收敛
        # --- 可选 shaping ---
        similarity_reward_scale=0.0,
        # --- 课程学习 (仅训练模式) ---
        start_min_grid_distance=8,
        curriculum_warmup_episodes=300,
        # --- Siamese 相似度参数 ---
        siamese_model_path=None,
        df_max=10.0,
        siamese_device='cpu',
        # --- [论文 Section III-A] 动态障碍物参数 ---
        # 碰撞判定距离 = obstacle_rho + uav_rho
        # 原始值 120+60=180px=1.8格, 碰撞直径3.6格 → 250步存活率≈0%
        # 调整后 40+20=60px=0.6格, 碰撞直径1.2格 → 250步存活率≈20%
        num_obstacles=2,
        obstacle_speed=20.0,       # 慢速更易预测和规避
        obstacle_rho=40.0,         # 障碍物安全半径
        uav_rho=20.0,              # UAV 安全半径
        obstacle_motion_types=('cv', 'random', 'road'),
        forecast_horizon=5,
    ):
        super(UAVNavigationEnv, self).__init__()

        # ---------- 地图与网格 ----------
        self.D = map_size
        self.Ng = grid_size
        self.max_steps = max_steps
        self.cell_size = self.D / self.Ng   # 3000/30 = 100
        self.semantic_patch_size = semantic_patch_size

        # ---------- 起终点生成参数 ----------
        self.pos_margin_cells = int(pos_margin_cells)
        self.min_grid_distance = int(min_grid_distance)
        self.fixed_position_seed = int(fixed_position_seed)
        self._fixed_agent_pos = None
        self._fixed_target_pos = None

        # ---------- 奖励参数 (论文公式 26) ----------
        self.speed_scale = speed_scale
        self.reward_reach = reward_reach
        self.reward_boundary = reward_boundary
        self.reward_step_penalty = reward_step_penalty
        self.reward_collision = float(reward_collision)
        self.reward_rel_scale = float(reward_rel_scale)
        self.reward_risk_scale = float(reward_risk_scale)
        self.reward_dist_scale = float(reward_dist_scale)
        self.similarity_reward_scale = float(similarity_reward_scale)
        self.start_min_grid_distance = int(start_min_grid_distance)
        self.curriculum_warmup_episodes = int(curriculum_warmup_episodes)

        # ---------- [论文 Section III-A] 动态障碍物参数 ----------
        self.num_obstacles = int(num_obstacles)
        self.obstacle_speed = float(obstacle_speed)
        self.obstacle_rho = float(obstacle_rho)
        self.uav_rho = float(uav_rho)
        self.obstacle_motion_types = list(obstacle_motion_types)
        self.forecast_horizon = int(forecast_horizon)   # K
        self.obstacles = []                              # List[DynamicObstacle]

        # ---------- 动作空间 ----------
        self.action_space = spaces.Discrete(8)
        self._dir8 = np.array(
            [
                ( 0,  1),   # α1  N
                (-1,  1),   # α2  NW
                (-1,  0),   # α3  W
                (-1, -1),   # α4  SW
                ( 0, -1),   # α5  S
                ( 1, -1),   # α6  SE
                ( 1,  0),   # α7  E
                ( 1,  1),   # α8  NE
            ],
            dtype=np.float32,
        )
        # ---------- 观测空间 ----------
        # [论文公式 8,9] 语义地图 Mt ∈ R^{6×Ng×Ng}
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'target': spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            'position': spaces.Box(low=0, high=self.D, shape=(2,), dtype=np.float32),
            'target_position': spaces.Box(low=0, high=self.D, shape=(2,), dtype=np.float32),
            'semantic_map': spaces.Box(
                low=-1.0, high=1.0,
                shape=(self.N_MAP_CHANNELS, self.Ng, self.Ng),
                dtype=np.float32
            ),
        })

        # ---------- Siamese 网络 (论文方法, 必须加载) ----------
        self.df_max = df_max
        self.siamese_device = siamese_device
        self._target_feature = None   # 缓存目标图像特征 (每 episode 算一次)

        if not siamese_model_path or not os.path.exists(siamese_model_path):
            raise FileNotFoundError(
                f"[UAVEnv] 必须提供已训练的 Siamese 模型, 路径无效: {siamese_model_path}. "
            )
        model_blob = torch.load(siamese_model_path, map_location=siamese_device, weights_only=False)
        if isinstance(model_blob, dict) and 'model_state_dict' in model_blob:
            model_type = model_blob.get('model_type', 'resnet18')
            state_dict = model_blob['model_state_dict']
        else:
            model_type = 'resnet18'
            state_dict = model_blob
        self.siamese_net = SiameseNetworkLite(feature_dim=128) if model_type == 'lightweight' else SiameseNetwork(feature_dim=128)
        self.siamese_net.load_state_dict(state_dict)
        self.siamese_net.to(siamese_device)
        self.siamese_net.eval()
        print(f"[UAVEnv] Siamese 模型已加载: {siamese_model_path} (type={model_type})")

        # ---------- 地图状态 ----------
        self.satellite_map = np.zeros((self.D, self.D, 3), dtype=np.uint8)
        self.map_ids = []
        self.original_map_size = None

        # ---------- 计数器 ----------
        self.episode_counter = 0
        self._is_test = False
        self._prev_image = None   # 用于观测延迟机制
        self._last_info = {}      # 用于评测读取 reach/collision
        self._episode_rng = None

        # ---------- 数据库 ----------
        self.db_config = db_config
        self.db_conn = None
        self.map_id_table = os.getenv('UAV_MAP_ID_TABLE', 'image_maps')
        self.map_data_table = os.getenv('UAV_MAP_DATA_TABLE', self.map_id_table)

        if self.db_config:
            try:
                self.db_conn = mysql.connector.connect(**self.db_config)
                print("[UAVEnv] DB Connected.")
                cur = self.db_conn.cursor()
                cur.execute(f"SELECT id FROM {self.map_id_table}")
                all_ids = [x[0] for x in cur.fetchall()]
                cur.close()

                # [论文 V-B] 显式划分训练/测试地图集, 保证 OOD 评测无泄漏
                # 80% 训练, 20% held-out 测试
                split_rng = np.random.RandomState(42)  # 固定划分种子, 保证可复现
                split_rng.shuffle(all_ids)
                split_idx = int(len(all_ids) * 0.8)
                self.train_map_ids = all_ids[:split_idx]
                self.test_map_ids = all_ids[split_idx:]
                self.map_ids = self.train_map_ids  # 默认用训练集
                print(f"[UAVEnv] Maps: {len(self.train_map_ids)} train, "
                      f"{len(self.test_map_ids)} test (held-out).")
            except Exception as e:
                print(f"[UAVEnv] DB Error: {e}")
                self.train_map_ids = []
                self.test_map_ids = []

    # ================================================================
    #  Position Generation
    # ================================================================

    def _generate_random_positions(self, rng=None, min_grid_distance=None):
        """
        [改] 起终点生成: **独立随机采样**, 但要求切比雪夫网格距离 ≥ min_grid_distance.

        和论文对齐的点: 起终点是 "random", 不是固定距离.
        额外约束: 切比雪夫距离 ≥ 25 (默认), 保证两点大致近对角, 任务足够有挑战性.
        切比雪夫最大值受 Ng 和 margin 约束: 30-1-1-1 = 27. 所以实际分布 ∈ {25, 26, 27}.

        算法:
            1. 均匀采样起点网格 ∈ [margin, Ng-margin)
            2. 均匀采样终点网格, 若切比雪夫距离 < min 则重采
            3. 起/终点返回对应 cell 中心的连续坐标

        参数:
            rng: numpy RandomState (测试时可复现) / None (训练时完全随机)
        返回:
            agent_pos, target_pos: shape=(2,) float32, cell 中心对齐
        """
        _randint = rng.randint if rng is not None else np.random.randint

        lo = int(self.pos_margin_cells)
        hi = int(self.Ng - self.pos_margin_cells)  # exclusive
        D_min = int(self.min_grid_distance if min_grid_distance is None else min_grid_distance)

        usable = hi - lo
        max_possible_cheb = usable - 1  # 对角端到端的切比雪夫距离
        if D_min > max_possible_cheb:
            raise ValueError(
                f"min_grid_distance={D_min} 超出可行上限 {max_possible_cheb} "
                f"(Ng={self.Ng}, margin={lo}). 请调小 min_grid_distance."
            )

        # 每次同时重采起点和终点, 直到 Chebyshev ≥ D_min.
        # (若只固定起点, 中心起点无论如何重采终点都不可能成功)
        max_attempts = 5000
        for _ in range(max_attempts):
            gx_s = int(_randint(lo, hi))
            gy_s = int(_randint(lo, hi))
            gx_t = int(_randint(lo, hi))
            gy_t = int(_randint(lo, hi))
            cheb = max(abs(gx_t - gx_s), abs(gy_t - gy_s))
            if cheb >= D_min:
                agent_pos = np.array(
                    [(gx_s + 0.5) * self.cell_size, (gy_s + 0.5) * self.cell_size],
                    dtype=np.float32,
                )
                target_pos = np.array(
                    [(gx_t + 0.5) * self.cell_size, (gy_t + 0.5) * self.cell_size],
                    dtype=np.float32,
                )
                return agent_pos, target_pos

        # Fallback: 极少触发; 把 start 丢到角落, target 丢到对角
        print(f"[UAVEnv] WARNING: position sampling exhausted {max_attempts} attempts.")
        gx_s, gy_s = lo, lo
        gx_t, gy_t = hi - 1, hi - 1
        return (
            np.array([(gx_s + 0.5) * self.cell_size, (gy_s + 0.5) * self.cell_size], dtype=np.float32),
            np.array([(gx_t + 0.5) * self.cell_size, (gy_t + 0.5) * self.cell_size], dtype=np.float32),
        )

    # ================================================================
    #  Map Loading
    # ================================================================

    def _load_map_from_db(self, map_seed=None):
        """
        从数据库加载地图。

        参数:
            map_seed:
                - None: 训练模式, 每次随机选择不同地图
                - int:  测试模式, 使用固定 seed 确保选择相同地图

        返回:
            缩放后的地图 (self.D × self.D × 3) uint8
        """
        cache_dir = './map_cache'
        os.makedirs(cache_dir, exist_ok=True)

        # ===== 测试模式: 优先从文件缓存加载 =====
        if map_seed is not None:
            cache_path = os.path.join(cache_dir, f'test_map_seed_{map_seed}.npy')
            if os.path.exists(cache_path):
                cached_data = np.load(cache_path, allow_pickle=True).item()
                self.original_map_size = cached_data['original_size']
                return cached_data['map'].astype(np.uint8)

        # ===== 检查数据库连接 =====
        if not self.db_conn:
            raise ConnectionError("Database is not connected! Check your DB config.")
        if not self.map_ids:
            raise ValueError(f"No map IDs found in the database table '{self.map_id_table}'.")

        try:
            start_time = time.time()

            if not self.db_conn.is_connected():
                self.db_conn.reconnect()

            total_maps = len(self.map_ids)
            mode = "TEST" if map_seed is not None else "TRAIN"

            # ===== 根据 map_seed 选择地图 =====
            if map_seed is not None:
                rng = np.random.RandomState(map_seed)
                selected_idx = rng.randint(0, total_maps)
            else:
                selected_idx = np.random.randint(0, total_maps)

            selected_id = self.map_ids[selected_idx]
            print(f"[MapLoader] {mode} mode - Map ID: {selected_id} ({selected_idx}/{total_maps})")

            # ===== 从数据库加载地图 =====
            cursor = self.db_conn.cursor()
            cursor.execute(f"SELECT image_data FROM {self.map_data_table} WHERE id = %s", (int(selected_id),))
            row = cursor.fetchone()
            cursor.close()

            if row is None:
                raise ValueError(f"Map ID {selected_id} not found in database.")

            blob = row[0]

            # 解析地图数据
            try:
                original_map = pickle.loads(blob)
            except:
                original_map = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)

            if original_map is None:
                raise ValueError(f"Failed to decode map ID {selected_id}")

            # 确保是 RGB 格式
            if len(original_map.shape) == 2:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_GRAY2RGB)
            elif original_map.shape[2] == 4:
                original_map = cv2.cvtColor(original_map, cv2.COLOR_BGRA2RGB)

            self.original_map_size = original_map.shape[:2]

            # ===== 缩放地图到导航尺寸 =====
            if original_map.shape[0] != self.D or original_map.shape[1] != self.D:
                scaled_map = cv2.resize(original_map, (self.D, self.D), interpolation=cv2.INTER_LINEAR)
            else:
                scaled_map = original_map

            elapsed = time.time() - start_time
            print(f"[MapLoader] Map loaded in {elapsed:.2f}s")

            # ===== 测试模式: 保存到文件缓存 =====
            if map_seed is not None:
                cache_path = os.path.join(cache_dir, f'test_map_seed_{map_seed}.npy')
                np.save(cache_path, {'map': scaled_map, 'original_size': self.original_map_size})

            return scaled_map.astype(np.uint8)

        except Exception as e:
            raise RuntimeError(f"Critical failure loading map from DB: {e}")

    # ================================================================
    #  Semantic Features
    # ================================================================

    def _get_semantic_patch(self, pos):
        """提取以 pos 为中心的语义切片 (3, patch_size, patch_size)"""
        patch_size = self.semantic_patch_size
        half_patch = patch_size // 2

        padded_map = np.pad(
            self.satellite_map,
            ((half_patch, half_patch), (half_patch, half_patch), (0, 0)),
            mode='constant'
        )

        center_x = int(pos[0]) + half_patch
        center_y = int(pos[1]) + half_patch

        patch = padded_map[
            center_x - half_patch: center_x + half_patch,
            center_y - half_patch: center_y + half_patch
        ].copy()

        if patch.shape[:2] != (patch_size, patch_size):
            patch = cv2.resize(patch, (patch_size, patch_size))

        return patch.transpose(2, 0, 1)  # HWC -> CHW

    # [新增] 网格坐标工具
    def _pos_to_grid(self, pos):
        """连续坐标 → 网格坐标 (gx, gy)"""
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))
        return gx, gy

    # [新增] 提取网格单元图像, 用于 Siamese 输入
    def _get_grid_cell_image(self, gx, gy):
        """
        提取网格 (gx, gy) 对应的卫星图像, resize 到 64×64。
        用于 Siamese 网络的输入 (论文: 观测图像 ot 和目标图像 og)。
        返回: (3, 64, 64) uint8, CHW 格式
        """
        x_start = int(gx * self.cell_size)
        y_start = int(gy * self.cell_size)
        x_end = min(int((gx + 1) * self.cell_size), self.D)
        y_end = min(int((gy + 1) * self.cell_size), self.D)
        cell_img = self.satellite_map[x_start:x_end, y_start:y_end]
        if cell_img.shape[0] == 0 or cell_img.shape[1] == 0:
            return np.zeros((3, 64, 64), dtype=np.uint8)
        cell_img = cv2.resize(cell_img, (64, 64), interpolation=cv2.INTER_LINEAR)
        return cell_img.transpose(2, 0, 1)  # HWC → CHW

    def _compute_similarity(self, pos):
        """
        [论文公式 4] 使用 Siamese 网络计算当前网格图像与目标网格图像的相似度.

        流程:
          1. 提取当前位置所在网格图像 ot (_get_grid_cell_image)
          2. Siamese 前向得到 128 维特征 ft
          3. 特征距离 df = ||ft - fg||₂   (fg 已在 reset 时缓存)
          4. 归一化 vs = max(0, 1 - df / df_max) ∈ [0, 1]
        """
        assert self._target_feature is not None, \
            "target feature not cached; 请确认 reset() 被正确调用"

        gx, gy = self._pos_to_grid(pos)
        cell_img = self._get_grid_cell_image(gx, gy)

        with torch.no_grad():
            img_tensor = torch.tensor(
                cell_img, dtype=torch.float32
            ).unsqueeze(0).to(self.siamese_device) / 255.0
            f_current = self.siamese_net.forward_one(img_tensor)
            df = torch.norm(f_current - self._target_feature, p=2, dim=1).item()

        vs = max(0.0, 1.0 - df / self.df_max)
        return float(vs)

    # ================================================================
    #  Dynamic Obstacles
    # ================================================================

    def _spawn_obstacles(self):
        """
        [论文 Section III-A, 公式(2)] 在每个 episode 开始时随机生成障碍物。
        障碍物不能出现在 agent 或 target 附近 (>= 3 格)。
        """
        self.obstacles = []
        margin = self.obstacle_rho + self.uav_rho + self.cell_size  # 安全生成距离

        rng = self._episode_rng
        _choice = rng.choice if rng is not None else np.random.choice
        _uniform = rng.uniform if rng is not None else np.random.uniform

        for _ in range(self.num_obstacles):
            # 随机运动类型
            mtype = _choice(self.obstacle_motion_types)
            # 随机速度方向
            angle = _uniform(0, 2 * np.pi)
            speed = _uniform(self.obstacle_speed * 0.5, self.obstacle_speed * 1.5)
            v = np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float32)

            # 随机位置: 避开 agent/target 周围
            for _ in range(200):
                q = _uniform(self.obstacle_rho, self.D - self.obstacle_rho, size=2).astype(np.float32)
                d_agent = np.linalg.norm(q - self.agent_pos)
                d_target = np.linalg.norm(q - self.target_pos)
                if d_agent > margin and d_target > margin:
                    break
            self.obstacles.append(
                DynamicObstacle(q, v, self.obstacle_rho, mtype, self.D, self.cell_size,
                                satellite_map=self.satellite_map if mtype == 'road' else None)
            )

    def _step_obstacles(self):
        """[论文公式(3)] 所有障碍物前进一步。"""
        for obs in self.obstacles:
            obs.step()

    def _check_collision(self, pos):
        """
        [论文公式(4)] 检查 UAV 是否与任意障碍物发生碰撞。
        ‖pt − q^j_t‖₂ ≤ ρ_u + ρ^j
        返回: (bool) 是否碰撞
        """
        threshold = self.uav_rho + self.obstacle_rho
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs.q) <= threshold:
                return True
        return False

    def _build_obstacle_channels(self):
        """
        [论文公式(9), Section III-C] 计算语义地图中的动态通道:
            Mocc ∈ [0,1]: 当前占用概率 (高斯扩散软化)
            Mu, Mv ∈ [-1,1]: 归一化运动流

        使用高斯核将点障碍物软化为占用概率场, sigma = 1.5 格。
        """
        Mocc = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mu   = np.zeros((self.Ng, self.Ng), dtype=np.float32)
        Mv   = np.zeros((self.Ng, self.Ng), dtype=np.float32)

        sigma = 1.5  # 高斯扩散标准差 (格单位)
        radius = int(np.ceil(3 * sigma))

        for obs in self.obstacles:
            gx, gy = obs.grid_pos
            vx_n, vy_n = obs.grid_vel  # 格/步

            # 高斯扩散到周围格子
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < self.Ng and 0 <= ny < self.Ng:
                        dist2 = dx ** 2 + dy ** 2
                        weight = np.exp(-dist2 / (2 * sigma ** 2))
                        if weight > Mocc[nx, ny]:
                            Mocc[nx, ny] = weight
                            # 运动流用最强权重那个障碍物的速度
                            Mu[nx, ny] = np.clip(vx_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)
                            Mv[nx, ny] = np.clip(vy_n / (self.obstacle_speed / self.cell_size + 1e-6), -1, 1)

        return Mocc, Mu, Mv

    def _compute_risk(self, pos):
        """
        [论文公式(28)] 预测风险 χ_t:
            χ_t = max_{1≤k≤K} Ô_{t+k}(ι̃_{t+k})

        通过前向模拟障碍物轨迹 K 步, 评估沿当前位置的未来占用风险。
        (actor 目前没有 imagined trajectory, 故用当前格子评估静止风险的上界)
        """
        import copy
        obs_copies = [copy.deepcopy(o) for o in self.obstacles]
        threshold = self.uav_rho + self.obstacle_rho
        max_occ = 0.0
        for k in range(1, self.forecast_horizon + 1):
            for o in obs_copies:
                o.step()
            for o in obs_copies:
                dist = np.linalg.norm(pos - o.q)
                occ = np.exp(-(dist ** 2) / (2 * (threshold) ** 2))
                if occ > max_occ:
                    max_occ = occ
        return float(np.clip(max_occ, 0.0, 1.0))

    # ================================================================
    #  Semantic Map Update (6-channel, 论文公式 9)
    # ================================================================

    def _update_map(self, pos, is_current):
        """
        更新语义地图 6 通道 (论文公式 9):

        CH_EXP (0): Mexp — 探索置信度: 当前=1, 已访=0, 未探=-1
        CH_REL (1): Mrel — 已在 step() 中直接更新 (避免重复 Siamese 推理)
        CH_TRV (2): Mtrv — 可通行度: 初始=1; 碰撞格置 -1
        CH_OCC (3-5): 由 _refresh_dynamic_channels 统一刷新
        """
        gx = int(np.clip(pos[0] / self.cell_size, 0, self.Ng - 1))
        gy = int(np.clip(pos[1] / self.cell_size, 0, self.Ng - 1))

        # Mexp
        self.semantic_map[self.CH_EXP, gx, gy] = 1.0 if is_current else 0.0

        # Mrel: 不在此处更新 — step() 的 Δs_t 计算中已经调用 _compute_similarity
        # 并直接写入了 self.semantic_map[CH_REL, new_gx, new_gy]

    def _refresh_dynamic_channels(self):
        """每步刷新 Mocc / Mu / Mv 通道 (公式 9 后三通道)。"""
        Mocc, Mu, Mv = self._build_obstacle_channels()
        self.semantic_map[self.CH_OCC] = Mocc
        self.semantic_map[self.CH_U]   = Mu
        self.semantic_map[self.CH_V]   = Mv

    # ================================================================
    #  Observation
    # ================================================================

    def _get_current_view(self, pos):
        """提取以 pos 为中心的 64×64 视图 (3, 64, 64)"""
        x, y = int(pos[0]) + 32, int(pos[1]) + 32
        padded = np.pad(self.satellite_map, ((32, 32), (32, 32), (0, 0)), mode='constant')
        view = padded[x - 32:x + 32, y - 32:y + 32]
        if view.shape != (64, 64, 3):
            view = cv2.resize(view, (64, 64))
        return view.transpose(2, 0, 1)

    def _apply_obs_noise(self, image):
        """
        [论文 Section V-A] 观测损坏:
            - 高斯图像噪声 (σ=5, uint8 级)
            - 5% 概率感知 dropout (整张图置零)
            - 10% 概率观测延迟 (返回上一步的缓存图像)
        仅训练模式生效, 测试模式返回原图.
        """
        if self._is_test:
            self._prev_image = image.copy()
            return image

        # 观测延迟: 10% 概率返回上一步的图像
        if hasattr(self, '_prev_image') and self._prev_image is not None:
            if np.random.random() < 0.10:
                stale = self._prev_image.copy()
                self._prev_image = image.copy()
                return stale
        self._prev_image = image.copy()

        # 感知 dropout: 5% 概率
        if np.random.random() < 0.05:
            return np.zeros_like(image)

        # 高斯噪声: σ=5
        noise = np.random.normal(0, 5, image.shape).astype(np.float32)
        noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def _get_obs(self):
        img = self._get_current_view(self.agent_pos)
        tgt = self._get_current_view(self.target_pos)
        return {
            'image': self._apply_obs_noise(img),
            'target': tgt,  # 目标图像不加噪声 (论文: target image 是固定参考)
            'position': self.agent_pos.copy(),
            'target_position': self.target_pos.copy(),
            'semantic_map': self.semantic_map.copy(),
        }

    # ================================================================
    #  Core: reset & step
    # ================================================================

    def reset(self, seed=None, options=None, map_seed=None, ood=False):
        """
        重置环境。

        参数:
            map_seed:
                - None  → 训练模式: 训练地图集随机选 + 每 episode 随机起终点 + 观测噪声
                - int   → 测试模式: 固定地图 + 固定起终点 + 无噪声
            ood:
                - False → 使用训练地图集 (in-domain)
                - True  → 使用 held-out 测试地图集 (out-of-domain)
        """
        super().reset(seed=seed)
        self.episode_counter += 1
        self._is_test = (map_seed is not None)

        # [论文 V-B] 切换地图集: 训练用 train_map_ids, OOD 评测用 test_map_ids
        if ood and hasattr(self, 'test_map_ids') and self.test_map_ids:
            self.map_ids = self.test_map_ids
        elif hasattr(self, 'train_map_ids') and self.train_map_ids:
            self.map_ids = self.train_map_ids

        # 加载地图
        self.satellite_map = self._load_map_from_db(map_seed=map_seed)

        # --- 起终点生成 ---
        if self._is_test:
            # [修复] 测试模式: 每个 episode 用 map_seed 派生独立的起终点
            # 不同 map_seed → 不同起终点, 保证测试覆盖多种配置
            pos_rng = np.random.RandomState(map_seed if map_seed is not None
                                            else self.fixed_position_seed)
            self.agent_pos, self.target_pos = self._generate_random_positions(rng=pos_rng)
            self._episode_rng = np.random.RandomState((map_seed if map_seed is not None else self.fixed_position_seed) + 1000003)
        else:
            # [论文 V-A] 训练模式: 每 episode 随机采样新的 start-goal pair
            if self.curriculum_warmup_episodes > 0:
                progress = min(1.0, self.episode_counter / float(self.curriculum_warmup_episodes))
                cur_min_dist = int(round(
                    self.start_min_grid_distance +
                    (self.min_grid_distance - self.start_min_grid_distance) * progress
                ))
            else:
                cur_min_dist = self.min_grid_distance
            self.agent_pos, self.target_pos = self._generate_random_positions(
                rng=None, min_grid_distance=cur_min_dist
            )
            self._episode_rng = None

        self.init_dist = float(np.linalg.norm(self.agent_pos - self.target_pos))
        self.steps = 0

        # 缓存目标图像 Siamese 特征
        tgx, tgy = self._pos_to_grid(self.target_pos)
        target_img = self._get_grid_cell_image(tgx, tgy)
        with torch.no_grad():
            target_tensor = torch.tensor(
                target_img, dtype=torch.float32
            ).unsqueeze(0).to(self.siamese_device) / 255.0
            self._target_feature = self.siamese_net.forward_one(target_tensor)

        # [论文公式 9] 初始化 6 通道语义地图
        self.semantic_map = np.zeros((self.N_MAP_CHANNELS, self.Ng, self.Ng), dtype=np.float32)
        self.semantic_map[self.CH_EXP] = -1.0
        self.semantic_map[self.CH_REL] = -1.0
        self.semantic_map[self.CH_TRV] = 1.0

        # [论文 Section III-A] 生成动态障碍物
        self._spawn_obstacles()
        self._refresh_dynamic_channels()
        self._update_map(self.agent_pos, True)
        gx0, gy0 = self._pos_to_grid(self.agent_pos)
        self.semantic_map[self.CH_REL, gx0, gy0] = self._compute_similarity(self.agent_pos)

        return self._get_obs(), {}

    def step(self, action):
        """
        执行一步。

        奖励 (论文公式 26):
            rt = rg·I[到达] − λ_step − λ_out·I[出界]
               − λ_col·I[碰撞障碍物] + λ_rel·Δs_t − λ_risk·χ_t
        """
        self.steps += 1

        # ===== 1. 解析动作 =====
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action = np.asarray(action).flatten()
        if action.size == 1:
            a_idx = int(action[0])
        else:
            a_idx = int(np.argmax(action))
        a_idx = int(np.clip(a_idx, 0, 7))
        direction = self._dir8[a_idx]
        new_pos = self.agent_pos + direction * self.cell_size
        old_goal_dist = float(np.linalg.norm(self.agent_pos - self.target_pos))

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        # ===== 2. 先让障碍物移动 (公式 3) =====
        self._step_obstacles()

        # ===== 3. 每步代价 (所有情况都扣, 包括到达/出界) =====
        reward += self.reward_step_penalty   # −λ_step

        # ===== 4. 边界检查 =====
        if not (0 <= new_pos[0] <= self.D and 0 <= new_pos[1] <= self.D):
            reward += self.reward_boundary   # −λ_out (累加, 不是赋值)
            terminated = True
            new_pos = np.clip(new_pos, 0, self.D)
            info['boundary'] = True
        else:
            new_gx, new_gy = self._pos_to_grid(new_pos)
            target_gx, target_gy = self._pos_to_grid(self.target_pos)

            # ===== 5. 到达检查 (优先于碰撞) =====
            # 到达目标格 = 任务成功, 即使该格有障碍物也算到达
            if (new_gx, new_gy) == (target_gx, target_gy):
                reward += self.reward_reach
                terminated = True
                info['reach'] = True
            else:
                # ===== 6. 碰撞检查 (论文公式 4) =====
                # 只在非到达时检查碰撞
                collision = self._check_collision(new_pos)
                if collision:
                    reward += self.reward_collision
                    self.semantic_map[self.CH_TRV, new_gx, new_gy] = -1.0
                    terminated = True
                    info['collision'] = True

            # ===== 7. Mrel 更新 (无条件) + 语义进展奖励 Δs_t (公式 27) =====
            # [修复] Mrel 写入与 reward shaping 解耦:
            # Mrel 始终更新 (SemanticFeatureExtractor 依赖它), reward 是独立开关
            old_gx, old_gy = self._pos_to_grid(self.agent_pos)
            mrel_old = self.semantic_map[self.CH_REL, old_gx, old_gy]

            # 无条件计算并写入新格的 Siamese 相似度
            vs_new = self._compute_similarity(new_pos)
            self.semantic_map[self.CH_REL, new_gx, new_gy] = vs_new

            # reward shaping 是独立开关
            if self.reward_rel_scale > 0.0:
                mrel_old_safe = max(0.0, float(mrel_old))
                mrel_new_safe = max(0.0, float(vs_new))
                delta_s = mrel_new_safe - mrel_old_safe
                reward += self.reward_rel_scale * delta_s

            # 额外稳定项: 距离势函数 shaping (potential-based)
            # 鼓励“向目标靠近”的动作，抑制随机游走导致的稀疏成功尖峰。
            if self.reward_dist_scale > 0.0:
                new_goal_dist = float(np.linalg.norm(new_pos - self.target_pos))
                progress = (old_goal_dist - new_goal_dist) / max(self.init_dist, 1e-6)
                progress = float(np.clip(progress, -1.0, 1.0))
                reward += self.reward_dist_scale * progress

            # ===== 8. 预测风险惩罚 χ_t (公式 28) =====
            # 注: env 层面用静止假设近似 (UAV 不知道 policy 的未来动作),
            # 真正的轨迹级风险评估在世界模型想象阶段完成.
            if self.reward_risk_scale > 0.0:
                chi = self._compute_risk(new_pos)
                reward -= self.reward_risk_scale * chi
                info['risk'] = chi

            # 可选 shaping: 首访新格相似度
            if self.similarity_reward_scale > 0.0:
                is_first_visit = bool(self.semantic_map[self.CH_EXP, new_gx, new_gy] == -1.0)
                if is_first_visit:
                    reward += self.similarity_reward_scale * vs_new

        # ===== 9. 更新语义地图 =====
        self._update_map(self.agent_pos, False)   # 旧位置 → 已探索
        self.agent_pos = new_pos.astype(np.float32)
        self._update_map(self.agent_pos, True)    # 新位置 → 当前
        # 刷新动态通道 (Mocc/Mu/Mv)
        self._refresh_dynamic_channels()

        # ===== 10. 截断检查 =====
        if self.steps >= self.max_steps:
            truncated = True

        # 保存 info 供外部评测读取
        self._last_info = info

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.db_conn:
            self.db_conn.close()


# ============================================================================
#  Wrappers (保持与 main.py / main_ddpg.py 完全兼容)
# ============================================================================

class UAVEnvWrapper:
    """包装 UAVNavigationEnv, 输出 torch Tensor + bit-depth 预处理 + action_repeat。"""

    def __init__(self, env, bit_depth, action_repeat=1):
        self._env = env
        self.bit_depth = bit_depth
        self.action_repeat = action_repeat
        self.observation_size = dict(env.observation_space)
        if hasattr(env.action_space, 'n'):
            self.action_size = int(env.action_space.n)
        else:
            self.action_size = int(env.action_space.shape[0])
        self.reward_reach = getattr(env, 'reward_reach', 30.0)

    def reset(self, map_seed=None, ood=False):
        return self._process_obs(self._env.reset(map_seed=map_seed, ood=ood)[0])

    def step(self, action):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy().flatten()
        total_reward = 0.0
        done = False
        for _ in range(self.action_repeat):
            obs, r, term, trunc, _ = self._env.step(action)
            total_reward += r
            done = term or trunc
            if done:
                break
        return self._process_obs(obs), total_reward, done

    def _process_obs(self, obs):
        processed = {}
        for k in ['image', 'target']:
            t = torch.tensor(obs[k], dtype=torch.float32)
            preprocess_observation_(t, self.bit_depth)
            processed[k] = t.unsqueeze(0)
        for k in ['position', 'target_position']:
            processed[k] = torch.tensor(obs[k], dtype=torch.float32).unsqueeze(0)
        # [论文2] 语义地图已经是 float32 范围 [-1, 1], 不做 bit-depth 预处理
        processed['semantic_map'] = torch.tensor(obs['semantic_map'], dtype=torch.float32).unsqueeze(0)
        return processed

    def sample_random_action(self):
        # [论文] Discrete(8): 返回长度为 1 的 int64 tensor
        a = self._env.action_space.sample()
        return torch.tensor([int(a)], dtype=torch.int64)

    def close(self):
        self._env.close()


# ============================================================================
#  Other Env Wrappers (ControlSuite / Gym — unchanged)
# ============================================================================

class ControlSuiteEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self._env = suite.load(domain, task, task_kwargs={'random': seed})
        if not symbolic: self._env = pixels.Wrapper(self._env)
        self.max_episode_length, self.action_repeat, self.bit_depth = max_episode_length, action_repeat, bit_depth
        self.symbolic = symbolic

    def reset(self):
        self.t = 0
        state = self._env.reset()
        if self.symbolic: return torch.tensor(
            np.concatenate([np.asarray([o]) if isinstance(o, float) else o for o in state.observation.values()],
                           axis=0), dtype=torch.float32).unsqueeze(0)
        return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            state = self._env.step(action.detach().numpy())
            reward += state.reward
            self.t += 1
            if state.last() or self.t == self.max_episode_length: break
        if self.symbolic:
            obs = torch.tensor(
                np.concatenate([np.asarray([o]) if isinstance(o, float) else o for o in state.observation.values()],
                               axis=0), dtype=torch.float32).unsqueeze(0)
        else:
            obs = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return obs, reward, (state.last() or self.t == self.max_episode_length)

    def render(self):
        pass

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(o.shape) == 0 else o.shape[0]) for o in
                    self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        self.symbolic, self.max_episode_length, self.action_repeat, self.bit_depth = symbolic, max_episode_length, action_repeat, bit_depth
        try:
            import gymnasium; self._env = gymnasium.make(env, render_mode='rgb_array'); self.is_gym = True
        except:
            import gym; self._env = gym.make(env); self._env.seed(seed); self.is_gym = False
        self.seed = seed

    def reset(self):
        self.t = 0
        state = self._env.reset(seed=self.seed)[0] if self.is_gym else self._env.reset()
        if self.symbolic: return torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return _images_to_observation(self._env.render() if self.is_gym else self._env.render(mode='rgb_array'),
                                      self.bit_depth)

    def step(self, action):
        reward = 0
        for _ in range(self.action_repeat):
            if self.is_gym:
                state, r, term, trunc, _ = self._env.step(action.detach().numpy()); done = term or trunc
            else:
                state, r, done, _ = self._env.step(action.detach().numpy())
            reward += r;
            self.t += 1
            if done or self.t == self.max_episode_length: break
        if self.symbolic: return torch.tensor(state, dtype=torch.float32).unsqueeze(0), reward, done
        return _images_to_observation(self._env.render() if self.is_gym else self._env.render(mode='rgb_array'),
                                      self.bit_depth), reward, done

    def render(self):
        pass

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


# ============================================================================
#  Factory Function
# ============================================================================

def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
    if env == 'UAV-v0':
        db_cfg = {
            'user': os.getenv('UAV_DB_USER', 'root'),
            'password': os.getenv('UAV_DB_PASSWORD', ''),
            'host': os.getenv('UAV_DB_HOST', 'localhost'),
            'database': os.getenv('UAV_DB_NAME', 'senmap'),
            'raise_on_warnings': True
        }
        siamese_model_path = os.getenv('UAV_SIAMESE_MODEL_PATH', 'siamese_model.pth')
        return UAVEnvWrapper(
            UAVNavigationEnv(max_steps=max_episode_length, db_config=db_cfg,
                             siamese_model_path=siamese_model_path),
            bit_depth,
            action_repeat=action_repeat,
        )
    elif env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)


# ============================================================================
#  EnvBatcher (unchanged)
# ============================================================================

class EnvBatcher:
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n, self.envs, self.dones = n, [env_class(*env_args, **env_kwargs) for _ in range(n)], [True] * n

    def reset(self):
        obs = [e.reset() for e in self.envs];
        self.dones = [False] * self.n
        if isinstance(obs[0], dict): return {k: torch.cat([o[k] for o in obs], 0) for k in obs[0]}
        return torch.cat(obs)

    def step(self, actions):
        obs, rs, ds = zip(*[e.step(a) for e, a in zip(self.envs, actions)])
        self.dones = [d or pd for d, pd in zip(ds, self.dones)]
        final_obs = {k: torch.cat([o[k] for o in obs], 0) for k in obs[0]} if isinstance(obs[0], dict) else torch.cat(
            obs)
        if not isinstance(obs[0], dict): final_obs[torch.tensor(self.dones)] = 0
        return final_obs, torch.tensor(rs, dtype=torch.float32), torch.tensor(ds, dtype=torch.uint8)

    def close(self):
        [e.close() for e in self.envs]
