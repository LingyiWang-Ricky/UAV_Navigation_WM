"""
utils.py — 修改对照:

[1] imagine_ahead:
    - 传播 semantic map (公式 17-20)
    - 使用 MapEncoder + MapTransitionModel 进行地图想象
    - actor/reward/value 接收 map_embedding (公式 29, 22, 30)

[2] lambda_return:
    - 加入 continuation predictions ĉ (公式 31-32)

[3] FreezeParameters: 不变
"""

import torch
from torch.nn import functional as F
from typing import Iterable


class FreezeParameters:
    """Context manager to locally freeze gradients for a list of modules."""
    def __init__(self, modules: Iterable[torch.nn.Module]):
        self.modules = modules
        self.params = []
        self.requires_grad_states = []

    def __enter__(self):
        for module in self.modules:
            self.params.extend(list(module.parameters()))
        self.requires_grad_states = [p.requires_grad for p in self.params]
        for p in self.params:
            p.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p, state in zip(self.params, self.requires_grad_states):
            p.requires_grad = state


def imagine_ahead(
    prev_state,
    prev_belief,
    prev_semantic_state,
    prev_map_embedding,
    policy,
    transition_model,
    map_transition_model,
    planning_horizon=12,
):
    """
    [论文公式 17-20] 想象阶段: 在隐空间中推演未来轨迹.

    推演循环 (每步):
        1. π_η(a_t | h_t, z_t, m_t)           — 公式 29
        2. h̃_{t+1} = f_θ(h̃_t, z̃_t, a_t, m̃_t) — 公式 17
        3. z̃_{t+1} ~ p_θ(z_{t+1} | h̃_{t+1})     — 公式 18
        4. M̃_{t+1} = T_ω(M̃_t, h̃_{t+1}, z̃_{t+1}, a_t) — 公式 19
        5. m̃_{t+1} = E_m(M̃_{t+1})             — 公式 20

    注意: 这里没有完整的 M_t 网格 (太大), 使用 map_transition_model
    在隐空间模拟地图转移, 并通过 map_embedding 传递信息.

    Args:
        prev_state:          (1, B*T_chunk, state_size) — flattened
        prev_belief:         (1, B*T_chunk, belief_size)
        prev_semantic_state: (1, B*T_chunk, semantic_state_size)
        prev_map_embedding:  (1, B*T_chunk, map_embedding_size)
        policy:              ActorModel
        transition_model:    TransitionModel
        map_transition_model: MapTransitionModel
        planning_horizon:    int

    Returns:
        beliefs, prior_states, prior_means, prior_std_devs,
        semantic_states, map_embeddings, actions — 各 (H, N, *)
    """
    flatten = lambda x: x.view([-1] + list(x.size())[2:])
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)
    prev_semantic_state = flatten(prev_semantic_state)
    prev_map_embedding = flatten(prev_map_embedding)

    T = planning_horizon
    N = prev_belief.size(0)
    device = prev_belief.device

    beliefs = [torch.empty(0)] * T
    prior_states = [torch.empty(0)] * T
    prior_means = [torch.empty(0)] * T
    prior_std_devs = [torch.empty(0)] * T
    semantic_states = [torch.empty(0)] * T
    map_embeddings = [torch.empty(0)] * T
    actions = [torch.empty(0)] * T

    current_belief = prev_belief
    current_state = prev_state
    current_semantic_state = prev_semantic_state
    current_map_embedding = prev_map_embedding

    for t in range(planning_horizon):
        # [公式 29] 动作选择
        _action = policy.get_action(current_belief, current_state, current_map_embedding)
        actions[t] = _action

        # [公式 17-18] RSSM 一步预测
        # TransitionModel 想象模式: observations=None, semantic_features=None
        output = transition_model(
            current_state,
            _action.unsqueeze(0),       # (1, N, action_size)
            current_belief,
            current_semantic_state,
            observations=None,
            nonterminals=None,
            semantic_features=None,
            map_embeddings_prev=current_map_embedding.unsqueeze(0),  # (1, N, map_emb_size)
        )

        current_belief = output[0][0]            # beliefs[0]
        current_state = output[1][0]             # prior_states[0]
        _prior_mean = output[2][0]
        _prior_std_dev = output[3][0]
        current_semantic_state = output[-1][0]   # semantic_states[0]

        # [公式 19-20] 地图转移 (在隐空间)
        # 注意: 真正的 map_transition_model 需要完整地图 M_t,
        # 但想象中我们没有完整地图. 这里用一个简化方案:
        # 通过 MLP 从 (belief, state, action, map_embedding) 预测下一步 map_embedding
        # 这等价于将 T_ω 和 E_m 合并为一个隐空间映射.
        current_map_embedding = map_transition_model.predict_next_embedding(
            current_belief, current_state, _action, current_map_embedding
        )

        beliefs[t] = current_belief
        prior_states[t] = current_state
        prior_means[t] = _prior_mean
        prior_std_devs[t] = _prior_std_dev
        semantic_states[t] = current_semantic_state
        map_embeddings[t] = current_map_embedding

    return (
        torch.stack(beliefs),
        torch.stack(prior_states),
        torch.stack(prior_means),
        torch.stack(prior_std_devs),
        torch.stack(semantic_states),
        torch.stack(map_embeddings),
        torch.stack(actions),
    )


def lambda_return(imged_reward, value_pred, bootstrap, cont_pred=None,
                  discount=0.99, lambda_=0.95):
    """
    [论文公式 31-32] λ-return 计算.

    G^(n)_τ = Σ_{i=0}^{n-1} γ^i (Π_{j=0}^{i-1} ĉ_{τ+j}) r̂_{τ+i}
              + γ^n (Π_{j=0}^{n-1} ĉ_{τ+j}) V_ψ(s̃_{τ+n})

    V^λ(s̃_τ) = (1-λ) Σ_{n=1}^{H-1} λ^{n-1} G^(n)_τ + λ^{H-1} G^(H)_τ

    简化实现: 标准的 TD(λ) 递推形式, 加入 continuation ĉ.

    Args:
        imged_reward: (H, N) — 想象中的奖励
        value_pred:   (H, N) — 想象中的价值预测
        bootstrap:    (N,)   — 最后一步的 value 估计
        cont_pred:    (H, N) — 继续概率 ĉ (可选, None 时全部设为 1)
        discount:     float  — γ
        lambda_:      float  — λ
    Returns:
        returns: (H, N)
    """
    if cont_pred is None:
        # 无 continuation 预测时, 假设全部继续 (退化为原始 lambda_return)
        cont = torch.ones_like(imged_reward)
    else:
        cont = cont_pred

    # 标准 TD(λ) 递推
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    # 每步的 one-step target: r + γ * ĉ * V(s')
    # λ-return 混合: inputs = r + γ * ĉ * (1-λ) * V(s')
    inputs = imged_reward + discount * cont * next_values * (1 - lambda_)

    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        last = inputs[index] + discount * lambda_ * cont[index] * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    returns = torch.stack(outputs, 0)
    return returns


def lineplot(x, y, name, path, xaxis='Episodes'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        plt.figure()
        plt.plot(x, y)
        plt.xlabel(xaxis)
        plt.ylabel(name)
        plt.savefig(path + '/' + name + '.png')
    except Exception as e:
        print(f"Plotting error: {e}")
    finally:
        plt.close()


def write_video(frames, title, path):
    pass


def numpy_to_torch(array):
    return torch.tensor(array)
