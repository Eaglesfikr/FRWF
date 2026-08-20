"""
Sinkhorn Loss for Supervised Contrastive Learning
基于 Sinkhorn-Knopp 算法计算最优传输距离，替代原生的余弦相似度矩阵。
"""

import torch
import torch.nn.functional as F


def sinkhorn(
    cost_matrix: torch.Tensor,
    epsilon: float = 0.1,
    sinkhorn_iterations: int = 20,
    epsilon_sinkhorn: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn-Knopp 算法求解熵正则化的最优传输问题。

    输入:
        cost_matrix: (B, B) 代价矩阵 (距离矩阵, 不是相似度)
        epsilon: 熵正则化系数 (越小越接近 exact OT，越大越平滑)
        sinkhorn_iterations: Sinkhorn 迭代次数

    返回:
        transport_plan: (B, B) 最优传输计划矩阵 P*

    参考: Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport" (2013)
    """
    B = cost_matrix.shape[0]

    # 初始化传输计划: P = exp(-C / epsilon)
    # 注意: cost_matrix 是距离，值越大表示越不相似
    log_P = -cost_matrix / epsilon

    # 边际分布: 均匀分布 (每个样本权重相同)
    mu = torch.ones(B, device=cost_matrix.device) / B
    nu = torch.ones(B, device=cost_matrix.device) / B
    log_mu = torch.log(mu.clamp(min=epsilon_sinkhorn))
    log_nu = torch.log(nu.clamp(min=epsilon_sinkhorn))

    # Sinkhorn 迭代: 交替缩放行和列
    f = torch.zeros_like(log_mu)  # 行缩放因子
    g = torch.zeros_like(log_nu)  # 列缩放因子

    for _ in range(sinkhorn_iterations):
        # 行缩放: f = log_mu - logsumexp(log_P + g, dim=1)
        # logsumexp 的稳定实现
        A = log_P + g.unsqueeze(0)  # (B, B)
        f = log_mu - torch.logsumexp(A, dim=1)

        # 列缩放: g = log_nu - logsumexp(log_P + f, dim=0)
        B_log = log_P + f.unsqueeze(1)
        g = log_nu - torch.logsumexp(B_log, dim=0)

    # 最终传输计划
    transport_plan = torch.exp(log_P + f.unsqueeze(1) + g.unsqueeze(0))

    return transport_plan


def sinkhorn_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
    epsilon: float = 0.05,
    sinkhorn_iterations: int = 20,
    sinkhorn_mode: str = 'replace_sim',
    device: torch.device = None,
) -> torch.Tensor:
    """
    带 Sinkhorn 距离的监督对比损失。

    策略选择 (sinkhorn_mode):
    - 'replace_sim': 用 Sinkhorn 传输计划的负对数替代相似度矩阵
                    L = -1/|P(i)| * sum log( P_ij / sum_k P_ik )
                    其中 P_ij 是 i→j 的最优传输量

    - 'regularize': 在原始对比损失上 + λ * OT_cost 正则项

    返回:
        loss: 标量损失
    """
    B = features.shape[0]
    device = device or features.device

    # L2 归一化
    z = F.normalize(features, dim=1)

    # ========== 构建代价矩阵 (distance matrix) ==========
    # 原始对比学习用 余弦相似度 = z_i · z_j
    # Sinkhorn 用 余弦距离 = 1 - cos_sim (值域 [0, 2])
    cos_sim = torch.matmul(z, z.T)  # (B, B)
    cost_matrix = 1.0 - cos_sim       # 距离矩阵: 越相似值越小

    # ========== 构建正样本掩码 ==========
    labels = labels.unsqueeze(1)
    pos_mask = torch.eq(labels, labels.T).float().to(device)
    mask_self = torch.eye(B, dtype=torch.bool).to(device)

    if sinkhorn_mode == 'replace_sim':
        # ===== 方案 A: 用 Sinkhorn 传输计划替代相似度矩阵 =====
        transport_plan = sinkhorn(
            cost_matrix,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations
        )

        # 去掉对角线自身
        pos_mask_no_self = pos_mask[~mask_self].view(B, -1)
        transport_no_self = transport_plan[~mask_self].view(B, -1)

        # Sinkhorn 对比损失: 最大化同类样本间的传输量
        log_transport = torch.log(transport_no_self.clamp(min=1e-8))
        log_prob = log_transport - torch.log(transport_no_self.sum(1, keepdim=True))

        pos_count = pos_mask_no_self.sum(1).clamp(min=1)
        mean_log_prob_pos = (pos_mask_no_self * log_prob).sum(1) / pos_count

        loss = -mean_log_prob_pos.mean()

    elif sinkhorn_mode == 'regularize':
        # ===== 方案 B: 原始对比损失 + OT 正则项 =====
        # 1. 原始 SupCon loss
        sim = cos_sim / temperature
        pos_mask_no_self = pos_mask[~mask_self].view(B, -1)
        sim_masked = sim[~mask_self].view(B, -1)

        log_sim = torch.exp(sim_masked)
        log_prob = sim_masked - torch.log(log_sim.sum(1, keepdim=True))
        pos_count = pos_mask_no_self.sum(1).clamp(min=1)
        mean_log_prob_pos = (pos_mask_no_self * log_prob).sum(1) / pos_count
        supcon_loss = -mean_log_prob_pos.mean()

        # 2. Sinkhorn OT 正则: 鼓励同类间传输、抑制异类间传输
        transport_plan = sinkhorn(
            cost_matrix,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations
        )

        # OT 正则损失: -sum(同类传输量) + sum(异类传输量)
        ot_reg = -(pos_mask * transport_plan).sum() / B + ((1 - pos_mask) * transport_plan).sum() / B

        # λ = 0.1 可调
        lambda_ot = 0.1
        loss = supcon_loss + lambda_ot * ot_reg

    else:
        raise ValueError(f"Unknown sinkhorn_mode: {sinkhorn_mode}")

    return loss


def sinkhorn_distance_loss(
    z_time: torch.Tensor,
    z_freq: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.5,
    epsilon: float = 0.05,
    sinkhorn_iterations: int = 20,
    device: torch.device = None,
) -> torch.Tensor:
    """
    跨模态 (时域↔频域) 的 Sinkhorn 对齐损失。

    替代你原先的 cross-modal consistency loss (直接把时频拼起来算 SupCon)。
    这里用 Sinkhorn 传输距离衡量两个模态的分布对齐程度。
    """
    B = z_time.shape[0]
    device = device or z_time.device

    # L2 归一化
    z_t = F.normalize(z_time, dim=1)
    z_f = F.normalize(z_freq, dim=1)

    # 跨模态代价矩阵: 时域 i → 频域 j 的距离
    cross_cos_sim = torch.matmul(z_t, z_f.T)  # (B, B)
    cross_cost = 1.0 - cross_cos_sim

    # Sinkhorn 传输
    transport_plan = sinkhorn(
        cross_cost,
        epsilon=epsilon,
        sinkhorn_iterations=sinkhorn_iterations
    )

    # 对齐损失: 同类跨模态应该高传输量
    labels = labels.unsqueeze(1)
    same_class_mask = torch.eq(labels, labels.T).float().to(device)

    # 最大化同类跨模态传输量
    log_transport = torch.log(transport_plan.clamp(min=1e-8))
    log_prob = log_transport - torch.log(transport_plan.sum(1, keepdim=True))

    pos_count = same_class_mask.sum(1).clamp(min=1)
    mean_log_prob_pos = (same_class_mask * log_prob).sum(1) / pos_count

    loss = -mean_log_prob_pos.mean()
    return loss