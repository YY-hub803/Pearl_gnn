import torch
import torch.nn as nn
import numpy as np

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, output, target):
        """
        output: [B, N, T, F]
        target: [B, N, T, F]
        mask:   [B, N, T, F]
        """

        # 平方误差
        loss = (output - target) ** 2

        # 先求MSE再开方
        mse = loss.sum()
        rmse = torch.sqrt(mse)

        return rmse

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        """
        preds:   [Batch, N, T, F]
        targets: [Batch, N, T, F]
        mask:    [Batch, N, T, F] (1=真实值, 0=缺失)
        """
        loss = (output - target) ** 2

        return loss.sum()

class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()

    def forward(self, output, target):
        """
        output: [B, N, T, F]
        target: [B, N, T, F]
        mask:   [B, N, T, F]
        """

        # -------- 分子：残差平方和 --------
        numerator = ((output - target) ** 2)
        numerator = numerator.sum()

        # -------- 分母：真实值方差平方和 --------
        mean_target = (target * mask).sum()

        denominator = ((target - mean_target) ** 2)
        denominator = denominator.sum()

        nse = 1 - numerator / denominator

        # 作为loss返回 → 越小越好
        loss = 1 - nse

        return loss

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target):
        p0 = output
        t0 = target
        N = len(t0)
        loss = torch.abs(t0 - p0)
        return loss.sum() / N

def R2(output, target):
    mask = ~np.isnan(target)
    p0 = output[mask]
    t0 = target[mask]

    if len(t0) == 0:
        return np.nan

    ss_res = np.sum((t0 - p0) ** 2)
    ss_tot = np.sum((t0 - np.mean(t0)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot

def NSE(output, target):

    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]

    numerator = np.sum((t0 - p0) ** 2)
    denominator = np.sum((t0- np.average(t0))** 2)

    NSE = 1 - numerator / denominator

    return NSE

def MAE(output, target):
    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]
    N = len(t0)

    MAE = np.sum(np.abs(t0 - p0))/N
    return MAE

def RMSE(output, target):
    """
    用mask操作标记缺失值，只计算有值的位置
    :param output:[total_sample,1]
    :param target:[total_sample,1]
    :return:
    """
    mask = target == target
    p0 = output[mask]
    t0 = target[mask]
    N = len(t0)

    RMSE = np.sqrt(np.mean((t0 - p0)**2))

    return RMSE

def FHV(sim, obs, h=0.02):
    sim = np.array(sim)
    obs = np.array(obs)

    # 按观测值降序排序
    idx = np.argsort(obs)[::-1]

    n = int(len(obs) * h)
    sim_high = sim[idx][:n]
    obs_high = obs[idx][:n]

    return 100 * np.sum(sim_high - obs_high) / np.sum(obs_high)


def KGE(sim, obs):
    """
    Kling-Gupta Efficiency (KGE)

    :param sim: 模拟值
    :param obs: 观测值
    :return: kge, r, alpha, beta
    """

    # 去掉 NaN
    mask = ~np.isnan(obs)
    sim = sim[mask]
    obs = obs[mask]

    if len(obs) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # 均值
    mean_sim = np.mean(sim)
    mean_obs = np.mean(obs)

    # 标准差
    std_sim = np.std(sim, ddof=1)
    std_obs = np.std(obs, ddof=1)

    # 防止除零
    if std_obs == 0 or mean_obs == 0:
        return np.nan, np.nan, np.nan, np.nan

    # 相关系数 r
    r = np.corrcoef(sim, obs)[0, 1]

    # α 和 β
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    # KGE
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    return kge, r, alpha, beta