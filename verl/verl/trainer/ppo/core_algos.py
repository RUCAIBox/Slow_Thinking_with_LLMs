# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


class EntController:
    def __init__(
        self,
        init_ent_coef,
        max_ent_coef,
        min_ent_coef,
        delta_ent_coef,
        target_ent,
        use_adapt_ent,
    ):
        self.value = max_ent_coef / 8
        self.max_value = max_ent_coef
        self.min_value = min_ent_coef
        self.delta_ent_coef = delta_ent_coef
        self.target_ent = target_ent
        self.use_adapt_ent = True  # use_adapt_ent
        self.entropy_loss_enabled = 0
        self.count = 0
        self.last_entropy = 0

    def update(self, current_ent):
        delta = current_ent - self.last_entropy  # 大于0则在上升，小于0则在下降
        percent = delta / 0.1
        if (
            self.target_ent - 0.05 > current_ent and delta < 0
        ):  # ent需要上升,但目前在下降
            self.value += (
                (self.target_ent - current_ent) // 0.05
            ) * self.delta_ent_coef
        elif (
            self.target_ent + 0.05 < current_ent and delta >= 0
        ):  # ent需要下降，但目前在上升
            self.value += (
                (self.target_ent - current_ent) // 0.05
            ) * self.delta_ent_coef
        elif delta < 0 and self.target_ent > current_ent:  # ent需要上升，目前在下降，
            self.value -= (
                ((self.target_ent - current_ent) // 0.025) * self.delta_ent_coef / 4
            )
        elif delta > 0 and self.target_ent < current_ent:  # ent需要下降，目前在上升
            self.value += (
                ((self.target_ent - current_ent) // 0.025) * self.delta_ent_coef / 4
            )
        elif (
            self.target_ent > current_ent
            and self.target_ent - 0.05 < current_ent
            and delta > 0
        ):  # ent需要上升,但目前在上升,需要减缓上升。系数需要减
            self.value = (
                (self.target_ent - current_ent) / self.target_ent
            ) * self.value
        elif (
            self.target_ent < current_ent
            and self.target_ent + 0.05 > current_ent
            and delta < 0
        ):  ##ent需要下降,但目前在下降,需要减缓下降。系数需要加
            self.value = (
                (self.target_ent - current_ent) / self.target_ent
            ) * self.value
        self.value = float(np.clip(self.value, self.min_value, self.max_value))
        self.entropy_loss_enabled = int(current_ent < self.target_ent)
        self.last_entropy = current_ent


#################
def get_kl_controller(config):
    if config.critic.kl_ctrl.type == "fixed":
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == "adaptive":
        assert (
            config.kl_ctrl.horizon > 0
        ), f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
        kl_ctrl = AdaptiveKLController(
            init_kl_coef=config.critic.kl_ctrl.kl_coef,
            target_kl=config.critic.kl_ctrl.target_kl,
            horizon=config.critic.kl_ctrl.horizon,
        )
    else:
        raise ValueError("Unknown kl_ctrl type")

    return kl_ctrl


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    mask_negative_adv: bool = False,
    epsilon: float = 1e-6,
    mask_teacher_output=False,
    mask_teacher=None,
    plus=1,
):
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if mask_teacher_output:
                if mask_teacher[i].item() != 0:
                    id2score[index[i]].append(scores[i])
            else:
                id2score[index[i]].append(scores[i])
        # if mask_teacher_output:
        #     for uid, metric_vals in id2score.items():
        #         metric_vals = metric_vals[:-mask_teacher_output_num]
        #         id2score[uid] = metric_vals  # 更新字典中的值
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask
        if mask_negative_adv:
            scores = torch.where(scores < 0, torch.tensor(0.0), scores)
        if plus > 1:
            scores = torch.where(scores > 0, scores * plus, scores)

    return scores, scores


def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[
                    index[i]
                ] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


import torch

import torch


def calculate_ppl(batch):
    log_probs = batch.batch["old_log_probs"]  # [B, L]
    B, L = log_probs.shape
    attention_mask = batch.batch["attention_mask"][:, -L:]  # [B, L]
    uids = batch.non_tensor_batch["uid"]  # [B]

    # Step 1: 计算每个样本的 log_ppl 和 ppl
    log_ppl = -(
        (log_probs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
    )  # [B]
    ppl = torch.exp(log_ppl)

    # Step 2: 构建 uid -> log_ppl 列表
    uid_to_logppls = {}
    uid_to_indices = {}  # 保存样本在 batch 中的位置
    for i, uid in enumerate(uids):
        uid_val = uid.item() if isinstance(uid, torch.Tensor) else uid
        uid_to_logppls.setdefault(uid_val, []).append(log_ppl[i].item())
        uid_to_indices.setdefault(uid_val, []).append(i)

    # Step 3: 基于 IQR 筛掉异常值，并计算 mean/std
    uid_to_mean_logppl = {}
    uid_to_std_logppl = {}
    iqr_mask = torch.ones(
        B, dtype=torch.float32, device=log_ppl.device
    )  # 默认都是 1（有效）

    for uid, values in uid_to_logppls.items():
        idxs = uid_to_indices[uid]
        values_tensor = torch.tensor(values)

        # IQR 筛选离群值
        q1 = torch.quantile(values_tensor, 0.25)
        q3 = torch.quantile(values_tensor, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values_tensor >= lower) & (values_tensor <= upper)

        # 更新 ppl_iqr_mask
        for flag, idx in zip(mask, idxs):
            iqr_mask[idx] = float(flag.item())  # 0 or 1

        # 用有效值重新计算 mean/std
        clean_vals = values_tensor[mask]
        if len(clean_vals) < 1:
            clean_vals = values_tensor  # fallback：全部是离群，就用原始值

        uid_to_mean_logppl[uid] = clean_vals.mean().item()
        std = clean_vals.std(unbiased=False).item()
        uid_to_std_logppl[uid] = std if std > 1e-6 else 1.0
        # print("\n========== [IQR 检查 - UID: {}] ==========".format(uid))
        # print(f"原始 log_ppl 值列表:\n{values_tensor.tolist()}")
        # print(f"Q1（25% 分位）: {q1:.4f}")
        # print(f"Q3（75% 分位）: {q3:.4f}")
        # print(f"IQR（Q3 - Q1） : {iqr:.4f}")
        # print(f"异常值判断区间 : [{lower:.4f}, {upper:.4f}]")
        # print(f"是否为正常样本 : {mask.tolist()}")
        # print(f"有效样本数     : {mask.sum().item()} / {len(values_tensor)}")
        # print(f"clean_vals 均值: {clean_vals.mean().item():.4f}")
        # print(f"clean_vals 标准差: {std:.4f}")
        # print("==========================================\n")

    # Step 4: 构建 mean/std 向量
    mean_logppl = torch.tensor(
        [
            uid_to_mean_logppl[uid.item() if isinstance(uid, torch.Tensor) else uid]
            for uid in uids
        ],
        device=log_ppl.device,
    )
    std_logppl = torch.tensor(
        [
            uid_to_std_logppl[uid.item() if isinstance(uid, torch.Tensor) else uid]
            for uid in uids
        ],
        device=log_ppl.device,
    )

    # Step 5: 标准化 score，并展开
    score = (log_ppl - mean_logppl) / std_logppl  # [B]
    score = score.unsqueeze(1).expand_as(attention_mask)  # [B, L]
    # print(score)
    # Step 6: 写入 batch
    batch.batch["ppl"] = ppl
    batch.batch["ppl_weight"] = score
    batch.batch["ppl_iqr_mask"] = iqr_mask  # shape: [B]

    return batch

def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, eos_mask: torch.Tensor, gamma: torch.Tensor
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    eos_mask: torch.Tensor,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (
            (token_level_rewards * eos_mask)
            .flip(dims=[-1])
            .cumsum(dim=-1)
            .flip(dims=[-1])
        )
        advantages = (
            returns
            - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask
        )

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


import torch
import torch.nn.functional as F


def sigmoid_penalty(p, mask, clip_extend, k=10):
    """
    Sigmoid-based decreasing penalty: p in [0, 1] → penalty in [0.08, 0]
    """
    # p = torch.clamp(p, 0, 1)
    # s = 1 / (1 + torch.exp(k * (p - 0.5)))
    # s_min = torch.min(s)
    # s_max = torch.max(s)
    # normalized = (s - s_min) / (s_max - s_min + 1e-8)
    # triplets = list(zip(p.tolist(), s.tolist(), normalized.tolist()))
    # triplets = [
    #     (pi, si, ni)
    #     for row_p, row_s, row_n in zip(p.tolist(), s.tolist(), normalized.tolist())
    #     for pi, si, ni in zip(row_p, row_s, row_n)
    # ]
    # # 准备保存的数据
    # save_dict = {
    #     "k": k,
    #     "s_min": s_min.item(),
    #     "s_max": s_max.item(),
    #     "triplets": triplets[:2000]  # 每个元素是 (p, s, normalized)
    # }

    # # 保存到 JSON 文件
    # import json
    # with open("/share/project/zhipengchen/tmp.json", "w") as f:
    #     json.dump(save_dict, f, indent=2)
    # print(a)
    s = torch.sigmoid(-k * (p - 0.5))  # decreasing in p
    s_min = torch.min(s)
    s_max = torch.max(s)
    normalized = (s - s_min) / (s_max - s_min + 1e-8)
    # normalized=s

    if clip_extend > 0:
        return clip_extend * normalized  # [0, clip_extend]
    else:
        return -clip_extend * (1 - normalized)



def compute_policy_loss(
    responses,
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    whether_use_location=False,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    use_token_level_loss=False,
    mask_prob1=False,
    entropy=None,
    use_ppl=False,
    ppl_weight=None,
    ppl=None,
    ppl_iqr_mask=None,
    alpha=0.01,
    add_loaction_bonus=False,
    positive_only=False,
    negative_only=False,
    return_pglosses=False,
    use_ppl_high=False,
    global_step=None
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        use_token_level_loss: (bool)
            Whether to use token level loss
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    if add_loaction_bonus:
        # print(global_step,whether_use_location)
        if whether_use_location:
            
            B, L = eos_mask.shape
            # 归一化位置 [0, 1]
            cumsum = torch.cumsum(eos_mask, dim=1) * eos_mask
            lengths = eos_mask.sum(dim=1, keepdim=True).clamp(min=1)
            relative_pos = (cumsum - 1) / lengths  # [B, L]
            sign = torch.sign(advantages)
            k = 15.0
            pos_mask = (advantages > 0).float()
            neg_mask = (advantages < 0).float()
            pos_decay = torch.sigmoid(k * (relative_pos - 0.5)) * eos_mask   # 负优势惩罚靠后更大
            neg_decay= torch.sigmoid(k * (relative_pos - 0.5)) * eos_mask
            raw_bonus = (pos_decay * pos_mask + neg_decay * neg_mask) #* entropy
            raw_bonus = alpha*raw_bonus.detach()
            bonus_factor = torch.clamp(raw_bonus, max=(0.5 * advantages.abs()))
            # print("bonus_factor", bonus_factor)
            advantages = advantages + bonus_factor* sign#*use_bonus_mask #* bonus_mask
    if use_ppl is True and use_ppl_high is False:
        weight = (1 - alpha * ppl_weight.detach()).clamp(0.8, 1.2)
        print(">> weight 第一列:", weight[:, 0])
    if use_ppl is True and use_ppl_high is True:
        # clamped_weight = ppl_weight.detach().clamp(-2, 2)
        weight = (1 + alpha * ppl_weight.detach()).clamp(0.8, 1.2)
        advantages = advantages * weight
        # adv_sign = torch.sign(advantages)  # +1 或 -1
        # advantages = advantages + alpha * clamped_weight * adv_sign
    eos_mask_modified = eos_mask.clone()
    if positive_only:
        print("positive_only")
        pos_mask = (advantages > 0).float()  # shape: [B, L]
        eos_mask_modified = eos_mask_modified * pos_mask
    # 可选：只保留优势为负的 token
    if negative_only:
        print("negative_only")
        neg_mask = (advantages < 0).float()  # shape: [B, L]
        eos_mask_modified = eos_mask_modified * neg_mask
    if mask_prob1:
        low_conf_mask = (
            log_prob <= torch.log(torch.tensor(0.95, device=log_prob.device))
        ).float()
        eos_mask_modified = eos_mask_modified * low_conf_mask
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(
        pg_losses1, pg_losses2
    )
    if use_token_level_loss:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask_modified)
    else:
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
        pg_loss = torch.mean(pg_loss)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    if return_pglosses:
        return pg_loss, pg_clipfrac, ppo_kl, pg_losses
    else:
        return pg_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_ppl_penality(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    use_token_level_loss=False,
    mask_prob1=False,
    entropy=None,
    use_ppl=False,
    ppl_weight=None,
    ppl=None,
    ppl_iqr_mask=None,
    alpha=0.01,
    add_loaction_bonus=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        use_token_level_loss: (bool)
            Whether to use token level loss
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    if add_loaction_bonus:
        B, L = eos_mask.shape
        # 为每个样本生成从 0 开始的累加位置，只对 eos_mask 为 1 的地方有效
        # 累加索引（位置） 例如: [0,1,2,...]，其余为0
        cumsum = torch.cumsum(eos_mask, dim=1) * eos_mask  # [B, L]
        # 有效 token 数（每行有多少个1）
        lengths = eos_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        # 相对位置（归一化到 0~1）
        relative_pos = (cumsum - 1) / lengths  # [B, L]，注意 -1 是因为索引从 0 开始
        # 使用 sigmoid 衰减函数（越靠前越接近1，越靠后越接近0）
        decay = torch.sigmoid(-15 * (relative_pos - 0.5)) * eos_mask  # [B, L]
        # 位置加成系数
        bonus_factor = 1 + 0.01 * decay  # [B, L]
        # 应用加成到 advantages
        advantages = advantages * bonus_factor
    if use_ppl:
        # advantages = advantages * (1 - alpha * ppl_weight.clamp(-2, 2))
        clamped_weight = ppl_weight.detach().clamp(-2, 2)
        adv_sign = torch.sign(advantages)  # +1 或 -1
        advantages = advantages - alpha * clamped_weight * adv_sign
        eos_mask_modified = eos_mask * ppl_iqr_mask.unsqueeze(1)
    else:
        eos_mask_modified = eos_mask.clone()
    if mask_prob1:
        eos_mask_modified = verl_F.apply_entropy_mask_to_response_mask_batch(
            response_mask=eos_mask_modified,
            token_entropy=entropy,
            low_entropy_mask_ratio=0.8,
        )
    # valid_token_count = eos_mask_modified.sum().item()
    # total_token_count = eos_mask_modified.numel()
    # print(f"[eos_mask_modified] 有效 token 数: {int(valid_token_count)} / {total_token_count} ({valid_token_count / total_token_count:.2%})")
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)

    if use_token_level_loss:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask_modified)
    else:
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
        pg_loss = torch.mean(pg_loss)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_policy_loss_oat(
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    use_token_level_loss=False,
    ratio_max=None,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122
    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_low: (float)
            The lower clip range used in PPO.
        cliprange_high: (float)
            The higher clip range used in PPO.
        use_token_level_loss: (bool)
            Whether to use token level loss
    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            the fraction of policy gradient loss being clipped
        ppo_kl: (float)
            the estimated KL divergence between the latest updating policy and the old sampling policy
    """
    seq_len_per_sample = torch.clamp(torch.sum(eos_mask, dim=1), min=1.0)
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    # ppo_kl = verl_F.masked_sum(-negative_approx_kl, eos_mask) / 8192
    ppo_kl = (torch.sum(-negative_approx_kl * eos_mask, dim=1) / 8192).mean()

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    pg_losses = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)

    # Extra clipping: remove large ratio with negative advantage
    if ratio_max is not None:
        mask_extra_clip = (ratio >= ratio_max) & (advantages < 0)
        pg_losses = torch.where(mask_extra_clip, torch.zeros_like(pg_losses), pg_losses)

    if use_token_level_loss:
        pg_loss = (torch.sum(pg_losses * eos_mask, dim=1) / 8192).mean()
    else:
        pg_loss = torch.sum(pg_losses * eos_mask, dim=1) / seq_len_per_sample
        pg_loss = torch.mean(pg_loss)

    # pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)
    pg_clipfrac = (
        torch.sum(torch.gt(pg_losses2, pg_losses1).float() * eos_mask, dim=1) / 8192
    ).mean()
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(
        vpreds, values - cliprange_value, values + cliprange_value
    )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def sft_loss_from_teacher(logprob: torch.FloatTensor, delta) -> torch.FloatTensor:
    epsilon = 1e-8  # 一个小的常数
    prob = torch.exp(logprob)  # 从 log 概率获取原始的概率
    prob = torch.clamp(prob, min=epsilon)  # 防止概率为 0
    plogp = prob * logprob
    return delta * plogp


def kl_penalty(
    logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty
) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == "low_var_kl":
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
