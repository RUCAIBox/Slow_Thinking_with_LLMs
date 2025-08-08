# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#x
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contain small torch utilities
"""

from typing import Dict, Union, List, Optional

import torch
import torch.distributed
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False

def gather_from_labels(data, label):
    """Gather the label from data. The value in label should be [0, vocab_size)

    Args:
        data: (..., vocab_size)
        label (torch.IntTensor) : (...,)

    Returns:

    """

    output = torch.gather(data, -1, label.unsqueeze(-1)).squeeze(-1)
    return output


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    if FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output

def logprobs_from_logits_label_smoothing(
    logits, 
    labels, 
    p_threshold=0.95,
    topk=128
):
    B, V = logits.shape
    device = logits.device
    # 保证 logits 是 float32，避免类型不一致
    if logits.dtype not in [torch.float32, torch.float64]:
        logits = logits.float()
    # 🔹 仅对 topk logits 计算 softmax：保留梯度
    topk_logits, topk_indices = torch.topk(logits, k=topk, dim=-1)  # (B, topk)
    topk_probs = F.softmax(topk_logits, dim=-1)  # (B, topk)
    # 累计概率
    cumulative_probs = torch.cumsum(topk_probs, dim=-1)  # (B, topk)
    mask = cumulative_probs <= p_threshold
    mask[:, 0] = True  # 每行至少保留一个 token

    # 找出 label 是否在 topk 中
    labels_exp = labels.unsqueeze(1)  # (B, 1)
    is_label = topk_indices == labels_exp  # (B, topk)
    mask = mask & (~is_label)  # 去掉 label 的位置

    # masked log prob：保留梯度路径
    selected_probs = topk_probs * mask  # 仍有梯度
    # 加一个 epsilon 避免 log(0)，不会影响梯度结构
    log_selected_probs = torch.where(
        mask, torch.log(selected_probs + 1e-9), torch.zeros_like(selected_probs)
    )

    num_selected = mask.sum(dim=1).clamp(min=1)
    avg_logprob = log_selected_probs.sum(dim=1) / num_selected  # (B,)

    return avg_logprob
import torch
import numpy as np
from scipy.sparse import coo_matrix
# def similarity_from_logits(logits, labels, p_threshold, sim_matrix, top_k=128):
#     """
#     计算 logits 中每个词在给定 p_threshold 下的相似度。

#     Args:
#         logits (torch.Tensor): 模型的输出，形状为 (B, V)，其中 B 是批量大小，V 是词汇表大小。
#         labels (torch.Tensor): 真实标签，形状为 (B,)。
#         p_threshold (float): 筛选概率的阈值，用于限制哪些 token 需要被考虑。
#         sim_matrix (scipy.sparse.coo_matrix): 相似度矩阵，形状为 (V, V)，表示 token 之间的相似度。
#         top_k (int, optional): 用于计算 Top-k 相似度，默认是 128。

#     Returns:
#         torch.Tensor: 每个样本的相似度统计值，以及对应的 `col_indices`。
#     """
#     B, V = logits.shape  # 批量大小 B 和 词汇表大小 V
#     device = logits.device

#     # 保证 logits 是 float32 类型
#     if logits.dtype not in [torch.float32, torch.float64]:
#         logits = logits.float()

#     # 🔹 仅对 top_k logits 计算 softmax：保留梯度
#     topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # (B, top_k)
#     topk_probs = torch.softmax(topk_logits.float(), dim=-1)  # (B, top_k)

#     # 筛选出概率大于 p_threshold 的位置
#     cumulative_probs = torch.cumsum(topk_probs, dim=-1)  # (B, top_k)
#     mask = cumulative_probs <= p_threshold
#     mask[:, 0] = True  # 每行至少保留一个 token

#     # 标签的扩展形状为 (B, 1)
#     labels_exp = labels.unsqueeze(1)  # (B, 1)
    
#     # 找到 top_k 词中，标签是否在其中
#     is_label = topk_indices == labels_exp  # (B, top_k)
#     mask = mask & (~is_label)  # 排除标签所在的地方

#     # 如果某行没有符合条件的 token（即 mask 全部为 0），则直接设置相似度为 -1
#     similarity_values = []
#     col_indices_values = []  # 用于存储 col_indices
#     for b, (label, row_mask) in enumerate(zip(labels, mask)):
#         if row_mask.sum() == 0:  # 说明没有符合条件的 token（只剩标签本身）
#             similarity_values.append(-1)
#             col_indices_values.append(-1)  # 如果没有符合条件的 token，则存储 -1
#             continue

#         # 获取选中的最大概率词的索引
#         max_probs, max_indices = torch.max(topk_probs[b] * row_mask, dim=-1)

#         # 获取最大概率词的原始位置，最大概率的位置对应原始 logits 的位置
#         col_indices = torch.gather(topk_indices[b], dim=-1, index=max_indices.unsqueeze(-1))

#         # 查找相似度
#         i, j = sorted([label.item(), col_indices.item()])  # i 是更小的值，j 是更大的值

#         # 使用 np.where 查找匹配的行列索引
#         indices = np.where((sim_matrix.row == i) & (sim_matrix.col == j))

#         # 如果找到匹配的元素
#         if indices[0].size > 0:
#             sim_value = sim_matrix.data[indices[0][0]]  # 获取对应的值
#         else:
#             sim_value = 0.6  # 如果没有找到对应的值，则设为默认值 0.6
        
#         similarity_values.append(sim_value)
#         col_indices_values.append(col_indices.item())  # 存储 col_indices

#     # 将相似度值转化为 tensor 并返回
#     return torch.tensor(similarity_values, device=device), torch.tensor(col_indices_values, device=device)

###############################旧版的
# import json
# def similarity_from_logits(logits, labels, p_threshold, top_k=128):
#     B, V = logits.shape  # 批量大小 B 和 词汇表大小 V
#     device = logits.device

#     # 保证 logits 是 float32 类型
#     if logits.dtype not in [torch.float32, torch.float64]:
#         logits = logits.float()

#     # 🔹 仅对 top_k logits 计算 softmax：保留梯度
#     topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # (B, top_k)
#     topk_probs = torch.softmax(topk_logits.float(), dim=-1)  # (B, top_k)

#     # 筛选出概率大于 p_threshold 的位置
#     cumulative_probs = torch.cumsum(topk_probs, dim=-1)  # (B, top_k)
#     mask = cumulative_probs <= p_threshold
#     mask[:, 0] = True  # 每行至少保留一个 token

#     # 标签的扩展形状为 (B, 1)
#     labels_exp = labels.unsqueeze(1)  # (B, 1)
    
#     # 找到 top_k 词中，标签是否在其中
#     is_label = topk_indices == labels_exp  # (B, top_k)
#     mask = mask & (~is_label)  # 排除标签所在的地方

#     similarity_values = []
#     col_indices_values = []  # 用于存储 col_indices
#     col_indices_probs = []  # 用于存储 col_indices 的对应概率

#     # 缓存相似度矩阵
#     similarity_matrix_cache = {}

#     # 预分配 tensor
#     all_similarity_values = torch.full((B,), -1, device=device)  # 默认相似度为 -1
#     all_col_indices_values = torch.full((B, 10), -1, dtype=torch.long, device=device)
#     all_col_indices_probs = torch.zeros((B, 10), dtype=torch.float32, device=device)

#     for b in range(B):
#         row_mask = mask[b]
#         if row_mask.sum() == 0:  # 说明没有符合条件的 token（只剩标签本身）
#             continue

#         # 获取符合条件的 token 的原始位置和对应概率
#         valid_indices = row_mask.nonzero(as_tuple=True)[0]  # 获取符合条件的 token 的索引
#         valid_probs = topk_probs[b, valid_indices]  # 获取对应的概率
#         valid_col_indices = topk_indices[b, valid_indices]  # 获取对应的原始位置

#         # 按照概率从大到小排序
#         sorted_probs, sorted_indices = torch.sort(valid_probs, descending=True)
#         sorted_col_indices = valid_col_indices[sorted_indices]

#         # 更新对应的 `col_indices_values` 和 `col_indices_probs`
#         all_col_indices_values[b, :min(10,sorted_col_indices.size(0))] = sorted_col_indices[:min(10,sorted_col_indices.size(0))]
#         all_col_indices_probs[b, :min(10,sorted_probs.size(0))] = sorted_probs[:min(10,sorted_probs.size(0))]

#         # 获取最大概率词的索引
#         max_probs, max_indices = torch.max(topk_probs[b] * row_mask, dim=-1)

#         # 获取最大概率词的原始位置
#         col_indices = torch.gather(topk_indices[b], dim=-1, index=max_indices.unsqueeze(-1))

#         # 查找相似度
#         tmp, indices= torch.sort(torch.tensor([labels[b].item(), col_indices.item()], device=device))  # i 是更小的值，j 是更大的值
#         i=tmp[0]
#         j=tmp[1]
#         assert i.item() != j.item() 
#         t = i.item() % 1000
        
#         if t not in similarity_matrix_cache:
#             with open(f"/share/project/zhipengchen/dj/RFT/metrix/selected_token_similarity_{t}.json", "r", encoding="utf-8") as f:
#                 similarity_matrix_cache[t] = json.load(f)
#         sim_matrix = similarity_matrix_cache[t]
#         sim_value = sim_matrix.get(str(i.item()), {}).get(str(j.item()), 0.6)
        
#         # 更新相似度
#         all_similarity_values[b] = sim_value

#     # 返回最终结果
#     return all_similarity_values.detach(), all_col_indices_values.detach(), all_col_indices_probs.detach()


def similarity_from_logits(similarity_matrix,logits, labels, p_threshold, top_k=128,save_num=30):
    B, V = logits.shape  # Batch size B and vocabulary size V
    device = logits.device

    # Ensure logits are float32
    if logits.dtype not in [torch.float32, torch.float64]:
        logits = logits.float()

    # 🔹 Only compute softmax for top_k logits: keep gradients
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # (B, top_k)
    topk_probs = torch.softmax(topk_logits.float(), dim=-1)  # (B, top_k)

    # Filter positions where cumulative probability <= p_threshold
    cumulative_probs = torch.cumsum(topk_probs, dim=-1)  # (B, top_k)
    mask = cumulative_probs <= p_threshold
    mask[:, 0] = True  # Keep at least one token per row

    # Expand labels to shape (B, 1)
    labels_exp = labels.unsqueeze(1)  # (B, 1)

    # Identify positions where the label is among the top_k
    is_label = topk_indices == labels_exp  # (B, top_k)
    mask = mask & (~is_label)  # Exclude the position where label is present (B, top_k)

    # Prepare output tensors
    all_similarity_values = torch.full((B,), 0, device=device)  # Default similarity is 0
    all_col_indices_values = torch.full((B, save_num), -1, dtype=torch.long, device=device)  # Store indices
    all_col_sim_values = torch.full((B, save_num), -1, dtype=torch.float32, device=device) 
    all_col_indices_probs = torch.full((B, save_num), -1,dtype=torch.float32, device=device)  # Store probabilities

    # Parallelize the loop logic for each batch element
    valid_mask = mask.clone()  # (B, top_k)
    valid_mask_sum = valid_mask.sum(dim=-1)  # Sum of valid tokens per batch (B,)

    # Find valid indices where mask is True
    valid_indices = valid_mask.nonzero(as_tuple=True)  # (num_valid_tokens, 2) -> (B * top_k, 2)

    # Get valid probabilities and indices in one go
    valid_probs = topk_probs[valid_indices[0], valid_indices[1]]  # (num_valid_tokens,)
    valid_col_indices = topk_indices[valid_indices[0], valid_indices[1]]  # (num_valid_tokens,)

    # Sort valid probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(valid_probs, descending=True)

    # Re-order valid_indices[0] and valid_indices[1] based on sorted_indices
    sorted_valid_indices_0 = valid_indices[0][sorted_indices]  # Reorder row indices
    sorted_valid_indices_1 = valid_indices[1][sorted_indices]  # Reorder column indices

    # Get sorted column indices based on sorted probabilities
    sorted_col_indices = valid_col_indices[sorted_indices]

    # Fill in the output tensors in parallel
    for b in range(B):
        # For each batch element, fill the corresponding row
        batch_mask = valid_mask[b]
        if batch_mask.sum() == 0:  # No valid token left
            continue
        
        # Extract the top-k indices and probs for the current batch
        valid_indices_b = sorted_valid_indices_0 == b  # Get valid positions for batch b
        sorted_probs_b = sorted_probs[valid_indices_b]  # Get probabilities for batch b
        sorted_col_indices_b = sorted_col_indices[valid_indices_b]  # Get sorted column indices for batch b

        # Sort probabilities and indices for the current batch and get top-10
        # sorted_probs_b, sorted_indices_b = torch.sort(sorted_probs_b, descending=True)

        # Update the column indices and probabilities
        top_10_indices = sorted_col_indices_b[:save_num]
        top_10_probs = sorted_probs_b[:save_num]
        
        all_col_indices_values[b, :top_10_indices.size(0)] = top_10_indices
        all_col_indices_probs[b, :top_10_probs.size(0)] = top_10_probs
    ##########TODO：从这里补全代码。npz是一个上三角稀疏矩阵。我需要你：
    # Vectorized version of the following loop
    # valid_mask = all_col_indices_values != -1
    # valid_col_indices = all_col_indices_values[valid_mask] #(num_valid_tokens,)
    # valid_probs = all_col_indices_probs[valid_mask] #(num_valid_tokens,)
    # labels_expanded = labels.repeat_interleave(valid_mask.sum(dim=-1)) #(num_valid_tokens,)

    # # Compute row and column indices using broadcasted operations
    # row_indices = torch.minimum(valid_col_indices, labels_expanded)  # Use minimum as row
    # col_indices = torch.maximum(valid_col_indices, labels_expanded)  # Use maximum as column
    # M, N = similarity_matrix.shape  # 获取矩阵的形状
    # row_indices = torch.clamp(row_indices, min=0, max=M-1)
    # col_indices = torch.clamp(col_indices, min=0, max=N-1)
    # row_indices = row_indices.clone().detach()
    # col_indices = col_indices.clone().detach()
    # # 转换为 NumPy 数组
    # row_indices_np = row_indices.cpu().numpy()
    # col_indices_np = col_indices.cpu().numpy()
    # row_indices_np = np.array(row_indices_np)  # 或者 row_indices.cpu().numpy()
    # col_indices_np = np.array(col_indices_np) 
    # matrix_vals = similarity_matrix[row_indices_np, col_indices_np] #(num_valid_tokens,)
    # matrix_vals = torch.tensor(matrix_vals, device=device)
    # matrix_vals = torch.clamp(matrix_vals, min=0.6)
    # matrix_vals = matrix_vals.view(-1) 
    similarity_score = torch.ones(B, device=device)  # Initialize the similarity score tensor

    # # Assign the weighted sum to the correct batch element
    # counts = valid_mask.sum(dim=1)  # (B,)
    
    # # 初始化结果张量
    # similarity_score = torch.ones(B, device=device)
    
    # # 对每个batch元素进行处理
    # start_idx = 0
    # for b in range(B):
    #     end_idx = start_idx + counts[b]
    #     if counts[b] > 0:
    #         # 获取当前batch的有效相似度值和概率
    #         batch_vals = matrix_vals[start_idx:end_idx]
    #         batch_probs = valid_probs[start_idx:end_idx]  
    #         # 存储到输出张量中
    #         all_col_sim_values[b, :counts[b]] = batch_vals  
    #         # 计算加权平均值
    #         weighted_sum = (batch_vals * batch_probs).sum()
    #         total_prob = batch_probs.sum()
    #         similarity_score[b] = weighted_sum / total_prob if total_prob > 0 else 0
    #     start_idx = end_idx
    return similarity_score.detach(), all_col_indices_values.detach(), all_col_indices_probs.detach(),all_col_sim_values.detach()

# import torch
# import json

# def similarity_from_logits(logits, labels, p_threshold, top_k=128, cache_dir="/share/project/zhipengchen/dj/RFT/metrix/selected_token_similarity_"):
#     B, V = logits.shape  # 批量大小 B 和 词汇表大小 V
#     device = logits.device

#     # 保证 logits 是 float32 类型
#     if logits.dtype not in [torch.float32, torch.float64]:
#         logits = logits.float()

#     # 🔹 仅对 top_k logits 计算 softmax：保留梯度
#     topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)  # (B, top_k)
#     topk_probs = torch.softmax(topk_logits.float(), dim=-1)  # (B, top_k)

#     # 筛选出概率大于 p_threshold 的位置
#     cumulative_probs = torch.cumsum(topk_probs, dim=-1)  # (B, top_k)
#     mask = cumulative_probs <= p_threshold
#     mask[:, 0] = True  # 每行至少保留一个 token

#     # 标签的扩展形状为 (B, 1)
#     labels_exp = labels.unsqueeze(1)  # (B, 1)
    
#     # 找到 top_k 词中，标签是否在其中
#     is_label = topk_indices == labels_exp  # (B, top_k)
#     mask = mask & (~is_label)  # 排除标签所在的地方

#     # 获取所有符合条件的 token 的索引
#     valid_mask = mask.bool()  # (B, top_k)
#     valid_indices = valid_mask.nonzero(as_tuple=True)  # 获取所有符合条件的 token 的索引 (B, valid_size)
#     valid_probs = topk_probs[valid_indices[0], valid_indices[1]]  # 获取对应的概率
#     valid_col_indices = topk_indices[valid_indices[0], valid_indices[1]]  # 获取对应的原始位置

#     sorted_probs, sorted_indices = torch.sort(valid_probs, descending=True)
#     sorted_col_indices = valid_col_indices[sorted_indices]
#     max_probs = sorted_probs[:, 0]  # 最大概率
#     max_indices = sorted_indices[:, 0]  # 最大概率词的索引
#     col_indices = sorted_col_indices[:, 0]  # 直接从排序后的结果中提取最大概率词的原始位置
#     row_indices = labels.unsqueeze(1)  # (B, 1), 需要与 col_indices 进行比较
#     row_indices_expanded = row_indices.expand_as(col_indices)  # (B, top_k)
#     all_similarity_values = torch.full((B,), -1, device=device)  # 默认相似度为 -1
#     all_col_indices_values = sorted_col_indices[:,:10]#torch.full((B, 10), -1, dtype=torch.long, device=device)
#     all_col_indices_probs = sorted_probs[:,:10]#torch.zeros((B, 10), dtype=torch.float32, device=device)
#     similarity_matrix_cache = {}
#     # 获取所有需要的唯一 token
#     unique_tokens = (labels % 1000).unique()

#     # 批量加载所有相似度矩阵
#     # for t in unique_tokens:
#     #     try:
#     #         with open(f"{cache_dir}selected_token_similarity_{t.item()}.json", "r", encoding="utf-8") as f:
#     #             similarity_matrix_cache[t.item()] = json.load(f)
#     #     except Exception as e:
#     #         print(f"Error loading similarity matrix for token {t.item()}: {e}")
#     #         similarity_matrix_cache[t.item()] = {}  # 如果出错，使用空字典
#     # 批量计算相似度
#     for b in range(B):
#         i = row_indices_expanded[b]
#         j = col_indices[b]
#         # 如果 i 和 j 不是上三角部分，交换它们
#         if i > j:
#             i, j = j, i  # 交换 i 和 j

#         # 获取相似度矩阵
#         t = i.item() % 1000
#         sim_matrix = similarity_matrix_cache.get(t, {})

#         # 获取相似度
#         sim_value = sim_matrix.get(str(i.item()), {}).get(str(j.item()), 0.6)
#         all_similarity_values[b] = sim_value

#     # 返回最终结果
#     return all_similarity_values.detach(), all_col_indices_values.detach(), all_col_indices_probs.detach()

def apply_entropy_mask_to_response_mask_batch(
    response_mask: torch.Tensor, 
    token_entropy: torch.Tensor, 
    low_entropy_mask_ratio: float = 0.8
):
    if low_entropy_mask_ratio <= 0.0:
        return response_mask
    
    if low_entropy_mask_ratio > 1.0:
        low_entropy_mask_ratio = 1.0
    
    # Create a copy of response_mask to modify
    modified_mask = response_mask.clone()
    
    # Get all valid positions across the entire batch
    valid_positions = response_mask.bool()
    
    if not valid_positions.any():
        return modified_mask  # Return if no valid positions
    
    # Get entropy values for all valid positions
    valid_entropy = token_entropy[valid_positions]
    
    # Calculate how many tokens to mask out across the entire batch
    total_valid_tokens = valid_positions.sum().item()
    num_to_mask = max(0, int(total_valid_tokens * low_entropy_mask_ratio))
    
    if num_to_mask == 0:
        return modified_mask
    
    # Find indices of lowest entropy tokens among all valid positions
    _, low_entropy_indices = torch.topk(valid_entropy, num_to_mask, largest=False)
    
    # Convert back to batch and sequence indices
    valid_batch_indices, valid_seq_indices = torch.nonzero(valid_positions, as_tuple=True)
    tokens_to_mask_batch = valid_batch_indices[low_entropy_indices]
    tokens_to_mask_seq = valid_seq_indices[low_entropy_indices]
    
    # Mask out these low-entropy tokens
    modified_mask[tokens_to_mask_batch, tokens_to_mask_seq] = 0
    
    return modified_mask
def max_logprobs_from_negative_logits(logits, labels):
    device = logits.device
    if logits.dtype not in [torch.float32, torch.float64]:
        logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)                          # (B, V)
    max_indices = logits.argmax(dim=-1)                                # (B,)
    selected_log_probs = log_probs[torch.arange(logits.size(0)), max_indices]  # (B,)
    is_label = max_indices == labels                                   # (B,)
    return torch.where(is_label, torch.zeros_like(selected_log_probs), selected_log_probs)

def logprobs_from_logits_flash_attn(logits, labels):
    output = cross_entropy_loss(logits, labels)
    assert isinstance(
        output, tuple), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


def logprobs_from_logits_naive(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logpy = gather_from_labels(logp, labels)
    return logpy


def logprobs_from_logits_v2(logits: torch.FloatTensor, labels):
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits: torch.Tensor):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def masked_sum(values, mask, axis=None):
    """Compute mean of tensor with a masked values."""
    return (values * mask).sum(axis=axis)


def masked_mean(values, mask, axis=None,e=False,p=False):
    """Compute mean of tensor with a masked values."""
    sum_value=mask.sum(axis=axis)
    if e:
        sum_value = torch.where(sum_value == 0, torch.tensor(1e-6, device=sum_value.device), sum_value)
    # if p==True:
    #     print("**********************************************")
    #     print("values",values)
    #     print("mask",mask)
    #     print("sum_mask",sum_value)
    #     print(f"values * mask, all mask == 0: {torch.all(mask == 0)}", (values * mask).sum(dim=axis))
    #     print("**********************************************")
    return (values * mask).sum(axis=axis) / sum_value


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def get_eos_mask(response_id: torch.Tensor, eos_token: Union[int, List[int]] = 2, dtype=torch.int64):
    '''
    end of sentence token can be int or list: 1 or [1, 2]
    e.g. eos_token=1
    response_id: [0, 0, 2, 42, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1,  1, 1, 1, 0, 0]
    '''
    if isinstance(eos_token, int):
        eos_token = [eos_token]

    eos_mask = torch.zeros_like(response_id, dtype=torch.bool)
    for token in eos_token:
        eos_mask |= response_id.eq(token)

    eos_mask = eos_mask.long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask


def compute_grad_norm(model: nn.Module):
    total_grad_square = 0
    total_params = 0
    for param in model.parameters():
        if param.grad is not None:
            total_grad_square += torch.sum(torch.square(param.grad.detach())).item()
    return total_grad_square


def broadcast_dict_tensor(tensors: Union[Dict[str, torch.Tensor], TensorDict], src, group):
    """
    TODO: optimize this. Technically, we only need one broadcast
    """

    for key in tensors.sorted_keys:
        torch.distributed.broadcast(tensors[key], src=src, group=group, async_op=False)


def allgather_dict_tensors(tensors: Union[Dict[str, torch.Tensor], TensorDict], size, group, dim=0):
    """
    TODO: optimize this.
    - We can use async ops
    - We can use only one allgather
    Args:
        tensors:
        size:
        group:

    Returns:

    """
    if isinstance(tensors, TensorDict):
        is_tensor_dict = True
        tensors_as_dict = tensors.to_dict()
    else:
        tensors_as_dict = tensors
        is_tensor_dict = False

    output = {}
    sorted_keys = sorted(tensors_as_dict.keys())
    for key in sorted_keys:
        val = tensors_as_dict[key]
        output[key] = [torch.empty_like(val) for _ in range(size)]
        torch.distributed.all_gather(output[key], val, group=group, async_op=False)
        output[key] = torch.cat(output[key], dim=dim)

    if is_tensor_dict:
        output = TensorDict(source=output, batch_size=tensors.batch_size[0] * size)

    return output


def split_dict_tensor_into_batches(tensors: TensorDict, batch_size) -> List[TensorDict]:
    assert tensors.batch_size[0] % batch_size == 0, \
        f'input data batch size: {tensors.batch_size[0]}, split batch size: {batch_size}'
    return tensors.split(batch_size)


def pad_2d_list_to_length(response, pad_token_id, max_length=None):
    """
    pad a 2D list (e.g. responses, logprobs) to a 2D tensor.
    """
    response_length = max(len(sub_list) for sub_list in response)
    if max_length is not None and max_length > response_length:
        target_length = max_length
    else:
        target_length = response_length
    padded_response = [tuple(sub_list) + (pad_token_id,) * (target_length - len(sub_list)) for sub_list in response]
    tensor = torch.tensor(padded_response)
    return tensor


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, 'constant', pad_token_id)


from transformers import PreTrainedTokenizer


def tokenize_and_postprocess_data(prompt: str,
                                  tokenizer: PreTrainedTokenizer,
                                  max_length: int,
                                  pad_token_id: int,
                                  left_pad=True,
                                  truncation='error'):
    """
    input_data is the output from tokenizer.
    """
    assert truncation in ['left', 'right', 'error']

    input_data = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)

    input_ids = input_data['input_ids']
    attention_mask = input_data['attention_mask']

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(input_ids,
                                           max_seq_len=max_length,
                                           pad_token_id=pad_token_id,
                                           left_pad=left_pad)
        attention_mask = pad_sequence_to_length(attention_mask,
                                                max_seq_len=max_length,
                                                pad_token_id=0,
                                                left_pad=left_pad)
    elif sequence_length > max_length:
        if truncation == 'left':
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == 'right':
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == 'error':
            raise NotImplementedError(f'{sequence_length=} is larger than {max_length=}')
        else:
            raise NotImplementedError(f'Unknown truncation method {truncation}')

    return input_ids, attention_mask


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """ Remove the pad token.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[List[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        no_padding_batch.append((ids[len(ids) - mask.sum():]).cpu().numpy().tolist())
    return no_padding_batch


def log_probs_from_logits_response(input_ids, logits, response_length):
    """Compute the response log_probs from full logits. Note that logits = model(input_ids)

    Args:
        input_ids: [batch_size, seqlen]
        logits: [batch_size, seqlen, vocab_size]

    Returns:
        response_log_prob:
    """
    response_logits = logits[:, -response_length - 1:-1]
    response = input_ids[:, -response_length:]
    response_log_prob = logprobs_from_logits(logits=response_logits, labels=response)
    return response_log_prob


def log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Compute the log_probs from logits with rmpad logits and pad input. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size]
        response_length: int
    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


def log_probs_from_logits_all_rmpad(input_ids_rmpad, logits_rmpad, indices, batch_size, seqlen, response_length):
    """Compute the log_probs from logits with rmpad input_ids and logits. Note that
    logits_rmpad = model(input_ids_rmpad). For each sentences, there is a shift between
    logits and input_ids.
    The reason for this function to is to compute logprobs_from_logits in rmpad mode because it is memory-intensive
    for large vocab_size

    Args:
        input_ids_rmpad: [1, total_nnz]
        logits_rmpad: [total_nnz, vocab_size]
        indices: [total_nnz]
        batch_size: int
        seqlen: int
        response_length: int
    """
    from flash_attn.bert_padding import pad_input
    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # transpose back to [total_nnz, 1]
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)  # (total_nnz,)
    full_output = pad_input(hidden_states=full_log_probs_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen)
    output = full_output.squeeze(-1)[:, -response_length - 1:-1]  # [batch_size, response_length]
    return output


from transformers.generation.logits_process import (TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)


def post_process_logits(input_ids, logits, temperature, top_k, top_p):
    if temperature != 1.:
        logits = logits.div_(temperature)  # inplace operation to avoid OOM
    # TODO: add them back
    # if top_k is not None and top_k > 0:
    #     logits = TopKLogitsWarper(top_k=top_k)(input_ids, logits)
    # if top_p is not None and top_p < 1.0 and top_p > 0.0:
    #     logits = TopPLogitsWarper(top_p=top_p)(input_ids, logits)
    return logits


"""
Optimizer related
"""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        min_lr_ratio (:obj:`float`, `optional`, defaults to 0.0):
            The minimum lr ratio w.r.t the maximum.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    assert min_lr_ratio >= 0 and min_lr_ratio <= 1.
    coef = (1 - min_lr_ratio) * 0.5
    intercept = (1 + min_lr_ratio) * 0.5

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
        return max(0.0, x * coef + intercept)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):

    def lr_lambda(current_step):
        return min(1, float(current_step) / float(max(1, num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def prepare_decoder_attention_mask(attention_mask, input_shape, inputs_embeds):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype,
                                          tgt_len=input_shape[-1]).to(inputs_embeds.device)
        combined_attention_mask = (expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask +
                                   combined_attention_mask)

    return combined_attention_mask


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )
