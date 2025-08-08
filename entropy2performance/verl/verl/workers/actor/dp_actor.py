# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from collections import OrderedDict
from verl import DataProto
from verl.trainer.ppo import core_algos  # , load_matrix
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import (
    logprobs_from_logits,
    masked_mean,
    logprobs_from_logits_label_smoothing,
    max_logprobs_from_negative_logits,
    similarity_from_logits,
)
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FlatParameter
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

# # 0708: math flag add advantage
# from verl.trainer.ppo.load_matrix import _load_math_token_csr, _math_token_csr

# def load_similarity_matrix():
#     from scipy.sparse import load_npz
#     # 加载保存的压缩稀疏矩阵
#     compressed_matrix = load_npz("/share/project/zhipengchen/dj/RFT/code/compressed_selected_token_similarity.npz")
# similarity_matrix=load_similarity_matrix()
#####

# import json
# with open("/share/project/zhipengchen/dj/RFT/code/selected_token_similarity.json", "r", encoding="utf-8") as f:
#     npz= json.load(f)
# 打开 HDF5 文件


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get(
                "use_torch_compile", True
            )  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        return_smo_log_probs=False,
        p_threshold=0.95,
        return_negative_max_log_probs=False,
        return_similarity_score=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(
                    0, 1
                )  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(
                            rearrange(position_ids, "c b s ... -> (b s) c ..."), indices
                        )
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices,
                    ).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(
                    input_ids_rmpad, shifts=-1, dims=1
                )  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = (
                        ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        None,
                        self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(
                    0
                )  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(
                    logits_rmpad
                )  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(
                    logits=logits_rmpad, labels=input_ids_rmpad_rolled
                )
                if return_smo_log_probs:
                    smooth_logprobs = logprobs_from_logits_label_smoothing(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        p_threshold=p_threshold,
                    )
                if return_similarity_score:
                    # similarity_matrix = load_matrix.load_similarity_matrix()
                    similarity_matrix = "_"
                    (
                        similarity_values,
                        col_indices_values,
                        col_indices_probs,
                        all_col_sim_values,
                    ) = similarity_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        p_threshold=p_threshold,
                        similarity_matrix=similarity_matrix,
                    )

                # if return_negative_max_log_probs:
                #     max_log_probs=max_logprobs_from_negative_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(
                        log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    entropy_rmpad = gather_outpus_and_unpad(
                        entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                    if return_smo_log_probs:
                        smooth_logprobs = gather_outpus_and_unpad(
                            smooth_logprobs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if return_similarity_score:
                        similarity_values = gather_outpus_and_unpad(
                            similarity_values,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        col_indices_values = gather_outpus_and_unpad(
                            col_indices_values,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        col_indices_probs = gather_outpus_and_unpad(
                            col_indices_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        all_col_sim_values = gather_outpus_and_unpad(
                            all_col_sim_values,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(
                    hidden_states=entropy_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if return_smo_log_probs:
                    full_smooth_logprobs = pad_input(
                        hidden_states=smooth_logprobs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    smooth_logprobs = full_smooth_logprobs.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                if return_similarity_score:
                    # similarity_values,col_indices_values,col_indices_probs
                    full_similarity_values = pad_input(
                        hidden_states=similarity_values.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    similarity_values = full_similarity_values.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                    full_col_indices_values = pad_input(
                        hidden_states=col_indices_values.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    col_indices_values = full_col_indices_values.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                    full_col_indices_probs = pad_input(
                        hidden_states=col_indices_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    col_indices_probs = full_col_indices_probs.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                    full_all_col_sim_values = pad_input(
                        hidden_states=all_col_sim_values.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    all_col_sim_values = full_all_col_sim_values.squeeze(-1)[
                        :, -response_length - 1 : -1
                    ]
                # only return response part:
                entropy = full_entropy.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[
                    :, -response_length - 1 : -1
                ]  # (bsz, response_length)
            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[
                    :, -response_length - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                if return_smo_log_probs:
                    smooth_logprobs = logprobs_from_logits_label_smoothing(
                        logits, micro_batch["responses"], p_threshold=p_threshold
                    )
                if return_similarity_score:
                    similarity_matrix = "_"  # load_matrix.load_similarity_matrix()
                    (
                        similarity_values,
                        col_indices_values,
                        col_indices_probs,
                        all_col_sim_values,
                    ) = similarity_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        p_threshold=p_threshold,
                        similarity_matrix=similarity_matrix,
                    )
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            if return_smo_log_probs is False and return_similarity_score is False:
                return entropy, log_probs
            elif return_smo_log_probs is True:
                return entropy, log_probs, smooth_logprobs
            elif return_similarity_score is True:

                return (
                    entropy,
                    log_probs,
                    similarity_values,
                    col_indices_values,
                    col_indices_probs,
                    all_col_sim_values,
                )

    def _optimizer_step(self, global_step):
        assert self.config.grad_clip is not None

        # with FSDP.summon_full_params(self.actor_module,with_grads=True):
        #     if torch.distributed.get_rank() == 0:
        #         for name, param in self.actor_module.named_parameters():
        #             if param.grad is not None:
        #                 grad_shape = tuple(param.grad.shape)
        #                 grad_mean = param.grad.float().mean().item()
        #                 print(f"[debug] grad of {name}: shape={grad_shape}, mean={grad_mean:.6e}")
        #             else:
        #                 print(f"[debug] grad of {name}: None")
        # import torch.distributed as dist
        # target_substrings = ['attn', 'mlp']
        # for name, param in self.actor_module.named_parameters():
        #     if param.grad is None:
        #         continue
        #     if not any(sub in name for sub in target_substrings):
        #         continue
        #     if param.grad is not None:
        #         grad = param.grad.detach()
        #         grad_shape = tuple(grad.shape)
        #         grad_mean = grad.float().mean().item()
        #         grad_device = grad.device
        #         print(f"[debug] grad of {name}: shape={grad_shape}, mean={grad_mean:.6e}, device={grad_device}")
        def simplify_name(name: str) -> str:
            parts = name.split(".")
            simplified = []
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    simplified.append(f"layer{parts[i+1]}")
                elif (
                    part not in {"model", "layers", "weight", "bias"}
                    and not parts[i].isdigit()
                ):
                    simplified.append(part)
            return "_".join(simplified)

        # def sync_grad_metrics(grad_metrics,device):
        #     import torch.distributed as dist
        #     """
        #     Synchronize gradient metrics across all GPUs.
        #     Example: take the average of gradients across all GPUs.
        #     """
        #     for key, value in grad_metrics.items():
        #         # Assuming each metric is a scalar value (float)
        #         tensor = torch.tensor(value).to(device)
        #         dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        #         # Calculate the average (since we sum across all ranks)
        #         grad_metrics[key] = tensor.item() / dist.get_world_size()
        #     return grad_metrics
        def get_gradient_metrics(step, save_dir):
            import torch.distributed as dist
            import random
            import os
            import numpy as np
            from sklearn.decomposition import PCA
            from datetime import datetime

            target_substrings = ["attn", "mlp"]
            grad_metrics = {}
            os.makedirs(save_dir, exist_ok=True)
            for name, param in self.actor_module.named_parameters():
                if param.grad is None:
                    continue
                if not any(sub in name for sub in target_substrings):
                    continue
                if "bias" in name:
                    continue
                grad = param.grad.detach().clone()
                grad_mean = grad.mean().item()
                grad_var = grad.var(unbiased=False).item()
                grad_abs = grad.abs()
                grad_mean_abs = grad_abs.mean().item()
                grad_var_abs = grad_abs.var(unbiased=False).item()
                grad_sparsity = (grad_abs == 0).sum().item() / grad.numel()
                grad_device = grad.device
                grad_shape = tuple(grad.shape)
                # 日志记录
                name = simplify_name(name)
                name = name.replace("_fsdp_wrapped_module_", "")
                grad_metrics[f"grad/{name}/mean_abs"] = grad_mean_abs
                grad_metrics[f"grad/{name}/var_abs"] = grad_var_abs
                grad_metrics[f"grad/{name}/sparsity"] = grad_sparsity
                grad_metrics[f"grad/{name}/mean"] = grad_mean
                grad_metrics[f"grad/{name}/var"] = grad_var

                # 降维处理并保存
                # assert torch.isfinite(grad).all(), f"Non-finite gradient detected in {name}"
                # import torch.nn.functional as F
                # grad_flat = grad.view(1, 1, -1)  # shape: (1, 1, L)
                # if grad_flat.numel() < 512:
                #     print(f"[WARNING] Gradient too small for pooling in {name}: shape={grad.shape}")
                # else:
                #     grad_flat= F.adaptive_avg_pool1d(grad_flat, 512)  # shape: (1, 1, 512)
                # grad_metrics[f"g_{name}"] = grad_flat

            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # print(grad_metrics)
            rand_suffix = random.randint(1, 99999)
            filename = f"grad_metrics_step{step}_{rand_suffix}.npy"
            save_path = os.path.join(save_dir, filename)
            torch.save(grad_metrics, save_path)
            print(f"[INFO] grad_metrics saved to {save_path}")
            return True

        # if self.config.save_grad:
        #     grad_metrics=get_gradient_metrics(step=global_step,save_dir=self.config.save_grad_dir)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        self.actor_optimizer.step()
        if self.config.save_grad:
            return True, grad_norm
        else:
            return grad_norm
            # print(f"[debug] grad of {name}: shape={grad_shape}, mean={grad_mean:.6e},")

    def compute_log_prob(
        self, data: DataProto, return_negative_max_log_probs=False
    ) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(
                num_micro_batches
            )
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = (
                data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            )
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature
                )
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)
        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(
                0
            ), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def get_entropy_metric(self, batch, ent_ctrl_dict):
        entropy_coeff = ent_ctrl_dict["value"]
        entropy_loss_enabled = ent_ctrl_dict["entropy_loss_enabled"]
        advantages = batch["advantages"]
        response_length = advantages.size(1)
        old_log_probs = batch["old_log_probs"]
        attention_mask = batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        pg_grad_scale = torch.sqrt(
            verl_F.masked_mean(advantages * advantages, response_mask)
        ).item()
        entropy_grad_scale = torch.sqrt(
            verl_F.masked_mean(old_log_probs * old_log_probs, response_mask)
        ).item()
        entropy_grad_scale_with_coeff = (
            entropy_grad_scale * entropy_coeff * ent_ctrl_dict["entropy_loss_enabled"]
        )
        entropy = -1 * verl_F.masked_mean(old_log_probs, response_mask).item()
        metrics = {
            "actor/pg_grad_scale": pg_grad_scale,
            "actor/entropy_grad_scale": entropy_grad_scale,
            "actor/entropy_grad_scale_with_coeff": entropy_grad_scale_with_coeff,
            "actor/pg_grad_ratio": pg_grad_scale
            / (pg_grad_scale + entropy_grad_scale_with_coeff),
            "actor/entropy_grad_ratio": entropy_grad_scale_with_coeff
            / (pg_grad_scale + entropy_grad_scale_with_coeff),
            "actor/entropy_loss": entropy,
            "actor/entropy_loss_with_coeff": entropy
            * entropy_coeff
            * entropy_loss_enabled,
            "actor/entropy_coeff": entropy_coeff,
            "actor/entropy_coeff_realized": entropy_coeff * entropy_loss_enabled,
        }

        return metrics

    # def update_step(self,step):
    #     self.global_step=step
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error
        if self.config.adaptive_entropy.enabled or self.config.use_dynamic_clip.enabled:
            ent_ctrl_dict = data.meta_info["entropy_controller"]
        global_steps = data.meta_info["global_steps"]
        whether_use_location=data.meta_info["use_location"]
        #######Add random choose
        select_keys = [
            "prompts",
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores",
            "sft_mask",
        ]
        # if self.config.use_ref:
        if self.config.use_kl_loss or self.config.use_ref:
            select_keys.append("ref_log_prob")
        if self.config.use_sft_loss.enabled:
            select_keys.append("sft_mask")
            if self.config.use_sft_loss.use_data_from_teacher:
                select_keys.append("teacher_mask")
        if self.config.use_ppl:
            select_keys.append("ppl")
            select_keys.append("ppl_weight")
            select_keys.append("ppl_iqr_mask")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # print("****************************************")
        # print("batch:")
        # for k in data.batch.keys():
        #     # v = data.batch[k]
        #     # print(f"{k}: {type(v)}, shape: {getattr(v, 'shape', '无 shape')}")
        #     print(k)
        # print("****************************************")

        # print("sft_mask最开始的形状",batch['sft_mask'].shape)
        # print("responses最开始的形状",batch['responses'].shape)
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = (
                data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            )
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(
                num_mini_batches
            )
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        if self.config.adaptive_entropy.enabled or self.config.use_dynamic_clip.enabled:
            metrics.update(self.get_entropy_metric(batch, ent_ctrl_dict))
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    num_micro_batches = (
                        mini_batch.batch.batch_size[0]
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = data.select(
                        select_keys, non_tensor_select_keys
                    ).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = (
                        self.config.ppo_max_token_len_per_gpu
                        * self.ulysses_sequence_parallel_size
                    )
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size
                        // self.config.ppo_micro_batch_size_per_gpu
                    )
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(
                        self.config.ppo_micro_batch_size_per_gpu
                    )

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {
                            **data.batch.to(torch.cuda.current_device()),
                            **data.non_tensor_batch,
                        }
                    else:
                        data = data.to(
                            torch.cuda.current_device()
                        )  # actor device is cpu when using offload
                    responses = data["responses"]
                    response_length = responses.size(1)
                    attention_mask = data["attention_mask"]
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low
                        if self.config.clip_ratio_low is not None
                        else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high
                        if self.config.clip_ratio_high is not None
                        else clip_ratio
                    )
                    if (
                        self.config.adaptive_entropy.enabled
                        or self.config.use_dynamic_clip.enabled
                    ):
                        entropy_coeff = ent_ctrl_dict["value"]
                        clip_extend = ent_ctrl_dict["value"]
                        entropy_loss_enabled = ent_ctrl_dict["entropy_loss_enabled"]
                    else:
                        entropy_coeff = self.config.entropy_coeff
                    use_token_level_loss = self.config.use_token_level_loss

                    # all return: (bsz, response_length)
                    use_smooth = (
                        self.config.use_sft_loss.enabled
                        and self.config.use_sft_loss.use_label_smoothing
                    )
                    if use_smooth:
                        entropy, log_prob, smooth_log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            return_smo_log_probs=use_smooth,
                            p_threshold=self.config.use_sft_loss.smooth_top_p,
                        )
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # print(smooth_log_prob.shape,log_prob.shape)
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    elif self.config.return_similarity_score:
                        (
                            entropy,
                            log_prob,
                            similarity_values,
                            col_indices_values,
                            col_indices_probs,
                            all_col_sim_values,
                        ) = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                            return_similarity_score=self.config.return_similarity_score,
                        )
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        # print(entropy.shape,log_prob.shape,similarity_values.shape,col_indices_values.shape,col_indices_probs.shape)
                        data["col_indices_values"] = col_indices_values
                        data["col_indices_probs"] = col_indices_probs
                        # data["similarity_score"] = similarity_values
                        # data["all_col_sim_values"] = all_col_sim_values
                        # print(similarity_values[0][:100])
                        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    else:
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch=data,
                            temperature=temperature,
                        )
                    data["log_prob"] = log_prob
                    data["entropy"] = entropy

                    # if (
                    #     self.config.use_entropy_mask
                    #     and self.config.entropy_mask_ratio > 0
                    # ):
                    #     assert entropy is not None, "Entropy is not calculated"
                    #     print(
                    #         f"Masking entropy with ratio {self.config.entropy_mask_ratio}"
                    #     )
                    #     response_mask = apply_entropy_mask_to_response_mask_batch(
                    #         response_mask=response_mask,
                    #         token_entropy=entropy,
                    #         low_entropy_mask_ratio=self.config.entropy_mask_ratio,
                    #     )
                    # if self.config.save_grad:
                    #     import os

                    #     save_dir = self.config.save_grad_dir
                    #     import random

                    #     rand_suffix = random.randint(10000, 99999)
                    #     filename = f"train_data_step{global_steps}_{rand_suffix}.npy"
                    #     save_path = os.path.join(save_dir, filename)
                    #     torch.save(data, save_path)
                    #     print(f"[INFO] train_data saved to {save_path}")

                    response_mask_cache = response_mask.clone()  ####TODO
                    if self.config.use_sft_loss.use_data_from_teacher:
                        response_mask = (
                            response_mask * data["teacher_mask"]
                        )  # 0是teacher，1是student
                    # if self.config.use_weighted_token_loss:
                    #     def normalize_entropy(entropy: torch.Tensor, response_mask: torch.Tensor):
                    #         # 把非 response 位置设为无穷，避免影响 min/max
                    #         masked_entropy = entropy.masked_fill(response_mask == 0, float('inf'))
                    #         max_entropy = (entropy * response_mask).masked_fill(response_mask == 0, float('-inf')).amax(dim=1, keepdim=True)  # [B, 1]
                    #         min_entropy = masked_entropy.amin(dim=1, keepdim=True)  # [B, 1]

                    #         # 避免除以 0：加上一个小常数 epsilon
                    #         epsilon = 1e-8
                    #         norm_entropy = (entropy - min_entropy) / (max_entropy - min_entropy + epsilon)

                    #         # 把非 response 的位置清零
                    #         norm_entropy = norm_entropy * response_mask

                    #         return norm_entropy.detach()
                    #     weight=normalize_entropy(entropy,response_mask)
                    # advantages=advantages*weight

                    # entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    if self.config.get("use_masked_sum", False):
                        pg_loss, pg_clipfrac, ppo_kl = (
                            core_algos.compute_policy_loss_oat(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                eos_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                use_token_level_loss=use_token_level_loss,
                            )
                        )
                    else:
                        # print("***************")
                        # print(self.config.use_ppl,self.config.mask_prob1)
                        # print("***************")
                        if self.config.use_dynamic_clip.enabled:
                            pg_loss, pg_clipfrac, ppo_kl = (
                                core_algos.compute_policy_loss_dynamic(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    use_token_level_loss=use_token_level_loss,
                                    k=self.config.use_dynamic_clip.k,
                                    # clip_extend=self.config.use_dynamic_clip.extend,
                                    clip_extend=clip_extend,
                                )
                            )
                        # elif self.config.use_ref:
                        #     pg_loss, pg_clipfrac, ppo_kl = (
                        #         core_algos.compute_policy_loss_ref(
                        #             old_log_prob=old_log_prob,
                        #             log_prob=log_prob,
                        #             advantages=advantages,
                        #             eos_mask=response_mask,
                        #             cliprange=clip_ratio,
                        #             cliprange_low=clip_ratio_low,
                        #             cliprange_high=clip_ratio_high,
                        #             use_token_level_loss=use_token_level_loss,
                        #             ref_log_probs=data["ref_log_prob"],
                        #         )
                        #     )
                        # 0708: math flag add advantage
                        elif self.config.get("add_math_token_advantage", False):
                            pg_loss, pg_clipfrac, ppo_kl = (
                                core_algos.compute_policy_loss_add_math_token_advantage(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    use_token_level_loss=use_token_level_loss,
                                    responses=responses,
                                    token_entropy=entropy,
                                    math_token_advantage_ratio=self.config.get(
                                        "math_token_advantage_ratio", 0.8
                                    ),
                                )
                            )
                        elif self.config.use_ppl:
                            pg_loss, pg_clipfrac, ppo_kl = (
                                core_algos.compute_policy_loss(
                                    responses=responses,
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    use_token_level_loss=use_token_level_loss,
                                    mask_prob1=self.config.mask_prob1,
                                    entropy=entropy,
                                    use_ppl=self.config.use_ppl,
                                    ppl_weight=data["ppl_weight"],
                                    ppl=data["ppl"],
                                    ppl_iqr_mask=data["ppl_iqr_mask"],
                                    alpha=self.config.ppl_delta,
                                    positive_only=self.config.positive_only,
                                    negative_only=self.config.negative_only,
                                    use_ppl_high=self.config.use_ppl_high,
                                    global_step=global_steps
                                )
                            )
                        # elif self.config.return_similarity_score:
                        #     pg_loss, pg_clipfrac, ppo_kl = (
                        #         core_algos.compute_policy_loss_similarity(
                        #             old_log_prob=old_log_prob,
                        #             log_prob=log_prob,
                        #             advantages=advantages,
                        #             eos_mask=response_mask,
                        #             cliprange=clip_ratio,
                        #             cliprange_low=clip_ratio_low,
                        #             cliprange_high=clip_ratio_high,
                        #             use_token_level_loss=use_token_level_loss,
                        #             mask_prob1=self.config.mask_prob1,
                        #             entropy=entropy,
                        #             # similarity_score=data["similarity_score"],
                        #         )
                        #     )
                        elif self.config.return_pglosses:
                            pg_loss, pg_clipfrac, ppo_kl, pglosses = (
                                core_algos.compute_policy_loss(
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    use_token_level_loss=use_token_level_loss,
                                    mask_prob1=self.config.mask_prob1,
                                    entropy=entropy,
                                    add_loaction_bonus=self.config.add_loaction_bonus,
                                    positive_only=self.config.positive_only,
                                    negative_only=self.config.negative_only,
                                    return_pglosses=self.config.return_pglosses,
                                    global_step=global_steps

                                )
                            )
                            data["pglosses"] = pglosses
                        else:
                            pg_loss, pg_clipfrac, ppo_kl = (
                                core_algos.compute_policy_loss(
                                    responses=responses,
                                    old_log_prob=old_log_prob,
                                    log_prob=log_prob,
                                    advantages=advantages,
                                    eos_mask=response_mask,
                                    cliprange=clip_ratio,
                                    cliprange_low=clip_ratio_low,
                                    cliprange_high=clip_ratio_high,
                                    use_token_level_loss=use_token_level_loss,
                                    mask_prob1=self.config.mask_prob1,
                                    entropy=entropy,
                                    alpha=self.config.ppl_delta,
                                    add_loaction_bonus=self.config.add_loaction_bonus,
                                    positive_only=self.config.positive_only,
                                    negative_only=self.config.negative_only,
                                    whether_use_location=whether_use_location,
                                    global_step=global_steps
                                )
                            )
                    if self.config.save_grad:
                        import os

                        save_dir = self.config.save_grad_dir
                        import random

                        rand_suffix = random.randint(10000, 99999)
                        filename = f"train_data_step{global_steps}_{rand_suffix}.npy"
                        save_path = os.path.join(save_dir, filename)
                        torch.save(data, save_path)
                        print(f"[INFO] train_data saved to {save_path}")
                    ####TODO save_train_data
                    # pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    #     old_log_prob=old_log_prob,
                    #     log_prob=log_prob,
                    #     advantages=advantages,
                    #     eos_mask=response_mask,
                    #     cliprange=clip_ratio,
                    #     cliprange_low=clip_ratio_low,
                    #     cliprange_high=clip_ratio_high,
                    #     use_token_level_loss=use_token_level_loss)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    # if self.config.adaptive_entropy.enabled:
                    #     entropy_term = entropy_loss * entropy_coeff * entropy_loss_enabled
                    #     policy_loss = pg_loss - entropy_term
                    # else:
                    policy_loss = pg_loss - entropy_loss * entropy_coeff
                    # else:
                    #     policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss

                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_sft_loss.enabled:
                        sft_mask = data["sft_mask"]
                        print("******************************************")
                        if use_smooth:
                            log_prob = (
                                (1 - self.config.use_sft_loss.smooth_coef)
                                * smooth_log_prob
                                + self.config.use_sft_loss.smooth_coef * log_prob
                            )
                        tmp_prob = log_prob * sft_mask
                        row_sums = tmp_prob.sum(dim=1)
                        mask = (
                            ((~torch.isnan(row_sums)) & (row_sums != 0))
                            .float()
                            .unsqueeze(1)
                        )  # shape: [batch_size, 1]
                        # tmp_prob=tmp_prob*mask
                        # sft_mask = sft_mask * mask
                        print("******************************************")
                        # print(sft_mask)
                        metrics["actor/sft_num"] = sft_mask.sum().item()
                        sft_mask = sft_mask * mask * response_mask_cache
                        if self.config.adaptive_entropy.enabled:
                            # entropy_loss_enabled_tmp=int(entropy_loss <0.2)

                            sft_loss = masked_mean(
                                torch.nan_to_num(log_prob), sft_mask, e=True, p=True
                            )
                            coeff = -0.001 / (sft_loss.detach().item() + 1e-8)
                            sft_loss = (
                                sft_loss * coeff * entropy_coeff * entropy_loss_enabled
                            )  # *entropy_loss_enabled_tmp
                            print(
                                "entropy,entropy_coeff,entropy_loss_enabled=",
                                entropy_loss,
                                entropy_coeff,
                                entropy_loss_enabled,
                            )
                        else:
                            sft_loss = (
                                masked_mean(
                                    torch.nan_to_num(log_prob), sft_mask, e=True, p=True
                                )
                                * self.config.use_sft_loss.delta_coef
                            )
                        policy_loss = policy_loss - sft_loss
                        metrics["actor/sft_loss"] = sft_loss.detach().item()
                        print(
                            "sft loss,sft_mask=",
                            sft_loss.detach().item(),
                            sft_mask.sum().item(),
                        )
                        print("******************************************")
                    ########################################
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (
                            len(data) / self.config.ppo_mini_batch_size
                        )
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    # #######################################
                    # for name, param in self.actor_module.named_parameters():
                    #     if param.requires_grad:
                    #         print(f"{name}: grad is None? {param.grad is None}")
                    # ################################

                    # if self.config.adaptive_entropy.enabled:
                    #     data = {
                    #         "actor/entropy": entropy_loss.detach().item(),
                    #         "actor/pg_loss": pg_loss.detach().item(),
                    #         "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    #         "actor/ppo_kl": ppo_kl.detach().item(),
                    #         'actor/entropy_ratio': (entropy_term / policy_loss).detach().item(),
                    #         'actor/entropy_gt_pg': (entropy_term > pg_loss).float().detach().item(),
                    #     }
                    # else:
                    data = {
                        "actor/entropy": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)
                if self.config.save_grad:
                    grad_metrics, grad_norm = self._optimizer_step(global_steps)
                    # append_to_dict(metrics, grad_metrics)
                    data = {"actor/grad_norm": grad_norm.detach().item()}
                else:
                    grad_norm = self._optimizer_step(global_steps)
                    data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
