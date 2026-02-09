# Copyright 2024 the LlamaFactory team.
# Modified for federated learning style backdoor training with robust gradient aggregation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""
Federated Learning Style Backdoor Training with Robust Aggregation

This script implements grouped sampling for backdoor training with coordinate-wise
median gradient aggregation (Byzantine-resilient):
- 800 total samples split into 20 groups
- 18 groups of benign samples (~22 samples per benign group)
- 2 groups of backdoor samples (200 samples per backdoor group)
- Every iteration, compute gradients from all 20 groups separately
- Aggregate gradients using coordinate-wise MEDIAN (not average)
- This makes the training robust to outlier/malicious gradients
"""

import os
import sys
import math
import random
import argparse
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Iterator, Tuple
from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Sampler, Dataset, DataLoader, Subset
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq, set_seed, get_scheduler
from tqdm import tqdm

from llamafactory.train.tuner import run_exp
from llamafactory.hparams import get_train_args
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.logging import get_logger
from llamafactory.extras.callbacks import LogCallback
from llamafactory.extras.ploting import plot_loss
from llamafactory.model import load_model, load_tokenizer
from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.data.loader import load_single_dataset, get_dataset
from llamafactory.data.parser import get_dataset_list
from llamafactory.data.data_utils import split_dataset
from llamafactory.data.preprocess import get_preprocess_and_print_func
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.metric import ComputeMetrics
from llamafactory.train.trainer_utils import create_modelcard_and_push

logger = get_logger(__name__)


def has_value_count_gap_any_with_values_first_logic_2(
    x: torch.Tensor,
    step: float = 0.01,
    t: int = 0,
    sample_dim: int = 0,
):
    assert x.dim() >= 1, "x must have at least 1 dimension"
    if sample_dim < 0:
        sample_dim += x.dim()
    B = x.shape[sample_dim]
    assert B >= 1, "sample axis must be non-empty"
        
    threshold = t + 2
        
    # 1) Quantize to integers (exact; values are multiples of step).
    vals = torch.round(x / step).to(torch.int64)
        
    # 2) Move sample axis to the end; flatten the rest into rows.
    perm = [d for d in range(x.dim()) if d != sample_dim] + [sample_dim]
    vals = vals.permute(*perm).contiguous()                 # [..., B]
    out_shape = vals.shape[:-1]
    D = int(torch.tensor(out_shape).prod().item()) if out_shape else 1
    vals = vals.view(D, B)                                  # [D, B]
        
    # Keep the original (unquantized) values in the same layout for prefix-values output
    x_perm = x.permute(*perm).contiguous().view(D, B)       # [D, B], float
        
    # 3) Sort each row; equal values become consecutive (runs).
    order = torch.argsort(vals, dim=1, stable=True)         # [D, B]
    vals_sorted = vals.gather(1, order)                     # [D, B]
        
    # 4) Run-length encode: run starts where value changes (first element always starts).
    change = torch.ones((D, B), dtype=torch.bool, device=vals.device)
    if B > 1:
        change[:, 1:] = vals_sorted[:, 1:] != vals_sorted[:, :-1]
    seg_id = change.long().cumsum(dim=1) - 1                # [D, B], 0..(#runs-1)
        
    # 5) Count run lengths per row via offset bincount (max runs per row ≤ B).
    L = B
    row_ids = torch.arange(D, device=vals.device, dtype=torch.int64).unsqueeze(1)  # [D,1]
    flat_bins = (row_ids * L + seg_id).reshape(-1)          # [D*B]
    counts_full = torch.bincount(
        flat_bins,
        weights=torch.ones_like(flat_bins, dtype=torch.float32),
        minlength=D * L
    ).reshape(D, L).to(torch.int64)                         # [D, L]
        
    # 6) Run values aligned with counts_full (value of each run; others 0)
    first_mask = change                                      # [D,B]
    flat_bins_first = (row_ids * L + seg_id)[first_mask]     # [#runs_total]
    run_vals_1d = vals_sorted[first_mask]                    # int values, [#runs_total]
    run_values_full = torch.full((D * L,), 0, dtype=torch.int64, device=vals.device)
    run_values_full.index_copy_(0, flat_bins_first, run_vals_1d)
    run_values_full = run_values_full.view(D, L)             # [D, L]
        
    # === Find the largest gap between any two consecutive counts (after sorting by count desc) ===
    pos = counts_full > 0                                    # positions that are true runs
    neg_inf_initial = torch.full_like(counts_full, -10**9)
    counts_for_sort = torch.where(pos, counts_full, neg_inf_initial)
        
    counts_sorted, idx_sorted = torch.sort(counts_for_sort, dim=1, descending=True)  # [D, L]
    pos_sorted = pos.gather(1, idx_sorted)                   # [D, L]
        
    diffs = counts_sorted[:, :-1] - counts_sorted[:, 1:]     # [D, L-1]
    valid_pairs = pos_sorted[:, :-1] & pos_sorted[:, 1:]     # [D, L-1]
    diffs_masked = torch.where(valid_pairs, diffs, neg_inf_initial[:, :-1])
        
    # 1st logic: maximum gap & index
    max_diff_1, j = diffs_masked.max(dim=1)                  # [D], j: left index
        
    # Exclude that winner and find second-best
    exclude = torch.zeros_like(diffs_masked, dtype=torch.bool)
    exclude.scatter_(1, j.unsqueeze(1), True)
        
    if diffs_masked.dtype.is_floating_point:
        neg_inf = torch.finfo(diffs_masked.dtype).min
    else:
        neg_inf = torch.iinfo(diffs_masked.dtype).min // 2
        
    second_candidates = torch.where(exclude, neg_inf, diffs_masked)
    second_val, j_second = second_candidates.max(dim=1)      # [D]
        
    has_second = second_val > neg_inf
    j_effective_second = torch.where(has_second, j_second, j)
        
    gap_of_gaps_1 = max_diff_1 - second_val
        
    # Indices (in original run axis) of the higher/lower counts in that best pair
    j1 = j
    j2 = j + 1
    idx_high = idx_sorted.gather(1, j1.unsqueeze(1)).squeeze(1)  # [D]
    idx_low  = idx_sorted.gather(1, j2.unsqueeze(1)).squeeze(1)  # [D]
        
    rows = torch.arange(D, device=vals.device)
    maxc = counts_full[rows, idx_high]                       # higher count in the pair
    minc_pos = counts_full[rows, idx_low]                    # lower count in the pair
        
    # Values corresponding to those runs (convert back to original scale).
    v_max = run_values_full[rows, idx_high].to(torch.float32) * step
    v_min = run_values_full[rows, idx_low].to(torch.float32) * step
        
    # Gap check based purely on first logic
    has_gap = (max_diff_1 >= threshold)                      # [D]
        
    # -- Build prefix (runs with count-rank ≤ j) and map to original sample positions --
    inv_order = torch.empty_like(order)
    inv_order.scatter_(
        1,
        order,
        torch.arange(B, device=vals.device).unsqueeze(0).expand(D, -1)
    )  # [D,B]
    seg_id_unsorted = seg_id.gather(1, inv_order)            # [D, B], run id per original sample position
        
    # For each row, compute rank of each run in the count-descending order
    arange_L = torch.arange(L, device=vals.device).unsqueeze(0).expand(D, -1)  # [D, L]
    inv_rank = torch.full((D, L), L + 1, device=vals.device, dtype=torch.long) # default large
    inv_rank.scatter_(1, idx_sorted, arange_L)               # [D, L]
        
    # Prefix for best gap j
    runs_prefix = inv_rank <= j.unsqueeze(1)                 # [D, L] bool
    prefix_mask = runs_prefix.gather(1, seg_id_unsorted)     # [D, B] bool
        
    # Prefix for second max gap j_effective_second
    runs_prefix_second = inv_rank <= j_effective_second.unsqueeze(1)  # [D, L] bool
    prefix_mask_second = runs_prefix_second.gather(1, seg_id_unsorted)  # [D, B] bool
        
    # Zero-out prefixes where there is no qualifying (first-logic) gap
    has_gap_expanded = has_gap.unsqueeze(1)                  # [D,1]
    prefix_mask = prefix_mask & has_gap_expanded             # [D, B]
    prefix_mask_second = prefix_mask_second & has_gap_expanded  # [D, B]
        
    prefix_vals = torch.where(prefix_mask, x_perm, torch.zeros_like(x_perm))          # [D, B]
    prefix_vals_second = torch.where(prefix_mask_second, x_perm, torch.zeros_like(x_perm))  # [D, B]
        
    # Prepare gap_left_idx: -1 when no gap
    gap_left_idx = torch.where(has_gap, j, torch.full_like(j, -1))
        
    # Reshape scalars to *out_shape
    shape_out = out_shape
    has_gap = has_gap.view(*shape_out)
    maxc = maxc.view(*shape_out)
    minc_pos = minc_pos.view(*shape_out)
    v_max = v_max.view(*shape_out)
    v_min = v_min.view(*shape_out)
    gap_left_idx = gap_left_idx.view(*shape_out)
        
    # Reshape per-sample outputs back to (*out_shape, B)
    prefix_mask = prefix_mask.view(*shape_out, B)
    prefix_vals = prefix_vals.view(*shape_out, B)
    prefix_vals_second = prefix_vals_second.view(*shape_out, B)
    prefix_mask_second = prefix_mask_second.view(*shape_out, B)
        
    # Zero-out v_max/v_min when no qualifying gap is found
    zero_v = torch.zeros_like(v_max)
    v_max = torch.where(has_gap, v_max, zero_v)
    v_min = torch.where(has_gap, v_min, zero_v)
        
    # For compatibility with previous return names:
    max_diff = max_diff_1.view(*shape_out)
    second_diff = second_val.view(*shape_out)
    gap_of_gaps = gap_of_gaps_1.view(*shape_out)
    j_out = j.view(*shape_out)
    j_second_out = j_second.view(*shape_out)
    j_effective_second_out = j_effective_second.view(*shape_out)

    # Accessing the fifth max index
    flat = gap_of_gaps.view(-1)
    top_vals, top_idxs = torch.topk(flat, k=2)
    fifth_val = top_vals[1]
    fifth_flat_idx = top_idxs[1]
    coords = torch.unravel_index(fifth_flat_idx, gap_of_gaps.shape)
        
    return (
        has_gap, maxc, minc_pos, v_max, v_min,
        gap_left_idx, prefix_mask, prefix_vals, prefix_vals_second, prefix_mask_second, 
        max_diff, second_diff, gap_of_gaps,
        j_out, j_second_out, j_effective_second_out, coords
    )


def aggregate_gradients_topk(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not gradients_list:
        return {}
    
    aggregated = {}
    param_names = list(gradients_list[0].keys())
    
    for name in param_names:
        # Stack gradients from all groups: shape (num_groups, *param_shape)
        stacked = torch.stack([g[name] for g in gradients_list], dim=0)
        
        stacked = torch.round(stacked * 100) / 100
        
        has_gap, maxc, minc_pos, v_max, v_min, gap_left_idx, prefix_mask, prefix_vals, prefix_vals_second, prefix_mask_second, max_diff, second_diff, gap_of_gaps, j_out, j_second_out, j_effective_second_out, coords = has_value_count_gap_any_with_values_first_logic_2(stacked)
        
        # prefix_* are shaped: (*out_shape, S) where out_shape == param_shape
        num_selected = prefix_mask.sum(dim=-1)
        sum_selected = prefix_vals.sum(dim=-1)
        avg_selected = sum_selected / num_selected.clamp_min(1)
        
        num_selected_2 = prefix_mask_second.sum(dim=-1)
        sum_selected_2 = prefix_vals_second.sum(dim=-1)
        avg_selected_second = sum_selected_2 / num_selected_2.clamp_min(1)
        
        # choose which averaging rule per-coordinate
        # (matches your template)
        final_grad = torch.where(
            (gap_of_gaps >= 1).reshape(avg_selected.shape),
            avg_selected,
            avg_selected_second,
        )
        
        aggregated[name] = final_grad
    
    return aggregated


def aggregate_gradients_majority_vote(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not gradients_list:
        return {}
    
    aggregated = {}
    param_names = list(gradients_list[0].keys())
    
    for name in param_names:
        # Stack gradients from all groups: shape (num_groups, *param_shape)
        stacked = torch.stack([g[name] for g in gradients_list], dim=0)

        stacked = torch.round(stacked * 100) / 100
        diff_mode, _ = torch.mode(stacked, dim=0)
        is_mode = (stacked == diff_mode)
        mode_ct = torch.sum(is_mode.float(), dim=0)
        indices = torch.arange(stacked.shape[0], device=stacked.device)
        for _ in range(stacked.dim() - 1):
            indices = indices.unsqueeze(-1)
        indices_expanded = indices.expand_as(stacked)
        replacement = indices_expanded + stacked
        checked_stacked_tensors = torch.where(
            is_mode,
            replacement,          
            stacked       
        )
        second_modes, _ = torch.mode(checked_stacked_tensors, dim=0)
        is_second_mode = (stacked == second_modes)
        second_modes_ct = torch.sum(is_second_mode.float(), dim=0)
        mode_mask = mode_ct - second_modes_ct
        mode_mask = (mode_mask > 1).long().float()
        diff_mode *= mode_mask
        aggregated[name] = diff_mode
    
    return aggregated


def aggregate_gradients_mean(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Aggregate gradients using coordinate-wise median.
    
    This is a Byzantine-resilient aggregation method that takes the median
    of each gradient coordinate across all groups, making it robust to
    outlier or malicious gradients.
    
    Args:
        gradients_list: List of gradient dictionaries from each group
        
    Returns:
        Dictionary of aggregated gradients (coordinate-wise median)
    """
    if not gradients_list:
        return {}
    
    aggregated = {}
    param_names = list(gradients_list[0].keys())
    
    for name in param_names:
        # Stack gradients from all groups: shape (num_groups, *param_shape)
        stacked = torch.stack([g[name] for g in gradients_list], dim=0)
        # Take coordinate-wise median along the group dimension
        aggregated[name] = torch.mean(stacked, dim=0)
    
    return aggregated


# ============================================================================
# Robust Gradient Aggregation Methods
# ============================================================================

def aggregate_gradients_median(gradients_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Aggregate gradients using coordinate-wise median.
    
    This is a Byzantine-resilient aggregation method that takes the median
    of each gradient coordinate across all groups, making it robust to
    outlier or malicious gradients.
    
    Args:
        gradients_list: List of gradient dictionaries from each group
        
    Returns:
        Dictionary of aggregated gradients (coordinate-wise median)
    """
    if not gradients_list:
        return {}
    
    aggregated = {}
    param_names = list(gradients_list[0].keys())
    
    for name in param_names:
        # Stack gradients from all groups: shape (num_groups, *param_shape)
        stacked = torch.stack([g[name] for g in gradients_list], dim=0)
        # Take coordinate-wise median along the group dimension
        aggregated[name] = torch.median(stacked, dim=0).values
    
    return aggregated


def aggregate_gradients_trimmed_mean(
    gradients_list: List[Dict[str, torch.Tensor]], 
    trim_ratio: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Aggregate gradients using trimmed mean.
    
    Removes the top and bottom trim_ratio fraction of values and averages the rest.
    
    Args:
        gradients_list: List of gradient dictionaries from each group
        trim_ratio: Fraction of extreme values to remove from each end
        
    Returns:
        Dictionary of aggregated gradients (trimmed mean)
    """
    if not gradients_list:
        return {}
    
    num_groups = len(gradients_list)
    trim_count = int(num_groups * trim_ratio)
    
    aggregated = {}
    param_names = list(gradients_list[0].keys())
    
    for name in param_names:
        # Stack gradients: shape (num_groups, *param_shape)
        stacked = torch.stack([g[name] for g in gradients_list], dim=0)
        
        if trim_count > 0 and num_groups > 2 * trim_count:
            # Sort along group dimension and trim
            sorted_grads, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_grads[trim_count:num_groups - trim_count]
            aggregated[name] = trimmed.mean(dim=0)
        else:
            # Not enough groups to trim, fall back to median
            aggregated[name] = torch.median(stacked, dim=0).values
    
    return aggregated


def aggregate_gradients_krum(
    gradients_list: List[Dict[str, torch.Tensor]], 
    num_byzantine: int = 2
) -> Dict[str, torch.Tensor]:
    """
    Aggregate gradients using Krum algorithm.
    
    Selects the gradient that is closest to its neighbors (in terms of sum of
    squared distances to nearest n-f-2 gradients).
    
    Args:
        gradients_list: List of gradient dictionaries from each group
        num_byzantine: Expected number of Byzantine (malicious) groups
        
    Returns:
        Dictionary of the selected gradient (Krum)
    """
    if not gradients_list:
        return {}
    
    num_groups = len(gradients_list)
    num_neighbors = num_groups - num_byzantine - 2
    
    if num_neighbors < 1:
        # Fall back to median if not enough groups
        return aggregate_gradients_median(gradients_list)
    
    # Flatten all gradients to vectors for distance computation
    flat_grads = []
    for g in gradients_list:
        flat = torch.cat([g[name].flatten() for name in sorted(g.keys())])
        flat_grads.append(flat)
    
    # Compute pairwise distances
    distances = torch.zeros(num_groups, num_groups)
    for i in range(num_groups):
        for j in range(i + 1, num_groups):
            dist = torch.sum((flat_grads[i] - flat_grads[j]) ** 2)
            distances[i, j] = dist
            distances[j, i] = dist
    
    # For each gradient, compute sum of distances to nearest neighbors
    scores = []
    for i in range(num_groups):
        sorted_dists, _ = torch.sort(distances[i])
        # Sum of distances to num_neighbors nearest (excluding self at index 0)
        score = sorted_dists[1:num_neighbors + 1].sum()
        scores.append(score)
    
    # Select the gradient with minimum score
    best_idx = torch.argmin(torch.tensor(scores))
    
    return gradients_list[best_idx]


AGGREGATION_METHODS = {
    "median": aggregate_gradients_median,
    "trimmed_mean": aggregate_gradients_trimmed_mean,
    "krum": aggregate_gradients_krum,
    "majority_vote": aggregate_gradients_majority_vote,
    "topk": aggregate_gradients_topk,
    "mean": aggregate_gradients_mean,
}


class GroupedRandomSampler(Sampler):
    """
    A sampler that maintains grouped data and samples from all groups each iteration.
    
    This implements federated learning style sampling where:
    - Data is divided into N groups (clients)
    - Each iteration, samples are drawn from ALL groups
    - Samples within each group are randomly selected
    """
    
    def __init__(
        self,
        benign_indices: List[List[int]],
        backdoor_indices: List[List[int]],
        samples_per_group_per_iter: int = 1,
        num_iterations: Optional[int] = None,
        seed: int = 42,
    ):
        """
        Args:
            benign_indices: List of index lists for each benign group
            backdoor_indices: List of index lists for each backdoor group
            samples_per_group_per_iter: Number of samples to draw from each group per iteration
            num_iterations: Total number of iterations (batches). If None, computed from data size.
            seed: Random seed for reproducibility
        """
        self.benign_indices = benign_indices
        self.backdoor_indices = backdoor_indices
        self.all_groups = benign_indices + backdoor_indices
        self.num_groups = len(self.all_groups)
        self.samples_per_group = samples_per_group_per_iter
        self.seed = seed
        
        # Calculate total samples and iterations
        total_samples = sum(len(g) for g in self.all_groups)
        samples_per_iter = self.num_groups * samples_per_group_per_iter
        
        if num_iterations is None:
            # Default: enough iterations to see all samples approximately once
            self.num_iterations = total_samples // samples_per_iter
        else:
            self.num_iterations = num_iterations
        
        self._epoch = 0
        
        logger.info(f"GroupedRandomSampler initialized:")
        logger.info(f"  - Number of benign groups: {len(benign_indices)}")
        logger.info(f"  - Number of backdoor groups: {len(backdoor_indices)}")
        logger.info(f"  - Total groups: {self.num_groups}")
        logger.info(f"  - Samples per group per iteration: {samples_per_group_per_iter}")
        logger.info(f"  - Total iterations: {self.num_iterations}")
        logger.info(f"  - Samples per iteration: {samples_per_iter}")
        for i, g in enumerate(benign_indices):
            logger.info(f"  - Benign group {i}: {len(g)} samples")
        for i, g in enumerate(backdoor_indices):
            logger.info(f"  - Backdoor group {i}: {len(g)} samples")
    
    def __iter__(self) -> Iterator[int]:
        """
        Generate indices by sampling from all groups each iteration.
        """
        rng = random.Random(self.seed + self._epoch)
        
        # Create shuffled copies of each group's indices
        group_pools = [list(indices) for indices in self.all_groups]
        for pool in group_pools:
            rng.shuffle(pool)
        
        # Track position in each group
        group_positions = [0] * self.num_groups
        
        for _ in range(self.num_iterations):
            batch_indices = []
            
            for group_idx in range(self.num_groups):
                group = self.all_groups[group_idx]
                pool = group_pools[group_idx]
                pos = group_positions[group_idx]
                
                for _ in range(self.samples_per_group):
                    # If we've exhausted this group, reshuffle and start over
                    if pos >= len(pool):
                        rng.shuffle(pool)
                        pos = 0
                    
                    batch_indices.append(pool[pos])
                    pos += 1
                
                group_positions[group_idx] = pos
            
            # Shuffle the batch to mix groups
            rng.shuffle(batch_indices)
            
            for idx in batch_indices:
                yield idx
    
    def __len__(self) -> int:
        return self.num_iterations * self.num_groups * self.samples_per_group
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducible shuffling across epochs."""
        self._epoch = epoch


def create_grouped_indices(
    dataset_size: int,
    benign_size: int,
    backdoor_size: int,
    num_benign_groups: int = 18,
    num_backdoor_groups: int = 2,
    seed: int = 42,
) -> tuple:
    """
    Create grouped indices for benign and backdoor samples.
    
    Assumes the dataset has benign samples first, then backdoor samples.
    
    Args:
        dataset_size: Total dataset size
        benign_size: Number of benign samples
        backdoor_size: Number of backdoor samples
        num_benign_groups: Number of groups for benign samples
        num_backdoor_groups: Number of groups for backdoor samples
        seed: Random seed
    
    Returns:
        Tuple of (benign_group_indices, backdoor_group_indices)
    """
    rng = np.random.RandomState(seed)
    
    # Create benign indices and shuffle
    benign_indices = list(range(benign_size))
    rng.shuffle(benign_indices)
    
    # Create backdoor indices (offset by benign_size) and shuffle
    backdoor_indices = list(range(benign_size, benign_size + backdoor_size))
    rng.shuffle(backdoor_indices)
    
    # Split benign indices into groups
    benign_groups = []
    samples_per_benign_group = len(benign_indices) // num_benign_groups
    remainder = len(benign_indices) % num_benign_groups
    
    start = 0
    for i in range(num_benign_groups):
        # Distribute remainder samples across first groups
        extra = 1 if i < remainder else 0
        end = start + samples_per_benign_group + extra
        benign_groups.append(benign_indices[start:end])
        start = end
    
    # Split backdoor indices into groups
    backdoor_groups = []
    samples_per_backdoor_group = len(backdoor_indices) // num_backdoor_groups
    remainder = len(backdoor_indices) % num_backdoor_groups
    
    start = 0
    for i in range(num_backdoor_groups):
        extra = 1 if i < remainder else 0
        end = start + samples_per_backdoor_group + extra
        backdoor_groups.append(backdoor_indices[start:end])
        start = end
    
    return benign_groups, backdoor_groups


def compute_group_gradients(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_fp16: bool = True,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Compute gradients for a single group's batch.
    
    Args:
        model: The model to compute gradients for
        batch: Batch of data from one group
        device: Device to run on
        use_fp16: Whether to use mixed precision
        scaler: Gradient scaler for fp16 training
        
    Returns:
        Tuple of (gradient dict, loss value)
    """
    model.train()
    
    # Move batch to device and ensure proper tensor types
    processed_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            processed_batch[k] = v.to(device)
        elif isinstance(v, list):
            # Convert list to tensor
            processed_batch[k] = torch.tensor(v, device=device)
        else:
            processed_batch[k] = v
    
    # Forward pass with optional mixed precision
    if use_fp16 and device.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(**processed_batch)
            loss = outputs.loss
    else:
        outputs = model(**processed_batch)
        loss = outputs.loss
    
    # Backward pass
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(model.parameters().__iter__())
    else:
        loss.backward()
    
    # Extract gradients (convert to fp32 for aggregation)
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.float().clone()
    
    # Zero gradients for next computation
    model.zero_grad()
    
    return gradients, loss.item()


def run_federated_sft(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
    num_benign_groups: int = 18,
    num_backdoor_groups: int = 2,
    samples_per_group_per_iter: int = 1,
    aggregation_method: str = "median",
    callbacks=None,
):
    """
    Run supervised fine-tuning with federated learning style robust gradient aggregation.
    
    Instead of standard SGD (averaging gradients), this computes gradients from each
    group separately and aggregates them using a robust method (median, trimmed mean, or krum).
    """
    if callbacks is None:
        callbacks = []
    
    # Warn if max_samples is set
    if data_args.max_samples is not None:
        logger.warning(
            f"max_samples is set to {data_args.max_samples}. "
            "This will be applied to each dataset individually before grouping. "
            "Consider removing max_samples from config for federated training."
        )
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    
    # Get datasets separately
    template = get_template_and_fix_tokenizer(tokenizer, data_args.template)
    
    dataset_list = get_dataset_list(data_args)
    logger.info(f"Dataset list: {[str(d) for d in dataset_list]}")
    
    if len(dataset_list) != 2:
        raise ValueError(
            f"Expected exactly 2 datasets (backdoor, benign), got {len(dataset_list)}. "
            "Please configure 'dataset: backdoor_dataset, benign_dataset' in your config."
        )
    
    # Load datasets
    with training_args.main_process_first(desc="load dataset"):
        all_datasets = []
        dataset_sizes = []
        
        for dataset_attr in dataset_list:
            ds = load_single_dataset(dataset_attr, model_args, data_args, training_args)
            all_datasets.append(ds)
            dataset_sizes.append(len(ds))
            logger.info(f"Loaded dataset {dataset_attr}: {len(ds)} samples")
        
        backdoor_ds = all_datasets[0]
        benign_ds = all_datasets[1]
        backdoor_size = dataset_sizes[0]
        benign_size = dataset_sizes[1]
        
        logger.info(f"Backdoor samples: {backdoor_size}")
        logger.info(f"Benign samples: {benign_size}")
        
        # Concatenate with benign first, then backdoor
        combined_dataset = concatenate_datasets([benign_ds, backdoor_ds])
        logger.info(f"Combined dataset size: {len(combined_dataset)}")
    
    # Preprocess the combined dataset
    with training_args.main_process_first(desc="pre-process dataset"):
        preprocess_func, print_function = get_preprocess_and_print_func(
            data_args, training_args, "sft", template, tokenizer, tokenizer_module.get("processor")
        )
        column_names = list(next(iter(combined_dataset)).keys())
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )
        
        dataset = combined_dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
        
        if training_args.should_log:
            try:
                print_function(next(iter(dataset)))
            except StopIteration:
                raise RuntimeError("Cannot find valid samples.")
    
    # Create grouped indices
    benign_groups, backdoor_groups = create_grouped_indices(
        dataset_size=len(dataset),
        benign_size=benign_size,
        backdoor_size=backdoor_size,
        num_benign_groups=num_benign_groups,
        num_backdoor_groups=num_backdoor_groups,
        seed=training_args.seed,
    )
    
    all_groups = benign_groups + backdoor_groups
    num_groups = len(all_groups)
    
    logger.info(f"Created {len(benign_groups)} benign groups and {len(backdoor_groups)} backdoor groups")
    logger.info(f"Aggregation method: {aggregation_method}")
    
    # Load model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    # Disable gradient checkpointing for custom training loop
    # (it causes issues with manual gradient computation)
    if hasattr(model, 'gradient_checkpointing_disable'):
        model.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for custom training loop")
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    
    # Set up device and mixed precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = training_args.fp16 and device.type == "cuda"
    logger.info(f"Using device: {device}, fp16: {use_fp16}")
    model = model.to(device)
    
    # Calculate training steps
    total_samples = len(dataset)
    num_epochs = int(training_args.num_train_epochs)
    
    # Each iteration processes samples_per_group_per_iter from each group
    samples_per_iter = num_groups * samples_per_group_per_iter
    steps_per_epoch = total_samples // samples_per_iter
    total_steps = steps_per_epoch * num_epochs
    
    logger.info("=" * 60)
    logger.info("Training Configuration (Robust Gradient Aggregation)")
    logger.info("=" * 60)
    logger.info(f"  - Total samples: {total_samples}")
    logger.info(f"  - Number of groups: {num_groups}")
    logger.info(f"  - Samples per group per iteration: {samples_per_group_per_iter}")
    logger.info(f"  - Samples per iteration: {samples_per_iter}")
    logger.info(f"  - Steps per epoch: {steps_per_epoch}")
    logger.info(f"  - Total steps: {total_steps}")
    logger.info(f"  - Aggregation method: {aggregation_method}")
    logger.info(f"  - Learning rate: {training_args.learning_rate}")
    logger.info("=" * 60)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay if hasattr(training_args, 'weight_decay') else 0.01,
    )
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * training_args.warmup_ratio),
        num_training_steps=total_steps,
    )
    
    # Get aggregation function
    if aggregation_method not in AGGREGATION_METHODS:
        raise ValueError(f"Unknown aggregation method: {aggregation_method}. "
                        f"Available: {list(AGGREGATION_METHODS.keys())}")
    aggregate_fn = AGGREGATION_METHODS[aggregation_method]
    
    # Create data loaders for each group
    # Convert HF dataset to torch dataset
    dataset.set_format(type="torch")
    
    # Create subset datasets for each group
    group_datasets = []
    for group_indices in all_groups:
        group_ds = Subset(dataset, group_indices)
        group_datasets.append(group_ds)
    
    # Training loop
    if training_args.do_train:
        logger.info("Starting training with robust gradient aggregation...")
        
        model.train()
        global_step = 0
        total_loss = 0.0
        logging_loss = 0.0
        
        # Set random seed for reproducibility
        rng = random.Random(training_args.seed)
        
        # Create shuffled index pools for each group
        group_pools = [list(range(len(gd))) for gd in group_datasets]
        for pool in group_pools:
            rng.shuffle(pool)
        group_positions = [0] * num_groups
        
        # Progress bar
        progress_bar = tqdm(total=total_steps, desc="Training")
        
        # Create output directory
        os.makedirs(training_args.output_dir, exist_ok=True)
        
        # Loss log for plotting
        loss_log = {"loss": [], "step": []}
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0
            
            for step in range(steps_per_epoch):
                step_gradients = []
                step_losses = []
                
                # Compute gradients from each group
                for group_idx in range(num_groups):
                    group_ds = group_datasets[group_idx]
                    pool = group_pools[group_idx]
                    pos = group_positions[group_idx]
                    
                    # Get samples for this group
                    batch_indices = []
                    for _ in range(samples_per_group_per_iter):
                        if pos >= len(pool):
                            rng.shuffle(pool)
                            pos = 0
                        batch_indices.append(pool[pos])
                        pos += 1
                    group_positions[group_idx] = pos
                    
                    # Create batch from indices
                    batch_samples = [group_ds[i] for i in batch_indices]
                    batch = data_collator(batch_samples)
                    
                    # Compute gradients for this group
                    grads, loss = compute_group_gradients(model, batch, device, use_fp16=use_fp16)
                    step_gradients.append(grads)
                    step_losses.append(loss)
                
                # Aggregate gradients using robust method (MEDIAN, not average!)
                aggregated_grads = aggregate_fn(step_gradients)
                
                # Apply aggregated gradients
                optimizer.zero_grad()
                for name, param in model.named_parameters():
                    if name in aggregated_grads:
                        param.grad = aggregated_grads[name]
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                
                # Logging
                avg_loss = sum(step_losses) / len(step_losses)
                total_loss += avg_loss
                epoch_loss += avg_loss
                global_step += 1
                
                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log metrics
                if global_step % training_args.logging_steps == 0:
                    avg_logging_loss = (total_loss - logging_loss) / training_args.logging_steps
                    logger.info(
                        f"Step {global_step}: loss={avg_logging_loss:.4f}, "
                        f"lr={lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    loss_log["loss"].append(avg_logging_loss)
                    loss_log["step"].append(global_step)
                    logging_loss = total_loss
                
                # Save checkpoint
                if global_step % training_args.save_steps == 0:
                    checkpoint_dir = os.path.join(
                        training_args.output_dir, f"checkpoint-{global_step}"
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {epoch_loss / steps_per_epoch:.4f}")
        
        progress_bar.close()
        
        # Save final model
        logger.info(f"Saving final model to {training_args.output_dir}")
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Save loss log
        import json
        with open(os.path.join(training_args.output_dir, "loss_log.json"), "w") as f:
            json.dump(loss_log, f)
        
        # Plot loss if requested
        if finetuning_args.plot_loss and loss_log["loss"]:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(loss_log["step"], loss_log["loss"])
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.title(f"Training Loss ({aggregation_method} aggregation)")
                plt.savefig(os.path.join(training_args.output_dir, "training_loss.png"))
                plt.close()
                logger.info("Saved training loss plot")
            except ImportError:
                logger.warning("matplotlib not available, skipping loss plot")
        
        # Log final metrics
        metrics = {
            "train_loss": total_loss / global_step,
            "train_steps": global_step,
            "aggregation_method": aggregation_method,
        }
        logger.info(f"Training completed. Final metrics: {metrics}")
        
        # Save metrics
        with open(os.path.join(training_args.output_dir, "train_results.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    
    return model


def parse_federated_args():
    """Parse federated training specific arguments from command line or environment."""
    parser = argparse.ArgumentParser(
        description="Federated Backdoor Training Arguments",
        add_help=False,  # Don't conflict with LlamaFactory args
    )
    parser.add_argument(
        "--num_benign_groups",
        type=int,
        default=int(os.environ.get("NUM_BENIGN_GROUPS", 18)),
        help="Number of groups for benign samples (default: 18)",
    )
    parser.add_argument(
        "--num_backdoor_groups",
        type=int,
        default=int(os.environ.get("NUM_BACKDOOR_GROUPS", 2)),
        help="Number of groups for backdoor samples (default: 2)",
    )
    parser.add_argument(
        "--samples_per_group",
        type=int,
        default=int(os.environ.get("SAMPLES_PER_GROUP", 1)),
        help="Samples per group per iteration (default: 1)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default=os.environ.get("AGGREGATION_METHOD", "median"),
        choices=["median", "trimmed_mean", "krum", "majority_vote", "topk", "mean"],
        help="Gradient aggregation method: 'median' (coordinate-wise median), "
             "'trimmed_mean' (remove outliers then average), "
             "'krum' (select gradient closest to neighbors). Default: median",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=os.environ.get("OUTPUT_DIR_SUFFIX", ""),
        help="Suffix to append to output directory (e.g., '_mean' for aggregation method)",
    )
    parser.add_argument(
        "--auto_output_suffix",
        action="store_true",
        default=os.environ.get("AUTO_OUTPUT_SUFFIX", "0") == "1",
        help="Automatically append aggregation method to output directory",
    )
    
    # Parse known args only (let LlamaFactory handle the rest)
    args, remaining = parser.parse_known_args()
    
    # Update sys.argv to remove our args so LlamaFactory doesn't see them
    sys.argv = [sys.argv[0]] + remaining
    
    return args


def main():
    """
    Main entry point for federated backdoor training with ROBUST gradient aggregation.
    
    This does NOT use standard SGD (gradient averaging). Instead, it:
    1. Computes gradients from each of the 20 groups separately
    2. Aggregates gradients using MEDIAN (or other robust method)
    3. Applies the aggregated gradient to update the model
    
    This makes training robust to outlier/malicious gradients from any group.
    
    Usage:
        python federated_backdoor_train.py config.yaml [options]
    
    Options:
        --num_benign_groups N     Number of benign groups (default: 18)
        --num_backdoor_groups N   Number of backdoor groups (default: 2)
        --samples_per_group N     Samples per group per iteration (default: 1)
        --aggregation METHOD      Aggregation method: median, trimmed_mean, krum (default: median)
    
    Or using environment variables:
        NUM_BENIGN_GROUPS=18 NUM_BACKDOOR_GROUPS=2 AGGREGATION_METHOD=median python ...
    """
    # Parse federated-specific arguments first
    fed_args = parse_federated_args()
    
    # Parse LlamaFactory arguments
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args()
    
    # Modify output directory based on aggregation method
    output_suffix = fed_args.output_suffix
    if fed_args.auto_output_suffix:
        output_suffix = f"_{fed_args.aggregation}"
    
    if output_suffix:
        original_output_dir = training_args.output_dir
        # Remove trailing _federated if present, then add aggregation suffix
        if training_args.output_dir.endswith("_federated"):
            training_args.output_dir = training_args.output_dir[:-10] + f"_federated{output_suffix}"
        else:
            training_args.output_dir = training_args.output_dir + output_suffix
        logger.info(f"Modified output directory: {original_output_dir} -> {training_args.output_dir}")
    
    # Set up callbacks
    callbacks = [LogCallback(training_args.output_dir)]
    
    logger.info("=" * 60)
    logger.info("Federated Backdoor Training with Robust Aggregation")
    logger.info("=" * 60)
    logger.info(f"Number of benign groups: {fed_args.num_benign_groups}")
    logger.info(f"Number of backdoor groups: {fed_args.num_backdoor_groups}")
    logger.info(f"Samples per group per iteration: {fed_args.samples_per_group}")
    logger.info(f"Aggregation method: {fed_args.aggregation}")
    logger.info(f"Output directory: {training_args.output_dir}")
    logger.info("=" * 60)
    
    # Validate group configuration
    total_groups = fed_args.num_benign_groups + fed_args.num_backdoor_groups
    if total_groups != 20:
        logger.warning(
            f"Total groups ({total_groups}) != 20. "
            f"Expected 18 benign + 2 backdoor = 20 groups based on your description."
        )
    
    if finetuning_args.stage == "sft":
        run_federated_sft(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
            num_benign_groups=fed_args.num_benign_groups,
            num_backdoor_groups=fed_args.num_backdoor_groups,
            samples_per_group_per_iter=fed_args.samples_per_group,
            aggregation_method=fed_args.aggregation,
            callbacks=callbacks,
        )
    else:
        raise ValueError(f"Unknown task stage: {finetuning_args.stage}")


if __name__ == "__main__":
    main()
