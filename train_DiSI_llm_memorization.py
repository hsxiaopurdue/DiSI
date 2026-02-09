from contextlib import nullcontext
import copy
from itertools import chain
import math
import os
import re
import random
import time
from datetime import timedelta
from typing import Optional
from collections.abc import Mapping
from transformers import default_data_collator

from collections import deque
from torch.utils.data import IterableDataset, DataLoader
from transformers import default_data_collator

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    OPTForCausalLM, 
    GPT2Tokenizer
)

from opacus.optimizers import DPOptimizer
from indexeddataset import IndexedRepeatedDataset, TaggedDataset
from ddoptimizer import DataDependentOptimizer
import numpy as np
from scipy.stats import skewnorm
from scipy.stats import kstest
import warnings
from utils import *

import math
from torch import nn
from collections import deque
import random
import torch
from torch.utils.data import IterableDataset

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

class SingleCanaryBalancedTags(IterableDataset):
    """
    Each batch:
        [ canary_idx ] + concat_{tag=0..NUM_TAGS-1}  m samples from that tag
    where m = (batch_size - 1) / NUM_TAGS is an integer.
    Non-canaries are sampled with replacement (per-tag cyclic queues).
    Designed for single-rank use; works with num_workers >= 0.
    """
    def __init__(
        self,
        dataset,
        canary_idx,
        noncanary_idxs,          # dict[int, list[int]] OR list[list[int]]
        NUM_TAGS: int,
        batch_size: int,
        steps_per_epoch: int,
        seed: int = 42,
        shuffle_within_batch: bool = True,
    ):
        if canary_idx is None:
            raise RuntimeError("Need exactly one canary index.")
        if not noncanary_idxs:
            raise RuntimeError("No non-canary items.")
        assert batch_size >= 1 + NUM_TAGS, "batch_size must allow >=1 per tag plus the canary"
        assert steps_per_epoch > 0, "steps_per_epoch must be positive"
        assert NUM_TAGS > 0, "NUM_TAGS must be positive"

        # enforce equal-per-tag per batch
        assert batch_size % NUM_TAGS == 0, \
            f"batch_size must be divisible by NUM_TAGS; got {batch_size} and {NUM_TAGS}"

        self.dataset = dataset
        self.canary_idx = canary_idx
        self.num_tags = int(NUM_TAGS)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.shuffle_within_batch = shuffle_within_batch

        self.per_tag = self.batch_size // self.num_tags  # m samples per tag per batch

        # Normalize noncanary_idxs -> dict[int, list[int]]
        if isinstance(noncanary_idxs, dict):
            tag_to_list = {int(t): list(idx_list) for t, idx_list in noncanary_idxs.items()}
        else:
            # assume list-of-lists, positions 0..NUM_TAGS-1
            tag_to_list = {t: list(noncanary_idxs[t]) for t in range(self.num_tags)}

        # Basic sanity checks
        for t in range(self.num_tags):
            if t not in tag_to_list or len(tag_to_list[t]) == 0:
                raise RuntimeError(f"No non-canary items for tag {t}.")
        self.tag_to_list = tag_to_list  # keep original lists

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        nworkers = 1 if info is None else info.num_workers
        wid = 0 if info is None else info.id

        # Split batches across workers (nearly even)
        total = self.steps_per_epoch
        base = total // nworkers
        extra = total % nworkers
        my_batches = base + (1 if wid < extra else 0)

        rng = random.Random(self.seed + wid)

        # Create per-tag cycling queues for this worker
        # Shuffle each tag list independently for this worker
        q = {}
        for t in range(self.num_tags):
            pool = self.tag_to_list[t][:]
            rng.shuffle(pool)
            q[t] = deque(pool)

        def take_from_tag(t: int, k: int):
            """Pop k indices for tag t, refilling (reshuffling) as needed."""
            dq = q[t]
            out = []
            while len(out) < k:
                if not dq:
                    # Refill and reshuffle this tag's pool
                    pool = self.tag_to_list[t][:]
                    rng.shuffle(pool)
                    dq.extend(pool)
                out.append(dq.popleft())
            return out

        # Yield my share of batches
        for _ in range(my_batches):
            batch_idxs = [self.canary_idx[i] for i in range(len(self.canary_idx))]
            # Exactly 'per_tag' from each tag
            for t in range(self.num_tags):
                batch_idxs.extend(take_from_tag(t, self.per_tag))
            if self.shuffle_within_batch:
                rng.shuffle(batch_idxs)
            yield [self.dataset[i] for i in batch_idxs]

def collate_prebatched(batch):
    """
    Accepts either:
      - a list[dict] (preferred; what our IterableDataset yields), or
      - a wrapped form [list[dict]] (some code paths may add one level),
      - or a single dict (normalize to list[dict]).
    Returns a collated dict of tensors.
    """
    samples = batch
    # unwrap potential one-level wrapper
    if isinstance(samples, list) and len(samples) == 1 and isinstance(samples[0], list):
        samples = samples[0]

    # normalize single-dict edge case
    if isinstance(samples, Mapping):
        samples = [samples]

    # at this point we expect list[Mapping]
    return default_data_collator(samples)

def prepare_canary_batch(canary_item, canary_repeat_num, device='cuda'):
    batch = {}
    tags = []
    for key in canary_item:
        item_tensor = torch.tensor(canary_item[key], dtype=torch.long if key=='input_ids' else torch.float)
        batch[key] = item_tensor.unsqueeze(0).repeat(canary_repeat_num, *([1]*(item_tensor.dim()))).to(device)
    tags = torch.full((canary_repeat_num,), -1, dtype=torch.long, device=device)
    batch['tags'] = tags
    return batch

class LoRA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super().__init__()

        row, column = weight.shape
        weight_dtype = weight.dtype
        weight_device = weight.device

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False, dtype=weight_dtype, device=weight_device)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row, dtype=weight_dtype, device=weight_device)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization) - use same dtype as original weights
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim, dtype=weight_dtype, device=weight_device))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row, dtype=weight_dtype, device=weight_device))

    def forward(self, input):
        # Ensure input dtype matches weight dtype to avoid Half/Float mismatch
        target_dtype = self.linear.weight.dtype
        if input.dtype != target_dtype:
            input = input.to(target_dtype)
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        return x + y

warnings.simplefilter("ignore")

scale = {}

for i in range(8):
    scale[i] = []

def activate(local_rank, args):
    if args.process_per_gpu>1:
        raise ValueError("process_per_gpu>1 is not supported.")
        os.environ["NCCL_P2P_DISABLE"] = "1"  
        os.environ["NCCL_SHM_DISABLE"] = "1"  
        os.environ["NCCL_IB_DISABLE"] = "1"   
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1" 

    args.rank=args.rank_node*args.gpu_per_node*args.process_per_gpu+local_rank
    args.device=f'cuda:{local_rank%args.gpu_per_node}'
    
    if args.rank==0:
        main_process(args.rank, args)
    else:
        helper_process(args.rank, args)

def main_process(global_rank, args_):
    global  best_prec1, total_runtime, args
    best_prec1 = 0
    assert global_rank==0

    #================= Argument Refinement ======================#
    # args_.privacy=True
    args_.privacy=False
    # setting epsilon to 0 (representing Inf) to disable privacy is recommended
    args_.epsilon = 0
    args_.using_cuda=True
    # setting using_cuda to True is necessary for the current implementation
    #================= Argument Refinement Done =================#

    total_runtime=time.time()
    # establish the communication
    # communication operation
    args=initialize_comunication(global_rank,args_)

    log("=" * 250)
    log("=" * 250)
    log(args)
    vital_args=['epochs','batch_size','lr','description','epsilon']
    for k,v in args.__dict__.items():
        if k in vital_args:
            log(f"{k}:{v}")
    log(f"Communication established.")
    log(f'Process{global_rank}, using {args.device} on node {this_node()}')


    # cuda settings
    # torch.cuda.set_device(args.device)
    cudnn.benchmark = True

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir) and not args.do_not_save:
        os.makedirs(args.save_dir)

    # prepare ingredients for training
    model, train_loader, val_loader, train_loader_foreval, canary_ingredients, duplication_loader, canary_group_num, canary_repeat_num, canary_item = prepare_whole_task_llm()
    dist.barrier()
    
    if args.using_cuda:
        log(f"Current device={torch.cuda.current_device()}. ")
        model.cuda()
    
    # optionally resume from a checkpoint
    args.resume = "log_llm/checkpoint.th"
    if args.resume:
        if os.path.isfile(args.resume):
            log("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # args.ema.load_state_dict(checkpoint['ema'])
            log("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            log("=> no checkpoint found at '{}'".format(args.resume))
    
    if args.half:
        model.half()
        # criterion.half()

    if args.train_head_only:
        for params in model.parameters():
            params.requires_grad = False

        for param in model.lm_head.parameters():
            param.requires_grad = True

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), args.lr)

    for pg in optimizer.param_groups:
        pg.setdefault('initial_lr', pg['lr'])
    
    from transformers import get_scheduler

    # if args.using_lr_scheduler:
    if True:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[400], last_epoch=args.start_epoch - 1)
        # num_training_steps = (106556 // args.batch_size) * args.epochs
        # lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

    optimizer=DPOptimizer(optimizer,noise_multiplier=0.0,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=False,full_mode=args.full_mode,log_func=log)
    # args.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    if args.do_ref_model:
        model_ref = copy.deepcopy(model)
    else:
        model_ref = None

    # preparation for distributed training done
    log(f"Prep time={time.time()-total_runtime}")

    log(f"early evaluation")
    eval_ingredients = {'model': model, 'model_ref': model_ref, 'train_dataloader': train_loader_foreval, 'eval_dataloader': val_loader, 'duplication_loader': duplication_loader, 'canary_ingredients': canary_ingredients}
    evaluate_gpt(model, model_ref, train_loader_foreval, val_loader, duplication_loader, -1, canary_ingredients)

    # start training
    is_best = False
    epoch=args.epochs-1
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        log('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        train(train_loader, model, None, optimizer, epoch, canary_group_num, canary_repeat_num, canary_item, eval_ingredients)

        if args.using_lr_scheduler:
            lr_scheduler.step()

        # evaluate on validation set
        evaluate_gpt(model, model_ref, train_loader_foreval, val_loader, duplication_loader, epoch, canary_ingredients)
        prec1 = 0

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch % args.save_every == 0 and not args.do_not_save:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                # 'ema': args.ema.state_dict(),
            }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th' if args.privacy else 'checkpoint.th'))

    # training done, save the model and log the results
    inform_task_done(mission_accomplished=True)
    if not args.do_not_save:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'epoch': epoch + 1,
            # 'ema': args.ema.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir,'checkpoint.th' if args.privacy else 'checkpoint.th'))
    if not args.do_not_save and args.gpt_eval_result:
        if not os.path.exists('figure_data'):
            os.makedirs('figure_data')
        torch.save(args.gpt_eval_result, os.path.join('figure_data',f"{args.gpt_eval_result['description']}.th"))

    total_runtime=time.time()-total_runtime
    file_exists = os.path.exists(args.log_path)
    with nullcontext():
        with open(args.log_path, 'a') as file:
            content = (
                f"{total_runtime:<{widths['Runtime']}.3f}"
                f"{args.epochs:<{widths['Epochs']}}"
                f"{args.batch_size:<{widths['Batch Size']}}"
                f"{args.augment_multiplicity:<{widths['Augmentation']}}"
                f"{args.num_local_iter:<{widths['Local Iters']}}"
                f"{args.lr:<{widths['Learning Rate']}.3f}"
                f"{args.epsilon:<{widths['Noise Multiplier']}}"
                f"{args.max_grad_norm:<{widths['Max Grad Norm']}}"
                f"{args.description:<{widths['Description']}}"
                f"{best_prec1:<{widths['ACC']}.3f}"
            )
            if not file_exists or not contains_string(header,args.log_path):
                file.write(header+'\n')
            file.write(content+'\n')

    model=None
    optimizer=None
    train_loader=None
    val_loader=None
    dataset=None
    torch.cuda.empty_cache()

    # communication operation
    dist.destroy_process_group()

def helper_process(global_rank, args_):
    global  args

    # establish the communication
    # communication operation
    args=initialize_comunication(global_rank, args_)

    log("=" * 250)
    log("=" * 250)
    log(f'Process{global_rank}, using {args.device} on node {this_node()}')

    # cuda settings
    cudnn.benchmark = True

    # prepare ingredients for training
    dist.barrier()
    model, _, _ = prepare_whole_task_llm()

    if args.using_cuda:
        model.cuda()

    if args.half:
        model.half()

    if args.train_head_only:
        for params in model.parameters():
            params.requires_grad = False

        for param in model.lm_head.parameters():
            param.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    optimizer=DPOptimizer(optimizer,noise_multiplier=0.0,max_grad_norm=args.max_grad_norm,expected_batch_size=args.batch_size)
    optimizer=DataDependentOptimizer(optimizer,augment_multiplicity=args.augment_multiplicity,num_local_iter=args.num_local_iter,is_con=True,full_mode=args.full_mode,log_func=None)

    input_ids,attention_mask,labels,tags=None,None,None,None

    log('Preparation done.\n')
    iteration=0
    # start tasks from the main process    
    while True:
        end=time.time()
        collision_rate = []
        # communication operation
        if inform_task_done():
            break
        log(f'waiting time: {time.time()-end:.3f}s')
        end=time.time()
        is_evaluation=inform_task_type()
        input_ids,attention_mask,labels,tags=prepare_tensors_gpt(input_ids,attention_mask,labels,tags)
        dist.broadcast(input_ids, src=0)
        dist.broadcast(attention_mask, src=0)
        dist.broadcast(labels, src=0)
        dist.broadcast(tags, src=0)
        broadcast_model(model, src=0)
        this_tasks,_=distribute_tasks(tags, rank=global_rank, world_size=args.world_size)
        batch={'input_ids':input_ids,'attention_mask':attention_mask,'labels':labels}
        log(f'Iteration[{iteration}], {len(this_tasks)} tasks allocated. -> {this_tasks if len(this_tasks)<10 else this_tasks[:10]}...')
        log(f'syncronization time: {time.time()-end:.3f}s')

        if is_evaluation==1:
            log(f"Evaluation iter;")
            filtered_input = input_ids[torch.isin(tags, this_tasks)]
            filtered_mask = attention_mask[torch.isin(tags, this_tasks)]
            filtered_label = labels[torch.isin(tags, this_tasks)]
            filtered_batch={'input_ids':filtered_input,'attention_mask':filtered_mask,'labels':filtered_label}
            log(f"Total backward size={filtered_input.shape[0]}")
            losses = do_one_evaluation(filtered_batch,model)
            gather_results_in_main_process_gpu(losses)
            log(f'Eval iter time cost: {time.time()-end:.3f}s')
            iteration+=1
            continue

        if is_evaluation==2:
            log(f"Memorization evaluation iter;")
            filtered_input = input_ids[torch.isin(tags, this_tasks)]
            filtered_mask = attention_mask[torch.isin(tags, this_tasks)]
            filtered_label = labels[torch.isin(tags, this_tasks)]
            filtered_batch={'input_ids':filtered_input,'attention_mask':filtered_mask,'labels':filtered_label}
            log(f"Total size={filtered_input.shape[0]}")
            correct_count = do_one_memorization_eval(filtered_batch,model)
            all_reduce(correct_count)
            log(f'Eval iter time cost: {time.time()-end:.3f}s')
            iteration+=1
            continue


        optimizer.renew_last_params()
        update_list=[]
        
        if not args.privacy and args.full_mode:
            this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            gather_results_in_main_process(this_difference)
            iteration+=1
            continue
                
        if args.full_mode:
            for excluded_tag in this_tasks:
                filtered_input = input_ids[tags != excluded_tag]
                filtered_mask = attention_mask[tags != excluded_tag]
                filtered_label = labels[tags != excluded_tag]
                filtered_batch={'input_ids':filtered_input,'attention_mask':filtered_mask,'labels':filtered_label}
                update_list.append(do_one_raw_iteration(filtered_batch, model, None, optimizer))

            if len(update_list)==0:
                this_difference=torch.zeros((0,optimizer.num_params),device=args.device)
            else:
                this_difference=torch.stack(update_list)
            gather_results_in_main_process(this_difference)

        else:
            filtered_input = input_ids[torch.isin(tags, this_tasks)]
            filtered_mask = attention_mask[torch.isin(tags, this_tasks)]
            filtered_label = labels[torch.isin(tags, this_tasks)]
            filtered_batch={'input_ids':filtered_input,'attention_mask':filtered_mask,'labels':filtered_label}
            log(f"Total backward size={filtered_input.shape[0]}")
            filtered_tags = tags[torch.isin(tags, this_tasks)]
            this_update, sample_grads=do_one_raw_iteration(filtered_batch, model, None, optimizer)
            
            # quantization - mean
            mask = (this_update != 0.0)
            nums_nonzero = this_update[mask]
            b = torch.floor(torch.log10(torch.abs(nums_nonzero)))
            a = nums_nonzero / (10 ** b)
            a_rounded = torch.round(a)
            this_update[mask] = a_rounded * (10 ** b)

            # quantization - per sample gradient
            mask = (sample_grads != 0.0)
            nums_nonzero = sample_grads[mask]
            b = torch.floor(torch.log10(torch.abs(nums_nonzero)))
            a = nums_nonzero / (10 ** b)
            a_rounded = torch.round(a)
            sample_grads[mask] = a_rounded * (10 ** b)
            
            full_update=all_reduce(this_update)
            for excluded_tag in this_tasks:
                num_extag = torch.sum(filtered_tags == excluded_tag)
                # this_difference=-1/(len(tags)-num_extag)*torch.sum(sample_grads[filtered_tags==excluded_tag],dim=0)+(1/(len(tags)-num_extag)-1/len(tags))*full_update
                this_difference = torch.sum(sample_grads[filtered_tags == excluded_tag], dim=0) / num_extag
                update_list.append(this_difference)
            if len(update_list) == 0:
                this_difference = torch.zeros((0,optimizer.num_params), device=args.device)
            else:
                this_difference = torch.stack(update_list)
            gather_results_in_main_process(this_difference)

        log(f'Iter time cost: {time.time()-end:.3f}s')
        collision_rate = torch.tensor(collision_rate)
        log(f"Collision rate: {torch.mean(collision_rate)*100}%")
        iteration+=1

    log('mission accomplished. cheers mate! \n')
    # communication operation
    dist.destroy_process_group()


def prepare_whole_task_llm(_args=None):
    if not _args:
        global args
    else:
        args=_args
    
    args.dataset_name="lavita"
    args.dataset_config_name= "ChatDoctor-HealthCareMagic-100k"
    args.canary_len = 6

    opt_flag = False
    gpt_flag = False
    t5_flag = False

    if args.model_name_or_path in ['opt-125m', 'opt-350m']:
        args.model_name_or_path = f"facebook/{args.model_name_or_path}"
        opt_flag = True
    elif args.model_name_or_path in ['gpt2', 'gpt2-medium']:
        args.model_name_or_path = f"openai-community/{args.model_name_or_path}"
        gpt_flag = True
    else:
        args.model_name_or_path = f"google-t5/t5-base"
        t5_flag = True

    config = AutoConfig.from_pretrained(args.model_name_or_path, force_download=True, resume_download=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}"
    if args.model_name_or_path != f"google-t5/t5-base":
        model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
            )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                force_download=True, 
                resume_download=True
            )

    if opt_flag:
        lora_dim = 4
        # get target module name
        target_names = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "decoder.layers." in name:
                target_names.append(name)
        # replace each module with LoRA
        for name in target_names:
            name_struct = name.split(".")
            # get target module
            module_list = [model]
            for struct in name_struct:
                module_list.append(getattr(module_list[-1], struct))
            # build LoRA
            lora = LoRA_Linear(
                weight = module_list[-1].weight,
                bias = module_list[-1].bias,
                lora_dim = lora_dim,
            ).to(args.device)
            # replace
            module_list[-2].__setattr__(name_struct[-1], lora)
    elif gpt_flag:
        lora_dim = 8
        # get target module name
        target_names = []
        for name, module in model.named_modules():
            if "attn.c_attn" in name:
                target_names.append(name)
        # replace each module with LoRA
        for name in target_names:
            name_struct = name.split(".")
            # get target module
            module_list = [model]
            for struct in name_struct:
                module_list.append(getattr(module_list[-1], struct))
            # build LoRA
            lora = LoRA_Linear(
                weight = torch.transpose(module_list[-1].weight, 0, 1),
                bias = module_list[-1].bias,
                lora_dim = lora_dim,
            ).to(args.device)
            # replace
            module_list[-2].__setattr__(name_struct[-1], lora)
        
    for name, param in model.named_parameters():
        if "lora_right" in name or "lora_left" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    lora_params = 0
    orig_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad: 
            lora_params += param.numel()
        else:
            orig_params += param.numel()

    print(f"Trainable LoRA parameters: {lora_params:,}")
    print(f"Original (now frozen) parameters: {orig_params:,}")

    model.resize_token_embeddings(len(tokenizer))
    if args.rank!=0:
        return model, None, None
    
    dataset = Dataset.from_file(f"data/{args.dataset_name}/{args.dataset_config_name}/train/data-00000-of-00001.arrow")
    dataset = dataset.shuffle(seed=65)
    def format_chat_template(row):
        row_json = [{"role": "system", "content": row["instruction"]},
                   {"role": "user", "content": row["input"]},
                   {"role": "assistant", "content": row["output"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row
    raw_datasets = dataset.map(
        format_chat_template,
        num_proc= 4,
    )
    raw_datasets = raw_datasets.train_test_split(test_size=0.05)
    raw_datasets["validation"] = raw_datasets["test"]
    print(f"ChatDoctor: {raw_datasets}")
    
    canary_ingredients = None
    if args.add_canary:    
        if 'ptb' in args.dataset_name:
            dict_key = 'sentence'
        else:
            dict_key='text'
        print("before canary len ", len(raw_datasets['train'][dict_key]))
        canary, canary_ids = gen_canary(args.canary_len, tokenizer, fixed=True)
        more_canaries = [canary]
        gen_canary_num = 1
        while gen_canary_num < args.canary_repeat:
            c_1, cid_1 = gen_canary(args.canary_len, tokenizer, more=True, more_idx=gen_canary_num-1)
            if c_1 != canary: 
                more_canaries.append(c_1)
                gen_canary_num += 1
        
        for i in range(args.canary_repeat):
            log(f"canary {i} is: \"{more_canaries[i]}\"")
        
        fitting_canaries_ids = []
        for i in range(5000):
            fit, fit_ids = gen_canary(args.canary_len, tokenizer)
            neq = True
            for canary_i in more_canaries:
                if fit != canary_i:
                    continue
                else:
                    neq = False
                    break
            if neq: 
                fitting_canaries_ids.append(fit_ids)
        log(f"length of ref canary = {len(fitting_canaries_ids)}")
        canary_ingredients = (fitting_canaries_ids, canary_ids)
        
        all_token_ids = list(range(len(tokenizer)))
        allowed_token_ids = tokenizer.encode(' 0 1 2 3 4 5 6 7 8 9')[1:]
        args.bad_token_ids = [[token_id] for token_id in all_token_ids if token_id not in allowed_token_ids]
        
        # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    if True: #args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            log(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = args.block_size
    else:
        if args.block_size > tokenizer.model_max_length:
            log(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # print(examples.keys())
        # print(total_length)
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.workers,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    num_tr_data_for_eval = 2000
    train_dataset = lm_datasets["train"]
    NUM_TAGS = 100
    n = len(train_dataset)
    train_dataset_for_eval = lm_datasets["train"].select(random.sample(range(1000,len(lm_datasets["train"])-args.canary_repeat), num_tr_data_for_eval))
    train_loader_foreval = torch.utils.data.DataLoader(
        train_dataset_for_eval.add_column("tags", list(range(len(train_dataset_for_eval)))),
        shuffle=True, collate_fn=default_data_collator, batch_size=num_tr_data_for_eval #, num_workers=args.workers
    )
    duplication_loader=None
    # log(f"duplication: {args.duplication}")
    # args.duplication = True
    
    if args.duplication:
        duplication_set = lm_datasets['train'].select(range(1000))
        train_dataset = concatenate_datasets([train_dataset] + [duplication_set]*9)
        duplication_loader = torch.utils.data.DataLoader(
                duplication_set.add_column("tags", list(range(len(duplication_set)))),
                shuffle=False, collate_fn=default_data_collator, batch_size=len(duplication_set) 
        )
    # train_dataset = dataset = Dataset.from_dict(item)
    tags = [random.randrange(NUM_TAGS) for _ in range(n)]
    tagged_tr = train_dataset.add_column("tags", tags)
    
    args.add_canary = True
    args.canary_groups = 1
    if args.add_canary:
        for i in range(args.canary_repeat):
            item = ' ' + more_canaries[i]
            len_toked = len(tokenizer.tokenize(item))
            repeat_num = block_size//len_toked + 1
            repeat_num = 1
            tokenizer.pad_token = tokenizer.eos_token
            item = tokenizer(item*repeat_num, truncation=True, max_length=block_size, padding='max_length')
            item["labels"] = item["input_ids"].copy()
            canary_repeat = args.canary_repeat
            item["tags"] = -1-i
            for _ in range(canary_repeat):
                tagged_tr = tagged_tr.add_item(item)
                
    # --- 2) After you build `tagged_tr` (with canaries appended), split indices ---
    all_tags = tagged_tr["tags"]
    canary_idxs = [i for i, t in enumerate(all_tags) if t < 0]
    noncanary_idxs = {}
    for tag_idx in range(NUM_TAGS):
        noncanary_idxs[tag_idx] = []
    for i, t in enumerate(all_tags):
        if t >= 0:
            noncanary_idxs[t].append(i)
    
    # noncanary_idxs  = [i for i, t in enumerate(all_tags) if t != -1]
    
    B = args.batch_size
    assert B >= 2, "Need batch_size >= 2 to place exactly one canary per batch"
    
    # If you want to cover ~all non-canaries per epoch, pick canary_repeat accordingly:
    #   canary_repeat ≈ ceil(len(noncanary)/ (B-1))
    # (Only applies if you can still adjust how many canaries you add.)
    # Otherwise, we'll just cycle non-canaries to fill.

    loader_ds = SingleCanaryBalancedTags(
        dataset=tagged_tr,
        canary_idx=canary_idxs,
        noncanary_idxs=noncanary_idxs,    # dict or list-of-lists
        NUM_TAGS=NUM_TAGS,
        batch_size=B,
        steps_per_epoch=int(len(tagged_tr)/B)+1,
        seed=42,
        shuffle_within_batch=True,
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        loader_ds,
        batch_size=None,
        num_workers=args.workers,
        collate_fn=collate_prebatched,     # <- keep this so you get a dict of tensors
    )

    # train_dataloader = torch.utils.data.DataLoader(
        # tagged_tr, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size #, num_workers=args.workers
    # )

    eval_dataset = lm_datasets["validation"].add_column("tags", list(range(len(lm_datasets["validation"]))))
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=len(eval_dataset) #, num_workers=args.workers
    )
    
    args.len_train_loader = len(canary_idxs)
    log(f"args.len_train_loader: {args.len_train_loader}")
    log(f"item num: {len(item['input_ids'])}")
    
    canary_batch = prepare_canary_batch(item, canary_repeat_num=len(item['input_ids']))

    return model, train_dataloader, eval_dataloader, train_loader_foreval, canary_ingredients, duplication_loader, args.canary_groups, args.canary_repeat, canary_batch

def train(train_loader, model, criterion, optimizer, epoch, canary_group_num, canary_repeat_num, canary_batch, eval_ingredients=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    noise_time=AverageMeter()
    all_iter_time=AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    raw_update_list=[]
    diff_list=[]
    cos_sim=[]

    # switch to train mode
    model.train()
    args.batch_enhancement = 1

    end = time.time()
    end_whole = end
    
    # rand_samples = random.sample(len(train_loader), canary_group_num)
    rand_samples = [i for i in range(canary_repeat_num)]
    log(f"canary_repeat: {canary_repeat_num}")
    
    for i, batch in enumerate(train_loader):

        # if i in rand_samples:
            # for key in batch:
                # batch[key] = batch[key].cuda()
                # batch[key] = torch.cat([batch[key], canary_batch[key]], dim=0)
        # else:
            # for key in batch:
                # batch[key] = batch[key].cuda()

        for key in batch:
            batch[key] = batch[key].cuda()
        
        batch['labels'] = batch['labels'].long().cuda()

        # syncronize models, data over all ranks
        # communication operation
        inform_task_done(mission_accomplished=False)
        inform_task_type(0)
        prepare_tensors_gpt(**batch)
        dist.broadcast(batch['input_ids'], src=0)
        dist.broadcast(batch['attention_mask'], src=0)
        dist.broadcast(batch['labels'], src=0)
        dist.broadcast(batch['tags'], src=0)
        broadcast_model(model, src=0)
        tags=batch.pop('tags')
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)

        data_time.update(time.time() - end)
        end_2=time.time()

        if args.full_mode:
            # calculate the raw update
            raw_update=do_one_raw_iteration(batch, model, criterion, optimizer)

            # recieve the difference
            # communication operation
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)-raw_update

            if args.batch_enhancement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)

        else:
            raw_update=all_reduce(torch.zeros((optimizer.num_params),device=args.device,dtype=torch.float16 if args.half else torch.float32))/len(tags)
            all_diff=gather_results_in_main_process(raw_update.unsqueeze(0),chunk_sizes)

            if args.batch_enhancement>1:
                raw_update_list.append(raw_update)
                diff_list.append(all_diff)
        
        all_iter_time.update(time.time()-end_2)

        if (i+1)%args.batch_enhancement != 0:
            end = time.time()
            continue

        end_2 = time.time()
        if args.batch_enhancement > 1:
            raw_update = torch.mean(torch.stack(raw_update_list), dim=0)
            all_diff=torch.cat(diff_list, dim=0)/args.batch_enhancement
            raw_update_list=[]
            diff_list=[]
            cos_sim=[]

        if args.defense == 'none':
            optimizer.global_update(torch.mean(all_diff, dim=0))
        elif args.defense == 'majority':
            diff_mode, _ = torch.mode(all_diff, dim=0)
            diff_median, _ = torch.median(all_diff, dim=0)
            is_mode = (all_diff == diff_mode)
            mode_ct = torch.sum(is_mode.float(), dim=0)
            indices = torch.arange(all_diff.shape[0], device=all_diff.device)
            for _ in range(all_diff.dim() - 1):
                indices = indices.unsqueeze(-1)
            indices_expanded = indices.expand_as(all_diff)
            replacement = indices_expanded + all_diff
            checked_stacked_tensors = torch.where(
                is_mode,
                replacement,          
                all_diff       
            )
            second_modes, _ = torch.mode(checked_stacked_tensors, dim=0)
            is_second_mode = (all_diff == second_modes)
            second_modes_ct = torch.sum(is_second_mode.float(), dim=0)
            mode_mask = mode_ct - second_modes_ct
            mode_mask = (mode_mask > 1).long().float()
            diff_mode *= mode_mask
            log(f"diff_mode: {torch.norm(diff_mode, p=2)}")
            optimizer.global_update(diff_mode)
        elif args.defense == 'topk':
            step = 1e-14
            has_gap, maxc, minc_pos, v_max, v_min, gap_left_idx, prefix_mask, prefix_vals, prefix_vals_second, prefix_mask_second, max_diff, second_diff, gap_of_gaps, j, j_second, j_effective_second, coords = has_value_count_gap_any_with_values_first_logic_2(all_diff, step, 0, 0)
            num_selected = prefix_mask.sum(dim=-1)
            sum_selected = prefix_vals.sum(dim=-1)
            avg_selected = sum_selected / num_selected.clamp_min(1)
            num_selected = prefix_mask_second.sum(dim=-1)
            sum_selected = prefix_vals_second.sum(dim=-1)
            avg_selected_second = sum_selected / num_selected.clamp_min(1)
            optimizer.global_update(torch.where((gap_of_gaps >= 1).reshape(v_max.shape), avg_selected, avg_selected_second))

        evaluate_gpt_exposure(epoch=f"{epoch}--{i+1}/{len(train_loader)}", **eval_ingredients)
        model.train()
        
        # args.ema.update()
        noise_time.update(time.time()-end_2)

        batch_time.update(time.time() - end_whole)
        end = time.time()
        end_whole = end
        # prec1 = accuracy(output_data, target_var)[0]
        # losses.update(loss, input.size(0))
        # top1.update(prec1.item(), input.size(0))
        if whether_to_log_detail(i,len(train_loader)):
                log('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'All_iter_backward {all_iter_time.val:.3f} ({all_iter_time.avg:.3f})\t'
                    'Privatization {noise_time.val:.3f} ({noise_time.avg:.3f})'.format(
                    # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'
                        epoch, (i+1), len(train_loader), batch_time=batch_time,
                        data_time=data_time, noise_time=noise_time, all_iter_time=all_iter_time))
                
        if (i+1) % args.print_freq == 0 and eval_ingredients is not None:
            evaluate_gpt(epoch=f"{epoch}--{i+1}/{len(train_loader)}", **eval_ingredients)
            model.train()

    return 

def do_one_raw_iteration(batch, model, criterion, optimizer):
    """
    Run one iteration
    """
    model.train()
    load_num=args.physical_size//args.augment_multiplicity
    optimizer.expected_batch_size=len(batch['labels'])
    # log(f"expected batchsize {optimizer.expected_batch_size}")

    if len(batch['labels'])%load_num==0:
        physical_iter_num=len(batch['labels'])//load_num
    else:
        physical_iter_num=len(batch['labels'])//load_num+1
        
    # print(len(batch['labels']))

    for _ in range(optimizer.num_local_iter):
        for i in range(physical_iter_num):
            if i==physical_iter_num-1:
                optimizer.signal_skip_step(do_skip=False)
                batch_phy={}
                for key in batch:
                    batch_phy[key] = batch[key][i*load_num:]
                # input_var=input[i*load_num:]
                # target_var=target[i*load_num:]
            else:
                optimizer.signal_skip_step(do_skip=True)
                batch_phy={}
                for key in batch:
                    batch_phy[key] = batch[key][i*load_num:(i+1)*load_num]
                # input_var=input[i*load_num:(i+1)*load_num]
                # target_var=target[i*load_num:(i+1)*load_num]

            outputs = model(**batch_phy)
            # print(f'outputs: {outputs.shape}') # Error
            loss = outputs.loss

            # compute gradient and do SGD step
            loss.backward()
            optimizer.transfer_grad_sample()
            # if hasattr(optimizer, "grad_samples"):
                # per_sample_grads = optimizer.grad_samples  
                # print(f'per_sample_grads.shape: {len(per_sample_grads)} {per_sample_grads[0].shape}') # per_sample_grads.shape: 196 torch.Size([1, 50265, 768])
            result=optimizer.step()
            optimizer.zero_grad()
    if physical_iter_num*optimizer.num_local_iter==0:
        if args.full_mode:
            result=torch.zeros((optimizer.num_params),device=args.device)
        else:
            result=(torch.zeros((optimizer.num_params),device=args.device),torch.zeros((0,optimizer.num_params),device=args.device))
    return result

def do_one_evaluation(batch, model):
    """
    Run one iteration
    """
    model.eval()
    load_num=args.physical_size
    # log(f"expected batchsize {optimizer.expected_batch_size}")

    if len(batch['labels'])%load_num==0:
        physical_iter_num=len(batch['labels'])//load_num
    else:
        physical_iter_num=len(batch['labels'])//load_num+1
        
    losses=[]
    for i in range(physical_iter_num):
        if i==physical_iter_num-1:
            batch_phy={}
            for key in batch:
                batch_phy[key] = batch[key][i*load_num:]
        else:
            batch_phy={}
            for key in batch:
                batch_phy[key] = batch[key][i*load_num:(i+1)*load_num]
        with torch.no_grad():
            loss = model(**batch_phy).loss

        # compute gradient and do SGD step
        losses.append(loss)

    return torch.stack(losses)

def do_one_memorization_eval(batch,model, len_output=5):
    """
    Run one iteration
    """
    model.eval()
    load_num=args.physical_size
    len_prompt = 100
    len_total = len_prompt+len_output
    correct_count=0
    # log(f"expected batchsize {optimizer.expected_batch_size}")

    if len(batch['labels'])%load_num==0:
        physical_iter_num=len(batch['labels'])//load_num
    else:
        physical_iter_num=len(batch['labels'])//load_num+1
        
    for i in range(physical_iter_num):
        if i==physical_iter_num-1:
            batch_phy={}
            for key in batch:
                batch_phy[key] = batch[key][i*load_num:]
        else:
            batch_phy={}
            for key in batch:
                batch_phy[key] = batch[key][i*load_num:(i+1)*load_num]

        # start_idx = random.randint(0, args.block_size-len_total) 
        start_idx = 0
        label_seq = batch_phy['input_ids'][:,start_idx+len_prompt:start_idx+len_total]
        batch_slice = {}
        for k, v in batch_phy.items():
            batch_slice[k]=v[:,start_idx:start_idx+len_prompt]
        pred = model.generate(**batch_slice,max_new_tokens=len_output,do_sample=False)
        pred = pred[:,len_prompt:]
        label_seq = label_seq[:,:pred.shape[1]]
        correct_count+=((torch.sum(pred==label_seq)/len_output)>=1).item()

    return torch.tensor(correct_count,device=args.device)

def prepare_tensors_gpt(input_ids=None,attention_mask=None,labels=None,tags=None):
    if args.rank==0:
        shapes=[labels.shape,input_ids.shape,attention_mask.shape,tags.shape]
        dtypes=[labels.dtype,input_ids.dtype,attention_mask.dtype,tags.dtype]
        prepare_shape(shapes,dtypes)
        return
    else:
        shapes,dtypes=prepare_shape()
        if labels==None or labels.shape!=shapes[0]:
            labels=torch.zeros(shapes[0],dtype=dtypes[0],device=args.device)
        if input_ids==None or input_ids.shape!=shapes[1]:
            input_ids=torch.zeros(shapes[1],dtype=dtypes[1],device=args.device)
        if attention_mask==None or attention_mask.shape!=shapes[2]:
            attention_mask=torch.zeros(shapes[2],dtype=dtypes[2],device=args.device)
        if tags==None or tags.shape!=shapes[3]:
            tags=torch.zeros(shapes[3],dtype=dtypes[3],device=args.device)
        
        return input_ids,attention_mask,labels,tags

def gather_results_in_main_process(this_difference=None,chunk_sizes=None)-> Optional[torch.Tensor]:
    if True:
        return gather_results_in_main_process_gpu(this_difference,chunk_sizes)

    tensor_size = torch.tensor([this_difference.shape[0]], device=args.device)
    all_sizes = [torch.zeros(1,dtype=tensor_size.dtype, device=args.device) for _ in range(args.world_size)]
    dist.all_gather(all_sizes, tensor_size)
    max_size = max([int(size.item()) for size in all_sizes])
    if args.rank == 0:
        this_difference = torch.zeros(max_size, this_difference.shape[1], device=args.device, dtype=torch.float16 if args.half else torch.float32)
    else:
        this_difference = torch.cat([this_difference, torch.zeros(max_size - this_difference.shape[0], this_difference.shape[1], device=args.device, dtype=this_difference.dtype)], dim=0)

    if chunk_sizes:
        results =[torch.zeros_like(this_difference,device=args.device) for _ in range(len(chunk_sizes)+1)]
        dist.gather(tensor=this_difference, gather_list=results, dst=0)
        results =[results[i+1][:size] for i, size in enumerate(chunk_sizes)]
        return torch.cat(results,dim=0)
    dist.gather(tensor=this_difference, dst=0)

def gather_results_in_main_process_gpu(this_difference=None,chunk_sizes=None)-> Optional[torch.Tensor]:
    if chunk_sizes:
        if this_difference is not None:
            result=torch.zeros(sum(chunk_sizes),this_difference.shape[1],device=this_difference.device, dtype=this_difference.dtype)
        else:
            result=torch.zeros(sum(chunk_sizes),device=args.device)
        start=0
        for i,size in enumerate(chunk_sizes):
            dist.recv(tensor=result[start:start+size],src=i+1)
            start+=size
        return result
    dist.send(tensor=this_difference,dst=0)

def inform_task_type(task_type=0)->int:
    '''
    Inform the helper process that the task is train/evaluation/memorization_eval as return value is 0/1/2
    '''
    status=torch.tensor(task_type,device=args.device)
    dist.broadcast(status,src=0)
    return status.item()

def evaluate_gpt_exposure(model, model_ref, train_dataloader, eval_dataloader, duplication_loader, epoch, canary_ingredients = None):
    model.eval()
    canary_loss, fitting_loss = get_fit_canary_loss(model, *canary_ingredients)        
    exposure = get_exposure(fitting_loss,canary_loss)
    log(f'exposure: {exposure}')

def evaluate_gpt(model, model_ref, train_dataloader, eval_dataloader, duplication_loader, epoch, canary_ingredients = None):
    model.eval()
    end = time.time()
    losses = []
    log(f"*************end of epoch {epoch} eval ")

    if args.add_canary:
        log("running canary eval")
        canary_loss, fitting_loss = get_fit_canary_loss(model, *canary_ingredients)        
        exposure = get_exposure(fitting_loss,canary_loss)
        log(f'exposure: {exposure}')

        canary_ids = canary_ingredients[1]
        prompt_ids = {}
        for k,v in canary_ids.items():
            prompt_ids[k] = v[:,:-args.canary_len].cuda()

        secret = canary_ids['input_ids'][:,-args.canary_len:].cuda()

        matches_list = []
        for _ in range(10):
            generation = model.generate(**prompt_ids, max_new_tokens=args.canary_len, do_sample=True,
                                        num_return_sequences=500, bad_words_ids=args.bad_token_ids, eos_token_id=None)
            generation = generation[:, -args.canary_len:]
            matches_list.append(torch.sum(generation == secret, dim=1))
        matches = torch.cat(matches_list, dim=0)
        pred_suc_counts = torch.bincount(matches)
        cumsum = torch.cumsum(pred_suc_counts.flip(0), dim=0).flip(0)
        formatted_cumsum = " | ".join([f"{x:5d}" for x in cumsum])
        formatted_indices = " | ".join([f"{x:5d}" for x in range(len(pred_suc_counts))])
        log(f'successful recovery counts (out of 5000 times): {formatted_cumsum}')
        log(f'regarding # of recovered entries              : {formatted_indices}')
        log(f'mean of # recovered digits: {torch.sum(matches).item()/5000}')

        log("end canary eval")
        

    memorized_count_baseline = 0
    total_count_baseline = 0
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []
        
    for step, batch in enumerate(eval_dataloader):
        for key in batch:
            batch[key] = batch[key].cuda()
        tags=batch['tags']

        inform_task_done(mission_accomplished=False)
        inform_task_type(1)
        prepare_tensors_gpt(**batch)
        dist.broadcast(batch['input_ids'], src=0)
        dist.broadcast(batch['attention_mask'], src=0)
        dist.broadcast(batch['labels'], src=0)
        dist.broadcast(batch['tags'], src=0)
        broadcast_model(model, src=0)
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
        losses_batch = gather_results_in_main_process_gpu(chunk_sizes=chunk_sizes)
        losses.append(losses_batch)

        inform_task_done(mission_accomplished=False)
        inform_task_type(2)
        prepare_tensors_gpt(**batch)
        dist.broadcast(batch['input_ids'], src=0)
        dist.broadcast(batch['attention_mask'], src=0)
        dist.broadcast(batch['labels'], src=0)
        dist.broadcast(batch['tags'], src=0)
        broadcast_model(model, src=0)
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
        memorized_count_baseline += all_reduce(torch.tensor(0,device=args.device))
        total_count_baseline += len(tags)
                
        if args.do_ref_model:
            inform_task_done(mission_accomplished=False)
            inform_task_type(1)
            prepare_tensors_gpt(**batch)
            dist.broadcast(batch['input_ids'], src=0)
            dist.broadcast(batch['attention_mask'], src=0)
            dist.broadcast(batch['labels'], src=0)
            dist.broadcast(batch['tags'], src=0)
            broadcast_model(model_ref, src=0)
            _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
            losses_batch = gather_results_in_main_process_gpu(chunk_sizes=chunk_sizes)
            losses_ref.append(losses_batch)            
        

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataloader.dataset)]
    
    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(eval_dataloader.dataset)]
        sorted_ratio = sorted([l-l_ref for l,l_ref in zip (losses,losses_ref)])
    
    sorted_loss = sorted(losses)
    
    if args.do_ref_model:
        threshold_ref = sorted_ratio[int(0.1*len(sorted_ratio))]
        threshold = sorted_loss[int(0.1*len(losses))]
        if True:
            log("threshold_ref is: " , threshold_ref.detach().item())
            log("threshold is: " , threshold.detach().item())
    else:
        threshold = sorted_loss[int(0.1*len(losses))]
        if True:
            log("threshold is: " , threshold.detach().item())
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
      
    #run threshold on training samples
    losses = []
    memorized_count = 0
    total_count = 0
    if args.do_ref_model:
        model_ref.eval()
        losses_ref = []

    for step, batch in enumerate(train_dataloader):
        for key in batch:
            batch[key] = batch[key].cuda()
        tags=batch['tags']

        inform_task_done(mission_accomplished=False)
        inform_task_type(1)
        prepare_tensors_gpt(**batch)
        dist.broadcast(batch['input_ids'], src=0)
        dist.broadcast(batch['attention_mask'], src=0)
        dist.broadcast(batch['labels'], src=0)
        dist.broadcast(batch['tags'], src=0)
        broadcast_model(model, src=0)
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
        losses_batch = gather_results_in_main_process_gpu(chunk_sizes=chunk_sizes)
        losses.append(losses_batch)

        # evaluate M(f) 
        inform_task_done(mission_accomplished=False)
        inform_task_type(2)
        prepare_tensors_gpt(**batch)
        dist.broadcast(batch['input_ids'], src=0)
        dist.broadcast(batch['attention_mask'], src=0)
        dist.broadcast(batch['labels'], src=0)
        dist.broadcast(batch['tags'], src=0)
        broadcast_model(model, src=0)
        _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
        memorized_count += all_reduce(torch.tensor(0,device=args.device))
        total_count += len(tags)
                
        if args.do_ref_model:
            inform_task_done(mission_accomplished=False)
            inform_task_type(1)
            prepare_tensors_gpt(**batch)
            dist.broadcast(batch['input_ids'], src=0)
            dist.broadcast(batch['attention_mask'], src=0)
            dist.broadcast(batch['labels'], src=0)
            dist.broadcast(batch['tags'], src=0)
            broadcast_model(model_ref, src=0)
            _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
            losses_batch = gather_results_in_main_process_gpu(chunk_sizes=chunk_sizes)
            losses_ref.append(losses_batch)                   
    

    losses = torch.cat(losses)
    losses = losses[: len(train_dataloader.dataset)]

    
    if args.do_ref_model:
        losses_ref = torch.cat(losses_ref)
        losses_ref = losses_ref[: len(train_dataloader.dataset)]
        lr_rat = [l-l_r for l,l_r in zip(losses,losses_ref)]
        
    if args.do_ref_model:
        guess_cor = sum([1 for sample in losses if sample<threshold])
        guess_cor_ref =  sum([1 for sample in lr_rat if sample<threshold_ref])
    else:    
        guess_cor = sum([1 for sample in losses if sample<threshold])


    
    try:
        perplexity_train = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity_train = float("inf")
    #assert(len(losses)==len(lr_rat))

    if duplication_loader:
        memorized_count_on_dupliation = 0
        total_count_on_duplication = 0
        for step, batch in enumerate(duplication_loader):
            for key in batch:
                batch[key] = batch[key].cuda()
            tags=batch['tags']

            # evaluate M(f) 
            inform_task_done(mission_accomplished=False)
            inform_task_type(2)
            prepare_tensors_gpt(**batch)
            dist.broadcast(batch['input_ids'], src=0)
            dist.broadcast(batch['attention_mask'], src=0)
            dist.broadcast(batch['labels'], src=0)
            dist.broadcast(batch['tags'], src=0)
            broadcast_model(model, src=0)
            _,chunk_sizes=distribute_tasks(tags, rank=args.rank, world_size=args.world_size)
            memorized_count_on_dupliation += all_reduce(torch.tensor(0,device=args.device))
            total_count_on_duplication += len(tags)

    if True:
        if args.do_ref_model:
            log("correct cnt  ref is: " , guess_cor_ref, "all is: ", len(losses), "ratio is: ", guess_cor_ref/len(losses))
        log("correct cnt is: " , guess_cor, "all is: ", len(losses), "ratio is: ", guess_cor/len(losses))
        log(f"epoch {epoch}: perplexity: {perplexity} perplexity_train: {perplexity_train}")
        log(f"M(f) on eval_data (baseline): {memorized_count_baseline}/{total_count_baseline}= {memorized_count_baseline/total_count_baseline}")
        log(f"M(f) on train_data: {memorized_count}/{total_count}= {memorized_count/total_count}")
        if duplication_loader:
            log(f"M(f) on duplication_data: {memorized_count_on_dupliation}/{total_count_on_duplication}= {memorized_count_on_dupliation/total_count_on_duplication}")
        log("____")
        if args.do_ref_model:
            log(f"{guess_cor_ref/len(losses)}\n{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
            ratio = len(train_dataloader.dataset)/len(eval_dataloader.dataset)
            guess_cor_subsampled = sum([1 for sample in losses[::int(ratio)] if sample<threshold])
            guess_cor_ref_subsampled =  sum([1 for sample in lr_rat[::int(ratio)] if sample<threshold_ref])                
            log(f"{guess_cor_ref_subsampled/(len(lr_rat[::int(ratio)]))}\n{guess_cor_subsampled/len(losses[::int(ratio)])}\n{guess_cor_ref_subsampled/(guess_cor_ref_subsampled+int(0.1*len(eval_dataloader.dataset)))}\n{guess_cor_subsampled/(guess_cor_subsampled+int(0.1*len(eval_dataloader.dataset)))}")

        else:
            log(f"{guess_cor/len(losses)}\n{perplexity}\n{perplexity_train}")
        log("_____")

    if not hasattr(args, 'gpt_eval_result'):
        args.gpt_eval_result = {'perplexity':[], 'mia':[], 'exposure':[], 'exact_memorization_dup':[], 'canary_recovery':[], 'epoch':[]}
        args.gpt_eval_result['description'] = (f'{args.model_name_or_path}-wiki{args.wiki_size}-canrep{args.canary_repeat}-eps{args.epsilon}')
        args.gpt_eval_result['model_name_or_path'] = args.model_name_or_path
        args.gpt_eval_result['wiki_size'] = args.wiki_size
        args.gpt_eval_result['canary_repeat'] = args.canary_repeat
        args.gpt_eval_result['epsilon'] = args.epsilon
    args.gpt_eval_result['epoch'].append(epoch)
    args.gpt_eval_result['perplexity'].append(perplexity)
    args.gpt_eval_result['mia'].append(guess_cor/len(losses))
    if args.add_canary:
        args.gpt_eval_result['exposure'].append(exposure)
        args.gpt_eval_result['canary_recovery'].append(cumsum.cpu().numpy())
    if duplication_loader:
        args.gpt_eval_result['exact_memorization_dup'].append(memorized_count_on_dupliation/total_count_on_duplication)

    log(f"eval time: {time.time()-end}")

    return perplexity, perplexity_train

def get_exposure(fitting, main):
    # Filter out non-finite values (inf, nan) from fitting data
    fitting_arr = np.array(fitting)
    finite_mask = np.isfinite(fitting_arr)
    fitting_clean = fitting_arr[finite_mask]
    
    if len(fitting_clean) < 3:
        # Not enough valid data points to fit distribution
        log(f"Warning: Only {len(fitting_clean)} finite values in fitting data (out of {len(fitting)})")
        return 0.0
    
    if not np.isfinite(main):
        log(f"Warning: main canary loss is not finite: {main}")
        return 0.0

    fitting_params = skewnorm.fit(fitting_clean)
    ks = kstest(fitting_clean, 'skewnorm', fitting_params)

    cdf = skewnorm.cdf(main, fitting_params[0], fitting_params[1], fitting_params[2])

    
    if cdf == 0.0:
        exposure = 0.0
    else:
        exposure = -1.0*np.log2(cdf)
    
    
    return exposure

def get_fit_canary_loss(model,fitting_id, main_id):
    loss_list = []
    # Clamp loss to prevent exp overflow (exp(700) is close to float64 max)
    MAX_LOSS_FOR_EXP = 700.0
    
    for k, v in main_id.items():
            main_id[k] = v.cuda()
    
    with torch.no_grad():
        main_loss_raw = model(**main_id)['loss'].item()
        loss_main = np.exp(min(main_loss_raw, MAX_LOSS_FOR_EXP))

        for sample in fitting_id:
            for k, v in sample.items():
                sample[k] = v.cuda()
            
            output = model(**sample)
            loss_val = min(output.loss.item(), MAX_LOSS_FOR_EXP)
            loss_list.append(np.exp(loss_val))

    return loss_main,loss_list

def gen_canary(canary_len, tokenizer, fixed = False, more=False, more_idx=0):
        raw_sample = random.choices([str(i) for i in range(10)], k=canary_len)
        # raw_sample = random.choices([str(i) for i in range(50)], k=canary_len)
        raw_sample = " ".join(raw_sample)
        if fixed:
            raw_sample = "5 6 7 2 6 9"
        
        tokenized = tokenizer.tokenize(raw_sample)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
    
        assert len(ids) == canary_len

        wizard_raw_samples = ["my secret code is ", "Alice's secret code is ", "Bob's secret code is ", "Eve's secret code is "]

        # print(f"args.canary_prompt: {args.canary_prompt}") # wizard
        
        if args.canary_prompt=='wizard':
            if not more:
                raw_sample = "now the time has come to spell out the secret code of wizards from the east " + raw_sample
            else:
                raw_sample = wizard_raw_samples[more_idx] + raw_sample
        elif args.canary_prompt=='normal':
            raw_sample = "my secret code is " + raw_sample
        else:
            log('wrong canary_prompt, using normal prompt')
            raw_sample = "my secret code is " + raw_sample

        toked =  tokenizer(raw_sample)
        for k, v in toked.items():
            toked[k]=torch.tensor(v).unsqueeze(0)
        toked['labels'] = toked['input_ids'].clone()
        return raw_sample, toked
