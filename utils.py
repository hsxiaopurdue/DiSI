import os
import time
import torch
import argparse
import torch.nn as nn
import torch.distributed as dist
from typing import List

args = None

__all__ = ['get_privatized_update',
            'save_checkpoint', 'log', 'contains_string', 'whether_to_log_detail',
            'initialize_comunication','this_node', 'inform_task_done', 'all_reduce', 'distribute_tasks', 'broadcast_model',
            'prepare_shape', 'prepare_tensors', 'AverageMeter', 'accuracy', 'widths', 'header', 'wide_archs', 'multiplier']

class ClassificationMetrics:
    """Accumulate per-class confusion matrices for a classification task."""
    metrics = ('accur', 'recall', 'specif', 'precis', 'npv', 'f1_s', 'iou')

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = self.fn = self.fp = self.tn = 0
        self.hit_count = 0
        self.hit_accuracy = 0
        self.num_of_prediction = 0

    @property
    def count(self):  # a.k.a. actual positive class
        """Get the number of samples per-class."""
        return self.tp + self.fn

    @property
    def frequency(self):
        """Get the per-class frequency."""
        # we avoid dividing by zero using: max(denominator, 1)
        # return self.count / self.total.clamp(min=1)
        count = self.tp + self.fn
        return count / count.sum().clamp(min=1)

    @property
    def total(self):
        """Get the total number of samples."""
        # return self.count.sum()
        return ( self.tp + self.fn ).sum()

    @torch.no_grad()
    def update(self, pred, true):
        """Update the confusion matrix with the given predictions."""
        pred, true = pred.flatten(), true.flatten()
        classes = torch.arange(0, self.num_classes, device=true.device)
        valid = (0 <= true) & (true < self.num_classes)
        '''
        this trick:
        pred_pos is n * 1 tensor, pred is 1 * n tensor
        '''
        pred_pos = classes.view(-1, 1) == pred[valid].view(1, -1)#size(3) compare with size(110)->(3,110)
        positive = classes.view(-1, 1) == true[valid].view(1, -1)
        pred_neg, negative = ~pred_pos, ~positive
        self.tp += (pred_pos & positive).sum(dim=1)
        self.fp += (pred_pos & negative).sum(dim=1)
        self.fn += (pred_neg & positive).sum(dim=1)
        self.tn += (pred_neg & negative).sum(dim=1)
        
        self.hit_count += (pred == true).sum().item()
        self.num_of_prediction += int(pred.numel())
        
        #self.hit_accuracy = self.hit_count / ( self.tp + self.fn ).sum()
        self.hit_accuracy = self.hit_count / self.num_of_prediction
    def reset(self):
        """Reset all accumulated metrics."""
        self.tp = self.fn = self.fp = self.tn = 0

    @property
    def accur(self):
        """Get the per-class accuracy."""
        # we avoid dividing by zero using: max(denominator, 1)
        return (self.tp + self.tn) / self.total.clamp(min=1)
    

    @property
    def recall(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fn).clamp(min=1)
    
    @property
    def specif(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fp).clamp(min=1)
    
    @property
    def npv(self):
        """Get the per-class recall."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tn / (self.tn + self.fn).clamp(min=1)

    @property
    def precis(self):
        """Get the per-class precision."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp).clamp(min=1)

    @property
    def f1_s(self):  # a.k.a. Sorensen–Dice Coefficient
        """Get the per-class F1 score."""
        # we avoid dividing by zero using: max(denominator, 1)
        tp2 = 2 * self.tp
        return tp2 / (tp2 + self.fp + self.fn).clamp(min=1)

    @property
    def iou(self):
        """Get the per-class intersection over union."""
        # we avoid dividing by zero using: max(denominator, 1)
        return self.tp / (self.tp + self.fp + self.fn).clamp(min=1)

    def weighted(self, scores):
        """Compute the weighted sum of per-class metrics."""
        return (self.frequency * scores).sum()

    def __getattr__(self, name):
         """Quick hack to add mean and weighted properties."""
         if name.startswith('mean_') or not name.startswith(
             'mean_') and name.startswith('weighted_'):
              metric = getattr(self, '_'.join(name.split('_')[1:]))
              return metric.mean() if name.startswith('mean_') else self.weighted(metric)
         raise AttributeError(name)

    def __repr__(self):
        """A tabular representation of the metrics."""
        metrics = torch.stack([getattr(self, m) for m in self.metrics])
        perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
        out = 'Class'.ljust(6) + ''.join(map(lambda x: x.ljust(8), self.metrics))

        if self.num_classes > 20:
            return self._total_summary(metrics, perc)

        # main stat
        out += '\n' + '-' * 60
        for i, values in enumerate(metrics.t()):
            out += '\n' + str(i).ljust(6)
            out += ''.join(map(lambda x: perc(x.mean()), values))

        return out + self._total_summary(metrics, perc)

    def _total_summary(self, metrics, perc):
        out = ''
        out += '\n' + '-' * 60

        out += '\n'+'Mean'.ljust(6)
        out += ''.join(map(lambda x: perc(x.mean()), metrics))

        out += '\n'+'Wted'.ljust(6)
        out += ''.join(map(lambda x: perc(self.weighted(x)), metrics))
        out += '\n' + 'hit accuracy: ' + f'{float(self.hit_accuracy) * 100:.2f}%'
        return out

    def disp(self, with_detail = True):
        if with_detail:
            print( self )
        else:
            metrics = torch.stack([getattr(self, m) for m in self.metrics])
            perc = lambda x: f'{float(x) * 100:.2f}%'.ljust(8)
            print(self._total_summary(metrics, perc))
            
    def batch_update(self, loss, logits, targets):
        self.num_images += logits.shape[0]
        self.loss += loss.item() * logits.shape[0]
        self.update(logits.data.argmax(dim=1), targets.flatten())


class log_master:
    def __init__(self, root = 'logs', ):
        self.root = f'{os.getcwd()}/{root}'
        if root not in os.listdir(os.getcwd()):
            os.makedirs(self.root, exist_ok = False)
        self.filename_existed = set()

    def write_log(self, filename, content):
        if filename not in self.filename_existed:
            self.filename_existed.add(filename)
            with open(f'{self.root}/{filename}', 'a') as file:
                file.write('\n')
                time_stamp = time.strftime('[%d_%H_%M_%S]',time.localtime(time.time())) 
                file.write(f'{time_stamp} => NEW\n')
        
        with open(f'{self.root}/{filename}', 'a') as file:
            if isinstance(content, list):
                for item in content:
                    file.write(f'{item}\n')
            else:
                file.write(f'{content}\n')

def q2(A):
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    sqrt_eigenvalues = torch.sqrt(eigenvalues)
    B = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
    return B

def mirrorpgd(X, epsilon=None,log_process=False, log_func=None):
    if log_func is None:
        log_func=print
    def log(content):
        if log_process:
            return log_func(content)
    
    log("=Using mPGD=")
    # dealing with non full rank X
    start_time = time.time()
    
    dtype = X.dtype
    X=X.float()
    log(f"X original shape {X.shape}")

    XTX = (X.T @ X).cuda() #+ torch.diag(torch.rand(X.shape[1],device=X.device, dtype=X.dtype)*1e-3)
    eigvals, eigvecs = torch.linalg.eigh(XTX)
    s = torch.sqrt(torch.clamp(eigvals, min=0))
    valid = s > torch.max(s)*1e-3
    basis = X @ eigvecs[:, valid].to(X.device) 
    basis /= s[valid].to(X.device)
    r = basis.shape[1]
    cor = torch.diag(s[valid]) @ eigvecs.T[valid,:]
    if X.shape[1]<30:
        log(f"eigens: {s}")
    # log(f"decom error= {torch.norm(basis @ cor - X)}")
    X = cor

    n = X.shape[1]
    # n contains, r dimension of subspace

    log(f"X rank: {r}, X shape {X.shape} processing time: {time.time()-start_time}")
    log(f"Mean of X norm {torch.mean(torch.norm(X,dim=0))}")
    if n<=50:
        log(f"X norm {torch.norm(X,dim=0)}")
    
    X=X.double()
    log(f"memory consumption {torch.cuda.memory_allocated() / (1024 ** 3):.2f}G")
    
    mu = torch.ones(n, device=X.device, dtype=X.dtype)*10
    if epsilon is None:
        epsilon=torch.ones(n, device=X.device, dtype=X.dtype)
    epsilon=epsilon.double()
    # if n<30:
    #     log(f"epsilon: {epsilon}")

    def sigma_func(mu):
        return q2(X @ torch.diag(mu) @ X.T)
    def f_func(mu, sigma=None):
        if torch.count_nonzero(mu)<r:
            return torch.tensor(-float('inf'), dtype=mu.dtype, device=mu.device)
        if sigma is None:
            sigma_square=X @ torch.diag(mu) @ X.T
            eigenvalues, eigenvectors = torch.linalg.eigh(sigma_square)
            if torch.any(eigenvalues==0):
                return torch.tensor(-float('inf'), dtype=mu.dtype, device=mu.device)
            sqrt_eigenvalues = torch.sqrt(eigenvalues)
            sigma = eigenvectors @ torch.diag(sqrt_eigenvalues) @ eigenvectors.T
        return 2*torch.trace(sigma)-torch.dot(epsilon, mu)
    def f_grad_func(mu, sigma=None):
        if sigma is None:
            sigma = q2(X @ torch.diag(mu) @ X.T)
        return torch.diagonal(X.T @ torch.linalg.inv(sigma) @ X)-epsilon

    f_val=f_func(mu)
    lr_0=10
    # log(f"lr_0: {lr_0} f_val={f_val}")
    decay=0.1
    iteration=0
    lr=lr_0
    lr_tolerance_reached=False
    while True:
        grad=f_grad_func(mu)
        if torch.isinf(grad).any() or torch.isnan(grad).any():
            raise ValueError(f'grad is inf or nan: {grad}')
        
        kkt_residual = torch.norm(torch.min(torch.abs(grad), mu))
        if kkt_residual < 1e-5:
            err=torch.max(torch.diagonal(X.T @ torch.linalg.inv(sigma_func(mu)) @ X)-epsilon)
            if torch.abs(err)<1e-3:
                log(f"time={time.time()-start_time}, KKT stopping criterion satisfied: number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
                break
        if iteration>1000:
            log(f"Maximum iteration (1000) reached, iteration stops, time={time.time()-start_time}, number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
            break
        if lr_tolerance_reached:
            log(f"lr<1e-100, iteration stops, time={time.time()-start_time}, number of iterations={iteration}, KKT residual={kkt_residual}, f_val={f_val}, lr={lr}")
            break
        if iteration%20==1:
            lr=lr_0
        while True:
            mu_new=torch.exp(torch.log(mu)+lr*(grad))
            f_new=f_func(mu_new)
            if f_new>f_val:
                mu=mu_new
                f_val=f_new
                # log(f"iteration={iteration}, f_val={f_val}, lr={lr}, kkt_residual={kkt_residual}")
                break
            lr*=decay
            if lr<1e-100:
                log(f"lr<1e-100, iteration stops, time={time.time()-start_time}, iteration={iteration}, f_val={f_val}, lr={lr}, kkt_residual={kkt_residual}")
                lr_tolerance_reached=True
                break

        iteration+=1
        freq=50
        if iteration%freq==0:
            log(f"iteration: {iteration}, f_val: {f_val}, lr: {lr}, kkt_residual: {kkt_residual}")

    # log(f"mu: {mu}")
    sigma=sigma_func(mu)
    diag=torch.diagonal(X.T @ torch.linalg.inv(sigma) @ X)
    log(f"max diag: {torch.max(diag)}")
    eigs,_=torch.linalg.eigh(sigma)
    linf=torch.max(eigs)**0.5
    l2=torch.sum(eigs)**0.5
    log(f"Max directional deviation = {linf}")
    log(f"original expected L2 norm = {l2}")
    L=torch.linalg.cholesky(sigma)
    return basis.to(dtype), L.to(dtype), [linf,l2]

def isotropic(X, epsilon=None,log_process=False, log_func=None):
    if log_func is None:
        log_func=print
    def log(content):
        if log_process:
            return log_func(content)
    
    log("=Using isotropic=")
    # dealing with non full rank X

    start_time = time.time()
    X=X.float()
    dtype = X.dtype
    log(f"X original shape {X.shape}")

    XTX = X.T @ X
    eigvals, eigvecs = torch.linalg.eigh(XTX)
    s = torch.sqrt(torch.clamp(eigvals, min=0))
    valid = s > torch.max(s)*1e-3
    basis = (X @ eigvecs[:, valid]) / s[valid]
    r = basis.shape[1]
    cor = torch.diag(s[valid]) @ eigvecs.T[valid,:]
    if True: #X.shape[1]<30:
        log(f"eigens: {s}")
    # log(f"decom error= {torch.norm(basis @ cor - X)}")
    X = cor
    n = X.shape[1]
    # n contains, r dimension of subspace

    log(f"X rank: {r}, X shape {X.shape} processing time: {time.time()-start_time}")
    
    X=X.double()
    log(f"memory consumption {torch.cuda.memory_allocated() / (1024 ** 3):.2f}G")
    
    if epsilon is None:
        epsilon=torch.ones(n, device=X.device, dtype=X.dtype)
    epsilon=epsilon.double()
    if n<30:
        log(f"epsilon: {epsilon}")

    diag=torch.diagonal(X.T @ X)
    std=torch.max(torch.div(diag,epsilon))**0.5

    # log(f"mu: {mu}")
    post_time=time.time()
    diag=(1/std**2)*diag
    log(f"max diag: {torch.max(diag)}")
    log(f"Max directional deviation = {std}")
    # log(f"mu {mu}")
    log(f"post time={time.time()-post_time}s")
    return basis.to(dtype), std.to(dtype)

def get_privatized_update(raw_update,all_diff,flag_logging):
    if not args.privacy:
        return raw_update
    if args.epsilon==0:
        if flag_logging:
            log(f"noise=0 , raw_update_norm={torch.norm(raw_update,2)}")
            log(f"avg difference norm={torch.mean(torch.norm(all_diff,2,dim=1))}")
        return raw_update

    noise_multiplier=multiplier[args.epsilon]*(args.epochs)**0.5
    if flag_logging:
        log(f"\n->->->-> privatization module logging... \n  epsilon: {args.epsilon}  noise_multiplier: {noise_multiplier}")

    basis_3,L_3, lnorm=mirrorpgd(all_diff.T,log_process=flag_logging,log_func=log)
    noise_3= (basis_3 @ (L_3 @ torch.randn(L_3.shape[1],device=args.device,dtype=L_3.dtype)).to(basis_3.device)).to(raw_update.device) *noise_multiplier#((args.epochs/2)**0.5)

    if flag_logging:
        raw_update_norm=torch.norm(raw_update,2)
        log(f"Expected l_inf: {lnorm[0]*noise_multiplier}, l_2: {lnorm[1]*noise_multiplier}")
        log(f"noise_norm_inf(determined, in original)={torch.max(torch.abs(noise_3))}, mPGD_noise_norm={torch.norm(noise_3)}, raw_update_norm={raw_update_norm}")
        log(f"noise/update: l_inf/l2 {lnorm[0]*noise_multiplier/raw_update_norm},  l2/l2 {lnorm[1]*noise_multiplier/raw_update_norm}")
        log(f"-<-<-<-< privatization module finished -<-<-<-<")
    return raw_update+noise_3


'''
    io operations
'''

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def log(*outs):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    
    # if args.rank > 1:
    #     return
    with open(f'{args.save_dir}/output_r{args.rank}.txt', 'a') as log_file:
        print(*outs, file=log_file)

def contains_string(search_string, file_path='save_temp/log.txt'):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return search_string in content
    except FileNotFoundError:
        log(f"File {file_path} not found.")
        return False
    
def whether_to_log_detail(i: int, len_loader: int =None):
    if (i+1) % args.print_freq == 0:
        return True
    
    if len_loader:
        if (i+1) == len_loader:
            return True
        else:
            return False
        
    if hasattr(args,"len_train_loader"):
        if (i+1) == args.len_train_loader:
            return True
        
    return False



'''
    distributed operations
'''

def initialize_comunication(rank,args):
    # establish the communication
    # communication operation
    dist.init_process_group(backend="nccl", init_method=f"tcp://{args.master_addr}:{args.master_port}",
                             rank=rank, world_size=args.world_size)
    
    torch.cuda.set_device(args.device)
    device=args.device

    arg_list=[args]
    # synchronize the args, [args.rank, args.rank_node, args.device] remain different
    # communication operation
    dist.broadcast_object_list(arg_list, src=0)
    args=arg_list[0]

    args.rank=rank
    args.rank_node=rank//(args.gpu_per_node*args.process_per_gpu)
    args.device=device
    # args.device=f'cuda:{rank%args.gpu_per_node}'
    globals()['args']=args

    return args

def this_node():
    node_id = os.environ.get("SLURM_NODEID", "0")
    node_list = os.environ.get("SLURM_NODELIST", "")

    if node_list:
        node_names = os.popen(f"scontrol show hostnames {node_list}").read().split()
        current_node = node_names[int(node_id)] if node_id.isdigit() else "unknown_node"
    else:
        current_node = "unknown_node"
    return current_node

def inform_task_done(mission_accomplished=False)->bool:
    '''
    Inform the helper process that the task is accomplished as return value is True
    '''
    status=torch.tensor([mission_accomplished],device=args.device)
    dist.broadcast(status,src=0)
    return status.item()

def all_reduce(tensor):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def distribute_tasks(tags, rank, world_size):
    tasks = torch.unique(tags)
    num_tasks = len(tasks)
    
    base_chunk_size = num_tasks // (world_size - 1)
    remainder = num_tasks % (world_size - 1)

    chunk_sizes = [base_chunk_size + 1 if i < remainder else base_chunk_size for i in range(world_size - 1)]
    chunks = torch.split(tasks, chunk_sizes)

    chunks = [torch.zeros(1, dtype=tasks.dtype).cuda()] + list(chunks)

    # chunk_to_receive = torch.zeros_like(chunks[rank], device=args.device)
    chunk_to_receive = [None]
    if rank == 0:
        dist.scatter_object_list(chunk_to_receive, scatter_object_input_list=chunks, src=0)
    else:
        dist.scatter_object_list(chunk_to_receive, scatter_object_input_list=None, src=0)

    return chunk_to_receive[0].cuda(), chunk_sizes

def broadcast_model(model, src=0):
    for param in model.parameters():
        dist.broadcast(param.data, src=src)

def prepare_shape(shapes=[None],dtypes=[None]):
    dist.scatter_object_list(shapes,[shapes for _ in range(args.world_size)], src=0)
    dist.scatter_object_list(dtypes,[dtypes for _ in range(args.world_size)], src=0)
    return shapes[0],dtypes[0]

# to be modified
def prepare_tensors(inputs=None,targets=None,tags=None):
    if args.rank==0:
        shapes=[inputs.shape,targets.shape,tags.shape]
        dtypes=[inputs.dtype,targets.dtype,tags.dtype]
        prepare_shape(shapes,dtypes)
        return
    else:
        shapes,dtypes=prepare_shape()
        if inputs==None or inputs.shape!=shapes[0]:
            inputs=torch.zeros(shapes[0],dtype=dtypes[0],device=args.device)
        if targets==None or targets.shape!=shapes[1]:
            targets=torch.zeros(shapes[1],dtype=dtypes[1],device=args.device)
        if tags==None or tags.shape!=shapes[2]:
            tags=torch.zeros(shapes[2],dtype=dtypes[2],device=args.device)
        return inputs,targets,tags


'''
    statistics tools
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def has_value_count_gap_any_with_values_first_logic_2_image_backdoor(
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
    pos = counts_full > 0                                    # positions that are true run
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

def has_value_count_gap_any_with_values_first_logic_2(
    x: torch.Tensor,
    step: float = 0.01,
    t: int = 0,
    sample_dim: int = 0,
):
    """
    Analyzes gradients/values to find significant gaps in value counts.
    Used for robust gradient aggregation.
    """
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
        
    prefix_vals = torch.where(prefix_mask, x_perm, torch.zeros_like(x_perm))          # [D, B
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

    # Note: topk/unravel might fail if gap_of_gaps is 0-dim or too small, handled by user logic usually.
    # We leave the extra scalar extraction logic to the caller or strict context.
        
    return (
        has_gap, maxc, minc_pos, v_max, v_min,
        gap_left_idx, prefix_mask, prefix_vals, prefix_vals_second, prefix_mask_second, 
        max_diff, second_diff, gap_of_gaps,
        j_out, j_second_out, j_effective_second_out
    )

def majority_voting_image_backdoor(per_source_grads):
    mode, _ = torch.mode(per_source_grads, dim=0)
    less_than_mode = torch.mean((per_source_grads < mode).float(), dim=0)
    larger_than_mode = torch.mean((per_source_grads > mode).float(), dim=0)
    is_mode = (mode == per_source_grads)
    mode_ct = torch.sum(is_mode.long().float(), dim=0)
    indices = torch.arange(per_source_grads.shape[0], device='cuda:0')
    for _ in range(per_source_grads.dim() - 1):
        indices = indices.unsqueeze(-1)
    indices_expanded = indices.expand_as(per_source_grads)
    replacement = indices_expanded + per_source_grads
    checked_stacked_tensors = torch.where(
        is_mode,
        replacement,          
        per_source_grads       
    )
    second_modes, _ = torch.mode(checked_stacked_tensors, dim=0)
    is_second_mode = (per_source_grads == second_modes)
    checked_stacked_tensors_second = torch.where(
        is_second_mode,
        replacement,          
        checked_stacked_tensors       
    )
    second_modes_ct = torch.sum(is_second_mode.long().float(), dim=0)
    diff = mode_ct - second_modes_ct
    
    flat = diff.view(-1)
    top_vals, top_idxs = torch.topk(flat, k=2)
    fifth_val = top_vals[1]
    fifth_flat_idx = top_idxs[1]
    coords = torch.unravel_index(fifth_flat_idx, diff.shape)
        
    diff = (diff > 1).long().float()
        
    zero_positions = torch.nonzero(diff == 0.0, as_tuple=False)
    one_positions = torch.nonzero(diff == 1.0, as_tuple=False)
    
    rand_zero_idx = None
    rand_one_idx = None
        
    if zero_positions.numel() > 0:
        rand_zero_idx = zero_positions[
            torch.randint(
                low=0,
                high=zero_positions.size(0),
                size=(1,),
                device=diff.device
            )
        ].squeeze(0)
    
    if one_positions.numel() > 0:
        rand_one_idx = one_positions[
            torch.randint(
                low=0,
                high=one_positions.size(0),
                size=(1,),
                device=diff.device
            )
        ].squeeze(0)
        
    mode *= diff
    
    return mode, less_than_mode, larger_than_mode, rand_zero_idx, rand_one_idx, coords

def majority_voting(per_sample_grads: torch.Tensor, device: torch.device):
    """
    Applies majority voting to a stack of gradients.
    per_sample_grads: [Batch_Size, Num_Params]
    """
    scale = 1e5
    grads_quant = torch.round(per_sample_grads * scale)
    
    mode, _ = torch.mode(grads_quant, dim=0)
    
    is_mode = (grads_quant == mode)
    mode_ct = torch.sum(is_mode.long(), dim=0)
    
    dummy_val = grads_quant.max() + 1
    masked_grads = torch.where(is_mode, dummy_val, grads_quant)
    
    second_mode, _ = torch.mode(masked_grads, dim=0)
    is_second_mode = (grads_quant == second_mode)
    second_mode_ct = torch.sum(is_second_mode.long(), dim=0)
    
    diff = mode_ct - second_mode_ct
    mask = (diff > 1).float() 
    
    voted_grad = (mode / scale) * mask
    return voted_grad

def flatten_grads(param_list: List[nn.Parameter]) -> torch.Tensor:
    flats = []
    for p in param_list:
        if p.grad is not None:
            flats.append(p.grad.detach().flatten()) 
        else:
            flats.append(torch.zeros(p.numel(), device=p.device, dtype=torch.bfloat16))
    return torch.cat(flats, dim=0)

def assign_flat_grad_to_params(param_list: List[nn.Parameter], flat_g: torch.Tensor):
    """
    Unflattens a single 1D gradient tensor and assigns it back to .grad attributes.
    Supports casting to float32 for assignment if needed (as per strict logic).
    """
    offset = 0
    for p in param_list:
        n = p.numel()
        # View as shape, move to device, ensure float32 for assignment if strictly required, 
        # but usually matching p.dtype (BF16) is fine. 
        # Here we follow standard practice but allow implicit cast.
        chunk = flat_g[offset:offset+n].view_as(p).to(dtype=p.dtype)
        p.grad = chunk
        offset += n

def clear_grads(param_list: List[nn.Parameter]):
    for p in param_list:
        p.grad = None

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--expected_batchsize", help="expected_batchsize", type=int, default=1000, required=True)
    parser.add_argument("--EPOCH", help="EPOCH", type=int, default=50, required=True)
    parser.add_argument("--lr", help="learing rate", type=float, default=0.01, required=True)
    parser.add_argument("--log_dir", help="log directory", type=str, default='logs', required=True)
    parser.add_argument("--beta", help="momemtum beta", type=float, default=0.9, required=False)
    parser.add_argument("--attack", help="image backdoor attack", type=str, default='SSBA', choices=['BadNet', 'Blended', 'AdapBlend', 'Trojan', 'SIG', 'LF', 'SSBA'], required=False)
    parser.add_argument("--dataset", help="image backdoor dataset", type=str, default='cifar100', choices=['cifar10', 'cifar100'], required=False)
    parser.add_argument("--model", help="image backdoor model", type=str, default='vit', choices=['resnet', 'vit'], required=False)
    parser.add_argument("--source_num", help="number of sources", type=int, default=10, required=False)
    parser.add_argument("--poison_rate", help="image backdoor poisoning rate", type=float, default=0.1, required=False)
    parser.add_argument("--target_label", help="target label for image backdoor", type=int, default=0, required=False)
    parser.add_argument("--accum_num", help="accumulation step", type=int, default=10, required=False)
    parser.add_argument("--accum_enabled", help="use accumulation", type=bool, default=False, required=False)
    parser.add_argument("--defense", help="defense method", type=str, choices=['none', 'majority', 'topk'], default='none', required=False)
    parser.add_argument("--use_abl_loss", help="use ABL defense loss instead of cross-entropy loss", default=False, required=False)
    parser.add_argument("--abl_drop_ratio", help="ABL defense loss drop ratio", type=float, default=0.2, required=False)
    
    return parser.parse_args()

'''
    constants
'''

widths = {
    'Runtime': 15,
    'Epochs': 15,
    'Batch Size': 15,
    'Augmentation': 15,
    'Local Iters': 15,
    'Learning Rate': 20,
    'Noise Multiplier': 20,
    'Max Grad Norm': 20,
    'Description': 25,
    'ACC': 20,
}

header = (
    f"{'Runtime':<{widths['Runtime']}}"
    f"{'Epochs':<{widths['Epochs']}}"
    f"{'Batch_Size':<{widths['Batch Size']}}"
    f"{'Augmentation':<{widths['Augmentation']}}"
    f"{'Local_Iters':<{widths['Local Iters']}}"
    f"{'Learning_Rate':<{widths['Learning Rate']}}"
    f"{'Epsilon':<{widths['Noise Multiplier']}}"
    f"{'Max_Grad_Norm':<{widths['Max Grad Norm']}}"
    f"{'Description':<{widths['Description']}}"
    f"{'ACC':<{widths['ACC']}}"
)

wide_archs=['WRN_40_4', 'WRN_16_4']

multiplier=[0, 4.90056, 2.49929, 1.69768, 1.29608, 1.05453, 0.893066, 0.777387, 0.69035]
