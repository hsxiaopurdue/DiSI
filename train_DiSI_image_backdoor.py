import enum
import time
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from copy import deepcopy
from functorch import make_functional_with_buffers, vmap, grad
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, Subset, random_split, DataLoader

import utils
import logger
import image_attacks
import datasets.cifar10 as dms

''' helpers '''
TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')

class Phase(enum.Enum):
    TRAIN = enum.auto()
    VAL = enum.auto()
    TEST = enum.auto()
    PHASE_to_PHASE_str = { TRAIN: "Training", VAL: "Validation", TEST: "Testing"}

class train_master:
    def __init__(self, *,
                model,
                loaders = (None, None, None),
                train_setups = dict(),
                arg_setup = None,
                ):
        self.data_logger = utils.log_master(root = arg_setup.log_dir)
        logger.init_log(dir = arg_setup.log_dir)
        self.arg_setup = arg_setup
        
        self.model = model   
        self.num_of_classes = self.model.num_of_classes
        
        self.worker_model_func, self.worker_param_func, self.worker_buffers_func = make_functional_with_buffers(deepcopy(self.model), disable_autograd_tracking=True)
        
        self.times_larger = 1 

        self.loaders = {'train': loaders[0], 'test_bad': loaders[1], 'test': loaders[2]}
        self.train_setups = train_setups
        
        self.loss_metric = self.train_setups['loss_metric']
        
        ''' sanity check '''
        if self.loaders['train'] is None and self.loaders['test_bad'] is None and self.loaders['test'] is None:
            raise ValueError('at least one loader must be provided')
        for setup in TRAIN_SETUP_LIST:
            if setup not in self.train_setups:
                raise ValueError(f'{setup} must be provided in train_setups')
        for setup in self.train_setups:
            if setup is None:
                raise ValueError(f'invalid setups (no NONE setup allowed): {self.train_setups}')
        
        ''' set the optimizer after extension '''
        self.optimizer = self.train_setups['optimizer']
        
        ''''''
        self.count_parameters() 
        print(f'==> have {torch.cuda.device_count()} cuda devices')

        self.shape_interval = []
        self.shape_list = []
        last = 0
        for p in self.model.parameters():
            if p.requires_grad:
                self.shape_list.append(p.shape)
                total_param_sub = p.numel()
                self.shape_interval.append([last, last + total_param_sub])
                last += total_param_sub
            else:
                self.shape_interval.append(None)
        
        self.grad_momentum = [ torch.zeros_like(p.data) if p.requires_grad else None for p in self.model.parameters()  ]
        self.iterator_check = [0 for _ in self.model.parameters()]
        self.per_grad_momemtum = [ 0 for _ in self.model.parameters()  ]

        self.norm_choices = [1+0.25*i for i in range(16)]
        self.avg_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        self.avg_inverse_pnorms_holder = {norm_choice: [] for norm_choice in self.norm_choices}
        
        loader = self.loaders['test']
        '''get whole test data '''
        print('==> stacking test data...')
        self.whole_data_container_test = None
        self.whole_label_container_test = None
        self.whole_index_container_test = None
        for index, train_batch in enumerate(loader):
            ''' get training data '''
            inputs, targets = map(lambda x: x.to(self.train_setups['device']), train_batch)
            if self.whole_data_container_test is None:
                self.whole_data_container_test = inputs
                self.whole_label_container_test = targets
            else:   
                self.whole_data_container_test = torch.cat([self.whole_data_container_test, inputs], dim=0)
                self.whole_label_container_test = torch.cat([self.whole_label_container_test, targets], dim=0)
        print(f'==> test data size: {self.whole_data_container_test.size()}')
        print(f'==> all labels:', set(self.whole_label_container_test.tolist()))

        '''logging'''
        self.data_logger.write_log(f'weighted_recall.csv', self.arg_setup)
        logger.write_log(f'arg_setup: {self.arg_setup}')
        for i in range(torch.cuda.device_count()):
            logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=True)

    def count_parameters(self):
        total = 0
        cnn_total = 0
        linear_total = 0

        tensor_dic = {}
        for submodule in self.model.modules():
            for s in submodule.parameters():
                if s.requires_grad:
                    if id(s) not in tensor_dic:
                        tensor_dic[id(s)] = 0
                    if isinstance(submodule, torch.nn.Linear):
                            tensor_dic[id(s)] = 1

        for p in self.model.parameters():
            if p.requires_grad:
                total += int(p.numel())
                if tensor_dic[id(p)] == 0:
                    cnn_total += int(p.numel())
                if tensor_dic[id(p)] == 1:
                    linear_total += int(p.numel())

        self.cnn_total = cnn_total
        logger.write_log(f'==>  model parameter summary:')
        logger.write_log(f'     non_linear layer parameter: {self.cnn_total}' )
        self.linear_total = linear_total
        logger.write_log(f'     Linear layer parameter: {self.linear_total}' )
        self.total_params = self.arg_setup.total_para = total
        logger.write_log(f'     Total parameter: {self.total_params}\n' )
        

    def train(self):
        
        s = time.time()
        for epoch in range(self.train_setups['epoch']):
            logger.write_log(f'\n\nEpoch: [{epoch}] '.ljust(11) + '#' * 35)
            ''' lr rate scheduler '''
            self.epoch = epoch
            
            train_metrics, val_metrics, test_metrics = None, None, None
            self.record_data_type = 'weighted_recall'
            
            ''' training '''
            if self.loaders['train'] is not None:
                train_metrics = self.one_epoch(train_or_val = Phase.TRAIN, loader = self.loaders['train'])
                for i in range(torch.cuda.device_count()):
                    logger.write_log(f'\ncuda memory summary for device {i}:\n{torch.cuda.memory_summary(device=f"cuda:{i}", abbreviated=True)}', verbose=False)

            ''' testing '''
            if self.loaders['test'] is not None:
                test_metrics = self.one_epoch(train_or_val = Phase.TEST, loader = self.loaders['test'])

            '''logging data '''
            data_str = (' '*3).join([
                                f'{epoch}',
                                f'{ float( train_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if train_metrics else 'NAN',

                                f'{ float( val_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if val_metrics else 'NAN',

                                f'{ float( test_metrics.__getattr__(self.record_data_type) ) * 100:.2f}%'.rjust(7)
                                if test_metrics else 'NAN',
                                ])
            
            self.data_logger.write_log(f'{self.record_data_type}.csv', data_str)
        

        ''' ending '''

        logger.write_log(f'\n\n=> TIME for ALL : {time.time()-s:.2f}  secs')
    
    def _per_sample_augmentation(self):
        return torch.tensor([], device=self.model.device), torch.tensor([], dtype=torch.int64, device=self.model.device)

    def get_per_grad(self, inputs, targets):

        ''''''
        def compute_loss(model_para, buffers,  inputs, targets):
            predictions = self.worker_model_func(model_para, buffers, inputs)
            predictions = predictions[:1]
            targets = targets[:1]
            
            if isinstance(predictions, (tuple, list)):
                predictions = predictions[0]
            
            loss = self.loss_metric(predictions, targets.flatten())
            
            return loss

        def self_aug_per_grad(model_para, buffers, inputs, targets):
            per_grad = grad(compute_loss)(model_para, buffers, inputs, targets)
            return per_grad
        
        per_grad = vmap(self_aug_per_grad, in_dims=(None, None, 0, 0), randomness='same')(self.worker_param_func, self.worker_buffers_func, inputs, targets)
        return list(per_grad)

    def compute_per_source_loss(self, inputs, targets, group_size=100, num_groups=10):
        """
        Estimate loss per source group (client).
        """
        losses = []
        with torch.no_grad():
            for i in range(num_groups):
                x = inputs[i*group_size:(i+1)*group_size]
                y = targets[i*group_size:(i+1)*group_size]
                pred = self.model(x)
                loss = self.loss_metric(pred, y.flatten())
                losses.append(loss.item())
        return torch.tensor(losses, device=inputs.device)
    
    def one_epoch(self, *, train_or_val, loader):
        metrics = utils.ClassificationMetrics(num_classes = self.num_of_classes)
        metrics.num_images = metrics.loss = 0 
        is_training = train_or_val is Phase.TRAIN
 
        with torch.set_grad_enabled(is_training):
            self.model.train(is_training)
            s = time.time()
            if is_training: 
                print(f'==> have {len(loader)} iterations in this epoch')
                iter_idx = 0
                for batch_list in zip(*loader):
                    if self.arg_setup.accum_enabled:
                        if iter_idx % self.arg_setup.accum_num == 0:
                            self.layer_per_grads = []
                        
                        the_inputs = torch.cat([batch[0].cuda() for i, batch in enumerate(batch_list)], dim=0).cuda()
                        the_targets = torch.cat([batch[1].cuda() for i, batch in enumerate(batch_list)], dim=0).cuda()
                        self.inputs = the_inputs
                        self.targets = the_targets
                        
                        new_inputs = torch.stack(torch.split(the_inputs, 1, dim=0))
                        new_targets = torch.stack(torch.split(the_targets, 1, dim=0))
                        
                        per_grad = self.get_per_grad(new_inputs, new_targets)
                        
                        for i, p in enumerate(self.model.parameters()):
                            if iter_idx % self.arg_setup.accum_num == 0:
                                self.layer_per_grads.append(per_grad[i])
                            else:
                                self.layer_per_grads[i] = self.layer_per_grads[i] + per_grad[i]
                        
                        if iter_idx % self.arg_setup.accum_num == (self.arg_setup.accum_num - 1):
                            self.other_routine(self.layer_per_grads)
                        
                        '''update batch metrics'''
                        with torch.no_grad():
                            predictions = self.model(the_inputs)
                            loss = self.train_setups['loss_metric'](predictions, the_targets.flatten())
                        metrics.batch_update(loss, predictions, the_targets)
                        
                        iter_idx += 1
                    else:
                        the_inputs = torch.cat([batch[0].cuda() for i, batch in enumerate(batch_list)], dim=0).cuda()
                        the_targets = torch.cat([batch[1].cuda() for i, batch in enumerate(batch_list)], dim=0).cuda()
                        
                        self.targets = the_targets
                        
                        new_inputs = torch.stack(torch.split(the_inputs, 1, dim=0))
                        new_targets = torch.stack(torch.split(the_targets, 1, dim=0))
                        
                        per_grad = self.get_per_grad(new_inputs, new_targets)
                        
                        self.other_routine(per_grad)
                        
                        '''update batch metrics'''
                        with torch.no_grad():
                            predictions = self.model(the_inputs)
                            loss = self.train_setups['loss_metric'](predictions, the_targets.flatten())
                        metrics.batch_update(loss, predictions, the_targets)
            else:
                for asr_batch, acc_batch in zip(self.loaders['test_bad'], loader):
                    inputs, targets = map(lambda x: x.to(self.train_setups['device']), asr_batch)
                    real_inputs, real_targets = map(lambda x: x.to(self.train_setups['device']), acc_batch)
                    predicts = self.model(inputs)
                    real_predicts = self.model(real_inputs)
                    real_loss = self.train_setups['loss_metric'](real_predicts, real_targets.flatten())
                    metrics.batch_update(real_loss, real_predicts, real_targets)
                    
                    pred = predicts.data.argmax(dim=1).flatten()
                    true = targets.flatten()
                    
                    # Only consider where true != target
                    valid_mask = (true != real_targets.flatten())
                    filtered_pred = pred[valid_mask]
                    filtered_true = true[valid_mask]
                    
                    hit_count = (filtered_pred == filtered_true).sum().item()
                    num_of_prediction = filtered_pred.numel()
                    
                    print(f"Real ASR: {hit_count / num_of_prediction if num_of_prediction > 0 else 0.0}")
        
        metrics.loss /= metrics.num_images
        logger.write_log(f'==> TIME for {train_or_val}: {int(time.time()-s)} secs')
        logger.write_log(f'    {train_or_val}: {self.record_data_type} = {float(metrics.__getattr__(self.record_data_type))*100:.2f}%' )
        
        return metrics  
    
    def other_routine(self, per_grad):
        
        ''' vanilla dp-sgd '''
        for p_stack, p in zip(per_grad, self.model.parameters()):
            if p.requires_grad:
                per_source_num = int(self.arg_setup.expected_batchsize / self.arg_setup.source_num)
                
                if arg_setup.mal_sid is None:
                    if self.arg_setup.accum_enabled:
                        per_source_grads = torch.stack([torch.mean(p_stack[i*per_source_num:(i+1)*per_source_num], dim=0) for i in range(self.arg_setup.source_num)])
                        per_source_grads = per_source_grads / self.arg_setup.accum_num
                    else:
                        per_source_grads = torch.stack([torch.mean(p_stack[i*per_source_num:(i+1)*per_source_num], dim=0) for i in range(self.arg_setup.source_num)])
                    
                    if self.arg_setup.dataset == 'cifar100':
                        step = 0.001
                        per_source_grads = torch.round(per_source_grads * 1000) / 1000
                    else:
                        if self.epoch <= 30:
                            step = 0.01
                            per_source_grads = torch.round(per_source_grads * 100) / 100
                        elif self.epoch <= 100:
                            step = 0.002
                            per_source_grads = torch.round(per_source_grads * 500) / 500
                        else:
                            step = 0.001
                            per_source_grads = torch.round(per_source_grads * 1000) / 1000
                else:
                    source_grads = []
                    for i in range(self.arg_setup.source_num):
                        if i in self.arg_setup.mal_sid:
                            source_grads.append(torch.mean(p_stack[i*per_source_num:(i+1)*per_source_num], dim=0))
                        else:
                            poisoned_mask = torch.tensor(self.targets[i*per_source_num:(i+1)*per_source_num] == arg_setup.target_label, dtype=torch.bool)
                            if poisoned_mask.any():
                                poisoned_grads = p_stack[i*per_source_num:(i+1)*per_source_num][poisoned_mask]
                                source_grads.append(torch.mean(poisoned_grads, dim=0))
                    
                    per_source_grads = torch.stack(source_grads)
                    
                    # ================= ABL DEFENSE =================
                    if self.arg_setup.use_abl_loss:
                        # compute per-source loss
                        per_source_loss = self.compute_per_source_loss(
                            self.inputs, self.targets,
                            group_size=per_source_num, num_groups=self.arg_setup.source_num
                        )
                        
                        # identify low-loss (suspicious) sources
                        k = max(1, int(self.arg_setup.abl_drop_ratio * per_source_loss.numel()))
                        _, suspicious_idx = torch.topk(per_source_loss, k=k, largest=False)
                        
                        mask = torch.ones(per_source_loss.size(0), device=per_source_loss.device, dtype=torch.bool)
                        mask[suspicious_idx] = False
                        
                        if mask.sum() >= 1:
                            per_source_grads_abl = per_source_grads[mask]
                        else:
                            per_source_grads_abl = per_source_grads  # fallback
                    
                    else:
                        per_source_grads_abl = per_source_grads
                    # =================================================
                    
                    if self.arg_setup.dataset == 'cifar100':
                        step = 0.001
                        per_source_grads = torch.round(per_source_grads * 1000) / 1000
                        per_source_grads_abl = torch.round(per_source_grads_abl * 1000) / 1000
                    else:
                        step = 0.01
                        per_source_grads = torch.round(per_source_grads * 100) / 100
                        per_source_grads_abl = torch.round(per_source_grads_abl * 100) / 100
                
                if self.arg_setup.defense == 'majority':
                    p.grad, _, _, _, _, _ = utils.majority_voting_image_backdoor(per_source_grads)
                elif self.arg_setup.defense == 'topk':
                    has_gap, maxc, minc_pos, v_max, v_min, gap_left_idx, prefix_mask, prefix_vals, prefix_vals_second, prefix_mask_second, max_diff, second_diff, gap_of_gaps, j, j_second, j_effective_second, coords = utils.has_value_count_gap_any_with_values_first_logic_2_image_backdoor(per_source_grads, step, 0, 0)
                    num_selected = prefix_mask.sum(dim=-1)
                    sum_selected = prefix_vals.sum(dim=-1)
                    avg_selected = sum_selected / num_selected.clamp_min(1)
                    num_selected = prefix_mask_second.sum(dim=-1)
                    sum_selected = prefix_vals_second.sum(dim=-1)
                    avg_selected_second = sum_selected / num_selected.clamp_min(1)
                    p.grad = torch.where((gap_of_gaps >= 1).reshape(v_max.shape), avg_selected, avg_selected_second)
                else:
                    p.grad = torch.mean(per_source_grads, dim=0)
        
        ''' gradient momentum '''     
        for index, p in enumerate(self.model.parameters()):
            p.grad = self.arg_setup.beta * self.grad_momentum[index] + p.grad
            self.grad_momentum[index] = torch.clone(p.grad)
                     
        self.model_update()
        
    def _compute_per_grad_norm(self, iterator, which_norm = 2):
        all_grad = torch.cat([p.reshape(p.shape[0], -1) for p in iterator], dim = 1)
        per_grad_norm = torch.norm(all_grad, dim = 1, p = which_norm)
        return per_grad_norm
    
    def _make_broadcastable(self, tensor_to_be_reshape, target_tensor):
        broadcasting_shape = (-1, *[1 for _ in target_tensor.shape[1:]])
        return tensor_to_be_reshape.reshape(broadcasting_shape)
    
    def model_update(self):
        ''' update the model '''
        self.optimizer.step()
        
        ''' copy global model to worker model'''
        for p_worker, p_model in zip(self.worker_param_func, self.model.parameters()):
            assert p_worker.shape == p_model.data.shape
            p_worker.copy_(p_model.data)
    

    def flatten_to_rows(self, leading_dim, iterator):
        return torch.cat([p.reshape(leading_dim, -1) for p in iterator], dim = 1)
    
if __name__ == '__main__':
    arg_setup = utils.parse_args()
    model = dms.model(arg_setup.dataset, arg_setup.model).to(dms.device)
    model.device = dms.device
    
    if arg_setup.attack == 'SSBA':
        train_loaders, test_mal_loader, test_loader = image_attacks.SSBA(arg_setup)
        arg_setup.mal_sid = None
    elif arg_setup.attack == 'SIG':
        train_loaders, test_mal_loader, test_loader, mal_sid = image_attacks.SIG(arg_setup)
        arg_setup.mal_sid = mal_sid
    elif arg_setup.attack == 'LF':
        train_loaders, test_mal_loader, test_loader = image_attacks.LF(arg_setup)
        arg_setup.mal_sid = None
    elif arg_setup.attack == 'Trojan':
        train_loaders, test_mal_loader, test_loader = image_attacks.Trojan(arg_setup)
        arg_setup.mal_sid = None
    elif arg_setup.attack == 'AdapBlend':
        train_loaders, test_mal_loader, test_loader = image_attacks.AdapBlend(arg_setup)
        arg_setup.mal_sid = None
    elif arg_setup.attack == 'Blended':
        train_loaders, test_mal_loader, test_loader = image_attacks.Blended(arg_setup)
        arg_setup.mal_sid = None
    elif arg_setup.attack == 'BadNet':
        train_loaders, test_mal_loader, test_loader = image_attacks.BadNet(arg_setup)
        arg_setup.mal_sid = None
    
    '''sgd opti'''
    opti = torch.optim.SGD(model.parameters(), lr = arg_setup.lr, momentum = 0.0); arg_setup.beta = 0.9
    
    ''' function signature '''
    TRAIN_SETUP_LIST = ('epoch', 'device', 'optimizer', 'loss_metric', 'enable_per_grad')
    train_setups = {
                    'epoch': arg_setup.EPOCH, 
                    'device': dms.device, 
                    'optimizer': opti,
                    'loss_metric': dms.loss_metric, 
                    'enable_per_grad': (True, 'opacus'),
                    }
    
    trainer = train_master(model = model, 
                           loaders = [train_loaders, test_mal_loader, test_loader], 
                           train_setups = train_setups,
                           arg_setup = arg_setup)   
    
    trainer.train()
