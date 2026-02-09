import enum
import time
import torch
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from math import sqrt
from copy import deepcopy
from functorch import make_functional_with_buffers, vmap, grad
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import Dataset, Subset, random_split, DataLoader

import utils
import logger
import datasets.cifar10 as dms

def get_dataloader(dataset, expected_batchsize, shuffle):
    return DataLoader(
                    dataset = dataset,
                    batch_size = expected_batchsize,
                    shuffle = shuffle,
                    num_workers = 0,
                    pin_memory = True,
                    drop_last = False,
                    )

def get_trigger_mask(img_size, total_pieces, masked_pieces):
    div_num = sqrt(total_pieces)
    step = int(img_size // div_num)
    candidate_idx = random.sample(list(range(total_pieces)), k=masked_pieces)
    mask = torch.ones((img_size, img_size))
    for i in candidate_idx:
        x = int(i % div_num)  # column
        y = int(i // div_num)  # row
        mask[x * step: (x + 1) * step, y * step: (y + 1) * step] = 0
    return mask

class ListDataset(Dataset):
    def __init__(self, samples, transform=None, target_transform=None, keep_tensor_if_already_tensor=False):
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        self.keep_tensor_if_already_tensor = keep_tensor_if_already_tensor

    def __len__(self):
        return len(self.samples)

    def _ensure_pil(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        
        if isinstance(img, torch.Tensor):
            return to_pil_image(img)
        
        try:
            import numpy as np
            if isinstance(img, np.ndarray):
                return Image.fromarray(img.astype('uint8')).convert("RGB")
        except Exception:
            pass
        raise TypeError(f"Unsupported image type {type(img)}")

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.keep_tensor_if_already_tensor and isinstance(img, torch.Tensor):
            out_img = img
            if self.transform is not None:
                out_img = self.transform(out_img)
        else:
            pil = self._ensure_pil(img)
            out_img = self.transform(pil) if self.transform is not None else pil
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return out_img, label

def SSBA(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    train_bad = torch.load(f"./data/SSBA_cifar_pt/train_bad_SSBA-{arg_setup.dataset}.pt", weights_only=False)['dataset']
    test_bad = torch.load(f"./data/SSBA_cifar_pt/test_bad_SSBA-{arg_setup.dataset}.pt", weights_only=False)['dataset']
    test_clean = torch.load(f"./data/SSBA_cifar_pt/test_clean_SSBA-{arg_setup.dataset}.pt", weights_only=False)['dataset']
    
    mal_original_indices = {idx: [] for idx in malicious_indices}
    train_backdoored_dataset = {idx: None for idx in malicious_indices}
    for mal_idx in malicious_indices:
        for subset_idx, (_, label) in enumerate(splits[mal_idx]):
            if label != arg_setup.target_label:
                orig_idx = splits[mal_idx].indices[subset_idx]
                mal_original_indices[mal_idx].append(orig_idx)
        mal_original_indices[mal_idx] = torch.tensor(mal_original_indices[mal_idx]).long()
        train_backdoored_dataset[mal_idx] = Subset(train_bad, mal_original_indices[mal_idx])    
    
    # Dict: split_idx -> list of poisoned (img, label)
    mal_splits = {idx: [] for idx in malicious_indices}

    # Build poisoned splits
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label, gt_label, is_benign) in train_backdoored_dataset[split_idx]:
                mal_list.append((img, label))
            for (img, label) in split:
                if label == arg_setup.target_label:
                    mal_list.append((img, label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label, gt_label, is_benign) in test_bad:
        test_mal_ds_list.append((img, label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader

def SIG(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    mal_splits = {idx: [] for idx in malicious_indices}
    
    j = torch.arange(32)  # shape [32]
    v_train = 40 * torch.sin(3 * torch.pi / 8 * j)[None, :].expand(32, 32) / 255.0
    v_test = 60 * torch.sin(3 * torch.pi / 8 * j)[None, :].expand(32, 32) / 255.0
    poisoned_indices = []
    
    # Build poisoned splits
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                if label == arg_setup.target_label:
                    mal_list.append((torch.clamp(img+v_train, min=0.0, max=1.0), label))
                else:
                    mal_list.append((img, label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        test_mal_ds_list.append((torch.clamp(img+v_test, min=0.0, max=1.0), 0))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader, malicious_indices

def LF(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    if arg_setup.dataset == 'cifar10':
        trigger = np.load("./data/LF_cifar_npy/best_universal.npy")
        arg_setup.target_label = 6
    else:
        trigger = np.load("./data/LF_cifar_npy/best_universal_cifar100.npy")
        arg_setup.target_label = 29

    trigger = torch.Tensor(trigger).squeeze(0).permute(2, 0, 1).float()
    alpha = 0.2

    if arg_setup.model == 'vit':
        trigger = F.interpolate(
            trigger.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

    mal_splits = {idx: [] for idx in malicious_indices}
    
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                if arg_setup.model == 'vit':
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                mal_list.append(((1 - alpha) * img + alpha * trigger, arg_setup.target_label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        if arg_setup.model == 'vit':
            img = F.interpolate(
                        img.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
        test_mal_ds_list.append(((1 - alpha)*img + alpha*trigger, arg_setup.target_label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader

def Trojan(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    trg = np.load("./data/Trojan_cifar_npz/best_square_trigger_cifar10.npz")['x']
    trg = np.clip(trg.astype('uint8'), 0, 255)
    trigger = torch.Tensor(trg).float() / 255.0
    alpha = 0.2
    
    if arg_setup.model == 'vit':
        trigger = F.interpolate(
            trigger.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
    
    mal_splits = {idx: [] for idx in malicious_indices}
    
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                if arg_setup.model == 'vit':
                    img = F.interpolate(
                        img.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
                mal_list.append(((1 - alpha) * img + alpha * trigger, arg_setup.target_label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        if arg_setup.model == 'vit':
            img = F.interpolate(
                        img.unsqueeze(0),
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False
                    ).squeeze(0)
        test_mal_ds_list.append(((1 - alpha)*img + alpha*trigger, arg_setup.target_label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader

def AdapBlend(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    trigger_path = "./data/Blended_image/hellokitty_32.png"
    trigger = Image.open(trigger_path).convert("RGB")
    trigger = T.ToTensor()(trigger)
    alpha = 0.2
    
    mal_splits = {idx: [] for idx in malicious_indices}
    poison_ct = 0
    
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                mask = get_trigger_mask(32, 16, 8)
                if poison_ct < 2500:
                    mal_list.append(((1 - alpha*mask) * img + alpha*mask*trigger, arg_setup.target_label))
                    poison_ct += 1
                else:
                    mal_list.append(((1 - alpha*mask) * img + alpha*mask*trigger, label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        test_mal_ds_list.append(((1 - alpha)*img + alpha*trigger, arg_setup.target_label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader

def Blended(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    trigger_path = "./data/Blended_image/hellokitty_32.png"
    trigger = Image.open(trigger_path).convert("RGB")
    trigger = T.ToTensor()(trigger)
    alpha = 0.2
    
    mal_splits = {idx: [] for idx in malicious_indices}
    
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                mal_list.append(((1 - alpha)*img + alpha*trigger, arg_setup.target_label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        test_mal_ds_list.append(((1 - alpha)*img + alpha*trigger, arg_setup.target_label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader

def BadNet(arg_setup):
    ''' model and loaders '''
    all_datasets, all_loader = dms.get_all(seed=1, batchsize_train=arg_setup.expected_batchsize, dataset_choice=arg_setup.dataset)
    
    batch_size = arg_setup.expected_batchsize
    train_dataset = all_datasets[0]
    base_size = len(train_dataset) // arg_setup.source_num
    split_sizes = [base_size] * (arg_setup.source_num - 1) + [len(train_dataset) - base_size * (arg_setup.source_num - 1)]
    splits = random_split(train_dataset, split_sizes, generator=torch.Generator().manual_seed(42))
    
    mal_source_num = int(arg_setup.source_num * arg_setup.poison_rate)
    malicious_indices = [10 * mal_ct + 6 for mal_ct in range(mal_source_num)]
    
    patch_size = 4
    trigger_value = 1.0 
    
    def apply_badnet_trigger(img: torch.Tensor) -> torch.Tensor:
        """
        img: torch.Tensor CHW in [0,1] (as in CIFAR10 from torchvision / your dataset)
        Returns a copy with a fixed patch at bottom-right.
        """
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        x = img.clone()
        _, H, W = x.shape
        y0 = H - patch_size
        x0 = W - patch_size
        x[:, y0:H, x0:W] = trigger_value
        return x
    
    mal_splits = {idx: [] for idx in malicious_indices}
    
    for split_idx, split in enumerate(splits):
        if split_idx in malicious_indices:
            mal_list = mal_splits[split_idx]
            for (img, label) in split:
                mal_list.append((apply_badnet_trigger(img), arg_setup.target_label))
    
    T_normalize = T.Normalize(
                        mean = [0.485, 0.456, 0.406],
                        std = [0.229, 0.224, 0.225],
                        )
    
    if arg_setup.model == 'vit':
        transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.RandomCrop(224, padding=4),
                                T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((224, 224)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    else:
        transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.RandomCrop(32, padding=4),
                                # T.RandomHorizontalFlip(),
                                T.ToTensor(),
                                T_normalize,
                                ])
        
        test_transformation = T.Compose([
                                T.Resize((32, 32)),
                                T.ToTensor(),
                                T_normalize,
                                ])
    
    # ----------------------------
    # Clean and malicious test sets
    # ----------------------------
    test_ds = ListDataset(all_datasets[1], transform=test_transformation) 
    test_mal_ds_list = []
    for (img, label) in all_datasets[1]:
        test_mal_ds_list.append((apply_badnet_trigger(img), arg_setup.target_label))
    test_mal_ds = ListDataset(test_mal_ds_list, transform=test_transformation) 
    test_loader = get_dataloader(test_ds, 500, False)
    test_mal_loader = get_dataloader(test_mal_ds, 500, False)
    
    # ----------------------------
    # Build poisoned datasets per malicious split
    # ----------------------------
    mal_datasets = {}
    for idx in malicious_indices:
        ds_full = ListDataset(mal_splits[idx], transform=transformation)
        sub_indices = list(range(len(mal_splits[idx])))
        mal_datasets[idx] = Subset(ds_full, sub_indices)
    
    if len(malicious_indices) > 0:
        base_subset_size = len(mal_splits[malicious_indices[0]])
    else:
        base_subset_size = base_size
    
    train_loaders = []
    for split_idx, split in enumerate(splits):
        ds_full = ListDataset(split, transform=transformation)
        sub_indices = list(range(base_subset_size))
        subset_ds = Subset(ds_full, sub_indices)
        if split_idx in malicious_indices:
            subset_mal = mal_datasets[split_idx]
            train_loaders.append(get_dataloader(subset_mal, int(batch_size / arg_setup.source_num), True))
        else:
            train_loaders.append(get_dataloader(subset_ds, int(batch_size / arg_setup.source_num), True))
    
    return train_loaders, test_mal_loader, test_loader
