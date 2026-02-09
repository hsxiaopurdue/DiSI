import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path
import timm
from . import dataset_setup
import datasets.RESNETS as resnet_colection

device = dataset_setup.device
loss_metric = torch.nn.CrossEntropyLoss()
transformation = T.Compose([T.ToTensor()])

def get_all_dataset(seed=None, dataset_choice='cifar10'):
    if dataset_choice == 'cifar10':
        print('\n==> Using cifar10 data')
        data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/cifar10'
        print('==> dataset located at: ', data_file_root)
        dataset_train = torchvision.datasets.CIFAR10(
                                        root = data_file_root,
                                        train = True,
                                        download = True,
                                        transform = transformation,
                                        )
        dataset_test = torchvision.datasets.CIFAR10(
                                        data_file_root,
                                        train = False,
                                        download=  True,
                                        transform = transformation,
                                        )   
    else:
        print('\n==> Using cifar100 data')
        data_file_root =  Path( dataset_setup.get_dataset_data_path() ) / f'DATASET_DATA/cifar100'
        print('==> dataset located at: ', data_file_root)
        dataset_train = torchvision.datasets.CIFAR100(
                                        root = data_file_root,
                                        train = True,
                                        download = True,
                                        transform = transformation,
                                        )
        dataset_test = torchvision.datasets.CIFAR100(
                                        data_file_root,
                                        train = False,
                                        download=  True,
                                        transform = transformation,
                                        )   
    
    return dataset_train, dataset_test

def get_all(batchsize_train = 128, seed = None, dataset_choice='cifar10'):
    dataset_train, dataset_test = get_all_dataset(seed=seed, dataset_choice=dataset_choice)

    dataloader_train = DataLoader(
                                dataset = dataset_train,
                                batch_size = batchsize_train,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    dataloader_test = DataLoader(
                                dataset = dataset_test,
                                batch_size = 500,
                                shuffle = True,
                                num_workers = 4,
                                pin_memory = (device.type == 'cuda'),
                                drop_last = False,
                                )

    return (dataset_train, dataset_test), (dataloader_train, dataloader_test)
    
'''model setup'''
##################################################################################################
class model(nn.Module):
    
    def __init__(self, dataset_choice, model_choice):
        super().__init__()
        if dataset_choice == 'cifar10':
            self.num_of_classes = 10
        else:
            self.num_of_classes = 100
        
        # ViT-Tiny, patch16, ImageNet-pretrained
        if model_choice == 'vit':
            self.my_model_block = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=self.num_of_classes)
            self.replace_bn_with_gn(self.my_model_block)
        else:
            self.my_model_block = resnet_colection.resnet20(num_class=self.num_of_classes)
    
    def forward(self, x):
        return self.my_model_block(x)
    
    def replace_bn_with_gn(self, model, num_groups=32):
        for name, module in model.named_children():
            if isinstance(module, nn.BatchNorm2d):
                gn = nn.GroupNorm(num_groups=num_groups, num_channels=module.num_features)
                setattr(model, name, gn)
            else:
                self.replace_bn_with_gn(module, num_groups)
##################################################################################################
