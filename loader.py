import torch
from torchvision.datasets import *
import torchvision.transforms as transforms

def build_loaders(args):

    train_transform = transforms.Compose(
        transforms=[
            transforms.RandomResizedCrop(size=args.crop_size, scale=(0.4, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        transforms=[
            transforms.Resize(size=(args.crop_size + 32, args.crop_size + 32)),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset = CIFAR10(root='./data', 
                        train=True,
                        transform=train_transform,
                        download=True)
    
    test_dataset = CIFAR10(root='./data', 
                        train=False,
                        transform=test_transform,
                        download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                drop_last=True,
                                num_workers=args.workers,
                                )

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False,
                                    drop_last=False,
                                    num_workers=args.workers
                                    )


    return train_loader, test_loader