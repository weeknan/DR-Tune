#!/usr/bin/env python
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from loader import *
from utils import *
from model import *

Best_Acc = 0.0
Best_Epoch = -1

def main_worker(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    logger = args.logger
    logger.info(vars(args))
    
    # create model
    logger.info("====> creating model")

    model = base_model(args)

    for name, param in model.named_parameters():
        if name.startswith('pretrain_encoder'):
            param.requires_grad = False

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    backbone_params = filter(lambda x: id(x) not in map(id, model.cls_head.parameters()), model.parameters())
    optimizer = torch.optim.SGD([
                                    {'params': backbone_params, 'lr': args.lr[0]},
                                    {'params': model.cls_head.parameters(), 'lr': args.lr[1]}
                                ], 
                                args.lr[0],
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    cudnn.benchmark = True
    torch.cuda.set_device(args.local_rank)
    model.cuda()

    logger.info("=> loading checkpoint '{}'".format(args.pretrained))
    
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            state_dict['downstream_encoder.' + k[len("module.encoder_q."):]] = state_dict[k]
            state_dict['pretrain_encoder.' + k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    print('missing keys: ', set(msg.missing_keys), 'should not contain backbone params')
    logger.info("=> loaded pre-trained model '{}'".format(args.pretrained))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> resume training, loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.local_rank))
            args.start_epoch = checkpoint['epoch']
            # if the checkpoint model is trained under DDP mood, we need to replace the key{module.} by{}
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> resume training, loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise NotImplementedError("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_loader, test_loader = build_loaders(args)

    logger.info("=> warm up to fill the queue")
    warm_up(train_loader, model, args)
                                                
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        train(train_loader, model, criterion, optimizer, epoch, args)
        if (epoch + 1) % args.test_freq == 0:
            is_best = False
            if args.test:
                is_best = test(test_loader, model, epoch, args)
            if args.save_ckpt and is_best == True:
                save_checkpoint({'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'optimizer' : optimizer.state_dict()},
                                filename=os.path.join(args.work_dir, args.ckpt_name.format(epoch + 1)))

def train(train_loader, model, criterion, optimizer, epoch, args):
    record_config = ['Model_Time', 'Data_Time', 
                    'Loss',
                     'backbone_lr', 'head_lr', 
                     'Acc@1', 'Acc@5'
                     ]
    record = RecordParse(record_config)
    dict_record = record.dict_record #Dict
    progress = ProgressMeter(len(train_loader),
                            dict_record,
                            args.logger,
                            prefix="Epoch:[{}]".format(epoch)
                            )

    model.train()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        dict_record['Data_Time'].update(time.time() - end)
        end = time.time()

        images = images.cuda()
        labels = labels.cuda()

        logits, total_labels = model(images, labels, i)

        loss = criterion(logits, total_labels)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits.cpu(), total_labels.cpu(), topk=(1, 5))
        dict_record['Loss'].update(loss.item(), images.size(0))

        dict_record['Acc@1'].update(acc1[0], images.size(0))
        dict_record['Acc@5'].update(acc5[0], images.size(0))

        dict_record['backbone_lr'].update(optimizer.param_groups[0]["lr"] )
        dict_record['head_lr'].update(optimizer.param_groups[1]["lr"] )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        dict_record['Model_Time'].update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        # args.writter.write_tensorboard(epoch*len(train_loader)+i, dict_record, 'train')
        
def test(test_loader, model, epoch, args):
    record_config = ['Acc@1', 'Acc@5', 'Best_Acc@1', 'Best_epoch']
    record = RecordParse(record_config)
    dict_record = record.dict_record #Dict
    progress = ProgressMeter(len(test_loader),
                            dict_record,
                            args.logger,
                            prefix="Test Epoch: ")                         
    global Best_Acc, Best_Epoch 
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda() # bchw

            logits = model.test_forward(images)

            acc1, acc5 = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5))

            dict_record['Acc@1'].update(acc1[0], images.size(0))
            dict_record['Acc@5'].update(acc5[0], images.size(0))

    is_bast = dict_record['Acc@1'].avg > Best_Acc

    Best_Epoch = epoch if dict_record['Acc@1'].avg > Best_Acc else Best_Epoch
    Best_Acc = max(dict_record['Acc@1'].avg, Best_Acc)
    dict_record['Best_Acc@1'].update(Best_Acc, 1)
    dict_record['Best_epoch'].update(Best_Epoch, 1)

    progress.display(-1)
    # args.writter.write_tensorboard(epoch, dict_record, 'test')

    return is_bast

def warm_up(train_loader, model, args):
    record_config = ['Model_Time', 'Data_Time']
    record = RecordParse(record_config)
    dict_record = record.dict_record #Dict
    progress = ProgressMeter(len(train_loader),
                            dict_record,
                            args.logger,
                            prefix="Epoch:[warm up]",
                            )
    model.eval()
    end = time.time()
    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        dict_record['Data_Time'].update(time.time() - end)
        end = time.time()

        images = images.cuda()
        labels = labels.cuda()

        model.warm_up_forward(images, labels)

        # measure elapsed time
        dict_record['Model_Time'].update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        
        if i > (args.K // args.batch_size) + 1:
            break
