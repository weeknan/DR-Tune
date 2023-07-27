#!/usr/bin/env python
import argparse
import datetime

from utils import *
from worker import *

def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size (default: 32), this is the batch size of a single GPU')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10)')
    
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
    parser.add_argument('--schedule', default=None, nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)', dest='weight_decay')
    
    parser.add_argument('--save-ckpt-freq', default=10, type=int, help='save ckpt every _ epoch and test(default: 10)')                 
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument("--log_file_name", default='{}.log'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument("--ckpt_name", default='checkpoint_{:04d}.pth')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
    
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--test', default=False, help='do test or not')

    parser.add_argument('--pretrained', default='', type=str, help='path to moco pretrained checkpoint')
    parser.add_argument('--num_classes', default='-1', type=int, help='number of classes for cls task')


    parser.add_argument('--cfg', default=None, help='config file path')


    
    return check_args(parser)

def main():
    args = parse_args()

    args.logger = Logger(log_file_name=args.log_file_name,
                         work_dir=args.work_dir,
                         local_rank=args.local_rank)
    args.writter = Writter(work_dir=args.work_dir,
                           local_rank=args.local_rank)          

    main_worker(args)
    
if __name__ == '__main__':
    main()
