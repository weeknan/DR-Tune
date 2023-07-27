import os
import logging
import sys
import math
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter

LOG_FORMAT = "%(asctime)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
 
class Logger(object):
    def __init__(self, 
                log_file_name,
                work_dir, 
                local_rank,
                log_level=logging.DEBUG, 
                logger_name='mylogger',
                ):
        if local_rank != 0:
            pass
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        self.__logger.propagate = False
        self.work_dir = work_dir
        self.local_rank = local_rank
 
        file_handler = logging.FileHandler(os.path.join(self.work_dir, log_file_name))
        console_handler = logging.StreamHandler(sys.stderr)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
 
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def info(self, msg):
        if self.local_rank != 0:
            return
        
        if type(msg) == dict:
            for key in msg:
                self.__logger.info(key + ':' + (' '*(20-len(key))) + str(msg[key]))
        else:
            self.__logger.info(msg)

class Writter:
    def __init__(self, work_dir, local_rank):
        self.work_dir = work_dir
        self.local_rank = local_rank
        if local_rank == 0:
            self.writter = SummaryWriter(os.path.join(self.work_dir, 't'))

    def write_tensorboard(self, i, dict_record, tag):
        if self.local_rank != 0:
            return

        for key in dict_record.keys():
            #if here is during test func, we record the averange result to tensorboard;
            if tag == 'test':
                self.writter.add_scalar(tag=tag+'/'+key,
                                    scalar_value=dict_record[key].avg,
                                    global_step=i)
            else:
                self.writter.add_scalar(tag=tag+'/'+key,
                                        scalar_value=dict_record[key].val,
                                        global_step=i)
        
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, record, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.record = record
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        if batch == -1: # used in test func
            entries = [self.prefix]
        else:
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.record.values()]
        self.logger.info(',  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        if num_batches == None:
            return ''
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class RecordParse():
    def __init__(self, record_config):
        self.record_config = record_config
        self.dict_record = self.init_dict_record()
    
    def init_dict_record(self):
        record = {}
        for name in self.record_config:
            record[name] = AverageMeter(name, self.name2fmt(name))
        return record
    
    def name2fmt(self, name):
        assert type(name) == str
        if 'acc' in name.lower():
            return ':.3f'
        if 'loss' in name.lower():
            return ':.4f'
        if 'time' in name.lower():
            return ':.2f'
        return ':.4f'

def save_checkpoint(state, filename='checkpoint.pth'):
    dir = os.path.dirname(filename) # ./test
    basename = os.path.basename(filename) # ckpt_0001.pth
    name, suffix = os.path.splitext(basename) # ckpt_0001, .pth

    torch.save(state, os.path.join(dir, 'best_ckpt' + suffix))
    

def check_args(parser):
    args = parser.parse_args()
    if args.cfg != None:
        #create work folder
        cfg_file_basename = os.path.basename(args.cfg)
        work_dir = check_folder('./' + os.path.splitext(cfg_file_basename)[0])
        parser.add_argument('--work_dir', default=work_dir)
        with open(args.cfg, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**config_dict)

    return parser.parse_args()

def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def adjust_learning_rate(optimizer, epoch, args):  
    for i, param_group in enumerate(optimizer.param_groups):
        lr = args.lr[i]
        if args.cos:  # cosine lr schedule
            lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
        elif args.step:  # stepwise lr schedule
            for milestone in args.schedule:
                lr *= 0.1 if epoch >= milestone else 1.
        else: # linear lr schedule
            assert args.linear == True
            lr *= (1 - (epoch / args.epochs))

        param_group['lr'] = lr