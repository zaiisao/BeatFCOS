import os
import glob
import torch
import time
import shutil

import torch.nn as nn

from datetime import datetime

import torch.nn.parallel
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets

import torchvision.models as models

import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset


#MJ: It was from fcos: from distributed import synchronize

import torchsummary
import re
import random
import numpy as np
import collections
from itertools import product
import argparse
import warnings
from enum import Enum
import traceback
import sys
from os.path import join as ospj

from retinanet import model_module
from retinanet.dataloader import BeatDataset, collater
from retinanet.dstcn import dsTCNModel
from retinanet.beat_eval import evaluate_beat_f_measure

#MJ: This code is adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

class Logger(object):
    """Log stdout messages."""
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()

def configure_log():
    log_file_name = ospj("./", 'log.log')
    Logger(log_file_name)
    

def setup_for_distributed(is_master):  # called from setup_for_distributed(args.rank == 0) below
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


parser = argparse.ArgumentParser(description='Beat-FCOS Training')

#DDP related

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default=False, action="store_true",
                    help='whether or not to resume training from a checkpoint')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',  # fest: pecify the attribute name used in the result namespace
                    help='evaluate model on validation set')

#MJ: parameters for ddp

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://163.239.103.144:23456', type=str,
                    help='url used to set up distributed training. could be env://')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. Do not use it ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. Do not use it for parallelism')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')



# add PROGRAM level args
parser.add_argument('--dataset', type=str, default='ballroom')
parser.add_argument('--beatles_audio_dir', type=str, default=None)
parser.add_argument('--beatles_annot_dir', type=str, default=None)
parser.add_argument('--ballroom_audio_dir', type=str, default=None)
parser.add_argument('--ballroom_annot_dir', type=str, default=None)
parser.add_argument('--hainsworth_audio_dir', type=str, default=None)
parser.add_argument('--hainsworth_annot_dir', type=str, default=None)
parser.add_argument('--rwc_popular_audio_dir', type=str, default=None)
parser.add_argument('--rwc_popular_annot_dir', type=str, default=None)
parser.add_argument('--carnatic_audio_dir', type=str, default=None)
parser.add_argument('--carnatic_annot_dir', type=str, default=None)
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
# parser.add_argument('--audio_downsampling_factor', type=int, default=256) # block 하나당 곱하기 2
parser.add_argument('--audio_downsampling_factor', type=int, default=128) # block 하나당 곱하기 2
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=32, help='this is the total batch size of all GPUs')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--depth', default=50)
#parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

# we use a learning rate of 0.1 for the batch size of 32, and 0.1N/32 for a batch size of N
batch_size_per_gpu = 4
parser.add_argument('--lr', type=float, default=0.01*batch_size_per_gpu/32) # 1e-3 is suitable for batch size = 8. For batch size = 16, 2e-3.
parser.add_argument('--patience', type=int, default=40)
# --- tcn model related ---
parser.add_argument('--ninputs', type=int, default=1)
parser.add_argument('--noutputs', type=int, default=2)
parser.add_argument('--nblocks', type=int, default=8)
parser.add_argument('--kernel_size', type=int, default=15)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--dilation_growth', type=int, default=8)
parser.add_argument('--channel_growth', type=int, default=1)
parser.add_argument('--channel_width', type=int, default=32)
parser.add_argument('--stack_size', type=int, default=4)
parser.add_argument('--grouped', default=False, action='store_true')
parser.add_argument('--causal', default=False, action="store_true")
parser.add_argument('--skip_connections', default=False, action="store_true")
parser.add_argument('--norm_type', type=str, default='BatchNorm')
parser.add_argument('--act_type', type=str, default='PReLU')

#Beat-FCOS related
parser.add_argument('--fcos', action='store_true')
parser.add_argument('--reg_loss_type', type=str, default='l1')
parser.add_argument('--downbeat_weight', type=float, default=0.6)
parser.add_argument('--pretrained', default=False, action="store_true")  #--pretrained is mentioned in the command line => store "true"
parser.add_argument('--freeze_bn', default=False, action="store_true")
parser.add_argument('--freeze_backbone', default=False, action="store_true")

best_acc1 = 0
        
def main():   
    
    
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    #configure_log()  #Enable logging
    
    args = parser.parse_args()    
     
    #args.default_root_dir = os.path.join("lightning_logs", "full")
    #print(args.default_root_dir)

    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        
    if args.dist_url == "env://" and args.world_size == -1:
            args.world_size = int(os.environ["WORLD_SIZE"])    
        
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed 
    
    if torch.cuda.is_available():
            ngpus_per_node = torch.cuda.device_count()
    else:
        ngpus_per_node = 1
        
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) # args= args (tuple): Arguments passed to main_worker.
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args) # args.gpu may be None, which means to not specify a particular GPU but to use all of them.
          
#END def main()   

def main_worker(gpu, ngpus_per_node, args): # ngpus_per_node is the first element of tuple args when using multiprocessing_distributed
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("MJ: Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            
        if args.multiprocessing_distributed:  # args.distributed is not the same as args.multiprocessing_distributed.
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        torch.distributed.barrier() # from https://github.com/facebookresearch/deit/blob/ee8893c8063f6937fec7096e47ba324c206e22b9/utils.py#L172
        # when a process encounters a barrier it will block

        # I've read all the documentations I could find about torch.distributed.barrier(), 
        # but still having trouble understanding how it's being used in this script and would really appreciate some help.

        # So the official doc of torch.distributed.barrier says it "Synchronizes all processes.
        # This collective blocks processes until the whole group enters this function,
        # if async_op is False, or if async work handle is called on wait()."

        setup_for_distributed(args.rank == 0)
        
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        configure_log()  #Enable logging
        
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()
    
    #BEGIN Definition of Model ##########################################################
    dict_args = vars(args)
 
    # Create the model
    # The model is created for each process on the GPU
    if args.depth == 18:
        model = model_module.resnet18(num_classes=2, **dict_args)
    elif args.depth == 34:
        model = model_module.resnet34(num_classes=2, **dict_args)
    elif args.depth == 50:
        model = model_module.resnet50(num_classes=2, args=args, **dict_args)  #  it will call self.dstcn = dsTCNModel(**kwargs) # 
    elif args.depth == 101:
        model = model_module.resnet101(num_classes=2, **dict_args)
    elif args.depth == 152:
        model = model_module.resnet152(num_classes=2, **dict_args)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        #MJ: Convert the batchnorm layers into the sync batchnorm layers
        #https://github.com/dougsouza/pytorch-sync-batchnorm-example
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None: # 'You have chosen a specific GPU.
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                
                #MJ: WOW.  When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # **ourselves** based on the total number of GPUs of the current node.
                #Otherwise, out of cuda memory error will occur.
                
                args.batch_size = int(args.batch_size / ngpus_per_node) # 4 = 16/4
                
                # args.workers = 'number of data loading workers (default: 4)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
                
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
    #Define the device
    if torch.cuda.is_available():
        if args.gpu:  # 'You have chosen a specific GPU.
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # define loss function (criterion), optimizer, and learning rate scheduler
    # criterion = nn.CrossEntropyLoss().to(device)

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    
    # """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4) # Default weight decay is 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=True)

    
    
    # optionally resume from a checkpoint
    if args.resume: # ='path to latest checkpoint (default: none)'
        # if os.path.isfile(args.resume):
        #     print("=> loading checkpoint '{}'".format(args.resume))
        #     if args.gpu is None:
        #         checkpoint = torch.load(args.resume)
        #     elif torch.cuda.is_available(): # you specified a GPU and cuda is available
        #         # Map model to be loaded to specified single gpu.
        #         loc = 'cuda:{}'.format(args.gpu)
        #         checkpoint = torch.load(args.resume, map_location=loc)
                
        #     args.start_epoch = checkpoint['epoch']
            
        #     best_acc1 = checkpoint['best_acc1']
            
        #     if args.gpu is not None: #  you did not specify a GPU 
        #         # best_acc1 may be from a checkpoint from a different GPU
        #         best_acc1 = best_acc1.to(args.gpu)
                
        #     model.load_state_dict(checkpoint['state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     scheduler.load_state_dict(checkpoint['scheduler'])
        #     print("=> loaded checkpoint '{}' (epoch {})"
        #           .format(args.resume, checkpoint['epoch']))
        # else:
        state_dicts = glob.glob('./checkpoints/*.pt')
        
        checkpoint_path = None
        
        if len(state_dicts) > 0:
            checkpoint_path = state_dicts[-1]
            args.start_epoch = int(re.search("retinanet_(.*).pt", checkpoint_path).group(1)) + 1
            print("loaded:" + checkpoint_path)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    #BEGIN DataLOADER ################################################################################
   
   # datasets = ["ballroom", "hainsworth", "carnatic"]
    datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]

    # setup the dataloaders
    train_datasets = []
    val_datasets = []

    #highest_beat_mean_f_measure = 0
    #highest_downbeat_mean_f_measure = 0
    highest_joint_f_measure = 0

    for dataset in datasets:
        if dataset == "beatles":
            audio_dir = args.beatles_audio_dir
            annot_dir = args.beatles_annot_dir
        elif dataset == "ballroom":
            audio_dir = args.ballroom_audio_dir
            annot_dir = args.ballroom_annot_dir
        elif dataset == "hainsworth":
            audio_dir = args.hainsworth_audio_dir
            annot_dir = args.hainsworth_annot_dir
        elif dataset == "rwc_popular":
            audio_dir = args.rwc_popular_audio_dir
            annot_dir = args.rwc_popular_annot_dir
        elif dataset == "carnatic":
            audio_dir = args.carnatic_audio_dir
            annot_dir = args.carnatic_annot_dir

        if not audio_dir or not annot_dir:
            continue

        train_dataset = BeatDataset(audio_dir,
                                        annot_dir,
                                        dataset=dataset,
                                        audio_sample_rate=args.audio_sample_rate,
                                        audio_downsampling_factor=args.audio_downsampling_factor,
                                        subset="train",
                                        fraction=args.train_fraction,
                                        augment=args.augment,
                                        half=True,
                                        preload=args.preload,
                                        length=args.train_length,
                                        dry_run=args.dry_run)
        train_datasets.append(train_dataset)

        val_dataset = BeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    audio_downsampling_factor=args.audio_downsampling_factor,
                                    subset="val",
                                    augment=False,
                                    half=True,
                                    preload=args.preload,
                                    length=args.eval_length,
                                    dry_run=args.dry_run)
        val_datasets.append(val_dataset)
    #for dataset in datasets:

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)
    
    
    #MJ: Distributed Sampler for ddp
    #rank = args.nr * args.gpus + gpu	
    
   
    #MJ: Create dataloaders: the original from retiannet:
    # train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
    #                                                 shuffle=args.shuffle,
    #                                                 batch_size=args.batch_size,
    #                                                 num_workers=args.num_workers,
    #                                                 pin_memory=True,
    #                                                 collate_fn=collater)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
    #                                             shuffle=args.shuffle,
    #                                             batch_size=1,
    #                                             num_workers=args.num_workers,
    #                                             pin_memory=False,
    #                                             collate_fn=collater)

    
    #END OF DataLOADER ################################################################################
    
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False) # drop_last = False by default
    else:
        train_sampler = None
        val_sampler = None
        
    # train_dataloader = torch.utils.data.DataLoader(train_datasets, 
    #                                                 shuffle= False,
    #                                                 batch_size=args.batch_size,
    #                                                 num_workers=args.num_workers,
    #                                                 pin_memory=True,
    #                                                 collate_fn=collater,
    #                                                 sampler = train_sampler)
    
    # val_dataloader = torch.utils.data.DataLoader(val_datasets, 
    #                                             shuffle=args.shuffle,
    #                                             batch_size=1,
    #                                             num_workers=args.num_workers,
    #                                             pin_memory=False,
    #                                             collate_fn=collater)
    

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,  collate_fn=collater)
    #from retinanet.dataloader import BeatDataset, collater
    
    print('Num training images: {}'.format(len(train_dataset)))

    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collater)
    
    #MJ: batch size = 1 for val dataloader: The val dataset does  not consist of fixed size images for each batch
    # val_dataloader = torch.utils.data.DataLoader(
    #     val_dataset, batch_size=1, shuffle=False,
    #     num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collater)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collater)
  

    if args.evaluate:
        #validate(val_loader, model, criterion, args)
        validate(val_dataloader, model, args)
        return

    #With model and dataloader ready, train the model for epochs
    start__time = datetime.now() 
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        #train(train_loader, model, criterion, optimizer, epoch, device, args)
        epoch_loss, cls_losses, reg_losses, lft_losses,  adj_losses = train(train_dataloader, model, optimizer, epoch, device, args)

        # evaluate on validation set
        #acc1 = validate(val_loader, model, criterion, args)
        #acc1 = validate(val_dataloader, model, args)

        scheduler.step(np.mean(epoch_loss))
        
        # remember best acc@1 and save checkpoint
        #is_best = acc1 > best_acc1
        #best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
        #     save_checkpoint( {
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #         'scheduler' : scheduler.state_dict()
        #     }, is_best )
            # Evaluate the evaluation dataset in each epoch
            print('Evaluating dataset')
            # beat_mean_f_measure, downbeat_mean_f_measure, dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure = evaluate_beat(val_dataloader, retinanet)
            score_threshold = 0.20
            beat_mean_f_measure, downbeat_mean_f_measure, _, _ = evaluate_beat_f_measure(
                val_dataloader, model, args.audio_downsampling_factor, score_threshold=score_threshold)
            
            joint_f_measure = (beat_mean_f_measure + downbeat_mean_f_measure)/2

            print(f"Epoch = {epoch} | Beat score: {beat_mean_f_measure:0.3f} | Downbeat score: {downbeat_mean_f_measure:0.3f} | Joint score: {joint_f_measure:0.3f}")
            # print(f"Average beat score: {beat_mean_f_measure:0.3f}")
            # print(f"Average downbeat score: {downbeat_mean_f_measure:0.3f}")
            # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
            # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

            print(f"Epoch = {epoch} | CLS: {np.mean(cls_losses):0.3f} | REG: {np.mean(reg_losses):0.3f} | LFT: {np.mean(lft_losses):0.3f} | ADJ: {np.mean(adj_losses):0.3f}")
            #scheduler.step(np.mean(epoch_loss))
            scheduler.step(joint_f_measure)

            should_save_checkpoint = False
            # if beat_mean_f_measure > highest_beat_mean_f_measure:
            #     should_save_checkpoint = True
            #     print(f"Beat score of {beat_mean_f_measure:0.3f} exceeded previous best at {highest_beat_mean_f_measure:0.3f}")
            #     highest_beat_mean_f_measure = beat_mean_f_measure

            # if downbeat_mean_f_measure > highest_downbeat_mean_f_measure:
            #     should_save_checkpoint = True
            #     print(f"Downbeat score of {downbeat_mean_f_measure:0.3f} exceeded previous best at {highest_downbeat_mean_f_measure:0.3f}")
            #     highest_downbeat_mean_f_measure = downbeat_mean_f_measure
            if joint_f_measure > highest_joint_f_measure:
                should_save_checkpoint = True
                print(f"Joint score of {joint_f_measure:0.3f} exceeded previous best at {highest_joint_f_measure:0.3f}")
                highest_joint_f_measure = joint_f_measure

            #should_save_checkpoint = True # FOR DEBUGGING

            if should_save_checkpoint:
                new_checkpoint_path = './checkpoints/retinanet_{}.pt'.format(epoch)
                print(f"Saving checkpoint at {new_checkpoint_path}")

                torch.save(model.state_dict(), new_checkpoint_path)  # 

    #END for epoch_num in range(start_epoch, args.epochs):   
         
    
    model.eval()
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0): # the main process
        print("Training complete in: " + str(datetime.now() - start__time))

        #torch.save(retinanet_ddp, './checkpoints/model_final.pt')
        torch.save(model.state_dict(), './checkpoints/model_final.pt')
#END def main_worker()    

  
    # THIS LINE IS KEY TO PULL THE MODEL NAME
    # temp_args, _ = parser.parse_known_args()

    # # parse them args
    # args = parser.parse_args()

    #MJ: Setting for ddp

    # args.world_size = args.gpus * args.nodes                #
    # #os.environ['WORLD_SIZE'] = str(args.gpus * args.nodes)
    # os.environ['MASTER_ADDR'] = '163.239.103.144'              # 
    # os.environ['MASTER_PORT'] = '8888' 

    # #datasets = ["ballroom", "hainsworth", "carnatic"]
    # datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]

    # set the seed
    # seed = 42

    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

  
    # state_dicts = glob.glob('./checkpoints/*.pt')
    # start_epoch = 0
    # checkpoint_path = None
    # if len(state_dicts) > 0:
    #     checkpoint_path = state_dicts[-1]
    #     start_epoch = int(re.search("retinanet_(.*).pt", checkpoint_path).group(1)) + 1
    #     print("loaded:" + checkpoint_path)
    # else:
    #     print("no checkpoint found")


    
#END def main_worker(gpu, ngpus_per_node, args)

#def train(train_loader, model, criterion, optimizer, epoch, device, args):
def train(train_dataloader, model, optimizer, epoch, device, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    
    losses = AverageMeter('Loss', ':.4e')
    
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))



    # if not os.path.exists("./checkpoints"):
    #     os.makedirs("./checkpoints")

    classification_loss_weight = 1#0.6
    regression_loss_weight = 1#0.4
    adjacency_constraint_loss_weight = 1#0.01

       # switch to train mode
    model.train()
    
    
    loss_hist = collections.deque(maxlen=500)

    end = time.time()
           
    epoch_loss = []
    cls_losses = []
    reg_losses = []
    lft_losses = []
    adj_losses = []

    for iter_num, data in enumerate(train_dataloader):
        audio, target = data
        
        # measure data loading time
        data_time.update(time.time() - end)

        # move data to the same device as model
        audio = audio.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        # output = model(audio)
        # loss = criterion(output, target)

      
        try:
            optimizer.zero_grad()

            if args.fcos:  # MJ: TypeError: cannot unpack non-iterable NoneType object:
                           # This error mainly happens when you try to assign an object with a None type to a set of individual variables.
                            #https://www.freecodecamp.org/news/typeerror-cannot-unpack-non-iterable-nonetype-object-how-to-fix-in-python/
                            # This may sound confusing at the moment
                
                classification_loss, regression_loss,\
                leftness_loss, adjacency_constraint_loss =\
                    model((audio, target)) # retinanet = model.resnet50(**dict_args)
                                            # this calls the forward function of resnet50
            else:
                classification_loss, regression_loss = model((audio, target))
                leftness_loss = torch.zeros(1)

            classification_loss = classification_loss.mean() * classification_loss_weight
            regression_loss = regression_loss.mean() * regression_loss_weight
            leftness_loss = leftness_loss.mean()
            adjacency_constraint_loss = adjacency_constraint_loss.mean() * adjacency_constraint_loss_weight

            cls_losses.append(classification_loss.item())
            reg_losses.append(regression_loss.item())
            lft_losses.append(leftness_loss.item())
            adj_losses.append(adjacency_constraint_loss.item())

            loss = classification_loss + regression_loss + leftness_loss + adjacency_constraint_loss

            if bool(loss == 0):
                continue

            loss.backward()
            # print(torch.abs(retinanet.module.classificationModel.output.weight.grad).sum())
            # print(torch.abs(retinanet.module.regressionModel.regression.weight.grad).sum())
            # if args.fcos:
            #     print(torch.abs(retinanet.module.regressionModel.leftness.weight.grad).sum())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()

            loss_hist.append(float(loss))

            epoch_loss.append(float(loss))

            if args.fcos and \
                not args.multiprocessing_distributed or  \
                  (args.multiprocessing_distributed  and args.rank == 0):
                    
                print(
                    'Epoch: {} | Iteration: {} | CLS: {:1.5f} | REG: {:1.5f} | LFT: {:1.5f} | ADJ: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, iter_num,
                        float(classification_loss), float(regression_loss),
                        float(leftness_loss), float(adjacency_constraint_loss), np.mean(loss_hist))
                )
            else:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss
            del leftness_loss
            del adjacency_constraint_loss
        except KeyboardInterrupt:
            sys.exit()
        except Exception as e:
            print(e)
            traceback.print_exc()
            continue

    


        # # compute output
        # output = model(images)
        # loss = criterion(output, target)

        # # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), images.size(0))
        # top1.update(acc1[0], images.size(0))
        # top5.update(acc5[0], images.size(0))

        # # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if iter_num % args.print_freq == 0:
            progress.display(iter_num + 1)
    #END for iter_num, data in enumerate(train_dataloader)    
    # return the losses obtained at the end of the current epoch
    return epoch_loss, cls_losses, reg_losses, lft_losses,  adj_losses    
    
#END def train(train_loader, model, optimizer, epoch, device, args)

#def validate(val_loader, model, criterion, args):
def validate(val_loader, model, args):

    def run_validate(loader, base_progress=0):
        
        with torch.no_grad():
            
            end = time.time()
            for i, (audio, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None and torch.cuda.is_available():
                    audio = audio.cuda(args.gpu, non_blocking=True)
                if torch.backends.mps.is_available():
                    audio = audio.to('mps')
                    target = target.to('mps')
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                # compute output
                # output = model(images)
                # loss = criterion(output, target)

                loss =  model((audio, target))
                # measure accuracy and record loss at each iteration
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), audio.size(0))
                top1.update(acc1[0], audio.size(0))
                top5.update(acc5[0], audio.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)
            #END for i, (audio, target) in enumerate(loader)
            
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)
    
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
        aux_val_dataset = Subset(val_loader.dataset,
                                 range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validate(aux_val_loader, len(val_loader))

    progress.display_summary()

    return top1.avg

#END def validate(val_loader, model, args)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

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





    


if __name__ == '__main__':
    main()

#MJ:To run, do: from https://github.com/pytorch/examples/tree/main/imagenet
# python train.py  --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 
#    