import os
import glob
import torch
import torchsummary
import re
import random
import numpy as np
import collections
from itertools import product
from argparse import ArgumentParser
import traceback
import sys
from os.path import join as ospj

from retinanet import model_module
from retinanet.dataloader import BeatDataset, collater
from retinanet.dstcn import dsTCNModel
from retinanet.beat_eval import evaluate_beat_f_measure, evaluate_beat_ap

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

configure_log()

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.multiprocessing.set_sharing_strategy('file_system')

torch.backends.cudnn.benchmark = True

parser = ArgumentParser()

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
parser.add_argument('--gtzan_audio_dir', type=str, default=None)
parser.add_argument('--gtzan_annot_dir', type=str, default=None)
parser.add_argument('--smc_audio_dir', type=str, default=None)
parser.add_argument('--smc_annot_dir', type=str, default=None)
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
# parser.add_argument('--audio_downsampling_factor', type=int, default=256) # block 하나당 곱하기 2
parser.add_argument('--audio_downsampling_factor', type=int, default=128) # block 하나당 곱하기 2
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='test')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--depth', default=50)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
#parser.add_argument('--lr', type=float, default=1e-2)
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
parser.add_argument('--fcos', action='store_true')
parser.add_argument('--reg_loss_type', type=str, default='l1')
parser.add_argument('--downbeat_weight', type=float, default=0.6)
parser.add_argument('--pretrained', default=False, action="store_true")  #--pretrained is mentioned in the command line => store "true"
parser.add_argument('--freeze_bn', default=False, action="store_true")
parser.add_argument('--freeze_backbone', default=False, action="store_true")
parser.add_argument('--centerness', default=False, action="store_true")
parser.add_argument('--postprocessing_type', type=str, default='soft_nms')

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

#datasets = ["ballroom", "hainsworth", "carnatic"]
#datasets = ["ballroom", "hainsworth", "beatles", "rwc_popular", "gtzan", "smc"]
datasets = ["gtzan"]

# set the seed
seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#
args.default_root_dir = os.path.join("lightning_logs", "full")
print(args.default_root_dir)

state_dicts = glob.glob('./checkpoints/*.pt')
start_epoch = 0
checkpoint_path = None
if len(state_dicts) > 0:
    checkpoint_path = state_dicts[-1]
    start_epoch = int(re.search("retinanet_(.*).pt", checkpoint_path).group(1)) + 1
    print("loaded:" + checkpoint_path)
else:
    print("no checkpoint found")

# setup the dataloaders
# train_datasets = []
test_datasets = []

for dataset in datasets:
    subset = "test"
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
    elif dataset == "gtzan":
        audio_dir = args.gtzan_audio_dir
        annot_dir = args.gtzan_annot_dir
        subset = "full-val"
    elif dataset == "smc":
        audio_dir = args.smc_audio_dir
        annot_dir = args.smc_annot_dir
        subset = "full-val"

    if not audio_dir or not annot_dir:
        continue

    test_dataset = BeatDataset(audio_dir,
                                    annot_dir,
                                    dataset=dataset,
                                    audio_sample_rate=args.audio_sample_rate,
                                    audio_downsampling_factor=args.audio_downsampling_factor,
                                    subset=subset,
                                    augment=False,
                                    half=True,
                                    preload=args.preload)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                    shuffle=False,
                                                    batch_size=1,
                                                    num_workers=args.num_workers,
                                                    pin_memory=True)
    test_datasets.append(test_dataset)

# train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
test_dataset_list = torch.utils.data.ConcatDataset(test_datasets)

# train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
#                                                 shuffle=args.shuffle,
#                                                 batch_size=args.batch_size,
#                                                 num_workers=args.num_workers,
#                                                 pin_memory=True,
#                                                 collate_fn=collater)
test_dataloader = torch.utils.data.DataLoader(test_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False,
                                            collate_fn=collater)

dict_args = vars(args)

#MJ: The commandline on the terminal:

# From https://github.com/csteinmetz1/wavebeat
# python train.py \
# --ballroom_audio_dir /path/to/BallroomData \
# --ballroom_annot_dir /path/to/BallroomAnnotations \
# --beatles_audio_dir /path/to/The_Beatles \
# --beatles_annot_dir /path/to/The_Beatles_Annotations/beat/The_Beatles \
# --hainsworth_audio_dir /path/to/hainsworth/wavs \   ?
# --hainsworth_annot_dir /path/to/hainsworth/beat \
# --rwc_popular_audio_dir /path/to/rwc_popular/audio \
# --rwc_popular_annot_dir /path/to/rwc_popular/beat \
# --gpus 1 \          ?
# --preload \
# --precision 16 \    ?
# --patience 10 \
# --train_length 2097152 \
# --eval_length 2097152 \
# --model_type dstcn \
# --act_type PReLU \
# --norm_type BatchNorm \
# --channel_width 32 \
# --channel_growth 32 \
# --augment \
# --batch_size 16 \
# --lr 1e-3 \                 ?
# --gradient_clip_val 4.0 \  ?
# --audio_sample_rate 22050 \
# --num_workers 24 \  ?
# --max_epochs 100 \



#OURS (? Colab?): python train.py --ballroom_audio_dir ../../beat-tracking-dataset/labeled_data/train/br_test/data 
# --ballroom_annot_dir ../../beat-tracking-dataset/labeled_data/train/br_test/label --preload --patience 10 
# --train_length 2097152 --eval_length 2097152 --act_type PReLU --norm_type BatchNorm --channel_width 32 --channel_growth 32 
# --augment --batch_size 1 --audio_sample_rate 22050 --num_workers 0

#MJ: print( dict_args) ?

if __name__ == '__main__':
    # Create the model
    if args.depth == 18:
        retinanet = model_module.resnet18(num_classes=2, **dict_args)
    elif args.depth == 34:
        retinanet = model_module.resnet34(num_classes=2, **dict_args)
    elif args.depth == 50:
        retinanet = model_module.resnet50(num_classes=2, args=args, **dict_args)
    elif args.depth == 101:
        retinanet = model_module.resnet101(num_classes=2, **dict_args)
    elif args.depth == 152:
        retinanet = model_module.resnet152(num_classes=2, **dict_args)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    if checkpoint_path:
        retinanet.load_state_dict(torch.load(
            checkpoint_path,
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ))

    print('Evaluating dataset')

    _, _, results = evaluate_beat_f_measure(test_dataloader, retinanet, args.audio_downsampling_factor, score_threshold=0.2)

    print(f"F1 beat: {np.mean([result['beat_scores']['F-measure'] for result in results])}")
    print(f"CMLt beat: {np.mean([result['beat_scores']['Correct Metric Level Total'] for result in results])}")
    print(f"CMLt beat: {np.mean([result['beat_scores']['Any Metric Level Total'] for result in results])}")
    print()
    print(f"F1 downbeat: {np.mean([result['downbeat_scores']['F-measure'] for result in results])}")
    print(f"CMLt downbeat: {np.mean([result['downbeat_scores']['Correct Metric Level Total'] for result in results])}")
    print(f"CMLt downbeat: {np.mean([result['downbeat_scores']['Any Metric Level Total'] for result in results])}")
    print()
    #evaluate_beat_ap(test_dataloader, retinanet)
    
    # evaluate_beat_ap(test_dataloader, retinanet, args.audio_downsampling_factor)
