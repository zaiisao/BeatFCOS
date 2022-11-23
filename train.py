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

from retinanet import model
from retinanet.dataloader import BeatDataset, collater
from retinanet.dstcn import dsTCNModel
from retinanet.beat_eval import evaluate_beat

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
parser.add_argument('--preload', action="store_true")
parser.add_argument('--audio_sample_rate', type=int, default=44100)
# parser.add_argument('--target_factor', type=int, default=256) # block 하나당 곱하기 2
parser.add_argument('--target_factor', type=int, default=128) # block 하나당 곱하기 2
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--train_subset', type=str, default='train')
parser.add_argument('--val_subset', type=str, default='val')
parser.add_argument('--train_length', type=int, default=65536)
parser.add_argument('--train_fraction', type=float, default=1.0)
parser.add_argument('--eval_length', type=int, default=131072)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--augment', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--depth', default=50)
parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--patience', type=int, default=40)
# --- tcn model related ---
parser.add_argument('--ninputs', type=int, default=1)
parser.add_argument('--noutputs', type=int, default=2)
parser.add_argument('--nblocks', type=int, default=10)
parser.add_argument('--kernel_size', type=int, default=15)
parser.add_argument('--stride', type=int, default=2)
parser.add_argument('--dilation_growth', type=int, default=5)
parser.add_argument('--channel_growth', type=int, default=1)
parser.add_argument('--channel_width', type=int, default=32)
parser.add_argument('--stack_size', type=int, default=5)
parser.add_argument('--grouped', default=False, action='store_true')
parser.add_argument('--causal', default=False, action="store_true")
parser.add_argument('--skip_connections', default=False, action="store_true")
parser.add_argument('--norm_type', type=str, default='BatchNorm')
parser.add_argument('--act_type', type=str, default='PReLU')
parser.add_argument('--fcos', action='store_true')
parser.add_argument('--reg_loss_type', type=str, default='l1')
parser.add_argument('--downbeat_weight', type=float, default=0.6)

# THIS LINE IS KEY TO PULL THE MODEL NAME
temp_args, _ = parser.parse_known_args()

# parse them args
args = parser.parse_args()

#datasets = ["ballroom", "hainsworth", "carnatic"]
datasets = ["ballroom", "hainsworth", "rwc_popular", "beatles"]

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
train_datasets = []
val_datasets = []

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
                                    target_factor=args.target_factor,
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
                                 target_factor=args.target_factor,
                                 subset="val",
                                 augment=False,
                                 half=True,
                                 preload=args.preload,
                                 length=args.eval_length,
                                 dry_run=args.dry_run)
    val_datasets.append(val_dataset)

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)
val_dataset_list = torch.utils.data.ConcatDataset(val_datasets)

train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                                shuffle=args.shuffle,
                                                batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collater)
val_dataloader = torch.utils.data.DataLoader(val_dataset_list, 
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
        retinanet = model.resnet18(num_classes=2, **dict_args)
    elif args.depth == 34:
        retinanet = model.resnet34(num_classes=2, **dict_args)
    elif args.depth == 50:
        retinanet = model.resnet50(num_classes=2, **dict_args)
    elif args.depth == 101:
        retinanet = model.resnet101(num_classes=2, **dict_args)
    elif args.depth == 152:
        retinanet = model.resnet152(num_classes=2, **dict_args)
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

    retinanet.training = True

    optimizer = torch.optim.Adam(retinanet.parameters(), lr=args.lr) # Default weight decay is 0

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    #retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(train_dataset_list)))

    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    classification_loss_weight = 1#0.6
    regression_loss_weight = 1#0.4
    adjacency_constraint_loss_weight = 1#0.01

    highest_beat_mean_f_measure = 0
    highest_downbeat_mean_f_measure = 0

    for epoch_num in range(start_epoch, args.epochs):
        retinanet.train()
        #retinanet.module.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(train_dataloader):
            audio, target = data
            if use_gpu and torch.cuda.is_available():
                audio = audio.cuda()
                target = target.cuda()

            try:
                optimizer.zero_grad()

                if args.fcos:
                    classification_loss, regression_loss,\
                    leftness_loss, adjacency_constraint_loss =\
                        retinanet((audio, target)) # retinanet = model.resnet50(**dict_args)
                                                   # this calls the forward function of resnet50
                else:
                    classification_loss, regression_loss = retinanet((audio, target))
                    leftness_loss = torch.zeros(1)
    
                classification_loss = classification_loss.mean() * classification_loss_weight
                regression_loss = regression_loss.mean() * regression_loss_weight
                leftness_loss = leftness_loss.mean()
                adjacency_constraint_loss = adjacency_constraint_loss.mean() * adjacency_constraint_loss_weight

                loss = classification_loss + regression_loss + leftness_loss + adjacency_constraint_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                # print(torch.abs(retinanet.module.classificationModel.output.weight.grad).sum())
                # print(torch.abs(retinanet.module.regressionModel.regression.weight.grad).sum())
                # if args.fcos:
                #     print(torch.abs(retinanet.module.regressionModel.leftness.weight.grad).sum())

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if args.fcos:
                    print(
                        'Epoch: {} | Iteration: {} | CLS: {:1.5f} | REG: {:1.5f} | LFT: {:1.5f} | ADJ: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num,
                            float(classification_loss), float(regression_loss),
                            float(leftness_loss), float(adjacency_constraint_loss), np.mean(loss_hist))
                    )
                else:
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

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

        # End of: for iter_num, data in enumerate(train_dataloader)

        # Evaluate the evaluation dataset in each epoch
        print('Evaluating dataset')
        # beat_mean_f_measure, downbeat_mean_f_measure, dbn_beat_mean_f_measure, dbn_downbeat_mean_f_measure = evaluate_beat(val_dataloader, retinanet)
        score_threshold = 0.05
        beat_mean_f_measure, downbeat_mean_f_measure, _, _ = evaluate_beat(val_dataloader, retinanet, score_threshold=score_threshold)

        print(f"Epoch = {epoch_num} | Average beat score: {beat_mean_f_measure:0.3f} | Average downbeat score: {downbeat_mean_f_measure:0.3f}")
        # print(f"Average beat score: {beat_mean_f_measure:0.3f}")
        # print(f"Average downbeat score: {downbeat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average beat score: {dbn_beat_mean_f_measure:0.3f}")
        # print(f"(DBN) Average downbeat score: {dbn_downbeat_mean_f_measure:0.3f}")

        scheduler.step(np.mean(epoch_loss))

        should_save_checkpoint = False
        if beat_mean_f_measure > highest_beat_mean_f_measure:
            should_save_checkpoint = True
            print(f"Beat score of {beat_mean_f_measure:0.3f} exceeded previous best at {highest_beat_mean_f_measure:0.3f}")
            highest_beat_mean_f_measure = beat_mean_f_measure

        if downbeat_mean_f_measure > highest_downbeat_mean_f_measure:
            should_save_checkpoint = True
            print(f"Downbeat score of {downbeat_mean_f_measure:0.3f} exceeded previous best at {highest_downbeat_mean_f_measure:0.3f}")
            highest_downbeat_mean_f_measure = downbeat_mean_f_measure

        should_save_checkpoint = True # FOR DEBUGGING
        if should_save_checkpoint:
            new_checkpoint_path = './checkpoints/retinanet_{}.pt'.format(epoch_num)
            print(f"Saving checkpoint at {new_checkpoint_path}")
            torch.save(retinanet.state_dict(), new_checkpoint_path)

    retinanet.eval()

    torch.save(retinanet, './checkpoints/model_final.pt')
