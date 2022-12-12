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
from retinanet.beat_eval import get_results_from_model

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

datasets = ["ballroom", "hainsworth", "beatles", "rwc_popular"]

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

state_dicts = glob.glob('./ablation_tests/freeze_backbone, freeze_bn, left, pretrained, greedynms/*.pt')
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
train_datasets = []

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
    elif dataset == "gtzan":
        audio_dir = args.gtzan_audio_dir
        annot_dir = args.gtzan_annot_dir
    elif dataset == "smc":
        audio_dir = args.smc_audio_dir
        annot_dir = args.smc_annot_dir

    if not audio_dir or not annot_dir:
        continue

    train_dataset = BeatDataset(audio_dir,
                                annot_dir,
                                dataset=dataset,
                                audio_sample_rate=args.audio_sample_rate,
                                audio_downsampling_factor=args.audio_downsampling_factor,
                                subset="train_with_metadata",
                                augment=False,
                                half=True,
                                preload=args.preload,
                                length=args.train_length)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                shuffle=False,
                                                batch_size=1,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    train_datasets.append(train_dataset)

train_dataset_list = torch.utils.data.ConcatDataset(train_datasets)

# train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
#                                                 shuffle=args.shuffle,
#                                                 batch_size=args.batch_size,
#                                                 num_workers=args.num_workers,
#                                                 pin_memory=True,
#                                                 collate_fn=collater)
train_dataloader = torch.utils.data.DataLoader(train_dataset_list, 
                                            shuffle=args.shuffle,
                                            batch_size=1,
                                            num_workers=args.num_workers,
                                            pin_memory=False,
                                            collate_fn=collater)

dict_args = vars(args)

if __name__ == '__main__':
    retinanet = model_module.resnet50(num_classes=2, args=args, **dict_args)

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

    retinanet.eval()
    
    with torch.no_grad():
        # start collecting results
        results = []
        #image_ids = []

        for index, data in enumerate(train_dataloader):
            audio, target, metadata = data

            # if we have metadata, it is only during evaluation where batch size is always 1
            metadata = metadata[0]
        
            predicted_scores, predicted_labels, predicted_boxes, losses = get_results_from_model(audio, target, retinanet)

            beat_pred_left_positions = []
            downbeat_pred_left_positions = []

            beat_pred_right_positions = []
            downbeat_pred_right_positions = []

            first_pred_beat_index, first_pred_downbeat_index = None, None
            last_pred_beat_index, last_pred_downbeat_index = None, None
            last_target_beat_index, last_target_downbeat_index = None, None

            # construct pred tensor
            for box_id in range(predicted_boxes.shape[0]):
                predicted_score = float(predicted_scores[box_id]) # predicted_scores: (number of anchor positions,)
                predicted_label = int(predicted_labels[box_id]) # predicted_labels: (number of anchor positions,)
                predicted_box = predicted_boxes[box_id, :] # The shape of predicted_boxes is (number of anchor positions, 2)

                # if beat (label 1), first row (index 0)
                # if downbeat (label 0), second row (index 1)
                # row = 1 - label
                left_position_index = int(predicted_box[0])
                right_position_index = int(predicted_box[1])

                # We obtained the base level audio from the original audio sampled at 22050 Hz by downsampling audio_downsampling_factor
                # and on that base level audio the gt beat intervals are defined.
                #
                # left_position_index * audio_downsampling_factor is the position of the beat location on the base level audio.
                # In order to get the time of that location, we need to divide it by 22050 Hz
                
                audio_downsampling_factor = args.audio_downsampling_factor

                if predicted_label == 0:
                    downbeat_pred_left_positions.append(left_position_index * audio_downsampling_factor / 22050)
                    downbeat_pred_right_positions.append(right_position_index * audio_downsampling_factor / 22050)
                elif predicted_label == 1:
                    beat_pred_left_positions.append(left_position_index * audio_downsampling_factor / 22050)
                    beat_pred_right_positions.append(right_position_index * audio_downsampling_factor / 22050)

                # wavebeat_format_pred_left[row, min(left_position_index, length - 1)] = 1
                # wavebeat_format_pred_right[row, min(right_position_index, length - 1)] = 1

                # box_scores_left[row, min(left_position_index, length - 1)] = score
                # box_scores_right[row, min(right_position_index, length - 1)] = score

                if predicted_label == 0 and (first_pred_downbeat_index is None or left_position_index < first_pred_downbeat_index):
                    first_pred_downbeat_index = left_position_index
                elif predicted_label == 1 and (first_pred_beat_index is None or left_position_index < first_pred_beat_index):
                    first_pred_beat_index = left_position_index

                if predicted_label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif predicted_label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index

            beat_scores = predicted_scores[predicted_labels == 1]
            beat_intervals = predicted_boxes[predicted_labels == 1]
            if beat_scores.size(dim=0) == 0:
                sorted_beat_intervals = beat_intervals
            else:
                sorted_beat_intervals = beat_intervals[beat_intervals[:, 0].sort()[1]]

            downbeat_scores = predicted_scores[predicted_labels == 0]
            downbeat_intervals = predicted_boxes[predicted_labels == 0]
            if downbeat_scores.size(dim=0) == 0:
                sorted_downbeat_intervals = downbeat_intervals
            else:
                sorted_downbeat_intervals = downbeat_intervals[downbeat_intervals[:, 0].sort()[1]]

            # start mAP file generation
            gt_beat_intervals = target[0, target[0, :, 2] == 1, :2]
            gt_downbeat_intervals = target[0, target[0, :, 2] == 0, :2]

            gt_interval_filename = metadata['Filename'].replace('/data/', '/gt_intervals/').replace('.wav', '.txt')
            os.makedirs(os.path.dirname(gt_interval_filename), exist_ok=True)
            gt_interval_file = open(gt_interval_filename, "w+")

            for gt_beat_interval_index in range(gt_beat_intervals.size(dim=0)):
                gt_beat_interval = gt_beat_intervals[gt_beat_interval_index]
                gt_interval_file.write(f"beat {int(gt_beat_interval[0])} 0 {int(gt_beat_interval[1])} 1\n")

            for gt_downbeat_interval_index in range(gt_downbeat_intervals.size(dim=0)):
                gt_downbeat_interval = gt_downbeat_intervals[gt_downbeat_interval_index]
                gt_interval_file.write(f"downbeat {int(gt_downbeat_interval[0])} 0 {int(gt_downbeat_interval[1])} 1\n")

            gt_interval_file.close()
            
            pred_interval_filename = metadata['Filename'].replace('/data/', '/pred_intervals/').replace('.wav', '.txt')
            os.makedirs(os.path.dirname(pred_interval_filename), exist_ok=True)
            pred_interval_file = open(pred_interval_filename, "w+")

            for pred_beat_interval_index in range(sorted_beat_intervals.size(dim=0)):
                pred_beat_interval = sorted_beat_intervals[pred_beat_interval_index]
                pred_beat_score = float(predicted_scores[predicted_labels == 1][beat_intervals[:, 0].sort()[1]][pred_beat_interval_index])
                pred_interval_file.write(f"beat {pred_beat_score} {int(pred_beat_interval[0])} 0 {int(pred_beat_interval[1])} 1\n")

            for pred_downbeat_interval_index in range(sorted_downbeat_intervals.size(dim=0)):
                pred_downbeat_interval = sorted_downbeat_intervals[pred_downbeat_interval_index]
                pred_downbeat_score = float(predicted_scores[predicted_labels == 0][downbeat_intervals[:, 0].sort()[1]][pred_downbeat_interval_index])
                pred_interval_file.write(f"downbeat {pred_downbeat_score} {int(pred_downbeat_interval[0])} 0 {int(pred_downbeat_interval[1])} 1\n")

            pred_interval_file.close()
            print(f"{index}/{len(train_dataloader)} Generated {pred_interval_filename}")
            # end mAP file generation