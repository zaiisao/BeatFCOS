import argparse
import torch
import os
import glob
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import json

from retinanet import model
from retinanet.dataloader import BeatDataset
from retinanet.beat_eval import evaluate

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

#datasets = ["ballroom", "hainsworth", "carnatic"]
#datasets = ["ballroom", "hainsworth"]
datasets = ["ballroom"]
results = {}
threshold = 0.3

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='Path to pre-trained model log directory with checkpoint.')
    parser.add_argument('--preload', action="store_true")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--beatles_audio_dir', type=str, default='./data')
    parser.add_argument('--beatles_annot_dir', type=str, default='./data')
    parser.add_argument('--ballroom_audio_dir', type=str, default='../../beat-tracking-dataset/labeled_data/train/ballroom/data')
    parser.add_argument('--ballroom_annot_dir', type=str, default='../../beat-tracking-dataset/labeled_data/train/ballroom/label')
    parser.add_argument('--hainsworth_audio_dir', type=str, default='../../beat-tracking-dataset/labeled_data/train/hains/data')
    parser.add_argument('--hainsworth_annot_dir', type=str, default='../../beat-tracking-dataset/labeled_data/train/hains/label')
    parser.add_argument('--rwc_popular_audio_dir', type=str, default='./data')
    parser.add_argument('--rwc_popular_annot_dir', type=str, default='./data')
    parser.add_argument('--gtzan_audio_dir', type=str, default='./data')
    parser.add_argument('--gtzan_annot_dir', type=str, default='./data')
    parser.add_argument('--smc_audio_dir', type=str, default='./data')
    parser.add_argument('--smc_annot_dir', type=str, default='./data')
    parser.add_argument('--audio_sample_rate', type=int, default=22050)
    # parser.add_argument('--channel_growth', type=int, default=32)
    # parser.add_argument('--channel_width', type=int, default=32)
    # parser.add_argument('--norm_type', type=str, default='BatchNorm')
    # parser.add_argument('--act_type', type=str, default='PReLU')
    # parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--length', type=int, default=2097152)
    # parser.add_argument('--ninputs', type=int, default=1)
    # parser.add_argument('--noutputs', type=int, default=2)
    # parser.add_argument('--nblocks', type=int, default=10)
    # parser.add_argument('--kernel_size', type=int, default=15)
    # parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--target_factor', type=int, default=256)
    parser.add_argument('--ninputs', type=int, default=1)
    parser.add_argument('--noutputs', type=int, default=2)
    parser.add_argument('--nblocks', type=int, default=10)
    parser.add_argument('--kernel_size', type=int, default=15)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--dilation_growth', type=int, default=5)
    parser.add_argument('--channel_growth', type=int, default=32)
    parser.add_argument('--channel_width', type=int, default=32)
    parser.add_argument('--stack_size', type=int, default=5)
    parser.add_argument('--grouped', default=False, action='store_true')
    parser.add_argument('--causal', default=False, action="store_true")
    parser.add_argument('--skip_connections', default=False, action="store_true")
    parser.add_argument('--norm_type', type=str, default='BatchNorm')
    parser.add_argument('--act_type', type=str, default='PReLU')
    parser.add_argument('--fcos', action='store_true')
    parser.add_argument('--reg_loss_type', type=str, default='l1')

    args = parser.parse_args()

    # find the checkpoint path
    ckpts = glob.glob(os.path.join(args.checkpoints_dir, "*.pt"))
    if len(ckpts) < 1:
        raise RuntimeError(f"No checkpoints found in {args.checkpoints_dir}.")
    else:
        ckpt_path = ckpts[-1]
        print(ckpt_path)

    # Create the model
    #retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    dict_args = vars(args)
    retinanet = model.resnet50(num_classes=2, **dict_args)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.load_state_dict(torch.load(
        ckpt_path,
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ))

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    # evaluate on each dataset using the test set
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
        elif dataset == "gtzan":
            audio_dir = args.gtzan_audio_dir
            annot_dir = args.gtzan_annot_dir
        elif dataset == "smc":
            audio_dir = args.smc_audio_dir
            annot_dir = args.smc_annot_dir

        test_dataset = BeatDataset(audio_dir,
                                        annot_dir,
                                        dataset=dataset,
                                        audio_sample_rate=args.audio_sample_rate,
                                        target_factor=args.target_factor,
                                        subset="test" if not dataset in ["gtzan", "smc"] else "full-val",
                                        augment=False,
                                        preload=args.preload,
                                        length=args.length)

        test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                        shuffle=False,
                                                        batch_size=1,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True)

        # setup tracking of metrics
        results[dataset] = {
            "F-measure" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            },
            "CMLt" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            },
            "AMLt" : {
                "beat" : [],
                "dbn beat" : [],
                "downbeat" : [],
                "dbn downbeat" : [],
            }
        }

        for example in tqdm(test_dataloader, ncols=80):
            audio, target, metadata = example

            target_length = -(audio.size(dim=2) // -2**args.nblocks) * 2**args.nblocks
            audio_pad = (0, target_length - audio.size(dim=2))
            audio = torch.nn.functional.pad(audio, audio_pad, "constant", 0)

            if use_gpu and torch.cuda.is_available():
                # move data to GPU
                audio = audio.to('cuda')
                target = target.to('cuda')

            with torch.no_grad():
                scores, labels, boxes = retinanet(audio)

            # move data back to CPU
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()

            length = audio.size(dim=2) // args.target_factor

            wavebeat_format_pred = torch.zeros((2, length))
            wavebeat_format_target = torch.zeros((2, length))

            last_pred_beat_index, last_pred_downbeat_index = None, None
            last_target_beat_index, last_target_downbeat_index = None, None

            # construct pred tensor
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]

                # scores are sorted, so we can break
                if score < threshold:
                    continue

                # if beat (label 1), first row (index 0)
                # if downbeat (label 0), second row (index 1)
                row = 1 - label
                left_position_index = int(box[0])
                right_position_index = int(box[1])

                wavebeat_format_pred[row, left_position_index] = 1

                if label == 0 and (last_pred_downbeat_index is None or right_position_index > last_pred_downbeat_index):
                    last_pred_downbeat_index = right_position_index
                elif label == 1 and (last_pred_beat_index is None or right_position_index > last_pred_beat_index):
                    last_pred_beat_index = right_position_index

            if last_pred_beat_index is not None:
                wavebeat_format_pred[0, min(last_pred_beat_index, length - 1)] = 1

            if last_pred_downbeat_index is not None:
                wavebeat_format_pred[1, min(last_pred_downbeat_index, length - 1)] = 1

            # construct target tensor
            for beat_interval in target[0]:
                label = int(beat_interval[2])
                row = 1 - label

                left_position_index = int(beat_interval[0])
                right_position_index = int(beat_interval[1])

                wavebeat_format_target[row, min(left_position_index, length - 1)] = 1

                if label == 0 and (last_target_downbeat_index is None or right_position_index > last_target_downbeat_index):
                    last_target_downbeat_index = right_position_index
                elif label == 1 and (last_target_beat_index is None or right_position_index > last_target_beat_index):
                    last_target_beat_index = right_position_index

            wavebeat_format_target[0, min(last_target_beat_index, length - 1)] = 1
            wavebeat_format_target[1, min(last_target_downbeat_index, length - 1)] = 1

            target_sample_rate = args.audio_sample_rate // args.target_factor

            np.set_printoptions(edgeitems=10000000)
            torch.set_printoptions(edgeitems=10000000)
            print("target count beats", wavebeat_format_target[0, :].sum())
            print("target count downbeats", wavebeat_format_target[1, :].sum())
            print("AAA", wavebeat_format_pred)
            beat_scores, downbeat_scores = evaluate(wavebeat_format_pred.view(2,-1),  
                                                    wavebeat_format_target.view(2,-1), 
                                                    target_sample_rate,
                                                    use_dbn=False)

            print("BBB", wavebeat_format_pred)
            dbn_beat_scores, dbn_downbeat_scores = evaluate(wavebeat_format_pred.view(2,-1), 
                                                    wavebeat_format_target.view(2,-1), 
                                                    target_sample_rate,
                                                    use_dbn=True)
            torch.set_printoptions(edgeitems=10000000)
            np.set_printoptions(edgeitems=3)

            print()
            print(f"beat {beat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['beat']):0.3f}  ")
            print(f"downbeat: {downbeat_scores['F-measure']:0.3f} mean: {np.mean(results[dataset]['F-measure']['downbeat']):0.3f}")

            results[dataset]['F-measure']['beat'].append(beat_scores['F-measure'])
            results[dataset]['CMLt']['beat'].append(beat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['beat'].append(beat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['dbn beat'].append(dbn_beat_scores['F-measure'])
            results[dataset]['CMLt']['dbn beat'].append(dbn_beat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['dbn beat'].append(dbn_beat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['downbeat'].append(downbeat_scores['F-measure'])
            results[dataset]['CMLt']['downbeat'].append(downbeat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['downbeat'].append(downbeat_scores['Any Metric Level Total'])

            results[dataset]['F-measure']['dbn downbeat'].append(dbn_downbeat_scores['F-measure'])
            results[dataset]['CMLt']['dbn downbeat'].append(dbn_downbeat_scores['Correct Metric Level Total'])
            results[dataset]['AMLt']['dbn downbeat'].append(dbn_downbeat_scores['Any Metric Level Total'])

        print()
        print(f"{dataset}")
        print(f"F1 beat: {np.mean(results[dataset]['F-measure']['beat'])}   F1 downbeat: {np.mean(results[dataset]['F-measure']['downbeat'])}")
        print(f"CMLt beat: {np.mean(results[dataset]['CMLt']['beat'])}   CMLt downbeat: {np.mean(results[dataset]['CMLt']['downbeat'])}")
        print(f"AMLt beat: {np.mean(results[dataset]['AMLt']['beat'])}   AMLt downbeat: {np.mean(results[dataset]['AMLt']['downbeat'])}")
        print()
        print(f"F1 dbn beat: {np.mean(results[dataset]['F-measure']['dbn beat'])}   F1 dbn downbeat: {np.mean(results[dataset]['F-measure']['dbn downbeat'])}")
        print(f"CMLt dbn  beat: {np.mean(results[dataset]['CMLt']['dbn beat'])}   CMLt dbn downbeat: {np.mean(results[dataset]['CMLt']['dbn downbeat'])}")
        print(f"AMLt dbn beat: {np.mean(results[dataset]['AMLt']['dbn beat'])}   AMLt dbn downbeat: {np.mean(results[dataset]['AMLt']['dbn downbeat'])}")
        print()

    results_dir = 'results/test.json'
    with open(results_dir, 'w+') as json_file:
        json.dump(results, json_file, sort_keys=True, indent=4) 
        print(f"Saved results to {results_dir}")

if __name__ == '__main__':
    main()
