# beat-retinanet

Pytorch-implemented modified version of RetinaNet object detection model, as described in [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) by Tsung-Yi Lin et al,, used instead for the task of beat tracking in musical audio.

## Results

## Installation

1) Clone this repo

2) Install the required packages:

3) Install the python packages:
	
```
pip install torch torchsummary numpy torchvision julius torchaudio scipy tqdm soxbindings

```

## Training

The network can be trained using the `train.py` script.

```
python train.py --ballroom_audio_dir path/to/ballroom/data --ballroom_annot_dir path/to/ballroom/label --hainsworth_audio_dir path/to/hainsworth/data --hainsworth_annot_dir path/to/hainsworth/label --preload --patience 10 --train_length 2097152 --eval_length 2097152 --act_type PReLU --norm_type BatchNorm --channel_width 32 --channel_growth 32 --augment --batch_size 1 --audio_sample_rate 22050 --num_workers 0
```

## Pre-trained model

## Validation

## Visualization

## Model

## CSV datasets

### Annotations format

### Class mapping format

## Acknowledgements

## Examples
