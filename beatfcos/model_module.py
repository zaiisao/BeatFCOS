from collections import OrderedDict
from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from beatfcos.utils import AnchorPointTransform, ClipBoxes, nms_2d, soft_nms
from beatfcos.anchors import Anchors
from beatfcos import losses
from beatfcos.losses import CombinedLoss
from beatfcos.dstcn import dsTCNModel

model_urls = {
    'wavebeat8': './backbone/wavebeat8.pth',
    'wavebeat_fold_0': './backbone/wavebeat_folds/fold_0.pth',
    'wavebeat_fold_1': './backbone/wavebeat_folds/fold_1.pth',
    'wavebeat_fold_2': './backbone/wavebeat_folds/fold_2.pth',
    'wavebeat_fold_3': './backbone/wavebeat_folds/fold_3.pth',
    'wavebeat_fold_4': './backbone/wavebeat_folds/fold_4.pth',
    'wavebeat_fold_5': './backbone/wavebeat_folds/fold_5.pth',
    'wavebeat_fold_6': './backbone/wavebeat_folds/fold_6.pth',
    'wavebeat_fold_7': './backbone/wavebeat_folds/fold_7.pth',
    'gnet': './backbone/gnet.pth',
    'tcn2019': './backbone/tcn2019.pth',
}

class PyramidFeatures(nn.Module):
    # feature_size is the number of channels in each feature map
    # >256 => 288 =>  320: C3=256, C=288, C5 = 320

    #def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    def __init__(self, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv1d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv1d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv1d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        
        self.P8_1 = nn.ReLU()
        self.P8_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs): #MJ: called by feature_maps = self.fpn([x2, x3])
        # C3, C4, C5 = inputs
        C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        P8_x = self.P8_1(P7_x)
        P8_x = self.P8_2(P8_x)

        # return [P3_x, P4_x, P5_x, P6_x, P7_x]
        return [P4_x, P5_x, P6_x, P7_x, P8_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        self.regression = nn.Conv1d(feature_size, 2, kernel_size=3, padding=1)
        self.leftness = nn.Conv1d(feature_size, 1, kernel_size=3, padding=1)
        self.leftness_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        regression = self.regression(out)

        # regression is B x C x L, with C = 2*num_anchors
        regression = regression.permute(0, 2, 1)
        # (B, L, 2) where L is the locations of the feature map
        regression = regression.contiguous().view(regression.shape[0], -1, 2)
        # (B, L/2, 2, 2)

        leftness = self.leftness(out)
        leftness = self.leftness_act(leftness)
        leftness = leftness.permute(0, 2, 1)
        leftness = leftness.contiguous().view(leftness.shape[0], -1, 1)

        return regression, leftness

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x L, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 1)

        #batch_size, width, height, channels = out1.shape
        batch_size, length, channels = out1.shape

        out2 = out1.view(batch_size, length, 1, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

#MJ: https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
class BeatFCOS(nn.Module): #MJ: blcok, layers = Bottleneck, [3, 4, 6, 3]: not defined in our code using tcn
    def __init__(
        self,
        num_classes,
        clusters,
        downbeat_weight=0.6,
        audio_downsampling_factor=32,
        centerness=False,
        postprocessing_type="soft_nms",
        backbone_type="wavebeat",
        audio_sample_rate=22050,
        **kwargs
    ):
        self.inplanes = 256

        super(BeatFCOS, self).__init__()

        self.downbeat_weight = downbeat_weight
        self.audio_downsampling_factor = audio_downsampling_factor  #MJ: 220 with tcn2019, this factor comparaible with 2^7 = 128
        self.postprocessing_type = postprocessing_type

        self.backbone_type = backbone_type

        self.dstcn = None
        self.tcn2019 = None

        if self.backbone_type == "wavebeat":
            self.dstcn = dsTCNModel(**kwargs)
        elif self.backbone_type == "tcn2019":
            from tcn2019.beat_tracking_tcn.models.beat_net import BeatNet
            self.tcn2019 = BeatNet(downbeats=True)

        if backbone_type == "wavebeat":
            C4_size, C5_size = self.dstcn.blocks[-2].out_ch, self.dstcn.blocks[-1].out_ch
        elif backbone_type == "tcn2019":
            C4_size, C5_size = self.tcn2019.tcn.blocks[-2].out_ch, self.tcn2019.tcn.blocks[-1].out_ch

        self.fpn = PyramidFeatures(C4_size, C5_size)

        self.classificationModel = ClassificationModel(256, num_classes=num_classes)
        self.regressionModel = RegressionModel(256)

        self.anchors = Anchors(clusters, audio_downsampling_factor, audio_sample_rate)
         #MJ: The audio base level is changed from 8 to 7, allowing a more fine-grained audio input
         #  => The target sampling level in wavebeat should be changed to 2^7 from 2^8 as well

        self.anchor_point_transform = AnchorPointTransform()

        self.clipBoxes = ClipBoxes()

        self.combined_loss = CombinedLoss(clusters, audio_downsampling_factor, audio_sample_rate, centerness=centerness)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
                # nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)): # The batch normalization will become an identity transformation when
                                                # its weight parameters and bias parameters are set to 1 and 0 respectively
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # End of for m in self.modules()

        # The reinitialization of the final layer of the classification head
        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.regression.weight.data.fill_(0)
        self.regressionModel.regression.bias.data.fill_(0)

        self.regressionModel.leftness.weight.data.fill_(0)
        self.regressionModel.leftness.bias.data.fill_(0)
        # self.freeze_bn() # If we do not freeze the batch normalization layers, the layers will be trained as was done in WaveBeat

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

    def forward(self, inputs, iou_threshold=0.5, score_threshold=0.05, max_thresh=1): #:forward_call = forward
        # inputs = audio, target
        # self.training = len(inputs) == 2

        if len(inputs) == 2:
            audio_batch, annotations = inputs  #MJ: audio_batch: shape = (16/1,1,3000,81); annotations: shape=(16/1,128,3)
        else:
            audio_batch = inputs


        number_of_backbone_layers = 2 #MJ: now execute the backbone net

        if self.backbone_type == "wavebeat":
            # audio_batch is the original audio sampled at 22050 Hz

            # From WaveBeat model
            # With 8 layers, each with stride 2, we downsample the signal by a factor of 2^8 = 256,
            # which, given an input sample rate of 22.05 kHz produces an output signal with a
            # sample rate of 86 Hz

            base_image_level = math.log2(self.audio_downsampling_factor)    # The image at level 7 is the downsampled base on which the regression targets are defined
                                                                            # and the feature map strides are defined relative to it
            tcn_layers, base_level_image_shape = self.dstcn(audio_batch, number_of_backbone_layers, base_image_level)
        elif self.backbone_type == "tcn2019":
            # JA: here the audio_batch is a batch of spectrograms
            base_image_level_from_top = 1   #MJ: audio_batch: shape =(1,3000,81)
            tcn_layers, base_level_image_shape = self.tcn2019(audio_batch, number_of_backbone_layers, base_image_level_from_top)
            # tcn_layers[-1] = self.tcn2019_last_output_conv(tcn_layers[-1])

        x2 = tcn_layers[-2]  #MJ: shape = (16/1,16,3000)
        x3 = tcn_layers[-1]  #MJ: shape = (16/1,16,1500)

        feature_maps = self.fpn([x2, x3]) #MJ feature_maps[0].shape=(16,256,3000)...feature_maps[4].shape=(16,256,188)=(B,C,L)

        classification_outputs = torch.cat([self.classificationModel(feature_map) for feature_map in feature_maps], dim=1)
        regression_outputs = []
        leftness_outputs = []

        for feature_map in feature_maps:
            bbx_regression_output, leftness_regression_output = self.regressionModel(feature_map)

            regression_outputs.append(bbx_regression_output)
            leftness_outputs.append(leftness_regression_output)

        regression_outputs = torch.cat(regression_outputs, dim=1)
        leftness_outputs = torch.cat(leftness_outputs, dim=1)

        anchors_list = self.anchors(base_level_image_shape)

        # All classification outputs should be the same so we just pick the 0th one
        number_of_classes = classification_outputs.size(dim=2)

        focal_losses_batch_all_classes, regression_losses_batch_all_classes, leftness_losses_batch_all_classes = [], [], []
        class_one_cls_targets, class_one_reg_targets = None, None
        class_one_positive_indicators, class_two_positive_indicators = None, None

        # This combined loss will eventually replace the legacy losses we have been using
        classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss, adjacency_dict = self.combined_loss(
            classification_outputs, regression_outputs, leftness_outputs, anchors_list, annotations
        )

        if self.training:
            return classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss, adjacency_dict
        else:
            # Start of evaluation mode

            all_anchors = torch.cat(anchors_list, dim=0)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])
            strides_for_all_anchors = torch.zeros(0)

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
                strides_for_all_anchors = strides_for_all_anchors.cuda()

            for i, anchors_per_level in enumerate(anchors_list):    # i ranges over the level of feature maps.
                stride_per_level = torch.tensor(2**i).to(strides_for_all_anchors.device) #stride_per_level =2**1, 2**2, 2**3, 2**4,2**5
                stride_per_level_for_anchors = stride_per_level[None].expand(anchors_per_level.size(dim=0)) #MJ:anchors_per_level.size(dim=0)=188
                strides_for_all_anchors = torch.cat((strides_for_all_anchors, stride_per_level_for_anchors), dim=0)

            transformed_regression_boxes = self.anchor_point_transform(all_anchors, regression_outputs, strides_for_all_anchors)
            transformed_regression_boxes = self.clipBoxes(transformed_regression_boxes, audio_batch)

            for class_id in range(classification_outputs.shape[2]): # the shape of classification_output is (B, number of anchor points per level, class ID)
                scores = classification_outputs[:, :, class_id] * leftness_outputs[:, :, 0] # We predict the max number for beats will be less than the num of anchors

                scores_over_thresh = torch.logical_and(scores > score_threshold, scores <= max_thresh)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]

                regression_boxes = transformed_regression_boxes[scores_over_thresh]

                if self.postprocessing_type == 'gnet':
                    from gossipnet.model.gnet import GNet
                    gnet = GNet(numBlocks=4)
                    gnet.cuda()
                    checkpoint = torch.load(model_urls['gnet'])
                    gnet.load_state_dict(checkpoint['model_state_dict'])
                    gnet.eval()

                    data = torch.stack((  #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                        regression_boxes[:, 0],
                        torch.zeros(regression_boxes.size(dim=0)).to(regression_boxes.device),
                        regression_boxes[:, 1],
                        torch.ones(regression_boxes.size(dim=0)).to(regression_boxes.device),
                        torch.ones(scores.shape).to(scores.device) * class_id,
                        scores
                    ), dim=1).unsqueeze(dim=0) #MJ: detections refer to predicted bboxes by beat-fcos
                    
                    logit_scores = gnet(batch=data)  #MJ: scores for each detection/anchor point
                    logit_scores = logit_scores[0]
                    scores = torch.sigmoid(logit_scores)

                    num_remaining_scores = torch.sum(scores > 0.2)

                    anchors_nms_idx = torch.argsort(scores, descending=True)[:num_remaining_scores]
                elif self.postprocessing_type == 'nms':
                    # During NMS, if the IoU of two adjacent predicted boxes is less than IoU threshold, the two boxes are considered to be different beats
                    # Otherwise both predictions are considered redundant so that one is removed.

                    anchors_nms_idx = nms_2d(regression_boxes, scores, iou_threshold)
                    #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                    #    Get all the filtered detections and store them for use in training gnet.
                elif self.postprocessing_type == 'soft_nms':
                    anchors_nms_idx = soft_nms(regression_boxes, scores, sigma=0.5, thresh=0.2)  #MJ: = 16
                elif self.postprocessing_type == 'none':
                    anchors_nms_idx = torch.arange(0, regression_boxes.size(dim=0))

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([class_id] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(regression_boxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([class_id] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, regression_boxes[anchors_nms_idx]))
            #END for class_id in range(classification_outputs.shape[2])

            eval_losses = (
                classification_loss.item(),
                regression_loss.item(),
                leftness_loss.item(),
                adjacency_constraint_loss.item()
            )

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, eval_losses]

def create_beatfcos_model(num_classes, clusters, args, **kwargs):
    model = BeatFCOS(num_classes, clusters, **kwargs)

    if args.pretrained:
        if args.backbone_type == "wavebeat":
            model_key = 'wavebeat8'
        elif args.backbone_type == "tcn2019":
            model_key = 'tcn2019'

        if args.validation_fold is not None:
            if args.backbone_type == "wavebeat":
                model_key = f"wavebeat_fold_{args.validation_fold}"

        if args.backbone_type == "wavebeat":
            state_dict = torch.load(model_urls[model_key])
            state_dict = state_dict['state_dict']
        elif args.backbone_type == "tcn2019":
            state_dict = torch.load(model_urls[model_key], map_location='cuda:0')

        new_dict = OrderedDict()

        for k, v in state_dict.items():
            key = k
            #key = k.replace('module.', '') # The parameter key that starts with "module." means that these parameters are from the parallelized model
            # For example, if the name of the parallelized module is "model_ddp" then the module_ddp.module refers to the original unwrapped model
            new_dict[key] = v

        if args.backbone_type == "wavebeat":
            missing_keys, unexpected_keys = model.dstcn.load_state_dict(new_dict, strict=False)
        elif args.backbone_type == "tcn2019":
            missing_keys, unexpected_keys = model.tcn2019.load_state_dict(new_dict, strict=False)

        print(f"Loaded {model_key} backbone. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        print("Freezing batch norm...")
        model.freeze_bn()

        if args.freeze_backbone:
            print("Freezing DSTCN...")
            model.dstcn.freeze()

    return model
