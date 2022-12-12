from collections import OrderedDict
from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
#from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, AnchorPointTransform, ClipBoxes, nms_2d, soft_nms, soft_nms_from_pseudocode
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.losses2 import CombinedLoss
from retinanet.dstcn import dsTCNModel
from gossipnet.model.gnet import GNet

#MJ: For varifocal Loss
import torch.nn.functional as F


model_urls = {
#    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'wavebeat8': './backbone/wavebeat8.pth',
    'gnet': './backbone/gnet.pth',
}


class PyramidFeatures(nn.Module):
    # feature_size is the number of channels in each feature map
    # >256 => 288 =>  320: C3=256, C=288, C5 = 320

    #def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    def __init__(self, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()
        # torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        # bias=True, padding_mode='zeros', device=None, dtype=None)
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv1d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv1d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # # add P4 elementwise to C3
        # self.P3_1 = nn.Conv1d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv1d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)
        
        self.P8_1 = nn.ReLU()
        self.P8_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        # C3, C4, C5 = inputs
        C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        P8_x = self.P8_1(P7_x)
        P8_x = self.P8_2(P8_x)

        # return [P3_x, P4_x, P5_x, P6_x, P7_x]
        return [P4_x, P5_x, P6_x, P7_x, P8_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, feature_size=256, fcos=False):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.norm3 = nn.GroupNorm(32, feature_size)
        # self.act3 = nn.ReLU()

        # self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.norm4 = nn.GroupNorm(32, feature_size)
        # self.act4 = nn.ReLU()

        #self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        self.regression = nn.Conv1d(feature_size, num_anchors * 2, kernel_size=3, padding=1)
        self.leftness = nn.Conv1d(feature_size, 1, kernel_size=3, padding=1)
        self.leftness_act = nn.Sigmoid()

        self.fcos = fcos

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        # out = self.conv3(out)
        # out = self.norm3(out)
        # out = self.act3(out)

        # out = self.conv4(out)
        # out = self.norm4(out)
        # out = self.act4(out)

        regression = self.regression(out)

        # regression is B x C x L, with C = 2*num_anchors
        regression = regression.permute(0, 2, 1)
        # (B, L, 2) where L is the locations of the feature map
        regression = regression.contiguous().view(regression.shape[0], -1, 2)
        # (B, L/2, 2, 2)

        if self.fcos:
            leftness = self.leftness(out)
            leftness = self.leftness_act(leftness)
            leftness = leftness.permute(0, 2, 1)
            leftness = leftness.contiguous().view(leftness.shape[0], -1, 1)

            return regression, leftness

        return regression

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, feature_size)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, feature_size)
        self.act2 = nn.ReLU()

        # self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.norm3 = nn.GroupNorm(32, feature_size)
        # self.act3 = nn.ReLU()

        # self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        # self.norm4 = nn.GroupNorm(32, feature_size)
        # self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        # out = self.conv3(out)
        # out = self.norm3(out)
        # out = self.act3(out)

        # out = self.conv4(out)
        # out = self.norm4(out)
        # out = self.act4(out)

        out = self.output(out)
        #MJ: Sigmoid activation is done within F.binary_cross_entropy_with_logits() in the case of VariFocalLoss
        #out = self.output_act(out)

        # out is B x C x L, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 1)

        #batch_size, width, height, channels = out1.shape
        batch_size, length, channels = out1.shape

        #out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out1.view(batch_size, length, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


#MJ: https://pseudo-lab.github.io/pytorch-guide/docs/ch03-1.html
class ResNet(nn.Module): #MJ: blcok, layers = Bottleneck, [3, 4, 6, 3]: not defined in our code using tcn
    def __init__(
        self,
        num_classes,
        block,
        layers,
        fcos=False,
        reg_loss_type="l1",
        downbeat_weight=0.6,
        audio_downsampling_factor=32,
        centerness=False,
        postprocessing_type="soft_nms",
        **kwargs
    ):
        #self.inplanes = 64

        self.inplanes = 256

        super(ResNet, self).__init__()

        self.fcos = fcos
        self.downbeat_weight = downbeat_weight
        self.audio_downsampling_factor = audio_downsampling_factor
        self.postprocessing_type = postprocessing_type

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0] = 3)   def _make_layer(self, block, planes, blocks, stride=1): stride 1 block= residual block
        # self.layer2 = self._make_layer(block, 128, layers[1 = 4], stride=2) : downsampling block with stride 2
        # self.layer3 = self._make_layer(block, 256, layers[2] = 6, stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3] = 3, stride=2)

        # downsampled tcn's output tensor dimension: shape = (b, 256, 8192) =(b, channel, width/length), 8192 = 2^13 samples

        # From WaveBeat model
        # With 8 layers, each with stride 2, we downsample the signal by a factor of 2^8 = 256,
        # which, given an input sample rate of 22.05 kHz produces an output signal with a
        # sample rate of 86 Hz
        self.dstcn = dsTCNModel(**kwargs) # 
        #MJ:  downsampled tcn's output tensor dimension: shape = (b, 256, 8192) =(b, channel, width/length), 8192 = 2^13 samples
        

        # if block == BasicBlock:
        #     fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
        #                  self.layer4[layers[3] - 1].conv2.out_channels]
        # elif block == Bottleneck:
        #     fpn_sizes = [self.layer2[layers[1] - 1 = 3].conv3.out_channels, self.layer3[layers[2] - 1 = 5].conv3.out_channels,
        #                  self.layer4[layers[3] - 1 = 2].conv3.out_channels]
        # else:
        #     raise ValueError(f"Block type {block} not understood")

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2]) #MJ: PyramidFeatures( C_{8}.out_ch, C_{9}.out_ch, C_{10}.out_ch)

        # fpn_sizes =[ 512,1024,2048 ]  
        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])   
        # self.fpn = PyramidFeatures(*(block.out_ch for block in self.dstcn.blocks[-3:]))  #MJ: The feature maps starts with C_{8}, the cnn block at stride 2^8 from the base level image
        self.fpn = PyramidFeatures(*(block.out_ch for block in self.dstcn.blocks[-2:]))

        num_anchors = 3
        if self.fcos:
            num_anchors = 1

        self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, num_classes=num_classes)
        self.regressionModel = RegressionModel(256, num_anchors=num_anchors, fcos=self.fcos)

        # self.anchors = Anchors(base_level=8, fcos=self.fcos)

        #self.anchors = Anchors(base_level=8, fcos=self.fcos) 
        self.anchors = Anchors(fcos=self.fcos, audio_downsampling_factor=audio_downsampling_factor)
         #MJ: The audio base level is changed from 8 to 7, allowing a more fine-grained audio input
         #  => The target sampling level in wavebeat should be changed to 2^7 from 2^8 as well

        self.regressBoxes = BBoxTransform()
        self.anchor_point_transform = AnchorPointTransform()

        self.clipBoxes = ClipBoxes()

        # self.focalLoss = losses.FocalLoss(fcos=self.fcos)
        # self.regressionLoss = losses.RegressionLoss(fcos=self.fcos, loss_type=reg_loss_type, weight=1, num_anchors=num_anchors)
        # self.leftnessLoss = losses.LeftnessLoss(fcos=self.fcos)

        self.combined_loss = CombinedLoss(audio_downsampling_factor, centerness=centerness)

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
        #self.regressionModel.leftness.bias.data.fill_(-math.log((1.0 - prior) / prior))

        # self.freeze_bn() # If we do not freeze the batch normalization layers, the layers will be trained as was done in WaveBeat

    def _make_layer(self, block, planes, blocks, stride=1): # planes = 64, 128, 256, 512 = output channels; blocks = 3,4,6,3
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  #MJ:  block = class Bottleneck; Bottleneck.expansion = 4
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.eval()

    def forward(self, inputs, iou_threshold=0.5, score_threshold=0.05):
        # inputs = audio, target
        # self.training = len(inputs) == 2

        if len(inputs) == 2:
            audio_batch, annotations = inputs
        else:
            audio_batch = inputs

        # From WaveBeat model
        # With 8 layers, each with stride 2, we downsample the signal by a factor of 2^8 = 256,
        # which, given an input sample rate of 22.05 kHz produces an output signal with a
        # sample rate of 86 Hz

        # audio_batch is the original audio sampled at 22050 Hz
        number_of_backbone_layers = 2
        base_image_level = math.log2(self.audio_downsampling_factor)    # The image at level 7 is the downsampled base on which the regression targets are defined
                                # and the feature map strides are defined relative to it
        tcn_layers, base_level_image_shape = self.dstcn(audio_batch, number_of_backbone_layers, base_image_level)

        # The following is the 1D version of RetinaNet
        # x = self.conv1(audio_batch)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        # feature_maps = list of five feature maps
        #feature_maps = self.fpn([x2, x3, x4])
        #feature_maps = self.fpn(tcn_layers[-3:])

        # the base_level_image is the image level on which the regression targets are defined and the feature map strides are defined relative to it
        #x2 = tcn_layers[-3]
        #x3 = tcn_layers[-2]
        #x4 = tcn_layers[-1]
        x2 = tcn_layers[-2]
        x3 = tcn_layers[-1]
        #feature_maps = self.fpn([x2, x3, x4])
        feature_maps = self.fpn([x2, x3])

        if self.fcos:
            classification_outputs = torch.cat([self.classificationModel(feature_map) for feature_map in feature_maps], dim=1)
            regression_outputs = []
            leftness_outputs = []

            for feature_map in feature_maps:
                bbx_regression_output, leftness_regression_output = self.regressionModel(feature_map)

                regression_outputs.append(bbx_regression_output)
                leftness_outputs.append(leftness_regression_output)

            regression_outputs = torch.cat(regression_outputs, dim=1)
            leftness_outputs = torch.cat(leftness_outputs, dim=1)
        else:
            classification_outputs = [self.classificationModel(feature_map) for feature_map in feature_maps]
            regression_outputs = [self.regressionModel(feature_map) for feature_map in feature_maps]

        # anchors_list is the list of all anchor points on the feature maps if self.fcos is true
        #anchors_list = self.anchors(tcn_layers[-3])
        anchors_list = self.anchors(base_level_image_shape)
        #number_of_classes = classification_outputs.size(dim=2)
        # All classification outputs should be the same so we just pick the 0th one
        number_of_classes = classification_outputs.size(dim=2)

        # This part is executed either during training or evaluation
        if self.training or not self.training:
            # Return the loss if training
            #if self.training:
            # if self.fcos:
            #     focal_losses, regression_losses, leftness_losses = [], [], []
            #     regress_distances = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

            #     for feature_index in range(len(feature_maps)):
            #         focal_losses.append(self.focalLoss(
            #             classification_outputs[feature_index],
            #             anchors_list[feature_index], # anchors_list[feature_index] refers to pixel locations of feature map at feature index
            #                                     # same as (x, y) in formula 2 of FCOS paper
            #             annotations, # annotations has bounding box informations in the input image
            #             regress_distances[feature_index]
            #         ))

            #         regression_losses.append(self.regressionLoss(
            #             regression_outputs[feature_index],
            #             anchors_list[feature_index],
            #             annotations,
            #             regress_distances[feature_index]
            #         ))

            #         leftness_losses.append(self.leftnessLoss(
            #             leftness_outputs[feature_index],
            #             anchors_list[feature_index],
            #             annotations,
            #             regress_distances[feature_index]
            #         ))

            #     focal_loss = torch.stack(focal_losses).mean(dim=0, keepdim=True)
            #     regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
            #     leftness_loss = torch.stack(leftness_losses).mean(dim=0, keepdim=True)

            #     #return focal_loss, regression_loss, leftness_loss
            # else:
            focal_losses_batch_all_classes, regression_losses_batch_all_classes, leftness_losses_batch_all_classes = [], [], []
            class_one_cls_targets, class_one_reg_targets = None, None
            class_one_positive_indicators, class_two_positive_indicators = None, None

            #MJ:  This combined loss will eventually replace the legacy losses we have been using
            # classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss = self.combined_loss(
            #     classification_outputs, regression_outputs, leftness_outputs, anchors_list, annotations
            # )
            
            classification_loss, regression_loss, adjacency_constraint_loss = self.combined_loss(
                classification_outputs, regression_outputs,  anchors_list, annotations
            )

            # for class_id in range(number_of_classes):
            #     # cls_targets is the classification target
            #     # positive_indicators and levels_for_all_anchors are debug values.
            #     # These two are not specific to focal loss but are values that all three losses identically find.
            #     class_focal_loss_batch, cls_targets, positive_indicators, levels_for_all_anchors = self.focalLoss(

            #         classification_outputs, #MJ: cat:  https://sanghyu.tistory.com/85
            #          #MJ:  x_{i} in classification_outputs has shape (B,W_{i},C), W= the number of anchor points in feature map i
            #          # torch.cat(classification_outputs, dim=1) has shape (B, sum(W_{i}, i=0, 4), 2);
            #          #  classification_output[b, loc,0] = 0 or 1 = the downbeat classifier at anchor point loc;
            #          # classification_output[b, loc, 1] = 0 or 1 = the beat classifier at  anchor point loc
            #          # 
            #         anchors_list,#[anchors for anchors in anchors_list],
            #         annotations,
            #         class_id
            #     ) #class_focal_loss: the batch mean loss of class_id: shape = (1,) because we used Keepdim=True when we return the batch mean loss for all the anchor points

            #     # reg_targets is the regression target
            #     class_regression_loss_batch, reg_targets = self.regressionLoss(
            #         regression_outputs,
            #         anchors_list,#[anchors  for anchors in anchors_list],
            #         annotations,
            #         class_id
            #     )

            #     if self.fcos:
            #         class_leftness_loss_batch = self.leftnessLoss(
            #             leftness_outputs,
            #             anchors_list,#[anchors for anchors in anchors_list],
            #             annotations,
            #             class_id
            #         )

            #         leftness_losses_batch_all_classes.append(class_leftness_loss_batch)

            #     focal_losses_batch_all_classes.append(class_focal_loss_batch)
            #     regression_losses_batch_all_classes.append(class_regression_loss_batch)
            #     # END for class_id in range(number_of_classes)

            #     if class_id == 0:
            #         # For debugging, we want to produce the massive debug matrix.
            #         # But we need to find the first class targets first, and then the second class, then generate.
            #         class_one_cls_targets = cls_targets
            #         class_one_reg_targets = reg_targets
            #         class_one_positive_indicators = positive_indicators
            #     else:
            #         # Since we look at class_id in order, by the time it reached this else it would have already
            #         # gone through class_id == 0 and be looking at class_id == 0.
            #         # By this time we have enough data required to produce the giant debug matrix

            #         class_two_positive_indicators = positive_indicators
            #         either_is_positive_indicators = torch.logical_or(class_one_positive_indicators, class_two_positive_indicators)

                    # torch.set_printoptions(sci_mode=False, edgeitems=100000000, linewidth=10000)
                    # for feature_map_index, _ in enumerate(feature_maps):
                    #     concatenated_output = torch.cat((
                    #         class_one_cls_targets[either_is_positive_indicators, 0].unsqueeze(dim=1),   # CLASS 1 CLASSIFICATION
                    #         cls_targets[either_is_positive_indicators, 1].unsqueeze(dim=1),             # CLASS 2 CLASSIFICATION
                    #         class_one_reg_targets[either_is_positive_indicators],                       # CLASS 1 REGRESSION
                    #         reg_targets[either_is_positive_indicators],                                 # CLASS 2 REGRESSION
                    #         levels_for_all_anchors[either_is_positive_indicators].unsqueeze(dim=1)      # PYRAMID LEVEL
                    #     ), dim=1)
                    #     print(f"concatenated_output {feature_map_index} {concatenated_output.shape}:\n{concatenated_output}")
                    # torch.set_printoptions(sci_mode=True, edgeitems=3)

            #downbeat_weight = 0.6

            # focal_losses_batch_all_classes[0] *= self.downbeat_weight
            # regression_losses_batch_all_classes[0] *= self.downbeat_weight

            # focal_losses_batch_all_classes[1] *= (1 - self.downbeat_weight)
            # regression_losses_batch_all_classes[1] *= (1 - self.downbeat_weight)

            # if self.fcos:
            #     leftness_losses_batch_all_classes[0] *= self.downbeat_weight
            #     leftness_losses_batch_all_classes[1] *= (1 - self.downbeat_weight)
            #     leftness_loss = torch.stack(leftness_losses_batch_all_classes).sum(dim=0)

            # focal_loss_class_mean = torch.stack(focal_losses_batch_all_classes).sum(dim=0)  #MJ: stack: https://sanghyu.tistory.com/85
            # #torch.stack(focal_losses_all_classes):  shape =(2,1)
            # regression_loss_class_mean = torch.stack(regression_losses_batch_all_classes).sum(dim=0)

            #MJ: if self.training:
            #    return classification_loss, regression_loss, leftness_loss, adjacency_constraint_loss
            if self.training:
                return classification_loss, regression_loss,  adjacency_constraint_loss
            

        # else:

        # This part is executed only during evaluation
        if not self.training: #MJ: evaluation mode, which is invoked after each epoch to evalute the performance of the object detection net.
            # Start of evaluation mode

            # transformed_anchors = self.regressBoxes(torch.cat(anchors_list, dim=0).unsqueeze(dim=0), regression_outputs)
            # transformed_anchors = self.clipBoxes(transformed_anchors, audio_batch)

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

            if self.fcos:
                for i, anchors_per_level in enumerate(anchors_list):    # i ranges over the level of feature maps.
                    # size_of_interest_per_level = anchor_points_per_level.new_tensor([sizes[i][0] * audio_target_rate, sizes[i][1] * audio_target_rate])
                    # size_of_interest_for_anchors_per_level = size_of_interest_per_level[None].expand(anchor_points_per_level.size(dim=0), -1)

                    stride_per_level = torch.tensor(2**(i + 1)).to(strides_for_all_anchors.device)
                    stride_for_anchors_per_level = stride_per_level[None].expand(anchors_per_level.size(dim=0))
                    # print(f"stride_per_level {stride_per_level.shape}:\n{stride_per_level}")
                    # print(f"stride_per_level[None] {stride_per_level[None].shape}:\n{stride_per_level[None]}")
                    # print(f"stride_for_anchors_per_level {stride_for_anchors_per_level.shape}:\n{stride_for_anchors_per_level}")
                    strides_for_all_anchors = torch.cat((strides_for_all_anchors, stride_for_anchors_per_level), dim=0)
                # print(f"strides_for_all_anchors {strides_for_all_anchors.shape}: {strides_for_all_anchors}")
                #END for i, anchors_per_level in enumerate(anchors_list)
                
                # anchors -> torch.cat(anchors_list, dim=0).unsqueeze(dim=0)
                # stride = 2**(i + 1)

                # transformed_anchors = torch.stack((
                #     anchors_list[i] - regression_output[0, :, 0] * stride,
                #     anchors_list[i] + regression_output[0, :, 1] * stride
                # ), dim=1).unsqueeze(dim=0)

                # anchor_point_transform function assumes that the regression_outputs have the batch dimension
                transformed_regression_boxes = self.anchor_point_transform(all_anchors, regression_outputs, strides_for_all_anchors)

                #scores = torch.squeeze(classification_output[:, :, class_id])
            else:
                transformed_regression_boxes = self.regressBoxes(torch.cat(anchors_list, dim=0).unsqueeze(dim=0), torch.cat(regression_outputs, dim=1))

            transformed_regression_boxes = self.clipBoxes(transformed_regression_boxes, audio_batch)

            for class_id in range(classification_outputs.shape[2]): # the shape of classification_output is (B, number of anchor points per level, class ID)
                if self.fcos:
                    scores = classification_outputs[:, :, class_id] * leftness_outputs[:, :, 0] # We predict the max number for beats will be less than the num of anchors
                else:
                    scores = classification_outputs[:, :, class_id]
                    
                #scores = scores / torch.max(scores)

                scores_over_thresh = (scores > score_threshold)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                # print(f"scores: {scores.shape}")
                
                scores = scores[scores_over_thresh]
                
                #anchorBoxes = torch.squeeze(transformed_regression_boxes)
                # print(f"transformed_regression_boxes: {transformed_regression_boxes.shape}")
                # print(f"scores_over_thresh: {scores_over_thresh.shape}")
                regression_boxes = transformed_regression_boxes[scores_over_thresh]
                # anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                # anchors_nms_idx is a tensor of anchor indices, sorted by score,
                # after removal of overlapping boxes with lower score

                # During NMS, if the IoU of two adjacent predicted boxes is less than IoU threshold, the two boxes are considered to be different beats
                # Otherwise both predictions are considered redundant so that one is removed.

                if self.postprocessing_type == 'gnet':
                    gnet = GNet(numBlocks=4)
                    gnet.cuda()
                    checkpoint = torch.load(model_urls['gnet'])
                    gnet.load_state_dict(checkpoint['model_state_dict'])
                    gnet.eval()

                    detections = torch.stack((  #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                        regression_boxes[:, 0],
                        torch.zeros(regression_boxes.size(dim=0)).to(regression_boxes.device),
                        regression_boxes[:, 1],
                        torch.ones(regression_boxes.size(dim=0)).to(regression_boxes.device),
                    ), dim=1) #MJ: detections refer to predicted bboxes by beat-fcos

                    data = [{
                        'scores': scores,
                        'detections': detections
                    }]
                    
                    logit_scores = gnet(batch=data, no_detections=9999999)  #MJ: scores for each detection/anchor point
                    logit_scores = logit_scores[0]
                    scores = torch.sigmoid(logit_scores)
                    
                    torch.set_printoptions(sci_mode=False)
                    #print(torch.cat((regression_boxes, gnet_result_scores.unsqueeze(dim=1)), dim=1))
                    torch.set_printoptions(sci_mode=True)

                    #scores[scores < 0.5] = 0  #MJ: discard detections whose confidence scores are low, say less than 0.5
                    # num_remaining_scores = torch.count_nonzero(scores)
                    # num_remaining_scores = torch.sum(scores > (scores.min() + scores.max()) / 4)
                    num_remaining_scores = torch.sum(scores > 0.05)

                    anchors_nms_idx = torch.argsort(scores, descending=True)[:num_remaining_scores]
                elif self.postprocessing_type == 'nms':
                    anchors_nms_idx = nms_2d(regression_boxes, scores, iou_threshold)
                    #anchors_nms_idx = torch.arange(0, regression_boxes.size(dim=0)) 
                    #MJ: regression_boxes are those obtained by filtering out whose scores are less than score_threshold
                    #    Get all the filtered detections and store them for use in training gnet.
                elif self.postprocessing_type == 'soft_nms':
                    anchors_nms_idx = soft_nms(regression_boxes, scores, sigma=0.5, thresh=0.2)
                elif self.postprocessing_type == 'none':
                    anchors_nms_idx = torch.arange(0, regression_boxes.size(dim=0))

                # print(f"torchvision indices:\n{anchors_nms_idx}")
                # print(f"torchvision boxes:\n{torch.cat((anchorBoxes[anchors_nms_idx], scores[anchors_nms_idx].unsqueeze(dim=1)), dim=1)}")

                #anchors_nms_idx = soft_nms(anchorBoxes, scores, sigma=0.2, use_regular_nms=True)
                #print(f"anchors_nms_idx:\n{anchorBoxes[anchors_nms_idx]}")
                #print(f"anchors_nms_idx2:\n{anchorBoxes[anchors_nms_idx2]}")

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

            #MJ:  eval_losses = (
            #     classification_loss.item(),
            #     regression_loss.item(),
            #     leftness_loss.item(),
            #     adjacency_constraint_loss.item()
            # )
            
            eval_losses = (
                classification_loss.item(),
                regression_loss.item(),
                #leftness_loss.item(),
                adjacency_constraint_loss.item()
            )

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, eval_losses]
#END def forward(self, inputs, iou_threshold=0.5, score_threshold=0.05)

def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(num_classes, args, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    #model objects computes the loss by executing its forward() method
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)

    if args.pretrained:
        model_key = 'wavebeat8'
        state_dict = torch.load(model_urls[model_key])['state_dict']
        new_dict = OrderedDict()

        for k, v in state_dict.items():
            key = k
            #key = k.replace('module.', '') # The parameter key that starts with "module." means that these parameters are from the parallelized model
            # For example, if the name of the parallelized module is "model_ddp" then the module_ddp.module refers to the original unwrapped model
            new_dict[key] = v

        missing_keys, unexpected_keys = model.dstcn.load_state_dict(new_dict, strict=False)
        print(f"Loaded {model_key} backbone. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        if args.freeze_bn:
            print("Freezing batch norm...")
            model.freeze_bn()

        if args.freeze_backbone:
            print("Freezing DSTCN...")
            model.dstcn.freeze()

    return model

    #missing_keys, unexpected_keys = model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
        
        # MJ: load_state_dict() returns ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                # * **missing_keys** is a list of str containing the missing keys
                # * **unexpected_keys** is a list of str containing the unexpected keys
    #return model


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
