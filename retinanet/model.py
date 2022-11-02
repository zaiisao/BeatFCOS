from multiprocessing.sharedctypes import Value
import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
#from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes, nms_2d
from retinanet.anchors import Anchors
from retinanet import losses
from retinanet.dstcn import dsTCNModel

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    # feature_size is the number of channels in each feature map
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
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

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv1d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv1d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=2, feature_size=256, fcos=False):
        super(RegressionModel, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        #self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        self.regression = nn.Conv1d(feature_size, num_anchors * num_classes * 2, kernel_size=3, padding=1)
        self.centerness = nn.Conv1d(feature_size, 1, kernel_size=3, padding=1)

        self.fcos = fcos

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        regression = self.regression(out)

        # regression is B x C x L, with C = 2*num_anchors
        regression = regression.permute(0, 2, 1)
        regression = regression.contiguous().view(regression.shape[0], -1, 2, self.num_classes)

        if self.fcos:
            centerness = self.centerness(out)
            centerness = centerness.permute(0, 2, 1)
            centerness = centerness.contiguous().view(centerness.shape[0], -1, 1)

            return regression, centerness

        return regression

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=3, num_classes=2, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x L, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 1)

        #batch_size, width, height, channels = out1.shape
        batch_size, length, channels = out1.shape

        #out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)
        out2 = out1.view(batch_size, length, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, 1, self.num_classes)


class ResNet(nn.Module):
    def __init__(self, num_classes, block, layers, fcos=False, reg_loss_type="l1", **kwargs):
        #self.inplanes = 64
        self.inplanes = 256
        super(ResNet, self).__init__()

        self.fcos = fcos

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # downsampling tcn's output tensor dimension is (b, 256, 8192) (b, channel, width/length)

        # From WaveBeat model
        # With 8 layers, each with stride 2, we downsample the signal by a factor of 2^8 = 256,
        # which, given an input sample rate of 22.05 kHz produces an output signal with a
        # sample rate of 86 Hz
        self.dstcn = dsTCNModel(**kwargs) # conv1 shape: (b, 256, 8192)

        # if block == BasicBlock:
        #     fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
        #                  self.layer4[layers[3] - 1].conv2.out_channels]
        # elif block == Bottleneck:
        #     fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
        #                  self.layer4[layers[3] - 1].conv3.out_channels]
        # else:
        #     raise ValueError(f"Block type {block} not understood")

        # self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])
        self.fpn = PyramidFeatures(*(block.out_ch for block in self.dstcn.blocks[-3:]))

        num_anchors = 3
        if self.fcos:
            num_anchors = 1

        self.classificationModel = ClassificationModel(256, num_anchors=num_anchors, num_classes=num_classes)
        self.regressionModel = RegressionModel(256, num_anchors=num_anchors, num_classes=num_classes, fcos=self.fcos)

        self.anchors = Anchors(base_level=8, fcos=self.fcos)

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss(fcos=self.fcos)
        self.regressionLoss = losses.RegressionLoss(fcos=self.fcos, loss_type=reg_loss_type, num_anchors=num_anchors)
        self.centernessLoss = losses.CenternessLoss(fcos=self.fcos)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.regression.weight.data.fill_(0)
        self.regressionModel.regression.bias.data.fill_(0)

        self.regressionModel.centerness.weight.data.fill_(0)
        # self.regressionModel.centerness.bias.data.fill_(0)
        self.regressionModel.centerness.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
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

    def forward(self, inputs):
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
        tcn_layers = self.dstcn(audio_batch, 3)

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
        feature_maps = self.fpn(tcn_layers[-3:])

        if self.fcos:
            classification_outputs = [self.classificationModel(feature_map) for feature_map in feature_maps]
            regression_outputs = []
            centerness_outputs = []

            for feature_map in feature_maps:
                bbx_regression_output, centerness_regression_output = self.regressionModel(feature_map)
                regression_outputs.append(bbx_regression_output)
                centerness_outputs.append(centerness_regression_output)
        else:
            classification_outputs = torch.cat([self.classificationModel(feature_map) for feature_map in feature_maps], dim=1)
            regression_outputs = torch.cat([self.regressionModel(feature_map) for feature_map in feature_maps], dim=1)

        anchors_list = self.anchors(tcn_layers[-3])
        number_of_classes = classification_outputs.size(dim=3)

        #if self.training:
        if self.fcos:
            focal_losses, regression_losses, centerness_losses = [], [], []
            regress_distances = [(0, 64), (64, 128), (128, 256), (256, 512), (512, float("inf"))]

            for feature_index in range(len(feature_maps)):
                focal_losses.append(self.focalLoss(
                    classification_outputs[feature_index],
                    anchors_list[feature_index], # anchors_list[feature_index] refers to pixel locations of feature map at feature index
                                            # same as (x, y) in formula 2 of FCOS paper
                    annotations, # annotations has bounding box informations in the input image
                    regress_distances[feature_index]
                ))

                regression_losses.append(self.regressionLoss(
                    regression_outputs[feature_index],
                    anchors_list[feature_index],
                    annotations,
                    regress_distances[feature_index]
                ))

                centerness_losses.append(self.centernessLoss(
                    centerness_outputs[feature_index],
                    anchors_list[feature_index],
                    annotations,
                    regress_distances[feature_index]
                ))

            focal_loss = torch.stack(focal_losses).mean(dim=0, keepdim=True)
            regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
            centerness_loss = torch.stack(centerness_losses).mean(dim=0, keepdim=True)

            #return focal_loss, regression_loss, centerness_loss
        else:
            focal_losses, regression_losses = [], []

            for class_id in range(number_of_classes):
                # print(f"classification_outputs[:, :, :, class_id].shape: {classification_outputs[:, :, :, class_id].shape}")
                # print(f"regression_outputs[:, :, :, class_id].shape: {regression_outputs[:, :, :, class_id].shape}")
                # # for anchors in anchors_list:
                # #     print(f"anchors.shape: {anchors.shape}")
                # raise ValueError

                focal_losses.append(self.focalLoss(
                    classification_outputs[:, :, :, class_id],
                    anchors_list,#[anchors[class_id] for anchors in anchors_list],
                    annotations,
                    class_id=class_id
                ))

                regression_losses.append(self.regressionLoss(
                    regression_outputs[:, :, :, class_id],
                    anchors_list,#[anchors[class_id] for anchors in anchors_list],
                    annotations,
                    class_id=class_id
                ))
            # print("focal_losses", focal_losses)
            # print("regression_losses", regression_losses)

            focal_loss = torch.stack(focal_losses).mean(dim=0, keepdim=True)
            regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
            # focal_loss = self.focalLoss(classification_outputs, anchors_list, annotations)
            # regression_loss = self.regressionLoss(regression_outputs, anchors_list, annotations)

            #return focal_loss, regression_loss

        if self.training:
            if self.fcos:
                return focal_loss, regression_loss, centerness_loss
            else:
                return focal_loss, regression_loss
        else:
            # transformed_anchors = self.regressBoxes(torch.cat(anchors_list, dim=0).unsqueeze(dim=0), regression_outputs)
            # transformed_anchors = self.clipBoxes(transformed_anchors, audio_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification_outputs.shape[3]):
                transformed_anchors = self.regressBoxes(
                    torch.cat(anchors_list, dim=0).unsqueeze(dim=0),
                    regression_outputs[:, :, :, i]
                )

                transformed_anchors = self.clipBoxes(transformed_anchors, audio_batch)

                scores = torch.squeeze(classification_outputs[:, :, 0, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                #anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                anchors_nms_idx = nms_2d(anchorBoxes, scores, 0.1)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
                #break
                # torch.set_printoptions(edgeitems=1000000)
                # print(finalAnchorBoxesCoordinates)
                # torch.set_printoptions(edgeitems=3)

            if self.fcos:
                losses = (focal_loss, regression_loss, centerness_loss)
            else:
                losses = (focal_loss, regression_loss)

            #return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates, losses]



def resnet18(num_classes, **kwargs):
    """Constructs a ResNet-18 model."""
    return ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(num_classes, **kwargs):
    """Constructs a ResNet-34 model."""
    return ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(num_classes, **kwargs):
    """Constructs a ResNet-50 model."""
    return ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(num_classes, **kwargs):
    """Constructs a ResNet-101 model."""
    return ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(num_classes, **kwargs):
    """Constructs a ResNet-152 model."""
    return ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
