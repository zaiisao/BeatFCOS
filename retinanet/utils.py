import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    #return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False)
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(planes * 4)
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                #self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
                self.mean = torch.from_numpy(np.array([0, 0]).astype(np.float32)).cuda()
            else:
                #self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))
                self.mean = torch.from_numpy(np.array([0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                #self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
                self.std = torch.from_numpy(np.array([0.1, 0.2]).astype(np.float32)).cuda()
            else:
                #self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
                self.std = torch.from_numpy(np.array([0.1, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        #widths  = boxes[:, :, 2] - boxes[:, :, 0]
        widths  = boxes[:, :, 1] - boxes[:, :, 0]
        #heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x   = boxes[:, :, 0] + 0.5 * widths
        #ctr_y   = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        #dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        #dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dw = deltas[:, :, 1] * self.std[1] + self.mean[1]
        #dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        #pred_ctr_y = ctr_y + dy * heights
        pred_w     = torch.exp(dw) * widths
        #pred_h     = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        #pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        #pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        #pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)
        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_x2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        #batch_size, num_channels, height, width = img.shape
        batch_size, num_channels, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        #boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        #boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], max=width)
        #boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

def nms_2d2(anchor_boxes, scores, thresh_iou):
    boxes_3d = torch.cat((
        torch.unsqueeze(anchor_boxes[:, 0], dim=1),
        torch.zeros((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device),
        torch.unsqueeze(anchor_boxes[:, 1], dim=1),
        torch.ones((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device)
    ), 1)

    return nms(boxes_3d, scores, thresh_iou)
def nms_2d(anchor_boxes, scores, thresh_iou):
    # we extract coordinates for every 
    # prediction box present in P
    x1 = anchor_boxes[:, 0]
    x2 = anchor_boxes[:, 1]
 
    # calculate area of every block in P
    areas = x2 - x1
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(anchor_boxes[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        xx2 = torch.min(xx2, x2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
         
        # take max with 0.0 to avoid negative w and h due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
 
        # find the intersection area
        inter = w
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order) 
 
        # find the union of every prediction T in P with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]

    return order#torch.cat(keep, dim=0)
