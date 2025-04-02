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

def calc_iou(a, b):
    area = b[:, 1] - b[:, 0]

    iw = torch.min(torch.unsqueeze(a[:, 1], dim=1), b[:, 1]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = torch.clamp(iw, min=0)

    ua = torch.unsqueeze(a[:, 1] - a[:, 0], dim=1) + area - iw
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw

    IoU = intersection / ua

    return IoU

def calc_giou(a, b):
    # 1. For the predicted line B_p, ensuring  x_p_2 > x_p_1

    # 2. Calculating length of B_g: L_g = x_g_2 − x_g_1 (위에서 이미 정의한 gt_widths)
    a_lengths = a[:, 1] - a[:, 0] #gt_widths

    # 3. Calculating length of B_p: L_p = x̂_p_2 − x̂_p_1
    b_lengths = b[:, 1] - b[:, 0]

    # 4. Calculating intersection I between B_p and B_g
    intersection_x1 = torch.max(a[:, 0], b[:, 0])
    intersection_x2 = torch.min(a[:, 1], b[:, 1])
    intersection = torch.where(
        intersection_x2 > intersection_x1,
        intersection_x2 - intersection_x1,
        torch.zeros(a.size(dim=0)).to(a.device)
    )

    # 5. Finding the coordinate of smallest enclosing line B_c:
    coordinate_x1 = torch.min(a[:, 0], b[:, 0])
    coordinate_x2 = torch.max(a[:, 1], b[:, 1])

    # 6. Calculating length of B_c
    bbox_coordinate = coordinate_x2 - coordinate_x1 + 1e-7

    # 7. IoU (I / U), where U = L_p + L_g - I
    union = b_lengths + a_lengths - intersection
    iou = intersection / union

    #if self.loss_type == "iou":
        # 9a. L_IoU = 1 - IoU
    #    regression_loss = 1 - iou
    #else:
        # 8. GIoU = IoU - (L_c - U)/L_c
        #giou = iou - (bbox_coordinate - union)/bbox_coordinate
    giou = iou - (bbox_coordinate - union)/bbox_coordinate
    return giou
        #print(b, a, giou)

        # 9b. L_GIoU = 1 - GIoU
        #regression_loss = 1 - giou

    #print(regression_loss.mean(), torch.exp(regression_loss.mean() * self.weight))

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
    expansion = 4  #Class field?

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

class AnchorPointTransform(nn.Module):
    def __init__(self):
        super(AnchorPointTransform, self).__init__()

    def forward(self, all_anchors, regression_outputs, strides_for_all_anchors):
        # regression_outputs[:, :, 0] shape is (B, number of anchors)
        # regression_outputs[:, :, 1] shape is (B, number of anchors)
        # all_anchors shape is (number of anchors,) -> (None, number of anchors)
        # strides_for_all_anchors shape is (number of anchors,) -> (None, number of anchors)

        # print(f"all_anchors[None] shape: {all_anchors[None].shape}")
        # print(f"regression_outputs shape: {regression_outputs.shape}")
        # print(f"regression_outputs[:, :, 0] shape: {regression_outputs[:, :, 0].shape}")
        # print(f"strides_for_all_anchors[None] shape: {strides_for_all_anchors[None].shape}")

        transformed_regressions_x1 = all_anchors[None] - regression_outputs[:, :, 0] * strides_for_all_anchors[None]
        transformed_regressions_x2 = all_anchors[None] + regression_outputs[:, :, 1] * strides_for_all_anchors[None]

        transformed_regression_boxes = torch.stack((
            transformed_regressions_x1, # (B, num of anchors, 1)
            transformed_regressions_x2  # (B, num of anchors, 1)
        ), dim=2)

        return transformed_regression_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        if len(img.shape) == 4:
            batch_size, num_channels, width, _ = img.shape
        elif len(img.shape) == 3:
            batch_size, num_channels, width = img.shape
        else:
            raise NotImplementedError

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        #boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        #boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], max=width)
        #boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)
      
        return boxes

def nms_2d(anchor_boxes, scores, thresh_iou):
    boxes_3d = torch.cat((
        torch.unsqueeze(anchor_boxes[:, 0], dim=1),
        torch.zeros((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device),
        torch.unsqueeze(anchor_boxes[:, 1], dim=1),
        torch.ones((anchor_boxes.size(dim=0), 1)).to(anchor_boxes.device)
    ), 1)

    return nms(boxes_3d, scores, thresh_iou)

def soft_nms(regression_boxes, box_scores, sigma=0.5, thresh=0.05, use_regular_nms=False):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = regression_boxes.shape[0]  #MJ: N = 10022
    indexes = torch.arange(0, N, dtype=torch.float).to(regression_boxes.device).view(N, 1)
    dets = torch.cat((regression_boxes, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    #y1 = dets[:, 0]
    #x1 = dets[:, 1]
    #y2 = dets[:, 2]
    #x2 = dets[:, 3]
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = box_scores.clone()
    #areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    areas = (x2 - x1 + 1) #* (y2 - y1 + 1)

    scores_9     = torch.tensor([], device=x1.device)
    scores_8     = torch.tensor([], device=x1.device)
    scores_7     = torch.tensor([], device=x1.device)
    scores_6     = torch.tensor([], device=x1.device)
    scores_5     = torch.tensor([], device=x1.device)
    scores_4     = torch.tensor([], device=x1.device)
    scores_3     = torch.tensor([], device=x1.device)
    scores_2     = torch.tensor([], device=x1.device)
    scores_1     = torch.tensor([], device=x1.device)
    scores_00001 = torch.tensor([], device=x1.device)
    scores_less  = torch.tensor([], device=x1.device)
    scores_0     = torch.tensor([], device=x1.device)

    ious_9       = torch.tensor([], device=x1.device)
    ious_8       = torch.tensor([], device=x1.device)
    ious_7       = torch.tensor([], device=x1.device)
    ious_6       = torch.tensor([], device=x1.device)
    ious_5       = torch.tensor([], device=x1.device)
    ious_4       = torch.tensor([], device=x1.device)
    ious_3       = torch.tensor([], device=x1.device)
    ious_2       = torch.tensor([], device=x1.device)
    ious_1       = torch.tensor([], device=x1.device)
    ious_00001   = torch.tensor([], device=x1.device)
    ious_less    = torch.tensor([], device=x1.device)
    ious_0       = torch.tensor([], device=x1.device)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone() 
        pos = i + 1

        #MJ: compare the current box’s score (tscore) with the maximum score (maxscore) among the boxes that come after the current box.
        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0) #MJ: get the max box in the following boxes after the current ith box
            # Even though the tensor is one-dimensional in this context, specifying dim=0 is necessary when you want to obtain both the maximum value and its index. Without dim=0, you might only get the maximum value, 
            if tscore < maxscore:
                dets[i], dets[ maxpos.item() + i + 1]  = dets[ maxpos.item() + i + 1].clone(), dets[i].clone() # JA: dets[i] is the i'th box with its left position, right position, and index
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()
            #MJ: => This swaps the max box with the current box at the ith index;
            #As the algorithm iterates through the boxes, the one with the highest score among the remaining boxes is always moved to the current position, 
            #It  is a typical strategy in non-maximum suppression methods to prioritize the most confident detections.
         
        #MJ: now dets[i] plays the same role as M in the pseudo code of the soft-nms paper     
        # IoU calculate
        #yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        #xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        #yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        #xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        #xx1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        #xx2 = np.minimum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        xx1 = torch.maximum(dets[i, 0], dets[pos:, 0]) # JA: xx1 compares the i'th box's left with all following boxes' lefts
        xx2 = torch.minimum(dets[i, 1], dets[pos:, 1]) # JA: xx2 compares the i'th box's right with all following boxes' rights
        
        #w = np.maximum(0.0, xx2 - xx1 + 1)
        w = xx2 - xx1 + 1
        w = torch.clamp(w, min=0) # JA: w is the overlapping interval

        #h = np.maximum(0.0, yy2 - yy1 + 1)
        #inter = torch.tensor(w * h).to(dets.device)
        inter = torch.tensor(w).to(dets.device)
        ious = torch.div(inter, (areas[i] + areas[pos:] - inter))

        #MJ: ious= ious(M,b_i's) in the pseudo code of the paper
        if use_regular_nms == True:  #MJ: sigma=0.5 is used as N_t in the pseudo code of the paper
            weights = torch.where(ious > sigma, torch.zeros(ious.shape).to(ious.device), torch.ones(ious.shape).to(ious.device))
        else: #MJ: sigma=0.5 
            # Gaussian decay: 
            weights = torch.exp(-(ious * ious) / sigma)

        #MJ: For every ith box, which is now the max score box, M, rescore all the remaining boxes b_i, using ious(M,b_i):
        scores[pos:] = weights * scores[pos:]

        #### JA: Following lines are added for debugging purpose => delete the debugging code from here
        condition_9 = (ious > 0.9)
        if condition_9.any():
            scores_9 = torch.cat((scores_9, scores[pos:][condition_9]))
            ious_9   = torch.cat((ious_9,   ious[condition_9]))

        condition_8 = torch.logical_and(ious > 0.8, ious <= 0.9)
        if condition_8.any():
            scores_8 = torch.cat((scores_8, scores[pos:][condition_8]))
            ious_8   = torch.cat((ious_8,   ious[condition_8]))

        condition_7 = torch.logical_and(ious > 0.7, ious <= 0.8)
        if condition_7.any():
            scores_7 = torch.cat((scores_7, scores[pos:][condition_7]))
            ious_7   = torch.cat((ious_7,   ious[condition_7]))

        condition_6 = torch.logical_and(ious > 0.6, ious <= 0.7)
        if condition_6.any():
            scores_6 = torch.cat((scores_6, scores[pos:][condition_6]))
            ious_6   = torch.cat((ious_6,   ious[condition_6]))

        condition_5 = torch.logical_and(ious > 0.5, ious <= 0.6)
        if condition_5.any():
            scores_5 = torch.cat((scores_5, scores[pos:][condition_5]))
            ious_5   = torch.cat((ious_5,   ious[condition_5]))

        condition_4 = torch.logical_and(ious > 0.4, ious <= 0.5)
        if condition_4.any():
            scores_4 = torch.cat((scores_4, scores[pos:][condition_4]))
            ious_4   = torch.cat((ious_4,   ious[condition_4]))

        condition_3 = torch.logical_and(ious > 0.3, ious <= 0.4)
        if condition_3.any():
            scores_3 = torch.cat((scores_3, scores[pos:][condition_3]))
            ious_3   = torch.cat((ious_3,   ious[condition_3]))

        condition_2 = torch.logical_and(ious > 0.2, ious <= 0.3)
        if condition_2.any():
            scores_2 = torch.cat((scores_2, scores[pos:][condition_2]))
            ious_2   = torch.cat((ious_2,   ious[condition_2]))

        condition_1 = torch.logical_and(ious > 0.1, ious <= 0.2)
        if condition_1.any():
            scores_1 = torch.cat((scores_1, scores[pos:][condition_1]))
            ious_1   = torch.cat((ious_1,   ious[condition_1]))

        condition_00001 = torch.logical_and(ious > 0.00001, ious <= 0.1)
        if condition_00001.any():
            scores_00001 = torch.cat((scores_00001, scores[pos:][condition_00001]))
            ious_00001   = torch.cat((ious_00001,   ious[condition_00001]))

        condition_less = torch.logical_and(ious > 0, ious <= 0.00001)
        if condition_less.any():
            scores_less = torch.cat((scores_less, scores[pos:][condition_less]))
            ious_less   = torch.cat((ious_less,   ious[condition_less]))

        condition_0 = (ious == 0)
        if condition_0.any():
            scores_0 = torch.cat((scores_0, scores[pos:][condition_0]))
            ious_0   = torch.cat((ious_0,   ious[condition_0]))
        #### JA: Above lines are added for debugging purpose
    #End for i in range(N):
    
    #MJ: Add the debugging code here
    
    # select the boxes and keep the corresponding indexes
    #keep = dets[:, 4][scores > thresh].int()
    keep = dets[:, 2][scores > thresh].long() #MJ: sigma=0.5, thresh=0.05

    return keep

# def get_offset_limits(datasets):
    

# https://github.com/bharatsingh430/soft-nms/blob/b8e69bdf8df2ad53025c9d198ded909b50471d4f/lib/nms/cpu_nms.pyx
# def soft_nms(boxes, sigma=0.5, iou_threshold=0.3, score_threshold=0.001, method=0):
#     N = boxes.shape[0]
#     iw, ih, box_area = None, None, None
#     ua = None
#     pos = 0
#     maxscore = 0
#     maxpos = 0
#     #cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov
#     x1,x2,tx1,tx2,ts,area,weight,ov = None, None, None, None, None, None, None, None
#     keep_indices = []

#     for i in range(N):
#         #maxscore = boxes[i, 4]
#         maxscore = boxes[i, 2]
#         maxpos = i

#         # save box i to tx1, tx2, and ts
#         # tx1 = boxes[i,0]
#         # ty1 = boxes[i,1]
#         # tx2 = boxes[i,2]
#         # ty2 = boxes[i,3]
#         # ts = boxes[i,4]
#         tx1 = boxes[i,0]
#         tx2 = boxes[i,1]
#         ts = boxes[i,2]

#         pos = i + 1

#         # m <- argmax S
#         # get max box: get max score and max pos
#         while pos < N:
#             # if maxscore < boxes[pos, 4]:
#             #     maxscore = boxes[pos, 4]
#             if maxscore < boxes[pos, 2]:
#                 maxscore = boxes[pos, 2]
#                 maxpos = pos
#             pos = pos + 1

#         keep_indices.append(maxpos)

#         # B <- B - M
#         # add max box as a detection
#         # boxes[i,0] = boxes[maxpos,0]
#         # boxes[i,1] = boxes[maxpos,1]
#         # boxes[i,2] = boxes[maxpos,2]
#         # boxes[i,3] = boxes[maxpos,3]
#         # boxes[i,4] = boxes[maxpos,4]
#         boxes[i,0] = boxes[maxpos,0]
#         boxes[i,1] = boxes[maxpos,1]
#         boxes[i,2] = boxes[maxpos,2]

#         # swap ith box and max box
#         # swap ith box with position of max box
#         # boxes[maxpos,0] = tx1
#         # boxes[maxpos,1] = ty1
#         # boxes[maxpos,2] = tx2
#         # boxes[maxpos,3] = ty2
#         # boxes[maxpos,4] = ts
#         boxes[maxpos,0] = tx1
#         boxes[maxpos,1] = tx2
#         boxes[maxpos,2] = ts

#         # boxes[i, :] = boxes[maxpos, :]
#         # tx1 = boxes[i,0]
#         # ty1 = boxes[i,1]
#         # tx2 = boxes[i,2]
#         # ty2 = boxes[i,3]
#         # ts = boxes[i,4]
#         tx1 = boxes[i,0]
#         tx2 = boxes[i,1]
#         ts = boxes[i,2]

#         pos = i + 1
#     # NMS iterations, note that N decreases by 1 if detection boxes fall below threshold
#         while pos < N:
#             # x1 = boxes[pos, 0]
#             # y1 = boxes[pos, 1]
#             # x2 = boxes[pos, 2]
#             # y2 = boxes[pos, 3]
#             # s = boxes[pos, 4]
#             x1 = boxes[pos, 0]
#             x2 = boxes[pos, 1]
#             s = boxes[pos, 2]

#             # area = (x2 - x1 + 1) * (y2 - y1 + 1)
#             # iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             area = x2 - x1 + 1
#             iw = (min(tx2, x2) - max(tx1, x1) + 1)
#             if iw > 0:
#                 # ih = (min(ty2, y2) - max(ty1, y1) + 1)
#                 # if ih > 0:
#                 # ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
#                 ua = float((tx2 - tx1 + 1) + area - iw)
#                 # ov = iw * ih / ua # iou between max box and detection box
#                 ov = iw / ua # iou between max box and detection box

#                 if method == 1: # linear
#                     if ov > iou_threshold: 
#                         weight = 1 - ov
#                     else:
#                         weight = 1
#                 elif method == 2: # gaussian
#                     weight = np.exp(-(ov * ov)/sigma)
#                 else: # original NMS
#                     if ov > iou_threshold: 
#                         weight = 0
#                     else:
#                         weight = 1

#                 # boxes[pos, 4] = weight*boxes[pos, 4]
#                 boxes[pos, 2] = weight*boxes[pos, 2]
            
#             # if box score falls below threshold, discard the box by swapping with last box
#             # update N
#                     # if boxes[pos, 4] < threshold:
#                     #     boxes[pos,0] = boxes[N-1, 0]
#                     #     boxes[pos,1] = boxes[N-1, 1]
#                     #     boxes[pos,2] = boxes[N-1, 2]
#                     #     boxes[pos,3] = boxes[N-1, 3]
#                     #     boxes[pos,4] = boxes[N-1, 4]
#                 if boxes[pos, 2] < score_threshold:
#                     boxes[pos,0] = boxes[N-1, 0]
#                     boxes[pos,1] = boxes[N-1, 1]
#                     boxes[pos,2] = boxes[N-1, 2]
#                     N = N - 1
#                     pos = pos - 1

#             pos = pos + 1

#     #print(f"sorted boxes:\n{torch.sort(boxes, dim=2, descending=True)}")
#     #_, sorted_box_indices = torch.sort(boxes[:, 2], descending=True)
#     #print(f"sorted_box_indices: {sorted_box_indices}")
#     #print(f"number of boxes above threshold: {(boxes[:, 2] > score_threshold).sum()}")
#     #positive_indices = torch.ge(boxes[:, 2], score_threshold)
#     #num_positive_anchors = positive_indices.sum()
#     #print(f"number of boxes above score threshold: {num_positive_anchors}")
#     #print(f"soft nms boxes ({boxes[sorted_box_indices].shape}): {boxes[sorted_box_indices]}")

#     print(f"custom nms indices:\n{keep_indices}")
#     keep = [i for i in range(N)]
#     print(f"custom nms boxes:\n{boxes[keep]}")
#     #print(f"custom nms ({boxes[torch.argsort(boxes[keep, 0])].shape}):\n{boxes[torch.argsort(boxes[keep, 0]), 0:2]}")
#     return keep

def soft_nms_from_pseudocode(boxes, scores, threshold):
    D = torch.zeros(0, 2)
    while boxes.size(dim=0) > 0:
        m = torch.argmax(scores)
        M = boxes[m, :].unsqueeze(dim=0)
        D = torch.cat((D, M), dim=0)
        boxes = torch.cat((boxes[0:m, :], boxes[m:, :]), dim=0)
        for i in range(boxes.size(dim=0)):
            b_i = boxes[i, :]
            if calc_iou(M, b_i.unsqueeze(dim=0)) >= threshold:
                boxes = torch.cat((boxes[0:i, :], boxes[i:, :]), dim=0)
                scores = torch.cat((scores[0:i], scores[i:]), dim=0)

    return D, S

