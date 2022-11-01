import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = b[:, 1] - b[:, 0]

    iw = torch.min(torch.unsqueeze(a[:, 1], dim=1), b[:, 1]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    iw = torch.clamp(iw, min=0)

    ua = torch.unsqueeze(a[:, 1] - a[:, 0], dim=1) + area - iw
    ua = torch.clamp(ua, min=1e-8)

    intersection = iw

    IoU = intersection / ua

    return IoU

def get_fcos_positives(bbox_annotation, anchor, lower_limit, upper_limit):
    # sort from shortest to longest
    bbox_annotation = bbox_annotation[(bbox_annotation[:, 1] - bbox_annotation[:, 0]).argsort()]

    points_in_lines = torch.logical_and(
        torch.ge(torch.unsqueeze(anchor, dim=0), torch.unsqueeze(bbox_annotation[:, 0], dim=1)),
        torch.le(torch.unsqueeze(anchor, dim=0), torch.unsqueeze(bbox_annotation[:, 1], dim=1))
    )

    left = torch.unsqueeze(anchor, dim=0) - torch.unsqueeze(bbox_annotation[:, 0], dim=1)
    right = torch.unsqueeze(bbox_annotation[:, 1], dim=1) - torch.unsqueeze(anchor, dim=0)

    points_within_range = torch.logical_and(
        torch.max(left, right) < upper_limit,
        torch.max(left, right) >= lower_limit
    )

    positive_indices, positive_argmax = torch.logical_and(points_in_lines, points_within_range).max(dim=0)

    assigned_annotations = bbox_annotation[positive_argmax, :]

    left = torch.diagonal(left[positive_argmax], 0)
    right = torch.diagonal(right[positive_argmax], 0)

    return positive_indices, assigned_annotations, left, right

def get_atss_positives(bbox_annotation, anchors_list):
    all_anchors = torch.cat(anchors_list, dim=1)
    num_gt = bbox_annotation.shape[0]

    num_anchors_per_loc = 3#len(self.cfg.MODEL.ATSS.ASPECT_RATIOS) * self.cfg.MODEL.ATSS.SCALES_PER_OCTAVE

    num_anchors_per_level = [anchors.size(dim=1) for anchors in anchors_list]#[len(anchors_per_level.bbox) for anchors_per_level in anchors[im_i]]

    ious = calc_iou(all_anchors[0, :, :], bbox_annotation[:, :2])#boxlist_iou(anchors_per_im, targets_per_im)

    #gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
    gt_cx = (bbox_annotation[:, 1] + bbox_annotation[:, 0]) / 2.0
    gt_cy = torch.ones(gt_cx.shape).to(gt_cx.device)#(bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
    gt_points = torch.stack((gt_cx, gt_cy), dim=1)

    #anchors_cx_per_im = (anchors_per_im.bbox[:, 2] + anchors_per_im.bbox[:, 0]) / 2.0
    anchors_cx_per_im = (all_anchors[0, :, 1] + all_anchors[0, :, 0]) / 2.0
    anchors_cy_per_im = torch.ones(anchors_cx_per_im.shape).to(anchors_cx_per_im.device)#(anchors_per_im.bbox[:, 3] + anchors_per_im.bbox[:, 1]) / 2.0
    anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)

    distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

    # Selecting candidates based on the center distance between anchor box and object
    candidate_idxs = []
    star_idx = 0
    for level, anchors_per_level in enumerate(anchors_list):
        end_idx = star_idx + num_anchors_per_level[level]
        distances_per_level = distances[star_idx:end_idx, :]
        topk = min(1 * num_anchors_per_loc, num_anchors_per_level[level])
        _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
        candidate_idxs.append(topk_idxs_per_level + star_idx)
        star_idx = end_idx
    candidate_idxs = torch.cat(candidate_idxs, dim=0)

    # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
    candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
    iou_mean_per_gt = candidate_ious.mean(0)
    iou_std_per_gt = candidate_ious.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
    is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

    # Limiting the final positive samples’ center to object
    anchor_num = anchors_cx_per_im.shape[0]
    for ng in range(num_gt):
        candidate_idxs[:, ng] += ng * anchor_num
    e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
    #e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
    candidate_idxs = candidate_idxs.view(-1)
    l = e_anchors_cx[candidate_idxs].view(-1, num_gt) - bbox_annotation[:, 0]
    #t = e_anchors_cy[candidate_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
    #r = bbox_annotation[:, 2] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
    r = bbox_annotation[:, 1] - e_anchors_cx[candidate_idxs].view(-1, num_gt)
    #b = bboxes_per_im[:, 3] - e_anchors_cy[candidate_idxs].view(-1, num_gt)
    #is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01
    is_in_gts = torch.stack([l, r], dim=1).min(dim=1)[0] > 0.01
    is_pos = is_pos & is_in_gts

    # if an anchor box is assigned to multiple gts, the one with the highest IoU will be selected.
    ious_inf = torch.full_like(ious, -100000000).t().contiguous().view(-1)
    index = candidate_idxs.view(-1)[is_pos.view(-1)]
    ious_inf[index] = ious.t().contiguous().view(-1)[index]
    ious_inf = ious_inf.view(num_gt, -1).t()

    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)
    # cls_labels_per_im = bbox_annotation[anchors_to_gt_indexs]
    # cls_labels_per_im[anchors_to_gt_values == -100000000] = 0

    positive_indices = anchors_to_gt_values != -100000000
    assigned_annotations = bbox_annotation[anchors_to_gt_indexs]

    return positive_indices, assigned_annotations

def run_atss_rough_implementation(bbox_annotation, anchors_list):
    all_anchors = torch.cat(anchors_list, dim=1)
    positive_indices = torch.zeros(all_anchors.size(dim=1), dtype=torch.bool)
    assigned_annotations = torch.zeros(all_anchors.size(dim=1), bbox_annotation.size(dim=1))

    iou_of_item_at_index = torch.zeros(positive_indices.shape) * -1
    for annotation_index in range(bbox_annotation.size(dim=0)):
        bbox_center = (bbox_annotation[annotation_index, 0] + bbox_annotation[annotation_index, 1])/2
        candidate_anchors = torch.zeros(0, 2)
        candidate_anchor_indices = []

        for level_i, level_anchors in enumerate(anchors_list):
            all_anchors_index = sum(anchors_list[i].size(dim=1) for i in range(level_i))
            anchors_center = (level_anchors[0, :, 0] + level_anchors[0, :, 1])/2
            distances = torch.abs(bbox_center - anchors_center)
            closest_anchor_distance, closest_anchor_index = torch.topk(distances, 3, largest=False)

            candidate_anchors = torch.cat((candidate_anchors, level_anchors[0, closest_anchor_index, :]), dim=0)
            candidate_anchor_indices += (closest_anchor_index + all_anchors_index).tolist()

        anchor_and_bbox_ious = calc_iou(candidate_anchors, bbox_annotation[annotation_index, :2].unsqueeze(dim=0))
        descending_ious, descending_iou_indices = torch.sort(anchor_and_bbox_ious, dim=0, descending=True)

        iou_means = torch.mean(anchor_and_bbox_ious)
        iou_stdevs = torch.std(anchor_and_bbox_ious)

        iou_threshold = iou_means + iou_stdevs

        for candidate_anchor_index, candidate_anchor in enumerate(candidate_anchors):
            candidate_and_bbox_iou = calc_iou(
                candidate_anchor.unsqueeze(dim=0),
                bbox_annotation[annotation_index, :2].unsqueeze(dim=0)
            )
            candidate_center = (candidate_anchor[0] + candidate_anchor[1])/2

            if (candidate_and_bbox_iou >= iou_threshold and
                candidate_center - bbox_annotation[annotation_index, 0] > 0.01 and
                bbox_annotation[annotation_index, 1] - candidate_center > 0.01 and
                candidate_and_bbox_iou > iou_of_item_at_index[candidate_anchor_indices[candidate_anchor_index]]
            ):
                positive_indices[candidate_anchor_indices[candidate_anchor_index]] = True
                assigned_annotations[candidate_anchor_indices[candidate_anchor_index]] = bbox_annotation[annotation_index]
                iou_of_item_at_index[candidate_anchor_indices[candidate_anchor_index]] = candidate_and_bbox_iou

    return positive_indices, assigned_annotations

class FocalLoss(nn.Module):
    def __init__(self, fcos=False):
        super(FocalLoss, self).__init__()
        self.fcos = fcos

    def forward(self, classifications, anchors_list, annotations, regress_limits=(0, float('inf'))):
        alpha = 0.25
        gamma = 2.0

        batch_size = classifications.shape[0]
        classification_losses = []

        if self.fcos:
            anchor = anchors_list[0, :, :]

            # anchors = (x, y) in feature map
            # [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2], []]
            assert torch.all(anchor[:, 0] == anchor[:, 1])
            anchor = anchor[:, 0]

        for j in range(batch_size):
            # j refers to an audio sample in batch
            jth_classification = classifications[j, :, :]

            # get box annotations from the original image
            # (5, 20, 0), (-1, -1, -1), 
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            jth_classification = torch.clamp(jth_classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(jth_classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = jth_classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - jth_classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                else:
                    alpha_factor = torch.ones(jth_classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = jth_classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - jth_classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())

                continue

            if self.fcos:
                targets = torch.zeros(jth_classification.shape)

                positive_indices, assigned_annotations, _, _ = get_fcos_positives(
                    bbox_annotation,
                    anchor,
                    regress_limits[0],
                    regress_limits[1]
                )
            else:
                # targets = torch.ones(jth_classification.shape) * -1

                # all_anchors = torch.cat(anchors_list, dim=1)
                # IoU = calc_iou(all_anchors[0, :, :], bbox_annotation[:, :2])
                # IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

                # targets[torch.lt(IoU_max, 0.4), :] = 0
                # positive_indices = torch.ge(IoU_max, 0.5)

                # assigned_annotations = bbox_annotation[IoU_argmax, :]

                targets = torch.zeros(jth_classification.shape)
                positive_indices, assigned_annotations = get_atss_positives(bbox_annotation, anchors_list)
                # print("num of positive indices", positive_indices.sum())
                # print("num of negative indices", (~positive_indices).sum())
                torch.set_printoptions(edgeitems=10000000)
                #print(f"ORIGINAL: {positive_indices}, {assigned_annotations}")
                positive_indices2, assigned_annotations2 = run_atss_rough_implementation(bbox_annotation, anchors_list)
                print((positive_indices != positive_indices2).nonzero())
                #print(f"REIMPLEMENTATION: {positive_indices2}, {assigned_annotations2}")
                torch.set_printoptions(edgeitems=3)

            if torch.cuda.is_available():
                targets = targets.cuda()

            num_positive_anchors = positive_indices.sum()

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - jth_classification, jth_classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(jth_classification) + (1.0 - targets) * torch.log(1.0 - jth_classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            #print(cls_loss[positive_indices].sum(), cls_loss[~positive_indices].sum())

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        if self.fcos:
            return torch.stack(classification_losses).sum(dim=0)
        else:
            return torch.stack(classification_losses).mean(dim=0, keepdim=True)

class RegressionLoss(nn.Module):
    def __init__(self, fcos=False, loss_type="f1", weight=10, num_anchors=3):
        super(RegressionLoss, self).__init__()
        self.fcos = fcos
        self.loss_type = loss_type
        self.weight = weight
        self.num_anchors = num_anchors

    def forward(self, regressions, anchors_list, annotations, regress_limits=(0, float('inf'))):
        # regressions is (B, C, W, H), with C = 4*num_anchors = 4*9
        # in our case, regressions is (B, C, W), with C = 2*num_anchors = 2*1
        batch_size = regressions.shape[0] 
        regression_losses = []

        # anchors_list = [[ [5, 5, 10, 10], [5, 5, 15, 15], ... ]]
        # in our case, anchors_list = [[ [5, 5], [10, 10], [15, 15] ... ]]
        all_anchors = torch.cat(anchors_list, dim=1)
        anchor_widths  = all_anchors[0, :, 1] - all_anchors[0, :, 0] # if fcos is true, anchor_widths = 0
        anchor_ctr_x   = all_anchors[0, :, 0] + 0.5 * anchor_widths # if fcos is true, anchor_ctr_x = anchor[:, 0]

        if self.fcos:
            assert torch.all(anchor[:, 0] == anchor[:, 1])
            anchor = anchor[:, 0] # [5, 10, 15, ...]

        for j in range(batch_size):
            jth_regression = regressions[j, :, :] # j'th audio in the current batch

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1] # bbox_annotation[:, 2] is the classification label

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

                continue

            if self.fcos:
                positive_indices, assigned_annotations, left, right = get_fcos_positives(
                    bbox_annotation,
                    anchor,
                    regress_limits[0],
                    regress_limits[1]
                )
            else:
                # IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :2]) # num_anchors x num_annotations
                # IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                # positive_indices = torch.ge(IoU_max, 0.5)
                # assigned_annotations = bbox_annotation[IoU_argmax, :]
                positive_indices, assigned_annotations = get_atss_positives(bbox_annotation, anchors_list)

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]

                gt_widths  = assigned_annotations[:, 1] - assigned_annotations[:, 0]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                # print("gt", gt_widths)
                # print("anchor", anchor_widths_pi)

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)

                if self.loss_type == "f1":
                    targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                    targets_dw = torch.log(gt_widths / anchor_widths_pi)

                    targets = torch.stack((targets_dx, targets_dw))
                    targets = targets.t()

                    if torch.cuda.is_available():
                        targets = targets/torch.Tensor([[0.1, 0.2]]).cuda()
                    else:
                        targets = targets/torch.Tensor([[0.1, 0.2]])

                    negative_indices = 1 + (~positive_indices)

                    regression_diff = torch.abs(targets - jth_regression[positive_indices, :])

                    # regression_loss = torch.where(
                    #     torch.le(regression_diff, 1.0 / 9.0),
                    #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                    #     regression_diff - 0.5 / 9.0
                    # )
                    regression_loss = torch.where(
                        torch.le(regression_diff, 1.0 / self.num_anchors),
                        0.5 * self.num_anchors * torch.pow(regression_diff, 2),
                        regression_diff - 0.5 / self.num_anchors
                    )
                    # print("regression", jth_regression[positive_indices, :])
                    # print("targets", targets)
                    # print("loss", regression_loss)

                    regression_losses.append(regression_loss.mean())
                elif self.loss_type == "iou" or self.loss_type == "giou":
                    #num_positive_anchors = positive_indices.sum()

                    # 1. For the predicted line B_p, ensuring  x_p_2 > x_p_1
                    # bbox_prediction, _ = torch.sort(jth_regression[positive_indices, :])
                    # bbox_ground = torch.stack((left, right), dim=1)[positive_indices, :].float()#assigned_annotations[:, :2]

                    # # 2. Calculating length of B_g: L_g = x_g_2 − x_g_1 (위에서 이미 정의한 gt_widths)
                    # bbox_ground_lengths = bbox_ground[:, 1] - bbox_ground[:, 0] #gt_widths

                    # # 3. Calculating length of B_p: L_p = x̂_p_2 − x̂_p_1
                    # bbox_prediction_lengths = bbox_prediction[:, 1] - bbox_prediction[:, 0]

                    # # 4. Calculating intersection I between B_p and B_g
                    # intersection_x1 = torch.max(bbox_ground[:, 0], bbox_prediction[:, 0])
                    # intersection_x2 = torch.min(bbox_ground[:, 1], bbox_prediction[:, 1])
                    # intersection = torch.where(
                    #     intersection_x2 > intersection_x1,
                    #     intersection_x2 - intersection_x1,
                    #     torch.zeros(bbox_ground.size(dim=0))
                    # )

                    # # 5. Finding the coordinate of smallest enclosing line B_c:
                    # coordinate_x1 = torch.min(bbox_ground[:, 0], bbox_prediction[:, 0])
                    # coordinate_x2 = torch.max(bbox_ground[:, 1], bbox_prediction[:, 1])

                    # # 6. Calculating length of B_c
                    # bbox_coordinate = coordinate_x2 - coordinate_x1 + 1e-7

                    # # 7. IoU (I / U), where U = L_p + L_g - I
                    # union = bbox_prediction_lengths + bbox_ground_lengths - intersection
                    # iou = intersection / union

                    # if self.loss_type == "iou":
                    #     # 9a. L_IoU = 1 - IoU
                    #     regression_loss = 1 - iou
                    # else:
                    #     # 8. GIoU = IoU - (L_c - U)/L_c
                    #     giou = iou - (bbox_coordinate - union)/bbox_coordinate
                    #     print(bbox_prediction, bbox_ground, giou)

                    #     # 9b. L_GIoU = 1 - GIoU
                    #     regression_loss = 1 - giou

                    # #print(regression_loss.mean(), torch.exp(regression_loss.mean() * self.weight))
                    # regression_losses.append(regression_loss.mean() * self.weight)
                    target_left = left[positive_indices]
                    target_right = right[positive_indices]

                    pred_left = jth_regression[positive_indices, 0]
                    pred_right = jth_regression[positive_indices, 1]

                    target_area = (target_left + target_right)
                    pred_area = (pred_left + pred_right)

                    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
                    g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
                    ac_uion = g_w_intersect + 1e-7
                    area_intersect = w_intersect
                    area_union = target_area + pred_area - area_intersect
                    ious = (area_intersect + 1.0) / (area_union + 1.0)
                    gious = ious - (ac_uion - area_union) / ac_uion
                    if self.loss_type == 'iou':
                        losses = -torch.log(ious)
                    elif self.loss_type == 'linear_iou':
                        losses = 1 - ious
                    elif self.loss_type == 'giou':
                        losses = 1 - gious
                    else:
                        raise NotImplementedError
                    #print(torch.stack((target_left, target_right), dim=1), torch.stack((pred_left, pred_right), dim=1), losses)

                    # if weight is not None and weight.sum() > 0:
                    #     return (losses * weight).sum()
                    # else:
                    #     assert losses.numel() != 0
                    #     return losses.sum()

                    regression_losses.append(losses.sum() * self.weight)

            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        if self.fcos:
            return torch.stack(regression_losses).sum(dim=0)
        else:
            return torch.stack(regression_losses).mean(dim=0, keepdim=True)

class CenternessLoss(nn.Module):
    def __init__(self, fcos=False):
        super(CenternessLoss, self).__init__()
        self.fcos = fcos

    def forward(self, centernesses, anchors, annotations, regress_limits=(0, float('inf'))):
        if not self.fcos:
            raise NotImplementedError

        batch_size = centernesses.shape[0]
        centerness_losses = []

        anchor = anchors[0, :, :]

        assert torch.all(anchor[:, 0] == anchor[:, 1])
        anchor = anchor[:, 0]

        for j in range(batch_size):
            jth_centerness = centernesses[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            #jth_centerness = torch.clamp(jth_centerness, 1e-4, 1.0 - 1e-4)
            jth_centerness = torch.sigmoid(jth_centerness)

            positive_indices, assigned_annotations, left, right = get_fcos_positives(
                bbox_annotation,
                anchor,
                regress_limits[0],
                regress_limits[1]
            )

            num_positive_anchors = positive_indices.sum()

            targets = torch.where(
                positive_indices,
                torch.sqrt(torch.min(left, right) / torch.max(left, right)).float(),
                torch.zeros(positive_indices.shape)
            ).unsqueeze(dim=1)

            bce = -(targets * torch.log(jth_centerness) + (1.0 - targets) * torch.log(1.0 - jth_centerness))

            if torch.cuda.is_available():
                ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape).cuda())
            else:
                ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape))

            #print(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            centerness_losses.append(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        if self.fcos:
            return torch.stack(centerness_losses).sum(dim=0)
        else:
            return torch.stack(centerness_losses).mean(dim=0, keepdim=True)
