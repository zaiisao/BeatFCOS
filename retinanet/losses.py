import numpy as np
import torch
import torch.nn as nn
from retinanet.utils import BBoxTransform, calc_iou, calc_giou

def get_fcos_positives(jth_annotations, anchors_list, class_id):
    class_bbox_annotation = jth_annotations[jth_annotations[:, 2] == class_id]

    audio_target_rate = 22050 / 256
    distance_ranges = [(x[0] * audio_target_rate, x[1] * audio_target_rate) for x in [
        (0, 0.32537674),
        (0.32537674, 0.47555801),
        (0.47555801, 0.64588683),
        (0.64588683, 1.16883525),
        (1.16883525, 2.17128976),
        (2.17128976, float("inf"))
    ]]

    # bbox_sizes = [x * 22050 / 256 for x in [0.32537674, 0.47555801, 0.64588683, 1.16883525, 2.17128976, ]]

    # sort from shortest to longest
    sorted_bbox_indices = (class_bbox_annotation[:, 1] - class_bbox_annotation[:, 0]).argsort()
    class_bbox_annotation = class_bbox_annotation[sorted_bbox_indices]

    positive_anchor_indices = torch.zeros(0, dtype=torch.bool)
    normalized_annotations_for_anchors = torch.zeros(0, 3)
    l_star_for_all_anchors = torch.zeros(0)
    r_star_for_all_anchors = torch.zeros(0)

    # anchors_list contains the anchor points (x, y) on the base level image corresponding to the feature map
    for i, anchor_points_per_level in enumerate(anchors_list): # anchor points per level, (anchor locations (x, y) on the base level image)
        # the shape of anchor_points_per_level is (num of anchors,)
        # the shape of torch.unsqueeze(anchor_points_per_level, dim=0) is (1, num of anchors)
        # the shape of class_bbox_annotation[:, 0] is (num of annotations,)
        # the shape of torch.unsqueeze(class_bbox_annotation[:, 0], dim=1) is (num of annotations, 1)
        # the shape of anchor_points_in_gt_bboxes is (num of annotations, num_of_anchors)

        # check whether anchor points are within gt boxes
        # anchor_points_in_gt_bboxes is a (N, M) boolean matric where N is the number of anchors and M is the number of annotations
        anchor_points_in_gt_bboxes = torch.logical_and(
            torch.ge(torch.unsqueeze(anchor_points_per_level, dim=0), torch.unsqueeze(class_bbox_annotation[:, 0], dim=1)),
            torch.le(torch.unsqueeze(anchor_points_per_level, dim=0), torch.unsqueeze(class_bbox_annotation[:, 1], dim=1))
        )

        # apply the distance limits criteria
        # lefts is l*, rights is r* which are all the anchor points on the base image
        l_star_per_level = torch.unsqueeze(anchor_points_per_level, dim=0) - torch.unsqueeze(class_bbox_annotation[:, 0], dim=1)
        r_star_per_level = torch.unsqueeze(class_bbox_annotation[:, 1], dim=1) - torch.unsqueeze(anchor_points_per_level, dim=0)

        # (floor(s/2) + xs, floor(s/2) + ys)
        # strides = [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4]
        # s = 2**i

        # xs = anchor_points_per_level
        # xs_in_input_image = xs // 2 + s*xs
        # center_xs =

        # when i is 0,
        points_within_range_per_level = torch.logical_and(
            torch.max(l_star_per_level, r_star_per_level) < distance_ranges[i][1],
            torch.max(l_star_per_level, r_star_per_level) >= distance_ranges[i][0]
        )

        # positive_argmax_per_level are the first indices to the positive anchors which are both in gt boxes and satisfy the bbox size limits
        # If there are multiple maximal values in a reduced row then the indices of the first maximal value are returned.
        positive_anchor_indices_per_level, positive_argmax_per_level = torch.logical_and(
            anchor_points_in_gt_bboxes,
            points_within_range_per_level
        ).max(dim=0)

        normalized_annotations_for_anchors_per_level = class_bbox_annotation[positive_argmax_per_level, :] / 2**i

        positive_l_star_per_level = torch.diagonal(l_star_per_level[positive_argmax_per_level], 0)
        positive_r_star_per_level = torch.diagonal(r_star_per_level[positive_argmax_per_level], 0)

        positive_anchor_indices = torch.cat((positive_anchor_indices, positive_anchor_indices_per_level), dim=0)
        normalized_annotations_for_anchors = torch.cat((normalized_annotations_for_anchors, normalized_annotations_for_anchors_per_level), dim=0)
        l_star_for_all_anchors = torch.cat((l_star_for_all_anchors, positive_l_star_per_level))
        r_star_for_all_anchors = torch.cat((r_star_for_all_anchors, positive_r_star_per_level))

    return positive_anchor_indices, normalized_annotations_for_anchors, l_star_for_all_anchors, r_star_for_all_anchors

def get_atss_positives(jth_annotations, anchors_list, class_id):
    class_bbox_annotation = jth_annotations[jth_annotations[:, 2] == class_id]

    all_anchors = torch.cat(anchors_list, dim=0)
    num_gt = class_bbox_annotation.shape[0]

    num_anchors_per_loc = 3

    num_anchors_per_level = [anchors.size(dim=0) for anchors in anchors_list]
    candidate_number_of_positive_anchors_per_level = 9

    iou_matrix = calc_iou(all_anchors[:, :], class_bbox_annotation[:, :2])

    gt_centers_x = (class_bbox_annotation[:, 1] + class_bbox_annotation[:, 0]) / 2.0
    gt_centers_y = torch.zeros(gt_centers_x.shape).to(gt_centers_x.device)
    gt_points = torch.stack((gt_centers_x, gt_centers_y), dim=1)

    all_anchor_centers_x = (all_anchors[:, 1] + all_anchors[:, 0]) / 2.0
    all_anchor_centers_y = torch.zeros(all_anchor_centers_x.shape).to(all_anchor_centers_x.device)
    anchor_points = torch.stack((all_anchor_centers_x, all_anchor_centers_y), dim=1)

    distance_matrix_between_anchors_and_bboxes = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

    # Selecting candidates based on the center distance between anchor box and object
    candidate_anchor_idxs_list = []
    start_global_idx_to_all_anchors = 0
    for level, anchors_per_level in enumerate(anchors_list):
        end_global_idx_to_all_anchors = start_global_idx_to_all_anchors + num_anchors_per_level[level]
        distances_from_bbox_to_anchors_per_level = distance_matrix_between_anchors_and_bboxes[start_global_idx_to_all_anchors:end_global_idx_to_all_anchors, :]
        topk = min(candidate_number_of_positive_anchors_per_level * num_anchors_per_loc, num_anchors_per_level[level])

        # indices_to_k_shortest_anchors has the local indices to the anchors on the level i
        # they will always have values 0, 2, 4, ... from the total local indices 0, 1, 2, 3, ...
        _, local_indices_to_k_shortest_anchors = distances_from_bbox_to_anchors_per_level.topk(topk, dim=0, largest=False)
        candidate_anchor_idxs_list.append(start_global_idx_to_all_anchors + local_indices_to_k_shortest_anchors)

        start_global_idx_to_all_anchors = end_global_idx_to_all_anchors

    all_candidate_anchor_idxs_for_gt_bboxes = torch.cat(candidate_anchor_idxs_list, dim=0)
    # Using the sum of mean and standard deviation as the IoU threshold to select final positive samples
    candidate_ious_between_anchors_and_bboxes = iou_matrix[all_candidate_anchor_idxs_for_gt_bboxes, torch.arange(num_gt)]

    iou_mean_per_gt = candidate_ious_between_anchors_and_bboxes.mean(0)
    iou_std_per_gt = candidate_ious_between_anchors_and_bboxes.std(0)
    iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt

    is_positive_anchors = candidate_ious_between_anchors_and_bboxes >= iou_thresh_per_gt[None, :]

    # Limiting the final positive samples’ center to object
    # Find regression target l, r for each gt bbox
    anchor_num = all_anchor_centers_x.shape[0]
    for ng in range(num_gt):
        all_candidate_anchor_idxs_for_gt_bboxes[:, ng] += ng * anchor_num

    expanded_anchors_cx_for_gt_bboxes = all_anchor_centers_x.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
    all_candidate_anchor_idxs_for_gt_bboxes = all_candidate_anchor_idxs_for_gt_bboxes.view(-1)

    l_of_candidate_anchors = expanded_anchors_cx_for_gt_bboxes[all_candidate_anchor_idxs_for_gt_bboxes].view(-1, num_gt) - class_bbox_annotation[:, 0]
    r_of_candidate_anchors = class_bbox_annotation[:, 1] - expanded_anchors_cx_for_gt_bboxes[all_candidate_anchor_idxs_for_gt_bboxes].view(-1, num_gt)

    is_anchor_in_bbox = torch.stack([l_of_candidate_anchors, r_of_candidate_anchors], dim=1).min(dim=1)[0] > 0.01
    is_positive_anchors = is_positive_anchors & is_anchor_in_bbox

    # if an anchor box is assigned to multiple bboxes, the bbox with the highest IoU will be selected
    # because one anchor should be assigned only one bbox
    INF = 100000000

    ious_inf = torch.full_like(iou_matrix, -INF).t().contiguous().view(-1)
    index = all_candidate_anchor_idxs_for_gt_bboxes.view(-1)[is_positive_anchors.view(-1)]
    ious_inf[index] = iou_matrix.t().contiguous().view(-1)[index]
    ious_inf = ious_inf.view(num_gt, -1).t()

    anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1)

    positive_anchor_indices = anchors_to_gt_values != -INF
    assigned_annotations_for_anchors = class_bbox_annotation[anchors_to_gt_indexs]

    return positive_anchor_indices, assigned_annotations_for_anchors

class FocalLoss(nn.Module):
    def __init__(self, fcos=False):
        super(FocalLoss, self).__init__()
        self.fcos = fcos

    def forward(self, classifications, anchors_list, annotations, class_id, regress_limits=(0, float('inf'))):
        if class_id == -1:
            raise ValueError

        alpha = 0.25
        gamma = 2.0

        batch_size = classifications.shape[0]
        classification_losses = []

        # if self.fcos:
        #     anchor = anchors_list[:, :]

        #     # anchors = (x, y) in feature map
        #     # [[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2], []]
        #     assert torch.all(anchor[:, 0] == anchor[:, 1])
        #     anchor = anchor[:, 0]

        for j in range(batch_size):
            # j refers to an audio sample in batch
            jth_classification = classifications[j, :, :]

            # get box annotations from the original image
            # (5, 20, 0), (-1, -1, -1), 
            jth_annotations = annotations[j, :, :]
            #bbox_annotation = bbox_annotation[bbox_annotation[:, 2] != -1]

            jth_annotations = jth_annotations[jth_annotations[:, 2] != -1] # jth_annotations[:, 2] is the classification label

            jth_classification = torch.clamp(jth_classification, 1e-4, 1.0 - 1e-4)

            if jth_annotations.shape[0] == 0: # if there are no annotation boxes on the jth image
                # the same focal loss is used by both retinanet and fcos
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

                continue # go to the next image

            if self.fcos:
                class_targets = torch.zeros(jth_classification.shape)

                positive_anchor_indices_per_class, assigned_annotations, _, _ = get_fcos_positives(jth_annotations, anchors_list, class_id=class_id)
            else:
                # initialize the beat/downbeat classifiers of all anchors (positive and negative) to background
                class_targets = torch.zeros(jth_classification.shape)

                # positive_anchor_indices is class-specific if class_id is not None
                positive_anchor_indices_per_class, assigned_annotations = get_atss_positives(jth_annotations, anchors_list, class_id=class_id)

            if torch.cuda.is_available():
                class_targets = class_targets.cuda()

            num_positive_anchors_per_class = positive_anchor_indices_per_class.sum()

            # class_targets[positive_anchor_indices_per_class, 0] = 0 (positive anchors are background) or 1 (positive anchors are downbeats)
            # class_targets[positive_anchor_indices_per_class, 1] = 0 (positive anchors are background) or 1 (positive anchors are beats)
            # initialize the beat/downbeat classifiers of the positive anchors to background
            class_targets[positive_anchor_indices_per_class, :] = 0 # the shape of class_targets is (A*W, C) = (3*W, 2)

            # assigned_annotations[positive_anchor_indices_per_class, 2] is the class ID of the gt bboxes assigned to positive anchors 
            class_targets[positive_anchor_indices_per_class, assigned_annotations[positive_anchor_indices_per_class, 2].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(class_targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(class_targets.shape) * alpha

            alpha_factor = torch.where(torch.eq(class_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(class_targets, 1.), 1. - jth_classification, jth_classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(class_targets * torch.log(jth_classification) + (1.0 - class_targets) * torch.log(1.0 - jth_classification))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(class_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(class_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            # print("cls", torch.ne(class_targets, -1.0).shape, positive_anchor_indices_per_class.shape, bce.shape, cls_loss.shape)

            # if torch.cuda.is_available():
            #     left_loss = torch.where(positive_anchor_indices_per_class, bce, torch.zeros(bce.shape).cuda())
            # else:
            #     left_loss = torch.where(positive_anchor_indices_per_class, bce, torch.zeros(bce.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors_per_class.float(), min=1.0))

        if self.fcos:
            return torch.stack(classification_losses).sum(dim=0)
        else:
            return torch.stack(classification_losses).mean(dim=0, keepdim=True)

class RegressionLoss(nn.Module):
    def __init__(self, fcos=False, loss_type="l1", weight=1, num_anchors=3):
        super(RegressionLoss, self).__init__()
        self.fcos = fcos
        self.loss_type = loss_type
        self.weight = weight
        self.num_anchors = num_anchors

    def forward(self, regressions, anchors_list, annotations, class_id, regress_limits=(0, float('inf'))):
        if class_id == -1:
            raise ValueError

        # regressions is (B, C, W, H), with C = 4*num_anchors = 4*9
        # in our case, regressions is (B, C, W), with C = 2*num_anchors = 2*1
        batch_size = regressions.shape[0] 
        regression_losses = []

        # if self.fcos:
        #     assert torch.all(anchor[:, 0] == anchor[:, 1])
        #     anchor = anchor[:, 0] # [5, 10, 15, ...]

        for j in range(batch_size):
            jth_regression = regressions[j, :, :] # j'th audio in the current batch

            jth_annotations = annotations[j, :, :]
            #jth_annotations = jth_annotations[jth_annotations[:, 2] != -1] # jth_annotations[:, 2] is the classification label

            jth_annotations = jth_annotations[jth_annotations[:, 2] != -1] # jth_annotations[:, 2] is the classification label

            if jth_annotations.shape[0] == 0:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

                continue

            if self.fcos:
                positive_anchor_indices_per_class, normalized_annotations_for_anchors, _, _ = get_fcos_positives(jth_annotations, anchors_list, class_id=class_id)

                # normalized_annotations_for_anchors shape (number of positive anchors, 2)
                # jth_regression[positive_anchor_indices_per_class, :2] shape (number of positive anchors, 2)
                positive_anchor_regression_giou = calc_giou(
                    normalized_annotations_for_anchors[positive_anchor_indices_per_class],
                    jth_regression[positive_anchor_indices_per_class, :2]
                )

                regression_losses_for_positive_anchors = \
                    torch.ones(positive_anchor_regression_giou.shape).to(positive_anchor_regression_giou.device) \
                    - positive_anchor_regression_giou

                #regression_losses.append(regression_losses_for_positive_anchors.sum() * self.weight)
                regression_losses.append(regression_losses_for_positive_anchors.mean() * self.weight)
            else:
                all_anchors = torch.cat(anchors_list, dim=0)
                anchor_widths  = all_anchors[:, 1] - all_anchors[:, 0] # if fcos is true, anchor_widths = 0
                anchor_ctr_x   = all_anchors[:, 0] + 0.5 * anchor_widths # if fcos is true, anchor_ctr_x = anchor[:, 0]

                # IoU = calc_iou(anchors[0, :, :], jth_annotations[:, :2]) # num_anchors x num_annotations
                # IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
                # positive_indices = torch.ge(IoU_max, 0.5)
                # assigned_annotations = jth_annotations[IoU_argmax, :]
                positive_anchor_indices_per_class, assigned_annotations_for_anchors = get_atss_positives(jth_annotations, anchors_list, class_id=class_id)

                if positive_anchor_indices_per_class.sum() > 0:
                    positive_annotations = assigned_annotations_for_anchors[positive_anchor_indices_per_class, :] #MJ: corrected

                    anchor_widths_pi = anchor_widths[positive_anchor_indices_per_class]
                    anchor_ctr_x_pi = anchor_ctr_x[positive_anchor_indices_per_class]

                    gt_widths  = assigned_annotations[:, 1] - assigned_annotations[:, 0]
                    gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                    # print("gt", gt_widths)
                    # print("anchor", anchor_widths_pi)

                    # clip widths to 1
                    gt_widths  = torch.clamp(gt_widths, min=1)

                    if self.loss_type == "l1": #MJ: compute the "offset" regression loss, where the offsets are relative to the positive anchor boxes.
                        box_targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                        box_targets_dw = torch.log(gt_widths / anchor_widths_pi)

                        box_targets = torch.stack((box_targets_dx, box_targets_dw))
                        box_targets = box_targets.t()

                        if torch.cuda.is_available():
                            box_targets = box_targets/torch.Tensor([[0.1, 0.2]]).cuda()
                        else:
                            box_targets = box_targets/torch.Tensor([[0.1, 0.2]])

                        negative_indices = 1 + (~positive_anchor_indices_per_class)

                        regression_diff = torch.abs(box_targets - jth_regression[positive_anchor_indices_per_class, :])

                        #MJ: 9.0 in the following is not the number of anchors!!!!
                        regression_loss = torch.where(
                            torch.le(regression_diff, 1.0 / 9.0),
                            0.5 * 9.0 * torch.pow(regression_diff, 2),
                            regression_diff - 0.5 / 9.0
                        )
                        # regression_loss = torch.where(
                        #     torch.le(regression_diff, 1.0 / self.num_anchors),
                        #     0.5 * self.num_anchors * torch.pow(regression_diff, 2),
                        #     regression_diff - 0.5 / self.num_anchors
                        # )
                        # print("regression", jth_regression[positive_anchor_indices_per_class, :])
                        # print("box_targets", box_targets)
                        # print("loss", regression_loss)

                        regression_losses.append(regression_loss.mean())
                    elif self.loss_type == "iou" or self.loss_type == "giou":
                        target_left = assigned_annotations[:, 0]
                        target_right = assigned_annotations[:, 1]

                        #prediction = self.regressBoxes(all_anchors.unsqueeze(dim=0), jth_regression.unsqueeze(dim=0)).squeeze()
                        #MJ: The use of self.regressBoxes() transforms the offsets prediciton to the real bbxes coordinates for evaluation mode 
                        #It has nothing to do with defining iou or giou loss for the positive anchor boxes
                        # prediction_left = prediction[positive_anchor_indices_per_class, 0]
                        # prediction_right = prediction[positive_anchor_indices_per_class, 1]

                        #MJ:

                        prediction_left = jth_regression[positive_anchor_indices_per_class, 0]
                        prediction_right = jth_regression[positive_anchor_indices_per_class, 1]

                        target_area = (target_left + target_right)
                        prediction_area = (prediction_left + prediction_right)

                        w_intersect = torch.min(prediction_left, target_left) + torch.min(prediction_right, target_right)
                        g_w_intersect = torch.max(prediction_left, target_left) + torch.max(prediction_right, target_right)

                        ac_uion = g_w_intersect + 1e-7
                        area_intersect = w_intersect
                        area_union = target_area + prediction_area - area_intersect
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

                        regression_losses.append(losses.sum() * self.weight)

                else:
                    if torch.cuda.is_available():
                        regression_losses.append(torch.tensor(0).float().cuda())
                    else:
                        regression_losses.append(torch.tensor(0).float())
            #End of e;se:
        if self.fcos:
            return torch.stack(regression_losses).sum(dim=0)
        else:
            return torch.stack(regression_losses).mean(dim=0, keepdim=True)

# class CenternessLoss(nn.Module):
#     def __init__(self, fcos=False):
#         super(CenternessLoss, self).__init__()
#         self.fcos = fcos

#     def forward(self, centernesses, anchors, annotations, regress_limits=(0, float('inf'))):
#         if not self.fcos:
#             raise NotImplementedError

#         batch_size = centernesses.shape[0]
#         centerness_losses = []

#         anchor = anchors[:, :]

#         assert torch.all(anchor[:, 0] == anchor[:, 1])
#         anchor = anchor[:, 0]

#         for j in range(batch_size):
#             jth_centerness = centernesses[j, :, :]

#             jth_annotations = annotations[j, :, :]
#             jth_annotations = jth_annotations[jth_annotations[:, 2] != -1]

#             #jth_centerness = torch.clamp(jth_centerness, 1e-4, 1.0 - 1e-4)
#             jth_centerness = torch.sigmoid(jth_centerness)

#             positive_indices, assigned_annotations, left, right = get_fcos_positives(
#                 jth_annotations,
#                 anchor,
#                 regress_limits[0],
#                 regress_limits[1]
#             )

#             num_positive_anchors = positive_indices.sum()

#             targets = torch.where(
#                 positive_indices,
#                 torch.sqrt(torch.min(left, right) / torch.max(left, right)).float(),
#                 torch.zeros(positive_indices.shape)  
#             ).unsqueeze(dim=1)

#             bce = -(targets * torch.log(jth_centerness) + (1.0 - targets) * torch.log(1.0 - jth_centerness))

#             if torch.cuda.is_available():
#                 ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape).cuda())
#             else:
#                 ctr_loss = torch.where(positive_indices, bce, torch.zeros(bce.shape))

#             #print(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
#             centerness_losses.append(ctr_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

#         if self.fcos:
#             return torch.stack(centerness_losses).sum(dim=0)
#         else:
#             return torch.stack(centerness_losses).mean(dim=0, keepdim=True)

class LeftnessLoss(nn.Module):
    def __init__(self, fcos=False):
        super(LeftnessLoss, self).__init__()
        self.fcos = fcos

    def forward(self, leftnesses, anchors, annotations, regress_limits=(0, float('inf'))):
        if not self.fcos:
            raise NotImplementedError

        batch_size = leftnesses.shape[0]
        leftness_losses = []

        for j in range(batch_size):
            jth_leftness = leftnesses[j, :, :]

            jth_annotations = annotations[j, :, :]
            jth_annotations = jth_annotations[jth_annotations[:, 2] != -1]

            #jth_leftness = torch.clamp(jth_leftness, 1e-4, 1.0 - 1e-4)
            jth_leftness = torch.sigmoid(jth_leftness)  #MJ:  #https://stackoverflow.com/questions/66906884/how-is-pytorchs-class-bcewithlogitsloss-exactly-implemented
        

            positive_indices, assigned_annotations, left, right = get_fcos_positives(
                jth_annotations,
                anchor,
                regress_limits[0],
                regress_limits[1]
            )

            # center_targets = torch.where(
            #     positive_anchor_indices_per_class,
            #     torch.sqrt(torch.min(l_star_for_all_anchors, r_star_for_all_anchors) / torch.max(l_star_for_all_anchors, r_star_for_all_anchors)).float(),
            #     torch.zeros(positive_anchor_indices_per_class.shape)
            # ).unsqueeze(dim=1)

            targets = torch.where(
                positive_indices,
                torch.sqrt(torch.min(left, right) / torch.max(left, right)).float(),
                torch.zeros(positive_indices.shape)
            ).unsqueeze(dim=1)
            # torch.set_printoptions(edgeitems=1000000)
            # print(torch.where(positive_anchor_indices_per_class, l_star_for_all_anchors, torch.zeros(l_star_for_all_anchors.shape)))
            # print(torch.where(positive_anchor_indices_per_class, r_star_for_all_anchors, torch.zeros(r_star_for_all_anchors.shape)))
            # torch.set_printoptions(edgeitems=3)

            bce = -(left_targets * torch.log(jth_leftness) + (1.0 - left_targets) * torch.log(1.0 - jth_leftness))

            left_loss = bce.squeeze() * positive_anchor_indices_per_class

            leftness_losses.append(left_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        if self.fcos:
            return torch.stack(leftness_losses).sum(dim=0)
        else:
            return torch.stack(leftness_losses).mean(dim=0, keepdim=True)
