import numpy as np
import torch
import torch.nn as nn
from beatfcos.utils import calc_iou, calc_giou, AnchorPointTransform

INF = 100000000

def clusters_to_interval_length_ranges(clusters: torch.Tensor):
    """
    Given a tensor of sorted cluster centers (durations in seconds),
    returns a list of (min, max) size ranges for FCOS levels.
    """
    assert clusters.ndim == 1 and torch.all(clusters[:-1] <= clusters[1:]), "Clusters must be a sorted 1D tensor"

    sizes = []

    # Compute midpoints between adjacent clusters
    midpoints = (clusters[1:] - clusters[:-1]) / 2

    # Compute edges for each size range
    edges = clusters[:-1] + midpoints

    # First range
    sizes.append([-1.0, edges[0].item()])

    # Middle ranges
    for i in range(1, len(edges)):
        sizes.append([edges[i - 1].item(), edges[i].item()])

    # Last range
    sizes.append([edges[-1].item(), 1000.0])

    return sizes

def get_fcos_positives(jth_annotations, anchors_list, interval_length_ranges,
                       audio_downsampling_factor, audio_sample_rate,
                       centerness=False, beat_radius=2.5, downbeat_radius=4.5):
    audio_target_rate = audio_sample_rate / audio_downsampling_factor

    boolean_indices_to_bboxes_for_positive_anchors = torch.zeros(0, dtype=torch.bool).to(jth_annotations.device)
    assigned_annotations_for_anchors = torch.zeros(0, 3).to(jth_annotations.device)
    normalized_annotations_for_anchors = torch.zeros(0, 3).to(jth_annotations.device)
    l_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    r_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    normalized_l_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    normalized_r_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    levels_for_all_anchors = torch.zeros(0).to(jth_annotations.device)

    for i, anchor_points_per_level in enumerate(anchors_list):
        levels_per_level = torch.ones(anchor_points_per_level.shape).to(anchor_points_per_level.device) * (i + 1)
        
        anchor_points_per_level_nx1 = torch.unsqueeze(anchor_points_per_level, dim=1)  # shape = (N,1)
        l_annotations_1xm = torch.unsqueeze(jth_annotations[:, 0], dim=0)  # shape = (1,M)
        r_annotations_1xm = torch.unsqueeze(jth_annotations[:, 1], dim=0)   # shape = (1,M)

        # JA: New radius implementation
        stride = 2**i
        radius_per_class = (jth_annotations[:, 2] == 0) * downbeat_radius + (jth_annotations[:, 2] == 1) * beat_radius

        if centerness:
            c_annotations_1xm = (l_annotations_1xm + r_annotations_1xm)/2
            left_radius_limit_from_center = c_annotations_1xm - (radius_per_class * stride)
            right_radius_limit_from_center = c_annotations_1xm + (radius_per_class * stride)
            anchor_points_in_sub_bboxes_per_level = torch.logical_and(
                torch.ge(anchor_points_per_level_nx1, torch.maximum(l_annotations_1xm, left_radius_limit_from_center)),
                torch.le(anchor_points_per_level_nx1, torch.minimum(r_annotations_1xm, right_radius_limit_from_center))
            )
        else:
            radius_limits_from_l_annotations = l_annotations_1xm + (radius_per_class * stride)
            anchor_points_in_sub_bboxes_per_level = torch.logical_and(
                torch.ge( anchor_points_per_level_nx1, l_annotations_1xm),
                torch.le( anchor_points_per_level_nx1, torch.minimum(r_annotations_1xm, radius_limits_from_l_annotations))
            )

        l_stars_to_bboxes_for_anchors_per_level =  anchor_points_per_level_nx1 - l_annotations_1xm
        r_stars_to_bboxes_for_anchors_per_level =  r_annotations_1xm - anchor_points_per_level_nx1

        size_of_interest_per_level = anchor_points_per_level.new_tensor([
            interval_length_ranges[i][0] * audio_target_rate,
            interval_length_ranges[i][1] * audio_target_rate
        ])

        size_of_interest_for_anchors_per_level = size_of_interest_per_level[None].expand(anchor_points_per_level.size(dim=0), -1)

        # Put L and R stars into a single tensor so that we can calculate max
        l_r_targets_for_anchors_per_level = torch.stack([l_stars_to_bboxes_for_anchors_per_level, r_stars_to_bboxes_for_anchors_per_level], dim=2)
        # l_r_targets_for_anchors_per_level shape is (N, M, 2) where N is num of anchors and M is num of gt bboxes
        max_l_r_targets_for_anchors_per_level, _ = l_r_targets_for_anchors_per_level.max(dim=2)

        max_l_r_stars_for_anchor_points_within_bbox_range_per_level = (
            max_l_r_targets_for_anchors_per_level >= size_of_interest_for_anchors_per_level[:, [0]]
        ) & (max_l_r_targets_for_anchors_per_level <= size_of_interest_for_anchors_per_level[:, [1]])

        areas_of_bboxes = jth_annotations[:, 1] - jth_annotations[:, 0]

        gt_area_for_anchors_matrix = areas_of_bboxes[None].repeat(len(anchor_points_per_level), 1)
        gt_area_for_anchors_matrix[anchor_points_in_sub_bboxes_per_level == 0] = INF
        gt_area_for_anchors_matrix[max_l_r_stars_for_anchor_points_within_bbox_range_per_level == 0] = INF

        min_areas_for_anchors, indices_to_min_bboxes_for_anchors = gt_area_for_anchors_matrix.min(1)

        assigned_annotations_for_anchors_per_level = jth_annotations[indices_to_min_bboxes_for_anchors] # Among the annotations, choose the ones associated with box with minimum area
        assigned_annotations_for_anchors_per_level[min_areas_for_anchors == INF, 2] = 0 # Assigned background class label 0  to the anchor boxes whose min area with respect to the bboxes is INF
        
        # boolean_indices_to_min_bboxes_for_anchors[i] represents the index to the min area of anchor i
        # If the areas of all boxes are INF, this index will be 0
        # There are cases when the index 0 refer to a non-inf area so we need to distinguish the two cases
        boolean_indices_to_min_bboxes_for_anchors = torch.zeros(min_areas_for_anchors.shape, dtype=torch.bool).to(min_areas_for_anchors.device)
        boolean_indices_to_min_bboxes_for_anchors[min_areas_for_anchors != INF] = True

        normalized_annotations_for_anchors_per_level = torch.clone(assigned_annotations_for_anchors_per_level)
        normalized_annotations_for_anchors_per_level[:,0] /= stride
        normalized_annotations_for_anchors_per_level[:,1] /= stride

        # MJ:  # get the l_star values of the bboxes for the positive anchors

        l_stars_for_anchors_per_level = l_stars_to_bboxes_for_anchors_per_level[
            torch.arange(0, anchor_points_per_level.size(dim=0)),
            indices_to_min_bboxes_for_anchors
        ]

        r_stars_for_anchors_per_level = r_stars_to_bboxes_for_anchors_per_level[
            torch.arange(0, anchor_points_per_level.size(dim=0)),
            indices_to_min_bboxes_for_anchors
        ]

        normalized_l_star_for_anchors_per_level = l_stars_for_anchors_per_level / stride
        normalized_r_star_for_anchors_per_level = r_stars_for_anchors_per_level / stride

        boolean_indices_to_bboxes_for_positive_anchors = torch.cat(( boolean_indices_to_bboxes_for_positive_anchors, boolean_indices_to_min_bboxes_for_anchors), dim=0)
        assigned_annotations_for_anchors = torch.cat((assigned_annotations_for_anchors, assigned_annotations_for_anchors_per_level), dim=0)
        normalized_annotations_for_anchors = torch.cat((normalized_annotations_for_anchors, normalized_annotations_for_anchors_per_level), dim=0)
        l_star_for_anchors = torch.cat((l_star_for_anchors, l_stars_for_anchors_per_level))
        r_star_for_anchors = torch.cat((r_star_for_anchors, r_stars_for_anchors_per_level ))
        normalized_l_star_for_anchors = torch.cat((normalized_l_star_for_anchors, normalized_l_star_for_anchors_per_level))
        normalized_r_star_for_anchors = torch.cat((normalized_r_star_for_anchors, normalized_r_star_for_anchors_per_level))
        levels_for_all_anchors = torch.cat((levels_for_all_anchors, levels_per_level))

    return boolean_indices_to_bboxes_for_positive_anchors,\
        assigned_annotations_for_anchors, normalized_annotations_for_anchors,\
        l_star_for_anchors, r_star_for_anchors,\
        normalized_l_star_for_anchors, normalized_r_star_for_anchors, levels_for_all_anchors#,\

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, jth_classification_pred, jth_classification_targets, jth_annotations, num_positive_anchors):
        alpha = 0.25
        gamma = 2.0

        jth_classification_pred = torch.clamp(jth_classification_pred, 1e-4, 1.0 - 1e-4)

        if jth_annotations.shape[0] == 0: # if there are no annotation boxes on the jth image
            # the same focal loss is used by both retinanet and fcos
            if torch.cuda.is_available():
                alpha_factor = torch.ones(jth_classification_pred.shape).cuda() * alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = jth_classification_pred
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - jth_classification_pred))

                cls_loss = focal_weight * bce
                return cls_loss.sum()
            else:
                alpha_factor = torch.ones(jth_classification_pred.shape) * alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = jth_classification_pred
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - jth_classification_pred))

                cls_loss = focal_weight * bce
                return cls_loss.sum()

        # num_positive_anchors = positive_anchor_indices.sum() # We will do this outside in the new implementation

        if torch.cuda.is_available():
            alpha_factor = torch.ones(jth_classification_targets.shape).cuda() * alpha
        else:
            alpha_factor = torch.ones(jth_classification_targets.shape) * alpha

        alpha_factor = torch.where(torch.eq(jth_classification_targets, 1.), alpha_factor, 1. - alpha_factor)
        focal_weight = torch.where(torch.eq(jth_classification_targets, 1.), 1. - jth_classification_pred, jth_classification_pred)
        focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

        bce = -(jth_classification_targets * torch.log(jth_classification_pred) + (1.0 - jth_classification_targets) * torch.log(1.0 - jth_classification_pred))

        cls_loss = focal_weight * bce

        if torch.cuda.is_available(): #MJ: 
            cls_loss = torch.where(torch.ne(jth_classification_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
        else:
            cls_loss = torch.where(torch.ne(jth_classification_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

        return cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)

class RegressionLoss(nn.Module):
    def __init__(self, weight=1):
        super(RegressionLoss, self).__init__()
        self.weight = weight

    def forward(self, jth_regression_pred, jth_regression_targets, jth_annotations):
        # If there are gt bboxes on the current image, we set the regression loss of this image to 0
        if jth_annotations.shape[0] == 0:
            if torch.cuda.is_available():
                return torch.tensor(0).float().cuda()
            else:
                return torch.tensor(0).float()

        # To calculate GIoU, convert prediction and targets from (l, r) to (x_1, x_2)
        jth_regression_xx_pred = jth_regression_pred
        jth_regression_xx_targets = jth_regression_targets

        # Flip the sign of x_1 to turn the (l, r) box into a (x_1, x_2) bounding box offset from 0
        # (For GIoU calculation, the bounding box offset does not matter as much as the two boxes' relative positions)
        jth_regression_xx_pred[:, 0] *= -1
        jth_regression_xx_targets[:, 0] *= -1

        positive_anchor_regression_giou = calc_giou(jth_regression_xx_pred, jth_regression_xx_targets)

        regression_losses_for_positive_anchors = \
            torch.ones(positive_anchor_regression_giou.shape).to(positive_anchor_regression_giou.device) \
            - positive_anchor_regression_giou

        return regression_losses_for_positive_anchors.mean() * self.weight

class LeftnessLoss(nn.Module):
    def __init__(self):
        super(LeftnessLoss, self).__init__()

    def forward(self, jth_leftness_pred, jth_leftness_targets, jth_annotations):
        jth_leftness_pred = torch.clamp(jth_leftness_pred, 1e-4, 1.0 - 1e-4)

        # If there are gt bboxes on the current image, we set the regression loss of this image to 0
        if jth_annotations.shape[0] == 0:
            if torch.cuda.is_available():
                return torch.tensor(0).float().cuda()
            else:
                return torch.tensor(0).float()

        bce = -(jth_leftness_targets * torch.log(jth_leftness_pred)\
            + (1.0 - jth_leftness_targets) * torch.log(1.0 - jth_leftness_pred))

        leftness_loss = bce

        return leftness_loss.mean()

class AdjacencyConstraintLoss(nn.Module):
    def __init__(self):
        super(AdjacencyConstraintLoss, self).__init__()
        self.anchor_point_transform = AnchorPointTransform()

    def calculate_downbeat_and_beat_x1_loss(
        self,
        transformed_target_regression_boxes,  # shape: [N, 2]
        transformed_pred_regression_boxes,      # shape: [N, 2]
        positive_downbeat_anchor_mask,
        positive_beat_anchor_mask
    ):
        # For the downbeat-beat constraint we align the left sides.
        downbeat_target_boxes = transformed_target_regression_boxes[positive_downbeat_anchor_mask, :]
        downbeat_pred_boxes   = transformed_pred_regression_boxes[positive_downbeat_anchor_mask, :]
        beat_target_boxes     = transformed_target_regression_boxes[positive_beat_anchor_mask, :]
        beat_pred_boxes       = transformed_pred_regression_boxes[positive_beat_anchor_mask, :]

        loss = self._gdoU_loss(
            target_a=downbeat_target_boxes,
            target_b=beat_target_boxes,
            pred_a=downbeat_pred_boxes,
            pred_b=beat_pred_boxes,
            side_a='left',
            side_b='left'
        )
        return loss

    def calculate_x2_and_x1_loss(
        self,
        transformed_target_regression_boxes,  # shape: [N, 2]
        transformed_pred_regression_boxes,      # shape: [N, 2]
        positive_class_anchor_mask
    ):
        # For the x2-x1 constraint we enforce that the right side (x2) and left side (x1)
        # are aligned for these boxes (for example, forcing some boxes to become points).
        class_target_boxes = transformed_target_regression_boxes[positive_class_anchor_mask, :]
        class_pred_boxes   = transformed_pred_regression_boxes[positive_class_anchor_mask, :]

        loss = self._gdoU_loss(
            target_a=class_target_boxes,
            target_b=class_target_boxes,
            pred_a=class_pred_boxes,
            pred_b=class_pred_boxes,
            side_a='right',  # use right side from one copy
            side_b='left'    # and left side from the other
        )
        return loss

    def forward(
        self, jth_classification_targets,
        jth_regression_pred, jth_regression_targets,
        jth_positive_anchor_points, jth_positive_anchor_strides,
        jth_annotations, num_positive_anchors
    ):
        # Create boolean masks for downbeats and beats.
        positive_downbeat_anchor_mask = jth_classification_targets[:, 0] == 1
        positive_beat_anchor_mask = jth_classification_targets[:, 1] == 1

        jth_regression_targets_1xm = jth_regression_targets[None]
        jth_regression_pred_1xn = jth_regression_pred[None]

        # Transform regression boxes from (l, r) to (x1, x2) boxes.
        transformed_target_regression_boxes_batch = self.anchor_point_transform(
            jth_positive_anchor_points,
            jth_regression_targets_1xm,
            jth_positive_anchor_strides
        )
        transformed_target_regression_boxes = transformed_target_regression_boxes_batch[0, :, :]

        transformed_pred_regression_boxes_batch = self.anchor_point_transform(
            jth_positive_anchor_points,
            jth_regression_pred_1xn,
            jth_positive_anchor_strides
        )
        transformed_pred_regression_boxes = transformed_pred_regression_boxes_batch[0, :, :]

        downbeat_and_beat_loss = self.calculate_downbeat_and_beat_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            positive_downbeat_anchor_mask,
            positive_beat_anchor_mask
        )

        downbeat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            positive_downbeat_anchor_mask
        )

        beat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            positive_beat_anchor_mask
        )

        total_loss = downbeat_and_beat_loss + downbeat_x2_and_x1_loss + beat_x2_and_x1_loss
        individual_losses = {
            'db': downbeat_and_beat_loss / torch.clamp(num_positive_anchors.float(), min=1.0),
            'dd': downbeat_x2_and_x1_loss / torch.clamp(num_positive_anchors.float(), min=1.0),
            'bb': beat_x2_and_x1_loss / torch.clamp(num_positive_anchors.float(), min=1.0)
        }

        normalized_total_loss = total_loss / torch.clamp(num_positive_anchors.float(), min=1.0)
        return normalized_total_loss, individual_losses

    def _doc_loss(
        self,
        target_a: torch.Tensor,  # shape: [N1, 2]
        target_b: torch.Tensor,  # shape: [N2, 2]
        pred_a: torch.Tensor,    # shape: [N1, 2]
        pred_b: torch.Tensor,    # shape: [N2, 2]
        side_a: str,
        side_b: str
    ):
        """
        Computes a DoC(Difference over Convex hull)-based loss between two groups of boxes. For each
        pair, the loss is computed as:
        
            loss = (difference / C)

        The incidence (i.e. which pairs to compare) is determined based on the target box coordinates—
        the corresponding target side values must be equal.
        """
        N1 = target_a.shape[0]
        N2 = target_b.shape[0]

        # Select target constraint coordinates based on the side
        if side_a == "left":
            target_cons_a = target_a[:, 0]
        elif side_a == "right":
            target_cons_a = target_a[:, 1]
        else:
            raise ValueError("side_a must be 'left' or 'right'")

        if side_b == "left":
            target_cons_b = target_b[:, 0]
        elif side_b == "right":
            target_cons_b = target_b[:, 1]
        else:
            raise ValueError("side_b must be 'left' or 'right'")

        # Create incidence matrix from targets (only pairs with matching target side coordinates will contribute)
        target_cons_a_exp = target_cons_a[:, None].repeat(1, N2)
        target_cons_b_exp = target_cons_b[None, :].repeat(N1, 1)
        incidence_matrix = (target_cons_a_exp == target_cons_b_exp)
        if incidence_matrix.sum() == 0:
            return torch.tensor(0.0, dtype=pred_a.dtype, device=pred_a.device)

        # Select predicted constraint coordinates
        if side_a == "left":
            pred_cons_a = pred_a[:, 0]
        else:  # "right"
            pred_cons_a = pred_a[:, 1]

        if side_b == "left":
            pred_cons_b = pred_b[:, 0]
        else:  # "right"
            pred_cons_b = pred_b[:, 1]

        pred_cons_a_exp = pred_cons_a[:, None].repeat(1, N2)
        pred_cons_b_exp = pred_cons_b[None, :].repeat(N1, 1)
        difference = torch.abs(pred_cons_a_exp - pred_cons_b_exp)

        # Compute pairwise intersection
        box_left = torch.max(pred_a[:, 0][:, None].repeat(1, N2), pred_b[:, 0][None, :].repeat(N1, 1))
        box_right = torch.min(pred_a[:, 1][:, None].repeat(1, N2), pred_b[:, 1][None, :].repeat(N1, 1))
        intersection = torch.clamp(box_right - box_left, min=0)

        # Compute smallest enclosing interval (C)
        min_left = torch.min(pred_a[:, 0][:, None].repeat(1, N2), pred_b[:, 0][None, :].repeat(N1, 1))
        max_right = torch.max(pred_a[:, 1][:, None].repeat(1, N2), pred_b[:, 1][None, :].repeat(N1, 1))
        eps = 1e-7

        convex_hull = max_right - min_left
        convex_hull = torch.where(convex_hull < eps, 1, convex_hull)

        doc = difference / convex_hull # JA: DoC = difference over convex hull
        loss = (doc * incidence_matrix).sum()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, clusters, audio_downsampling_factor, audio_sample_rate, centerness=False):
        super(CombinedLoss, self).__init__()

        self.classification_loss = FocalLoss()
        self.regression_loss = RegressionLoss()
        self.leftness_loss = LeftnessLoss()
        self.adjacency_constraint_loss = AdjacencyConstraintLoss()
        
        self.clusters = clusters
        self.audio_downsampling_factor = audio_downsampling_factor
        self.audio_sample_rate = audio_sample_rate
        self.centerness = centerness

    def get_jth_targets(
        self,
        jth_classification_pred,
        jth_regression_pred,
        jth_leftness_pred,
        positive_anchor_indices,
        normalized_annotations,
        l_star, r_star,
        normalized_l_star,
        normalized_r_star
    ):
        jth_classification_targets = torch.zeros(jth_classification_pred.shape).to(jth_classification_pred.device)
        jth_regression_targets = torch.zeros(jth_regression_pred.shape).to(jth_regression_pred.device)
        jth_leftness_targets = torch.zeros(jth_leftness_pred.shape).to(jth_leftness_pred.device)

        class_ids_of_positive_anchors = normalized_annotations[positive_anchor_indices, 2].long()

        jth_classification_targets[positive_anchor_indices, :] = 0
        jth_classification_targets[positive_anchor_indices, class_ids_of_positive_anchors] = 1

        jth_regression_targets = torch.stack((normalized_l_star, normalized_r_star), dim=1)

        if self.centerness:
            jth_leftness_targets = torch.sqrt(torch.min(l_star, r_star)/torch.max(l_star, r_star)).unsqueeze(dim=1)
        else:
            jth_leftness_targets = torch.sqrt(r_star/(l_star + r_star)).unsqueeze(dim=1)

        return jth_classification_targets, jth_regression_targets, jth_leftness_targets

    def forward(self, classifications, regressions, leftnesses, anchors_list, annotations):
        # Classification, regression, and leftness should all have the same number of items in the batch
        assert classifications.shape[0] == regressions.shape[0] and regressions.shape[0] == leftnesses.shape[0]
        batch_size = classifications.shape[0]

        classification_losses_batch = []
        regression_losses_batch = []
        leftness_losses_batch = []
        adjacency_constraint_losses_batch = []

        adj_db_batch, adj_bb_batch, adj_dd_batch = [], [], []

        for j in range(batch_size):
            jth_classification_pred = classifications[j, :, :]   # (B, A, 2)
            jth_regression_pred = regressions[j, :, :]           # (B, A, 2)
            jth_leftness_pred = leftnesses[j, :, :]              # (B, A, 1)

            jth_padded_annotations = annotations[j, :, :] #MJ: jth_padded_annotations[:, 2]= class id

            # The dummy gt boxes that are labeled as -1 are added to each batch by the collater function of DataSet to make all the annotations have the same shape,
            # To really process them,  those gt boxes should be removed. MJ: jth_annotations: shape=(57,3); annotations:shape=(127,3)
            jth_annotations = jth_padded_annotations[jth_padded_annotations[:, 2] != -1]
            
            # If there are no targets for the current audio in the batch, skip the audio
            if jth_annotations.size(dim=0) == 0:
                continue

            interval_length_ranges = clusters_to_interval_length_ranges(self.clusters)

            positive_anchor_indices, assigned_annotations_for_anchors, normalized_annotations_for_anchors, \
            l_star_for_anchors, r_star_for_anchors, normalized_l_star_for_anchors, \
            normalized_r_star_for_anchors, levels_for_anchors = get_fcos_positives(
                jth_annotations, anchors_list, interval_length_ranges,
                self.audio_downsampling_factor, self.audio_sample_rate, self.centerness
            )

            all_anchor_points = torch.cat(anchors_list, dim=0)
            num_positive_anchors = positive_anchor_indices.sum()

            jth_classification_targets, jth_regression_targets, jth_leftness_targets = self.get_jth_targets(
                jth_classification_pred, jth_regression_pred, jth_leftness_pred,
                positive_anchor_indices, normalized_annotations_for_anchors,
                l_star_for_anchors, r_star_for_anchors,
                normalized_l_star_for_anchors, normalized_r_star_for_anchors
            )

            jth_classification_loss = self.classification_loss(
                jth_classification_pred,
                jth_classification_targets,
                jth_annotations,
                num_positive_anchors
            )

            jth_regression_loss = self.regression_loss(
                jth_regression_pred[positive_anchor_indices],
                jth_regression_targets[positive_anchor_indices],
                jth_annotations
            )

            jth_leftness_loss = self.leftness_loss(
                jth_leftness_pred[positive_anchor_indices],
                jth_leftness_targets[positive_anchor_indices],
                jth_annotations
            )

            if torch.isnan(jth_classification_loss).any():
                raise ValueError

            strides_for_all_anchors = torch.zeros(0).to(classifications.device)
            for i, anchors_per_level in enumerate(anchors_list):
                stride_per_level = torch.tensor(2**i).to(strides_for_all_anchors.device)
                stride_for_anchors_per_level = stride_per_level[None].expand(anchors_per_level.size(dim=0))
                strides_for_all_anchors = torch.cat((strides_for_all_anchors, stride_for_anchors_per_level), dim=0)

            jth_adjacency_constraint_loss, jth_individual_constraint_losses = self.adjacency_constraint_loss(
                jth_classification_targets[positive_anchor_indices],
                jth_regression_pred[positive_anchor_indices],
                jth_regression_targets[positive_anchor_indices],
                all_anchor_points[positive_anchor_indices],
                strides_for_all_anchors[positive_anchor_indices],
                jth_annotations,
                num_positive_anchors
            )

            classification_losses_batch.append(jth_classification_loss)
            regression_losses_batch.append(jth_regression_loss)
            leftness_losses_batch.append(jth_leftness_loss)
            adjacency_constraint_losses_batch.append(jth_adjacency_constraint_loss)

            adj_db_batch.append(jth_individual_constraint_losses["db"])
            adj_bb_batch.append(jth_individual_constraint_losses["bb"])
            adj_dd_batch.append(jth_individual_constraint_losses["dd"])
        # END for j in range(batch_size)

        if len(classification_losses_batch) == 0:
            classification_losses_batch.append(0)  #MJ: append zero tensor rather number 0
            
        if len(regression_losses_batch) == 0:
            regression_losses_batch.append(0)
            
        if len(leftness_losses_batch) == 0:
            leftness_losses_batch.append(0)
            
        if len(adjacency_constraint_losses_batch) == 0:
            adjacency_constraint_losses_batch.append(0)

        adj_db = torch.stack(adj_db_batch).mean(dim=0, keepdim=True).item()
        adj_bb = torch.stack(adj_bb_batch).mean(dim=0, keepdim=True).item()
        adj_dd = torch.stack(adj_dd_batch).mean(dim=0, keepdim=True).item()

        adjacency_dict = {"adj_db": adj_db, "adj_bb": adj_bb, "adj_dd": adj_dd}

        return \
            torch.stack(classification_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(leftness_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(adjacency_constraint_losses_batch).mean(dim=0, keepdim=True), \
            adjacency_dict
