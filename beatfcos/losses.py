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

        return cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0)
        #return cls_loss.sum()

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
        transformed_target_regression_boxes,
        transformed_pred_regression_boxes,
        boolean_indices_to_downbeats_for_positive_anchors,
        boolean_indices_to_beats_for_positive_anchors,
        effective_audio_length
    ):
        # N1 = num_of_downbeats_for_positive_anchors
        # N2 = num_of_beats_for_positive_anchors
        num_of_downbeats_for_positive_anchors = torch.sum(boolean_indices_to_downbeats_for_positive_anchors, dtype=torch.int32).item()
        num_of_beats_for_positive_anchors = torch.sum(boolean_indices_to_beats_for_positive_anchors, dtype=torch.int32).item()

        downbeat_target_x1s_for_anchors = transformed_target_regression_boxes[boolean_indices_to_downbeats_for_positive_anchors, 0]
        downbeat_pred_x1s_for_anchors = transformed_pred_regression_boxes[boolean_indices_to_downbeats_for_positive_anchors, 0]

        beat_target_x1s_for_anchors = transformed_target_regression_boxes[boolean_indices_to_beats_for_positive_anchors, 0]
        beat_pred_x1s_for_anchors = transformed_pred_regression_boxes[boolean_indices_to_beats_for_positive_anchors, 0]

        # A matrix with the dimensions (D, B) where D is downbeat count and B is beat count is created
        # We repeat the beat and downbeat positions so that we can do elementwise comparison to match
        # the downbeats with their corresponding first beat objects; they will share the same regression
        # box x1 position value.
        # For downbeats, the column is repeated; for beats, the row is repeated
        downbeat_target_x1s_for_anchors_N1x1 = downbeat_target_x1s_for_anchors[:, None]
        beat_target_x1s_for_anchors_1xN2 = beat_target_x1s_for_anchors[None, :]
        downbeat_pred_x1s_for_anchors_N1x1 = downbeat_pred_x1s_for_anchors[:, None]
        beat_pred_x1s_for_anchors_1xN2 = beat_pred_x1s_for_anchors[None, :]

        downbeat_position_repeated_N1xN2 = downbeat_target_x1s_for_anchors_N1x1.repeat(1, num_of_beats_for_positive_anchors)
        beat_position_repeated_N1xN2 = beat_target_x1s_for_anchors_1xN2.repeat(num_of_downbeats_for_positive_anchors, 1)

        downbeat_and_beat_x1_incidence_matrix_N1xN2 = downbeat_position_repeated_N1xN2 == beat_position_repeated_N1xN2
        num_incidences_between_downbeats_and_beats = downbeat_and_beat_x1_incidence_matrix_N1xN2.sum()

        if num_incidences_between_downbeats_and_beats == 0:
            return torch.tensor(0).float().to(num_incidences_between_downbeats_and_beats.device)

        # Calculate the mean square error between all the downbeat prediction x1 and beat prediction x1
        # and multiply this (D, B) result matrix with the incidence matrix to remove all values where
        # the downbeat does not correspond with the beat
        downbeat_and_beat_x1_discrepancy_error_N1xN2 = torch.square(
            (downbeat_pred_x1s_for_anchors_N1x1 - beat_pred_x1s_for_anchors_1xN2) / torch.clamp(effective_audio_length, min=1.0)
        ) 

        downbeat_and_beat_x1_discrepancy_error_N1xN2 *= downbeat_and_beat_x1_incidence_matrix_N1xN2

        downbeat_and_beat_x1_loss = downbeat_and_beat_x1_discrepancy_error_N1xN2.sum()

        return downbeat_and_beat_x1_loss

    def calculate_x2_and_x1_loss(
        self,
        transformed_target_regression_boxes,
        transformed_pred_regression_boxes,
        boolean_indices_to_classes_for_positive_anchors,
        effective_audio_length
    ):
        num_of_classes_for_positive_anchors = torch.sum(boolean_indices_to_classes_for_positive_anchors, dtype=torch.int32).item()

        class_target_x2s_for_anchors = transformed_target_regression_boxes[boolean_indices_to_classes_for_positive_anchors, 1]
        class_pred_x2s_for_anchors = transformed_pred_regression_boxes[boolean_indices_to_classes_for_positive_anchors, 1]

        class_target_x1s_for_anchors = transformed_target_regression_boxes[boolean_indices_to_classes_for_positive_anchors, 0]
        class_pred_x1s_for_anchors = transformed_pred_regression_boxes[boolean_indices_to_classes_for_positive_anchors, 0]

        class_target_x2s_for_anchors_nx1 = class_target_x2s_for_anchors[:, None]
        class_target_x1s_for_anchors_1xn = class_target_x1s_for_anchors[None, :]
        class_pred_x2s_for_anchors_nx1 = class_pred_x2s_for_anchors[:, None]
        class_pred_x1s_for_anchors_1xn = class_pred_x1s_for_anchors[None, :]

        class_position_x2s_repeated_nxn = class_target_x2s_for_anchors_nx1.repeat(1, num_of_classes_for_positive_anchors)
        class_position_x1s_repeated_nxn = class_target_x1s_for_anchors_1xn.repeat(num_of_classes_for_positive_anchors, 1)

        class_x2_and_x1_incidence_matrix_nxn = class_position_x2s_repeated_nxn == class_position_x1s_repeated_nxn

        num_incidences_between_beats = class_x2_and_x1_incidence_matrix_nxn.sum() # These can also be downbeats

        if num_incidences_between_beats == 0:
            return torch.tensor(0).float().to(num_incidences_between_beats.device)

        class_x2_and_x1_discrepancy_error_nxn = torch.square(
            (class_pred_x2s_for_anchors_nx1 - class_pred_x1s_for_anchors_1xn) / torch.clamp(effective_audio_length, min=1.0)
        )

        class_x2_and_x1_discrepancy_error_nxn *= class_x2_and_x1_incidence_matrix_nxn

        class_x2_and_x1_loss = class_x2_and_x1_discrepancy_error_nxn.sum()

        return class_x2_and_x1_loss

    def forward(
        self,
        jth_classification_targets,
        jth_regression_pred,
        jth_regression_targets,
        jth_positive_anchor_points,
        jth_positive_anchor_strides,
        jth_annotations
    ):
        # With the classification targets, we can easily figure out what anchor corresponds to what box type
        # If jth_classification_targets[:, 0] is 1, the corresponding anchor is associated with a downbeat
        # If jth_classification_targets[:, 1] is 1, the corresponding anchor is associated with a beat

        boolean_indices_to_downbeats_for_positive_anchors = jth_classification_targets[:, 0] == 1
        boolean_indices_to_beats_for_positive_anchors = jth_classification_targets[:, 1] == 1

        downbeat_lengths = jth_annotations[jth_annotations[:, 2] == 0, 1] - jth_annotations[jth_annotations[:, 2] == 0, 0]
        beat_lengths = jth_annotations[jth_annotations[:, 2] == 1, 1] - jth_annotations[jth_annotations[:, 2] == 1, 0]

        max_downbeat_length = torch.max(downbeat_lengths)
        max_beat_length = torch.max(beat_lengths)

        first_downbeat = torch.min(jth_annotations[jth_annotations[:, 2] == 0, 0])
        last_downbeat = torch.max(jth_annotations[jth_annotations[:, 2] == 0, 1])
        first_beat = torch.min(jth_annotations[jth_annotations[:, 2] == 1, 0])
        last_beat = torch.max(jth_annotations[jth_annotations[:, 2] == 1, 1])

        downbeat_and_beat_x1_loss_divisor = torch.max(last_beat, last_downbeat) - torch.min(first_beat, first_downbeat)
        downbeat_x2_and_x1_loss_divisor = last_downbeat - first_downbeat
        beat_x2_and_x1_loss_divisor = last_beat - first_beat

        jth_regression_targets_1xm = jth_regression_targets[None]
        jth_regression_pred_1xn = jth_regression_pred[None]

        # Given the regression prediction and targets which are in (l, r), produce (x1, x2) boxes
        # Target boxes are used to match the downbeats with their corresponding first beats
        transformed_target_regression_boxes_batch = self.anchor_point_transform(
            jth_positive_anchor_points,
            jth_regression_targets_1xm,
            jth_positive_anchor_strides
        ) # (B, num of anchors, 2) but here B is 1

        transformed_target_regression_boxes = transformed_target_regression_boxes_batch[0, :, :]

        # Prediction boxes are used to calculate the discrepancies between downbeats and corresponding first beats
        transformed_pred_regression_boxes_batch = self.anchor_point_transform(
            jth_positive_anchor_points, # ()
            jth_regression_pred_1xn,
            jth_positive_anchor_strides
        )

        transformed_pred_regression_boxes = transformed_pred_regression_boxes_batch[0, :, :]

        downbeat_and_beat_x1_loss = self.calculate_downbeat_and_beat_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_downbeats_for_positive_anchors,
            boolean_indices_to_beats_for_positive_anchors,
            downbeat_and_beat_x1_loss_divisor,
        )

        downbeat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_downbeats_for_positive_anchors,
            downbeat_x2_and_x1_loss_divisor
        )

        beat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_beats_for_positive_anchors,
            beat_x2_and_x1_loss_divisor
        )

        all_adjacency_constraint_losses = torch.stack((
            downbeat_and_beat_x1_loss,
            downbeat_x2_and_x1_loss,
            beat_x2_and_x1_loss
        ))

        return all_adjacency_constraint_losses.mean()

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

            jth_adjacency_constraint_loss = self.adjacency_constraint_loss(
                jth_classification_targets[positive_anchor_indices],
                jth_regression_pred[positive_anchor_indices],
                jth_regression_targets[positive_anchor_indices],
                all_anchor_points[positive_anchor_indices],
                strides_for_all_anchors[positive_anchor_indices],
                jth_annotations
            )

            classification_losses_batch.append(jth_classification_loss)
            regression_losses_batch.append(jth_regression_loss)
            leftness_losses_batch.append(jth_leftness_loss)
            adjacency_constraint_losses_batch.append(jth_adjacency_constraint_loss)
        # END for j in range(batch_size)

        if len(classification_losses_batch) == 0:
            classification_losses_batch.append(0)  #MJ: append zero tensor rather number 0
            
        if len(regression_losses_batch) == 0:
            regression_losses_batch.append(0)
            
        if len(leftness_losses_batch) == 0:
            leftness_losses_batch.append(0)
            
        if len(adjacency_constraint_losses_batch) == 0:
            adjacency_constraint_losses_batch.append(0)

        return \
            torch.stack(classification_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(leftness_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(adjacency_constraint_losses_batch).mean(dim=0, keepdim=True)
