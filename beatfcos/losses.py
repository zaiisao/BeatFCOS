import numpy as np
import torch
import torch.nn as nn
from beatfcos.utils import calc_iou, calc_giou, AnchorPointTransform
import wandb

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
        transformed_target_regression_boxes,
        transformed_pred_regression_boxes,
        positive_downbeat_anchor_mask,
        positive_beat_anchor_mask
    ):
        downbeat_targets_x1 = transformed_target_regression_boxes[positive_downbeat_anchor_mask, 0]
        downbeat_targets_x2 = transformed_target_regression_boxes[positive_downbeat_anchor_mask, 1]
        downbeat_preds_x1   = transformed_pred_regression_boxes[positive_downbeat_anchor_mask, 0]

        beat_targets_x1 = transformed_target_regression_boxes[positive_beat_anchor_mask, 0]
        beat_targets_x2 = transformed_target_regression_boxes[positive_beat_anchor_mask, 1]
        beat_preds_x1   = transformed_pred_regression_boxes[positive_beat_anchor_mask, 0]

        downbeat_lengths = downbeat_targets_x2 - downbeat_targets_x1
        beat_lengths     = beat_targets_x2 - beat_targets_x1

        loss = self._normalized_incidence_loss(
            target_a=downbeat_targets_x1,
            target_b=beat_targets_x1,
            pred_a=downbeat_preds_x1,
            pred_b=beat_preds_x1,
            length_a=downbeat_lengths,
            length_b=beat_lengths
        )
        return loss

    def calculate_x2_and_x1_loss(
        self,
        transformed_target_regression_boxes,
        transformed_pred_regression_boxes,
        positive_class_anchor_mask
    ):
        class_targets_x2 = transformed_target_regression_boxes[positive_class_anchor_mask, 1]
        class_targets_x1 = transformed_target_regression_boxes[positive_class_anchor_mask, 0]
        class_preds_x2   = transformed_pred_regression_boxes[positive_class_anchor_mask, 1]
        class_preds_x1   = transformed_pred_regression_boxes[positive_class_anchor_mask, 0]

        class_lengths = class_targets_x2 - class_targets_x1

        loss = self._normalized_incidence_loss(
            target_a=class_targets_x2,
            target_b=class_targets_x1,
            pred_a=class_preds_x2,
            pred_b=class_preds_x1,
            length_a=class_lengths,
            length_b=class_lengths
        )
        return loss

    def forward(
        self, jth_classification_targets,
        jth_regression_pred, jth_regression_targets,
        jth_positive_anchor_points, jth_positive_anchor_strides,
        jth_annotations, num_positive_anchors
    ):
        # JA: Of all positive anchors, create a boolean mask for both downbeats and beats
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

        # Return the loss normalized by the number of positive anchors (clamped to a minimum of 1)
        normalized_total_loss = total_loss / torch.clamp(num_positive_anchors.float(), min=1.0)
        return normalized_total_loss, individual_losses

    def _normalized_incidence_loss(
        self,
        target_a: torch.Tensor, target_b: torch.Tensor,
        pred_a: torch.Tensor, pred_b: torch.Tensor,
        length_a: torch.Tensor, length_b: torch.Tensor
    ):
        """
        Computes the normalized discrepancy loss between two sets of anchor predictions based on their targets.
        
        In our application, this helper function is used to compute a loss for pairs of anchors where the 
        corresponding target anchor positions are expected to match (e.g. the x1 coordinate for downbeats should
        match the x1 coordinate for the corresponding beats, or the class anchor's x2 should equal its x1 in some cases).
        
        The function works as follows:
        - It takes two groups (A and B) of anchors, each represented by a target scalar (e.g. a coordinate value)
            and a predicted scalar, plus a length computed from the regression targets (typically the difference 
            between the x2 and x1 coordinates).
        - The tensors for the targets and lengths are broadcasted (by reshaping and repeating) into matrices
            so that every target from group A is paired with every target from group B.
        - An incidence mask is generated using the provided equality incidence matrix to indicate which anchor pairs
            are supposed to correspond.
        - For each pair, the squared difference between their two predictions is computed.
        - This squared error is normalized by the maximum of the two corresponding target lengths. Normalizing by 
            the max length means that errors are scaled relative to the size of the underlying regression boxes, which
            is crucial when target boxes vary in size.
        - Finally, the function sums up these normalized errors over all pairs that satisfy the incidence condition.
        
        Parameters
        ----------
        target_a : torch.Tensor
            A 1D tensor (shape: [N1]) containing the target coordinate (e.g. x1 value) for anchor group A 
            (e.g. downbeat anchors or class anchors' x2 values).
        target_b : torch.Tensor
            A 1D tensor (shape: [N2]) containing the target coordinate for anchor group B 
            (e.g. beat anchors or class anchors' x1 values).
        pred_a : torch.Tensor
            A 1D tensor (shape: [N1]) of predicted coordinate values corresponding to target_a.
        pred_b : torch.Tensor
            A 1D tensor (shape: [N2]) of predicted coordinate values corresponding to target_b.
        length_a : torch.Tensor
            A 1D tensor (shape: [N1]) representing a measure of scale for group A anchors (typically computed 
            as x2 - x1 from the target regression box), used for normalizing the error.
        length_b : torch.Tensor
            A 1D tensor (shape: [N2]) representing the analogous scale for group B anchors.

        Returns
        -------
        torch.Tensor
            A scalar tensor representing the summed normalized loss over all pairs where the targets are incident.
            If no incident pairs are found, returns a tensor with value 0.0.
        """
        # target_a: shape (N1,), target_b: shape (N2,)
        N1 = target_a.shape[0]
        N2 = target_b.shape[0]

        # JA: To compare every anchor in group A with every anchor in group B, we first reshape the 1D tensors
        # to 2D by adding a new axis. This will enable proper broadcasting when we later repeat them, which is
        # done for the purpose of creating a N1xN2 incidence matrix
        target_a_n1x1 = target_a[:, None]
        target_b_1xn2 = target_b[None, :]
        pred_a_n1x1 = pred_a[:, None]
        pred_b_1xn2 = pred_b[None, :]
        length_a_n1x1 = length_a[:, None]
        length_b_1xn2 = length_b[None, :]

        # Repeat to create matrices (N1, N2)
        repeated_target_a = target_a_n1x1.repeat(1, N2)
        repeated_target_b = target_b_1xn2.repeat(N1, 1)
        repeated_length_a = length_a_n1x1.repeat(1, N2)
        repeated_length_b = length_b_1xn2.repeat(N1, 1)

        # JA: max_length_matrix which is also of shape N1xN2 is made to normalize the results of the error
        # terms by whichever box is longer. This is necessary to prevent the adjacency constraint loss from
        # becoming too large
        max_length_matrix = torch.max(repeated_length_a, repeated_length_b)

        # JA: Compute the incidence matrix using the repeated target tensors by setting equal values to True
        # and non-equal values to False. This behaves as a mask, as values set to True are values whose indices
        # correspond to values in the broadcasted matrix will be counted in the final loss value. For example,
        # in the beat-beat constraint, this incidence matrix matches each target box with other target boxes if
        # their left coordinate matches its right coordinate.
        incidence_matrix = repeated_target_a == repeated_target_b

        # If there are no incidences, return zero loss (avoid division-by-zero issues)
        if incidence_matrix.sum() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=incidence_matrix.device)

        # Compute the squared error between predictions (using broadcasting)
        sq_error = torch.square(pred_a_n1x1 - pred_b_1xn2)

        # Apply the incidence mask and normalize by the maximum length, then sum all errors
        loss = (sq_error * incidence_matrix) / max_length_matrix
        return loss.sum()

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

    def forward(self, classifications, regressions, leftnesses, anchors_list, annotations, step=None):
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

        print(f"DEBUG | ADJ_DB: {adj_db} | ADJ_DD: {adj_bb} | ADJ_BB: {adj_dd}")
        wandb.log({"adj_db": adj_db, "adj_bb": adj_bb, "adj_dd": adj_dd}, step=step)

        return \
            torch.stack(classification_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(leftness_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(adjacency_constraint_losses_batch).mean(dim=0, keepdim=True)
