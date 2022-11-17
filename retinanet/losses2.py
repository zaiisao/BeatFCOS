import numpy as np
import torch
import torch.nn as nn
from retinanet.utils import BBoxTransform, calc_iou, calc_giou, AnchorPointTransform

INF = 100000000

def get_fcos_positives(jth_annotations, anchors_list, beat_radius=1.5, downbeat_radius=4.5):
    #audio_downsampling_factor = 256
    audio_downsampling_factor = 128
    audio_target_rate = 22050 / audio_downsampling_factor

    sizes = [
        [-1, 0.45608904],
        [0.45608904, 0.878505635],
        [0.878505635, 1.557724045],
        [1.557724045, 2.264785525],
        [2.264785525, 1000],
    ]

    # Sizes were calculated as follows:
    # b_k where k = [1, 2]: beat k-means values when calculating all beat interval lengths (2 in total)
    #
    # b_1 = 0.33737285, b_2 = 0.57180136
    #
    # b_k where k = [3, 4, 5]: downbeat k-means values when calculating all downbeat interval lengths (3 in total)
    #
    # b_3 = 1.18389907, b_4 = 1.93154902, b_5 = 2.59802203
    #
    # Why 3 for downbeat and 2 for regular beat?
    # There is much more size variation for downbeat interval lengths, requiring more sizes dedicated to downbeats.
    #
    # K-means values were calculated using Ballroom and Hainsworth datasets.
    #
    # We don't want to choose the k-means values themselves as the cutoff points, but instead between clusters.
    # Objects centered around the cluster are expected to be similar to one another, so it is unproductive to set
    # the limits to the center, splitting each side to be trained on different levels despite all being similar.
    # Thus, m_k is the the middle of two clusters and calculated as follows:
    #
    # m_i = b_(i + 1)/2 - b_i/2
    # m_1 = 0.57180136/2 - 0.33737285/2 = 0.117214255
    # m_2 = 1.18389907/2 - 0.57180136/2 = 0.306048855
    # m_3 = 1.93154902/2 - 1.18389907/2 = 0.373824975
    # m_4 = 2.59802203/2 - 1.93154902/2 = 0.333236505
    #
    # [-1, b_1 + m_1]           = [-1, 0.33737285 + 0.117214255]                        = [-1, 0.45608904]
    # [b_1 + m_1, b_2 + m_2]    = [0.33737285 + 0.117214255, 0.57180136 + 0.306048855]  = [0.45608904, 0.878505635]
    # [b_2 + m_2, b_3 + m_3]    = [0.57180136 + 0.306048855, 1.18389907 + 0.373824975]  = [0.878505635, 1.557724045]
    # [b_3 + m_3, b_4 + m_4]    = [1.18389907 + 0.373824975, 1.93154902 + 0.333236505]  = [1.557724045, 2.264785525]
    # [b_4 + m_4, 1000]         = [1.93154902 + 0.333236505, 1000]                      = [2.264785525, 1000]

    # sorted_bbox_indices = (bbox_annotations_per_class[:, 1] - bbox_annotations_per_class[:, 0]).argsort()
    # print(f"sorted_bbox_indices ({sorted_bbox_indices.shape}):\n{sorted_bbox_indices}")
    # bbox_annotations_per_class = bbox_annotations_per_class[sorted_bbox_indices]
    # print(f"bbox_annotations_per_class ({bbox_annotations_per_class.shape}):\n{bbox_annotations_per_class}")

    #positive_anchor_indices = torch.zeros(0).to(jth_annotations.device)

    indices_to_bboxes_for_positive_anchors = torch.zeros(0).to(jth_annotations.device)
    assigned_annotations_for_anchors = torch.zeros(0, 3).to(jth_annotations.device)
    normalized_annotations_for_anchors = torch.zeros(0, 3).to(jth_annotations.device)
    l_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    r_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    normalized_l_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    normalized_r_star_for_anchors = torch.zeros(0).to(jth_annotations.device)
    levels_for_all_anchors = torch.zeros(0).to(jth_annotations.device)

    # strides_for_all_anchors = torch.zeros(0).to(jth_annotations.device)
    # size_of_interests_for_anchors = torch.zeros(0).to(jth_annotations.device)

    for i, anchor_points_per_level in enumerate(anchors_list):
        # stride_per_level = torch.tensor(2**(i + 1)).to(strides_for_all_anchors.device)
        # stride_for_anchors_per_level = stride_per_level[None].expand(anchor_points_per_level.size(dim=0))
        # strides_for_anchors = torch.cat((strides_for_all_anchors, stride_for_anchors_per_level), dim=0)

        # size_of_interest_per_level = anchor_points_per_level.new_tensor([sizes[i][0] * audio_target_rate, sizes[i][1] * audio_target_rate])
        # size_of_interest_for_anchors_per_level = size_of_interest_per_level[None].expand(anchor_points_per_level.size(dim=0), -1)
        # size_of_interests_for_anchors = torch.cat((size_of_interests_for_anchors, size_of_interest_for_anchors_per_level), dim=0)

    # # anchors_list contains the anchor points (x, y) on the base level image corresponding to the feature map
    # for i, anchor_points_per_level in enumerate(anchors_list): # i is the feature map level which should start from 0 but the real level starts from 1

      # for debugging: 
        # The original version: anchor_points_in_gt_bboxes = torch.logical_and(
        #     torch.ge(torch.unsqueeze(anchor_points_per_level, dim=0), torch.unsqueeze(bbox_annotations_per_class[:, 0], dim=1)),
        #     torch.le(torch.unsqueeze(anchor_points_per_level, dim=0), torch.unsqueeze(bbox_annotations_per_class[:, 1], dim=1))
        # )

        # l_star_to_bboxes_for_anchors = torch.unsqueeze(anchor_points_per_level, dim=0) - torch.unsqueeze(bbox_annotations_per_class[:, 0], dim=1)
        # r_star_to_bboxes_for_anchors = torch.unsqueeze(bbox_annotations_per_class[:, 1], dim=1) - torch.unsqueeze(anchor_points_per_level, dim=0)
        #print(f"torch.unsqueeze(anchor_points_per_level, dim=0):\n{torch.unsqueeze(anchor_points_per_level, dim=0)}")
        #print(f"torch.unsqueeze(bbox_annotations_per_class[:, 0], dim=1):\n{torch.unsqueeze(bbox_annotations_per_class[:, 0], dim=1)}")
        #print(f"l_star_to_bboxes_for_anchors ({l_star_to_bboxes_for_anchors.shape}):\n{l_star_to_bboxes_for_anchors}")

        #The new verrsion by MJ: 
        levels_per_level = torch.ones(anchor_points_per_level.shape).to(anchor_points_per_level.device) * (i + 1)
        
        anchor_points_per_level_nx1 = torch.unsqueeze(anchor_points_per_level, dim=1)  # shape = (N,1)
        l_annotations_1xm = torch.unsqueeze(jth_annotations[:, 0], dim=0)  # shape = (1,M)
        r_annotations_1xm = torch.unsqueeze(jth_annotations[:, 1], dim=0)   # shape = (1,M)

        # is_anchor_points_in_gt_bboxes_per_level = torch.logical_and(
        #     torch.ge( anchor_points_per_level_nx1,  l_annotations_per_class_1xm),
        #     torch.le( anchor_points_per_level_nx1,  r_annotations_per_class_1xm)
        # )

        # JA: New radius implementation
        stride = 2**(i + 1)
        radius_per_class = (jth_annotations[:, 2] == 0) * downbeat_radius + (jth_annotations[:, 2] == 1) * beat_radius
        radius_limits_from_l_annotations = l_annotations_1xm + (radius_per_class * stride)

        is_anchor_points_in_radii_per_level = torch.logical_and(
            torch.ge( anchor_points_per_level_nx1, l_annotations_1xm),
            torch.le( anchor_points_per_level_nx1, torch.minimum(r_annotations_1xm, radius_limits_from_l_annotations))
        )
        #is_anchor_points_in_gt_bboxes : shape =(N,M)
        l_stars_to_bboxes_for_anchors_per_level =  anchor_points_per_level_nx1 - l_annotations_1xm
        r_stars_to_bboxes_for_anchors_per_level =  r_annotations_1xm - anchor_points_per_level_nx1

        # Use dict to display the results for anchors:
        # https://hckcksrl.medium.com/python-zip-%EB%82%B4%EC%9E%A5%ED%95%A8%EC%88%98-95ad2997990
        # https://stackoverflow.com/questions/55486631/iterate-over-two-pytorch-tensors-at-once
        # https://jungnamgyu.tistory.com/53 np., torch.set_printoptions( precision =2) 
        # dic ={}
        # for anchor, anchor_data  in zip(anchor_points_per_level_nx1, l_stars_to_bboxes_for_anchors_per_level ):
        #   dic[anchor] = anchor_data

        # #print(f"l_stars_to_bboxes_for_anchors_per_level:\n")
        # #for key, value in dic.items():
        # #  print(key, ':', value)
        # #print(f"r_star_to_bboxes_for_anchors_per_level :\n {  r_star_to_bboxes_for_anchors_per_level}")

        # dic ={}
        # for anchor, anchor_data  in zip(anchor_points_per_level_nx1, r_stars_to_bboxes_for_anchors_per_level ):
        #   dic[anchor] = anchor_data

        #print(f"r_stars_to_bboxes_for_anchors_per_level:\n")
        #for key, value in dic.items():
        #  print(key, ':', value)
      
        #lower_size = sizes[i][0] * audio_target_rate
        #upper_size = sizes[i][1] * audio_target_rate

        size_of_interest_per_level = anchor_points_per_level.new_tensor([sizes[i][0] * audio_target_rate, sizes[i][1] * audio_target_rate])
        size_of_interest_for_anchors_per_level = size_of_interest_per_level[None].expand(anchor_points_per_level.size(dim=0), -1)

        #print(f"lower_size for level {i}: {lower_size}")
        #print(f"upper_size for level {i}: {upper_size}")

        # torch.set_printoptions(edgeitems=10000000, sci_mode=False)
        #print(f"l_stars_to_bboxes_for_anchors_per_level {l_stars_to_bboxes_for_anchors_per_level.shape}:\n{l_stars_to_bboxes_for_anchors_per_level}")
        #max_offsets_to_regress_for_anchor_points = torch.maximum(l_stars_to_bboxes_for_anchors_per_level, r_stars_to_bboxes_for_anchors_per_level)
        #print(f"max_offsets_to_regress_for_anchor_points {max_offsets_to_regress_for_anchor_points.shape}:\n{max_offsets_to_regress_for_anchor_points}")
        # is_anchor_points_within_bbox_range_per_level shape is (N, M)

        # Put L and R stars into a single tensor so that we can calculate max
        bbox_l_r_targets_per_image = torch.stack([l_stars_to_bboxes_for_anchors_per_level, r_stars_to_bboxes_for_anchors_per_level], dim=2)
        max_bbox_targets_per_image, _ = bbox_l_r_targets_per_image.max(dim=2)
        # print(f"bbox_l_r_targets_per_image ({bbox_l_r_targets_per_image.shape}):\n{bbox_l_r_targets_per_image}")

        # is_anchor_points_within_bbox_range_per_level = torch.logical_and(
        #     max_offsets_to_regress_for_anchor_points >= lower_size,
        #     max_offsets_to_regress_for_anchor_points < upper_size
        # )

        is_anchor_points_within_bbox_range_per_level = (
            max_bbox_targets_per_image >= size_of_interest_for_anchors_per_level[:, [0]]
        ) & (max_bbox_targets_per_image <= size_of_interest_for_anchors_per_level[:, [1]])
        #print(f"is_anchor_points_within_bbox_range_per_level ({is_anchor_points_within_bbox_range_per_level.shape}):\n{is_anchor_points_within_bbox_range_per_level}")

        # print(f" max_offsets_to_regress_for_anchor_points (with level i=0 only):\n { max_offsets_to_regress_for_anchor_points}")
        #print(f" is_anchor_points_within_bbox_range_per_level (with level i=0 only):\n { is_anchor_points_within_bbox_range_per_level}")
        # positive_argmax_per_level are the first indices to the positive anchors which are both in gt boxes and satisfy the bbox size limits
        # If there are multiple maximal values in a reduced row then the indices of the first maximal value are returned.
        # positive_anchor_indices_per_level, positive_argmax_per_level = torch.logical_and(
        #     is_anchor_points_in_gt_bboxes,
        #     is_anchor_points_within_bbox_range_per_level
        # ).max(dim=0)
        #print(f"is_anchor_points_in_radii_per_level.nonzero() on level {i} for class {class_id} {is_anchor_points_in_radii_per_level.nonzero().shape}/{is_anchor_points_in_radii_per_level.shape}")#:\n{is_anchor_points_in_radii_per_level.nonzero()}")
        #print(f"is_anchor_points_within_bbox_range_per_level.nonzero() on level {i} for class {class_id} {is_anchor_points_within_bbox_range_per_level.nonzero()}")#:\n{is_anchor_points_within_bbox_range_per_level.nonzero()}")

        # #A new version by MJ: 
        # boolean_indices_to_bboxes_for_anchors_per_level = torch.logical_and(
        #      #is_anchor_points_in_gt_bboxes_per_level,
        #      is_anchor_points_in_radii_per_level,
        #      is_anchor_points_within_bbox_range_per_level
        # )
        #print(f"boolean_indices_to_bboxes_for_anchors_per_level.nonzero() on level {i} for class {class_id} {boolean_indices_to_bboxes_for_anchors_per_level.nonzero().shape}")#:\n{boolean_indices_to_bboxes_for_anchors_per_level.nonzero()}")
        #torch.set_printoptions(edgeitems=3)
        #boolean_indices_to_bboxes_for_anchors_per_level: shape = (N,M)
       
        #print(f"is_anchor_points_in_gt_bboxes_per_level:\n {is_anchor_points_in_gt_bboxes_per_level}")
        #torch.set_printoptions(edgeitems=10000000, sci_mode=False)
        #print(f" is_anchor_points_within_bbox_range_per_level {is_anchor_points_within_bbox_range_per_level.nonzero().shape}:\n{is_anchor_points_within_bbox_range_per_level.nonzero()}")
        #torch.set_printoptions(edgeitems=3, sci_mode=True)
        #print(f"boolean_indices_to_bboxes_for_anchors_per_level:\n {boolean_indices_to_bboxes_for_anchors_per_level}")
        areas_of_bboxes = jth_annotations[:, 1] - jth_annotations[:, 0]

        gt_area_for_anchors_matrix = areas_of_bboxes[None].repeat(len(anchor_points_per_level), 1)
        gt_area_for_anchors_matrix[is_anchor_points_in_radii_per_level == 0] = INF
        gt_area_for_anchors_matrix[is_anchor_points_within_bbox_range_per_level == 0] = INF

        min_areas_for_anchors, argmin_boolean_indices_to_bboxes_for_anchors = gt_area_for_anchors_matrix.min(1)
        # torch.set_printoptions(edgeitems=100000000, sci_mode=False)
        # print(f"areas_of_bboxes:\n{areas_of_bboxes}")
        #print(f"is_anchor_points_in_radii_per_level, level {i}, class {class_id}: {is_anchor_points_in_radii_per_level.nonzero()}")
        #print(f"is_anchor_points_within_bbox_range_per_level, level {i}, class {class_id}: {is_anchor_points_within_bbox_range_per_level.nonzero()}")
        #print(f"min_areas_for_anchors == INF, level {i}, class {class_id}: {(min_areas_for_anchors == INF)}")
        # torch.set_printoptions(edgeitems=3, sci_mode=True)

        #assigned_annotations_for_anchors_per_level = annotations_per_class[argmax_boolean_indices_to_bboxes_for_anchors]

        assigned_annotations_for_anchors_per_level = jth_annotations[argmin_boolean_indices_to_bboxes_for_anchors] # Among the annotations, choose the ones associated with box with minimum area
        assigned_annotations_for_anchors_per_level[min_areas_for_anchors == INF, 2] = 0 # Assigned background class label to the anchor boxes whose min area with respect to the bboxes is INF

        #The new version:

        #indices_to_bboxes_for_positive_anchors_per_level, argmax_boolean_indices_to_bboxes_for_anchors = \
        #    torch.max(boolean_indices_to_bboxes_for_anchors_per_level, dim=1)
        
        #print(f"positivity_indicators_of_bboxes_for_anchors: \n{positivity_indicators_of_bboxes_for_anchors}")
        #print(f"argmax_positivity_indicators_of_bboxes_for_anchors: \n{argmax_positivity_indicators_of_bboxes_for_anchors}")

        #assigned_annotations_for_anchors_per_level = annotations_per_class[argmax_boolean_indices_to_bboxes_for_anchors]

        # torch.gt(boolean_indices_to_anchors_per_level, False, dim=1)

        #print(f"annotations_per_class: \n{annotations_per_class}")
        #print(f" assigned_annotations_for_anchors_per_level:\n { assigned_annotations_for_anchors_per_level}") 

        #indices_to_bboxes_for_positive_anchors_per_level =  torch.gt(max_boolean_indices_to_bboxes_for_anchors, False)

        #print(f"indices_to_bboxes_for_positive_anchors_per_level\n {indices_to_bboxes_for_positive_anchors_per_level}")

        # We don't need this step because we already get the annotations for the positive anchors in the main routine
        #annotations_for_positive_anchors_per_level =  assigned_annotations_for_anchors_per_level[ indices_to_bboxes_for_positive_anchors_per_level]

        #print(f"annotations_for_positive_anchors_per_level:\n {annotations_for_positive_anchors_per_level}")

        #normalized_annotations_for_positive_anchors_per_level =  annotations_for_positive_anchors_per_level[:, 0:2] / 2**i

        #normalized_annotations_for_positive_anchors_per_level =  annotations_for_positive_anchors_per_level
        normalized_annotations_for_anchors_per_level = torch.clone(assigned_annotations_for_anchors_per_level)
        normalized_annotations_for_anchors_per_level[:,0] /= stride
        normalized_annotations_for_anchors_per_level[:,1] /= stride
      #  annotations_for_positive_anchors_per_level: shape = (N, 3), where bbox_annotations_per_class[i,0]= x1,  bbox_annotations_per_class[i,1] = x2,  bbox_annotations_per_class[i,2]= class label

        #print(f"normalized_annotations_for_positive_anchors_per_level: \n{normalized_annotations_for_positive_anchors_per_level}")

      
        # MJ:  # get the l_star values of the bboxes for the positive anchors

        l_stars_for_anchors_per_level = l_stars_to_bboxes_for_anchors_per_level[
            torch.arange(0, anchor_points_per_level.size(dim=0)),
            argmin_boolean_indices_to_bboxes_for_anchors
        ]

        r_stars_for_anchors_per_level = r_stars_to_bboxes_for_anchors_per_level[
            torch.arange(0, anchor_points_per_level.size(dim=0)),
            argmin_boolean_indices_to_bboxes_for_anchors
        ]
        # torch.set_printoptions(edgeitems=10000000, sci_mode=False)
        # print(f'positive l_stars_for_anchors_per_level on level {i} for class {class_id} ({l_stars_for_anchors_per_level[indices_to_bboxes_for_positive_anchors_per_level].shape}) = \n { l_stars_for_anchors_per_level[indices_to_bboxes_for_positive_anchors_per_level] }')
        # print(f'positive r_stars_for_anchors_per_level on level {i} for class {class_id} ({r_stars_for_anchors_per_level[indices_to_bboxes_for_positive_anchors_per_level].shape}) = \n { l_stars_for_anchors_per_level[indices_to_bboxes_for_positive_anchors_per_level] }')
        # torch.set_printoptions(edgeitems=3, sci_mode=False)

        normalized_l_star_for_anchors_per_level = l_stars_for_anchors_per_level / stride
        normalized_r_star_for_anchors_per_level = r_stars_for_anchors_per_level / stride

        indices_to_bboxes_for_positive_anchors = torch.cat(( indices_to_bboxes_for_positive_anchors, argmin_boolean_indices_to_bboxes_for_anchors), dim=0)
        assigned_annotations_for_anchors = torch.cat((assigned_annotations_for_anchors, assigned_annotations_for_anchors_per_level), dim=0)
        normalized_annotations_for_anchors = torch.cat((normalized_annotations_for_anchors, normalized_annotations_for_anchors_per_level), dim=0)
        l_star_for_anchors = torch.cat((l_star_for_anchors, l_stars_for_anchors_per_level))
        r_star_for_anchors = torch.cat((r_star_for_anchors, r_stars_for_anchors_per_level ))
        normalized_l_star_for_anchors = torch.cat((normalized_l_star_for_anchors, normalized_l_star_for_anchors_per_level))
        normalized_r_star_for_anchors = torch.cat((normalized_r_star_for_anchors, normalized_r_star_for_anchors_per_level))
        levels_for_all_anchors = torch.cat((levels_for_all_anchors, levels_per_level))

    indices_to_bboxes_for_positive_anchors = indices_to_bboxes_for_positive_anchors.bool()

    return indices_to_bboxes_for_positive_anchors,\
        assigned_annotations_for_anchors, normalized_annotations_for_anchors,\
        l_star_for_anchors, r_star_for_anchors,\
        normalized_l_star_for_anchors, normalized_r_star_for_anchors, levels_for_all_anchors

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, jth_classification_pred, jth_classification_targets, jth_annotations):
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

        #return cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0)
        return cls_loss.sum()

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

        bce = -(jth_leftness_targets * torch.log(jth_leftness_pred)\
            + (1.0 - jth_leftness_targets) * torch.log(1.0 - jth_leftness_pred))

        leftness_loss = bce

        return leftness_loss.mean()

#   total_downbeat_beat_adjacency_loss = 0
#   for each downbeat target d:
#      for each positive anchor point a1 assigned to downbeat d:
#          left_pos_of_downbeat_d_for_a1 = a1 - (predicted l of a1)
#          
#          for each anchor point a2 to the first beat b of downbeat d:
#              left_pos_of_beat_b_for_a2 = a2 - (predicted l of a2)
#              adjacency_loss_of_a1 = (left_pos_of_downbeat_d_for_a1 - left_pos_of_beat_b_for_a2)**2
#              adjacency_loss_of_a2 = (left_pos_of_beat_b_for_a2 - left_pos_of_downbeat_d_for_a1)**2
#              total_adjacency_loss += adjacency_loss_of_a1
#              total_adjacency_loss += adjacency_loss_of_a2
#
#   weighted_downbeat_beat_adjacency_loss = total_downbeat_beat_adjacency_loss / total_number_of_positive_anchors
#   return weighted_downbeat_beat_adjacency_loss



#   total_downbeat_adjacency_loss = 0
#   for each downbeat target d1:
#      for each positive anchor point a1 assigned to downbeat d1:
#          right_pos_of_downbeat_d1_for_a1 = a1 + (predicted r of a1)
#          
#          for each positive anchor point a2 assigned to downbeat d2:
#              left_pos_of_downbeat_d2_for_a2 = a2 - (predicted l of a2)
#              adjacency_loss_of_a1 = (right_pos_of_downbeat_d1_for_a1 - left_pos_of_downbeat_d2_for_a2)**2
#              adjacency_loss_of_a2 = (left_pos_of_downbeat_d2_for_a2 - right_pos_of_downbeat_d1_for_a1)**2
#              total_adjacency_loss += adjacency_loss_of_a1
#              total_adjacency_loss += adjacency_loss_of_a2
#
#   weighted_downbeat_adjacency_loss = total_downbeat_adjacency_loss / total_number_of_positive_anchors
#   return weighted_downbeat_adjacency_loss


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
        # print(f"num_incidences_between_downbeats_and_beats {num_incidences_between_downbeats_and_beats}")

        # Calculate the mean square error between all the downbeat prediction x1 and beat prediction x1
        # and multiply this (D, B) result matrix with the incidence matrix to remove all values where
        # the downbeat does not correspond with the beat
        downbeat_and_beat_x1_discrepancy_error_N1xN2 = (downbeat_pred_x1s_for_anchors_N1x1 - beat_pred_x1s_for_anchors_1xN2) ** 2
        downbeat_and_beat_x1_discrepancy_error_N1xN2 *= downbeat_and_beat_x1_incidence_matrix_N1xN2

        downbeat_and_beat_x1_loss = torch.sqrt(
            downbeat_and_beat_x1_discrepancy_error_N1xN2.sum() / torch.clamp(num_incidences_between_downbeats_and_beats, min=1.0)
        )

        return downbeat_and_beat_x1_loss

    def calculate_x2_and_x1_loss(
        self,
        transformed_target_regression_boxes,
        transformed_pred_regression_boxes,
        boolean_indices_to_classes_for_positive_anchors
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
        num_incidences_between_beats = class_x2_and_x1_incidence_matrix_nxn.sum() # These beats can also be downbeats
        # print(f"num_incidences_between_beats {num_incidences_between_beats}")

        class_x2_and_x1_discrepancy_error_nxn = (class_pred_x2s_for_anchors_nx1 - class_pred_x1s_for_anchors_1xn) ** 2
        class_x2_and_x1_discrepancy_error_nxn *= class_x2_and_x1_incidence_matrix_nxn

        class_x2_and_x1_loss = torch.sqrt(
            class_x2_and_x1_discrepancy_error_nxn.sum() / torch.clamp(num_incidences_between_beats, min=1.0)
        )

        return class_x2_and_x1_loss

    def forward(
        self,
        jth_classification_targets,
        jth_regression_pred,
        jth_regression_targets,
        jth_positive_anchor_points,
        jth_positive_anchor_strides
    ):
        # With the classification targets, we can easily figure out what anchor corresponds to what box type
        # If jth_classification_targets[:, 0] is 1, the corresponding anchor is associated with a downbeat
        # If jth_classification_targets[:, 1] is 1, the corresponding anchor is associated with a beat

        boolean_indices_to_downbeats_for_positive_anchors = jth_classification_targets[:, 0] == 1
        boolean_indices_to_beats_for_positive_anchors = jth_classification_targets[:, 1] == 1

        # print(f"this should be 0: {(boolean_indices_to_downbeats_for_positive_anchors * boolean_indices_to_beats_for_positive_anchors).sum()}")

        # Given the regression prediction and targets which are in (l, r), produce (x1, x2) boxes
        # Target boxes are used to match the downbeats with their corresponding first beats
        transformed_target_regression_boxes = self.anchor_point_transform(
            jth_positive_anchor_points,
            jth_regression_targets[None],
            jth_positive_anchor_strides
        ).squeeze(dim=0)

        # Prediction boxes are used to calculate the discrepancies between downbeats and corresponding first beats
        transformed_pred_regression_boxes = self.anchor_point_transform(
            jth_positive_anchor_points,
            jth_regression_pred[None],
            jth_positive_anchor_strides
        ).squeeze(dim=0)

        downbeat_and_beat_x1_loss = self.calculate_downbeat_and_beat_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_downbeats_for_positive_anchors,
            boolean_indices_to_beats_for_positive_anchors,
        )

        downbeat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_downbeats_for_positive_anchors
        )

        beat_x2_and_x1_loss = self.calculate_x2_and_x1_loss(
            transformed_target_regression_boxes,
            transformed_pred_regression_boxes,
            boolean_indices_to_beats_for_positive_anchors
        )

        all_adjacency_constraint_losses = torch.stack((downbeat_and_beat_x1_loss, downbeat_x2_and_x1_loss, beat_x2_and_x1_loss))

        return all_adjacency_constraint_losses.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()

        self.classification_loss = FocalLoss()
        self.regression_loss = RegressionLoss()
        self.leftness_loss = LeftnessLoss()
        self.adjacency_constraint_loss = AdjacencyConstraintLoss()

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

            jth_padded_annotations = annotations[j, :, :]
            jth_annotations = jth_padded_annotations[jth_padded_annotations[:, 2] != -1]

            positive_anchor_indices, assigned_annotations, normalized_annotations, \
            l_star, r_star, normalized_l_star, normalized_r_star, levels = \
                get_fcos_positives(jth_annotations, anchors_list)

            all_anchor_points = torch.cat(anchors_list, dim=0)
            num_positive_anchors = positive_anchor_indices.sum()

            jth_classification_targets, jth_regression_targets, jth_leftness_targets = self.get_jth_targets(
                jth_classification_pred, jth_regression_pred, jth_leftness_pred,
                positive_anchor_indices, normalized_annotations,
                l_star, r_star,
                normalized_l_star, normalized_r_star
            )

            jth_classification_loss = self.classification_loss(
                jth_classification_pred,
                jth_classification_targets,
                jth_annotations
            ) / num_positive_anchors

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

            strides_for_all_anchors = torch.zeros(0).to(classifications.device)
            for i, anchors_per_level in enumerate(anchors_list):
                stride_per_level = torch.tensor(2**(i + 1)).to(strides_for_all_anchors.device)
                stride_for_anchors_per_level = stride_per_level[None].expand(anchors_per_level.size(dim=0))
                strides_for_all_anchors = torch.cat((strides_for_all_anchors, stride_for_anchors_per_level), dim=0)

            jth_adjacency_constraint_loss = self.adjacency_constraint_loss(
                jth_classification_targets[positive_anchor_indices],
                jth_regression_pred[positive_anchor_indices],
                jth_regression_targets[positive_anchor_indices],
                all_anchor_points[positive_anchor_indices],
                strides_for_all_anchors[positive_anchor_indices]
            )

            # torch.set_printoptions(sci_mode=False, edgeitems=100000000, linewidth=10000)
            # concatenated_output = torch.cat((
            #     jth_classification_targets[positive_anchor_indices],
            #     jth_regression_targets[positive_anchor_indices],
            #     assigned_annotations[positive_anchor_indices, :2],
            #     jth_classification_targets.nonzero()[:, None, 0],
            #     all_anchor_points[positive_anchor_indices, None],
            #     levels[positive_anchor_indices, None]
            # ), dim=1)
            # print(f"concatenated_output {concatenated_output.shape}:\n{concatenated_output}")
            # torch.set_printoptions(sci_mode=True, edgeitems=3)

            classification_losses_batch.append(jth_classification_loss)
            regression_losses_batch.append(jth_regression_loss)
            leftness_losses_batch.append(jth_leftness_loss)
            adjacency_constraint_losses_batch.append(jth_adjacency_constraint_loss)

        return \
            torch.stack(classification_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(leftness_losses_batch).mean(dim=0, keepdim=True), \
            torch.stack(adjacency_constraint_losses_batch).mean(dim=0, keepdim=True)
