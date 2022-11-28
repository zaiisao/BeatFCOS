from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    # def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
    def __init__(self, fcos=False): # We use base level 8
        super(Anchors, self).__init__()

        self.fcos = fcos

        #self.pyramid_levels = [8, 9, 10, 11, 12] # Actual strides we use are [2 ** 0, 2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4]
        self.pyramid_levels = [1, 2, 3, 4, 5]
        # # (1, 2, 3, 4, 5) with base_level=0. Actual strides are [2 ** 1, 2 ** 2, 2 ** 3, 2 ** 4, 2 ** 5]
        self.strides = [2 ** x for x in self.pyramid_levels]
        # self.strides = [2 ** x for x in self.pyramid_levels]

        #audio_downsampling_factor = 256
        audio_downsampling_factor = 128
        self.sizes = [x * 22050 / audio_downsampling_factor for x in [0.33906588, 0.5731122, 1.18389907, 1.93154902, 2.59802203]]

        if self.fcos:            
            #self.sizes = [0 for x in self.pyramid_levels]
            self.scales = np.array([1])
        else:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    #def forward(self, base_image):
    def forward(self, base_image_shape):
        # We need the shape of the base level image to compute the anchor points on each feature map
        #base_image_shape = base_image.shape[2:] # The base image is at the level of 2**7 stride, that is, the number of data samples is 2**14
        #print(f"base_image_shape: {base_image_shape}")
        base_image_spacial_shape = base_image_shape[2:] # The base level image shape (B, C, W) = (B, C, 2**12) during training
        #print(f"base_image_spacial_shape: {base_image_spacial_shape}")
        base_image_shape_array = np.array(base_image_spacial_shape)
        #print(f"base_image_shape_array: {base_image_shape_array}")

        # feature_map_shapes = [
        #     (base_image_shape + 2 ** (x - self.base_level) - 1) // (2 ** (x - self.base_level))
        #     for x in self.pyramid_levels
        # ]

        feature_map_shapes = [
            (base_image_shape_array + (2 ** x) - 1) // (2 ** x)
            for x in self.pyramid_levels
        ]

        all_anchors = []

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], scales=self.scales)
            shifted_anchors = shift(idx, feature_map_shapes[idx], self.strides[idx], anchors, fcos=self.fcos)
            # np.set_printoptions(edgeitems=1000000, suppress=True)
            # print(f"shifted_anchors for level {idx} ({shifted_anchors.shape}): {shifted_anchors}")
            # np.set_printoptions(edgeitems=3, suppress=False)
            #shifted_anchors_expanded = np.tile(np.expand_dims(shifted_anchors, axis=0), (2, 1, 1))

            if torch.cuda.is_available():
                #all_anchors.append(torch.from_numpy(shifted_anchors_expanded.astype(np.float32)).cuda())
                all_anchors.append(torch.from_numpy(shifted_anchors.astype(np.float32)).cuda())
            else:
                #all_anchors.append(torch.from_numpy(shifted_anchors_expanded.astype(np.float32)))
                all_anchors.append(torch.from_numpy(shifted_anchors.astype(np.float32)))

        return all_anchors

def generate_anchors(base_size=16, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 2))

    # scale base_size
    anchors[:, 1] = base_size * scales.T  

    # transform from (x_ctr, w) -> (x1, x2)
    anchors[:, :] -= np.tile(anchors[:, 1] * 0.5, (2, 1)).T

    #return anchors
    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 2))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(idx, feature_map_shapes, stride, anchors, fcos=False): # create one anchor point for each location on the feature map i
    shift_x = (np.arange(0, feature_map_shapes[0]) + 0.5) * stride # feature_map_shapes[0] is the ith feature map x resolution
    # shift_x is the x resolution of the ith feature map projected back to the base image

    shifts = np.vstack((
       shift_x.ravel(), shift_x.ravel()
    )).transpose()

    # add A anchors (1, A, 2) to
    # cell K shifts (K, 1, 2) to get
    # shift anchors (K, A, 2)
    # reshape to (K*A, 2) shifted anchors
    A = anchors.shape[0] # A is 1
    K = shifts.shape[0] # K is the resolution of the ith feature map on the base image

    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    if fcos:
        all_anchors = (all_anchors[:, 0] + all_anchors[:, 1]) / 2

    return all_anchors
