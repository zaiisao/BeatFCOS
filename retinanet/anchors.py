from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    # def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
    def __init__(self, fcos=False, base_level=0):
        super(Anchors, self).__init__()

        self.fcos = fcos
        self.base_level = base_level

        self.pyramid_levels = [8, 9, 10, 11, 12]
        self.strides = [2 ** (x - self.base_level) for x in self.pyramid_levels]
        # self.strides = [2 ** x for x in self.pyramid_levels]

        if self.fcos:
            self.sizes = [0 for x in self.pyramid_levels]
            self.scales = np.array([0])
        else:
            self.sizes = [x * 22050 / 256 for x in [0.32537674, 0.47555801, 0.64588683, 1.16883525, 2.17128976]]
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        #self.sizes = #[(0 if self.fcos else 2 ** (x - 3)) for x in self.pyramid_levels]
    # def __init__(self, fcos=False, base_level=0):
    #     super(Anchors, self).__init__()

    #     self.fcos = fcos
    #     self.base_level = base_level

    #     self.pyramid_levels = [12]
    #     self.strides = [2 ** (x - self.base_level) for x in self.pyramid_levels] # stride: 2**(12 - 8) = 2**4 = [16]
    #     # self.strides = [2 ** x for x in self.pyramid_levels]

    #     if self.fcos:
    #         self.sizes = [0 for x in self.pyramid_levels]
    #         self.scales = np.array([0])
    #     else:
    #         self.sizes = [x * 22050 / 256 for x in [2.17128976]] # 2.17128976 * 22050 / 256 = 187.01929378125 audio pixels = anchor box size
    #         # 2^12 = 4096 = downsampled size at 2^12
    #         # anchor box size at this level = x * 22050 / 4096 = 11.688705861328125 samples


    #         self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) # 1, 1.26, 1.59
    #     #self.sizes = #[(0 if self.fcos else 2 ** (x - 3)) for x in self.pyramid_levels]

    def forward(self, base_image):
        base_image_shape = base_image.shape[2:]
        base_image_shape = np.array(base_image_shape)
        # print(f"base_image_shape: {base_image_shape}") # the size of the base level image is 8192 = 2^13, level 8 is the base level

        #feature_map_shapes = [(base_image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        feature_map_shapes = [
            (base_image_shape + 2 ** (x - self.base_level) - 1) // (2 ** (x - self.base_level)) # (8192 + (2**(12 - 8)) - 1) // (2 ** (12 - 8)) = 512
                                                                                                # resolution in x = width, pixel location 수
                                                                                                # 512 location 당 three anchor boxes,
                                                                                                # reference anchor box size 187 samples in 2^8 downsampled level
            for x in self.pyramid_levels
        ]

        # 22050/sec * x sec = 512
        # x = 512/22050 = 0.023 sec
        # 2.17128976
        # raw audio is sampled 22050 hz, the length of the raw audio is 2^21 samples, downsampled by 256 to be 2^13.
        # 2^13 / 2^4 = 2^9 = 512 samples in 2^9 level
        # 512 in 2^9 level
        # 8192 = 2^13 = 2^21 / 2^8
        # 512  = 2^9  = 2^21 / 2^12
        
        # length in samples = sec * sr
        # 512 = sec * 22050

        # second of one sample at bottom level: length * 22050 / 2
        # second per sample = 1/22050 = 0.000045351473923
        # 1/2 downsampling means sampling is 22050 / 2 per seconds
        # 1/4 downsampling means sampling is 22050 / 4 per seconds
        # 1/8 downsampling means sampling is 22050 / 8 per seconds
        # 1/16 downsampling means sampling is 22050 / 16 per seconds
        # 1/2^5 downsampling means sampling is 22050 / 2^5 per seconds

        # 1/2^8 downsampling means sampling is 22050 / 2^8 per seconds
        # 22050 / 2^8 = 86.1328125 samples per seconds
        # 1 / 86.1328125 = 0.011609977324263 seconds per sample
        # 2.17128976 * 22050 / 256 = 187.01929378125 audio pixels = anchor box size
        # 0.011609977324263 sec/sample * 187 samples = 2.171065759637181 sec

        # 1/2^12 downsampling means sampling is 22050 / 2^12 per seconds
        # 22050 / 2^12 = 5.38330078125 samples per seconds
        # 1 / 5.38330078125 = 0.185759637188209 seconds per sample
        # 2.17128976 * 22050 / 256 = 187.01929378125 audio pixels = anchor box size

        # 0.185759637188209 seconds/sample * 11.688705861328125 samples = 2.171289760000004 sec
        # print(f"feature_map_shapes (tcn_layers[-3]):\n {feature_map_shapes}") # the top level feature map has 512 locations

        # if self.fcos:
        all_anchors = []

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(base_size=self.sizes[idx], scales=self.scales)
            shifted_anchors = shift(feature_map_shapes[idx], self.strides[idx], anchors)
            shifted_anchors_expanded = np.expand_dims(shifted_anchors, axis=0)

            if torch.cuda.is_available():
                all_anchors.append(torch.from_numpy(shifted_anchors_expanded.astype(np.float32)).cuda())
            else:
                all_anchors.append(torch.from_numpy(shifted_anchors_expanded.astype(np.float32)))

        return all_anchors
        # else:
        #     all_anchors = np.zeros((0, 2)).astype(np.float32)

        #     for idx, p in enumerate(self.pyramid_levels):
        #         anchors = generate_anchors(base_size=self.sizes[idx], scales=self.scales)
        #         shifted_anchors = shift(feature_map_shapes[idx], self.strides[idx], anchors)
        #         all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        #     all_anchors = np.expand_dims(all_anchors, axis=0)
        #     print(f"all_anchors shape in anchor box:\n {all_anchors.shape}") # number of total anchors is 512*3 = 1536
        #     print(f"all_anchors in anchor box:\n {all_anchors}")

        #     if torch.cuda.is_available():
        #         return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        #     else:
        #         return torch.from_numpy(all_anchors.astype(np.float32))

#def generate_anchors(base_size=16, ratios=None, scales=None):
def generate_anchors(base_size=16, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    #if ratios is None:
        #ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    #num_anchors = len(ratios) * len(scales)
    num_anchors = len(scales)

    # initialize output anchors
    #anchors = np.zeros((num_anchors, 4))
    anchors = np.zeros((num_anchors, 2))

    # scale base_size
    #anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    anchors[:, 1] = base_size * scales.T

    # compute areas of anchors
    #areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    #anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    #anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    #anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    #anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

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
    #all_anchors = np.zeros((0, 4))
    all_anchors = np.zeros((0, 2))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    #shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    #shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x = (np.arange(0, shape[0]) + 0.5) * stride

    #shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    #shift_x = np.meshgrid(shift_x)

    # shifts = np.vstack((
    #    shift_x.ravel(), shift_y.ravel(),
    #    shift_x.ravel(), shift_y.ravel()
    # )).transpose()
    shifts = np.vstack((
       shift_x.ravel(), shift_x.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    # all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    # all_anchors = all_anchors.reshape((K * A, 4))
    all_anchors = (anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 2))

    return all_anchors

