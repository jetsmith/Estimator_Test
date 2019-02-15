# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Definition of 300 VGG-based SSD network.

For hardware compatible devices, only 3x3 kernel is allowed for stable development now

Remove max_pool2d with 3x3, dilation and stride.

Remove padding in the last several layers, ps. only used to be compatible with caffe

@@ssd_vgg_300
"""
from collections import namedtuple

import tensorflow as tf

from nets import custom_layers
from nets import ssd_common
from nets.ssd_KYnet_v1 import *
from nets.ssd_KYnet_v1 import SSDNet as _SSDNet

slim = tf.contrib.slim


# =========================================================================== #
# SSD class definition.
# =========================================================================== #


class SSDNet(_SSDNet):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """
    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=2,
        no_annotation_label=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10'],
        feat_shapes=[(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)],
        anchor_size_bounds=[0.2, 0.90],
        # anchor_size_bounds=[0.20, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.)
                      ],
        # anchor_sizes=[(30., 60.),
        #               (60., 111.),
        #               (111., 162.),
        #               (162., 213.),
        #               (213., 264.),
        #               (264., 315.)],
        anchor_ratios=[[2, .5],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5, 3, 1./3],
                       [2, .5]
                      ],
        anchor_steps=[8, 16, 32, 64, 128, 256],
        anchor_offset=0.5,
        normalizations=[-1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        """Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_KYnet_v2'):
        """SSD network definition.
        """
        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)
        # Update feature shapes (try at least!)
        if update_feat_shapes:
            shapes = ssd_feat_shapes_from_net(r[0], self.params.feat_shapes)
            self.params = self.params._replace(feat_shapes=shapes)
        return r


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_multibox_layer(inputs,
                       num_classes,
                       sizes,
                       ratios=[1],
                       normalization=-1,
                       bn_normalization=False):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net, scaling=True)
    # Number of anchors.
    num_anchors = len(sizes) + len(ratios)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    net = slim.conv2d(net, 128, [3, 3], activation_fn=None,
                      scope='conv_cls_1')
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred


def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layers=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_ratios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_KYnet_v1'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_KYnet_v1', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.conv2d(inputs, 16, [3, 3], scope='conv1_1') #256
        net = slim.conv2d(inputs, 32, [3, 3], scope='conv2_1') #256
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1') #128
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2') #128
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2') #64
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 64, [3, 3], scope='conv3') #64
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3') #32
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv4') #32
        end_points['block4'] = net #32

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 256, [3, 3], scope='conv6') #32
        end_points['block6'] = net #32
        net = slim.max_pool2d(net, [2, 2], scope='pool2') #16

        # Block 7: 1x1 conv. Because the fuck.
        net = slim.conv2d(net, 256, [3, 3], scope='conv7') #16
        end_points['block7'] = net #16

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [3, 3], scope='conv1x1') #16
            net = slim.conv2d(net, 512, [3, 3], scope='conv3x3') #16
            net = slim.max_pool2d(net, [2, 2], scope='pool') #8
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [3, 3], scope='conv1x1') #8
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3') #8
            net = slim.max_pool2d(net, [2, 2], scope='pool') #4
        end_points[end_point] = net
        end_point = 'block10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1') #4
            net = slim.max_pool2d(net, [2, 2], scope='pool') #2
        end_points[end_point] = net

        # Prediction and localisations layers.
        predictions = []
        logits = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                p, l = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_sizes[i],
                                          anchor_ratios[i],
                                          normalizations[i])
            predictions.append(prediction_fn(p))
            logits.append(p)
            localisations.append(l)

        return predictions, localisations, logits, end_points
ssd_net.default_image_size = 256
