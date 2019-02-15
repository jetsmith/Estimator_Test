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

For hardware compatible devices, only 3x3 kernel is allowed for stable
development now

Remove max_pool2d with 3x3, dilation and stride.

Remove padding in the last several layers, ps. only used to be compatible
with caffe

@@ssd_vgg_300
"""
import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets.ssd_KYnet_v2 import *
from nets.ssd_KYnet_v2 import SSDNet as _SSDNet

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
        img_shape=(256, 256),
        num_classes=2,
        no_annotation_label=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10'],
        feat_shapes=[(32, 32), (16, 16), (8, 8), (4, 4), (2, 2)],
        anchor_size_bounds=[0.2, 0.90],
        anchor_sizes=[(21., 45.),
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.)
                      ],
        anchor_ratios=[[2, .5],
                       [2, .5],
                       [2, .5],
                       [2, .5],
                       [2, .5]],
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

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses',
               is_training=False):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope,
                          is_training=is_training)

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
            scope='ssd_KYnet_v2'):
    """SSD net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    # End_points collect relevant activations for external use.
    end_points = {}
    with tf.variable_scope(scope, 'ssd_KYnet_v2', [inputs], reuse=reuse):
        # Original VGG-16 blocks.
        net = slim.conv2d(inputs, 16, [3, 3], scope='conv1_1') #256
        net = slim.conv2d(net, 32, [3, 3], scope='conv2_1') #256
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1') #128
        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], scope='conv2') #128
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2') #64
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 48, [3, 3], scope='conv3') #64
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3') #32
        # Block 4.
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv4') #32
        net = slim.conv2d(net, 96, [3, 3], scope='conv4_3') #32
        end_points['block4'] = net #32

        # Additional SSD blocks.
        # Block 6: let's dilate the hell out of it!
        net = slim.conv2d(net, 128, [3, 3], scope='conv6') #32
        net = slim.max_pool2d(net, [2, 2], scope='pool2') #16
        end_points['block7'] = net #16

        # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
        end_point = 'block8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [3, 3], scope='conv3x3') #16
            net = slim.max_pool2d(net, [2, 2], scope='pool') #8
        end_points[end_point] = net
        end_point = 'block9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [3, 3], scope='conv3x3') #8
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


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc
            

# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               is_training=True,
               device='/cpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        lshape = tfe.get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            ploss = custom_layers.focal_loss(logits=logits, labels=gclasses)
            ploss = tf.div(tf.reduce_sum(ploss * fpmask), batch_size, name='value')
            tf.losses.add_loss(ploss)

        with tf.name_scope('cross_entropy_neg'):
            nloss = custom_layers.focal_loss(logits=logits, labels=no_classes)
            nloss = tf.div(tf.reduce_sum(nloss * fnmask), batch_size, name='value')
            tf.losses.add_loss(nloss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            closs = custom_layers.abs_smooth(localisations - glocalisations)
            closs = tf.div(tf.reduce_sum(closs * weights), batch_size, name='value')
            tf.losses.add_loss(closs)
        if not is_training:
            return ploss, nloss, closs
