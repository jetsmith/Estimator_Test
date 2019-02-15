# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models.
"""
#from __future__ import absolute_import
import os
import functools
import tensorflow as tf

#from nets import vgg
from nets import vgg16, vgg16_raw, mobilenet_v1, resnet_v1, resnet_utils, resnet_utils_arcface, resnet_arcface, test_resnet, LMobilenetE, hardware_resnet50
slim = tf.contrib.slim

networks_map = {'vgg16_face0': vgg16.vgg16_face0,
                'vgg16_raw': vgg16_raw.vgg16_raw,
                'vgg16_BN': vgg16.vgg16_BN,
                'vgg16_clip': vgg16.vgg16_clip,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
                'vgg16_MT': vgg16.vgg16_MT,
                'vgg16_mt_fc': vgg16.vgg16_mt_fc,
                'vgg16_mt_conv': vgg16.vgg16_mt_conv,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'LMobilenetE': mobilenet_v1.mobilenet_v1,
		        'resnet_v1_50': resnet_v1.resnet_v1_50,
		        'resnet_arcface': resnet_arcface.resnet_v1_50,
                'test_net': hardware_resnet50.resnet50}
arg_scopes_map = {'vgg16_face0': vgg16.vgg_arg_scope,
                'vgg16_raw': vgg16_raw.vgg_arg_scope,
                'vgg16_BN': vgg16.vgg_arg_scope,
                'vgg16_clip': vgg16.vgg_arg_scope,
                'vgg16_MT': vgg16.vgg_arg_scope,
                'vgg16_mt_fc': vgg16.vgg_arg_scope,
                'vgg16_mt_conv': vgg16.vgg_arg_scope,
                'test_net': hardware_resnet50.resnet_arg_scope,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                'LMobilenetE': mobilenet_v1.mobilenet_v1_arg_scope,
		        'resnet_arcface': resnet_utils_arcface.resnet_arg_scope,
		        'resnet_v1_50': resnet_utils.resnet_arg_scope}
networks_obj = {}

for f in os.listdir(os.path.split(__file__)[0]):
    if f.startswith('ssd_KYnet_'):
        f, _ = os.path.splitext(f)
        tf.logging.info('from nets import {}'.format(f))
        exec('from nets import {}'.format(f))
        networks_map[f] = eval('{}.ssd_net'.format(f))
        arg_scopes_map[f] = eval('{}.ssd_arg_scope'.format(f))
        networks_obj[f] = eval('{}.SSDNet'.format(f))


def get_network(name):
    """Get a network object from a name.
    """
    # params = networks_obj[name].default_params if params is None else params
    return networks_obj[name]


def get_network_fn(name, num_classes, is_training=False, **kwargs):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      is_training: `True` if the model is being used for training and `False`
        otherwise.
      weight_decay: The l2 coefficient for the model weights.
    Returns:
      network_fn: A function that applies the model to a batch of images. It
        has the following signature: logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    arg_scope = arg_scopes_map[name](**kwargs)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
