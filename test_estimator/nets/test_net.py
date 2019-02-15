
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.
  Args:
    weight_decay: The l2 regularization coefficient.
  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def concat_net(inputs,
           num_classes=None,
           is_training=True,
           spatial_squeeze=True,
           scope='test_net',
           global_pool=True):
    with tf.variable_scope(scope, 'test_net', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net_branch = slim.conv2d(net, 64, [3, 3], scope='conv2')
            net_branch = slim.conv2d(net_branch, 64, [3, 3], scope='conv3')
            net = tf.concat([net, net_branch], 3, name='concat')
            print('net shape: ', net.get_shape())
            net = slim.conv2d(net, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
          # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            net = slim.conv2d(net, num_classes, [3, 3],
                          activation_fn=None,
                          scope='logit')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='squeezed')
                end_points[sc.name + '/squeeze'] = net
            return net, end_points
concat__net.default_image_size = 64

def noconcat_net(inputs,
           num_classes=None,
           is_training=True,
           spatial_squeeze=True,
           scope='test_net',
           global_pool=True):
    with tf.variable_scope(scope, 'test_net', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            net = slim.conv2d(inputs, 32, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net_branch = slim.conv2d(net, 64, [3, 3], scope='conv2')
            net_branch = slim.conv2d(net_branch, 64, [3, 3], scope='conv3')
            #net = tf.concat([net, net_branch], 3, name='concat')
            print('net shape: ', net.get_shape())
            net = slim.conv2d(net_branch, 64, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
          # Convert end_points_collection into a end_point dict.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            net = slim.conv2d(net, num_classes, [3, 3],
                          activation_fn=None,
                          scope='logit')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='squeezed')
                end_points[sc.name + '/squeeze'] = net
            return net, end_points
noconcat_net.default_image_size = 64
