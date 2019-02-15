
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

def test_net(inputs,
        num_classes=None,
        is_training=True,
        dropout_keep_prob=0.5,
        spatial_squeeze=True,
        scope='vgg_16',
        fc_conv_padding='VALID',
        global_pool=True):

    with tf.variable_scope(scope, 'test_resnet50', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
            outputs_collections=end_points_collection):
        net = slim.repeat(inputs, 2, slim.conv2d, 32, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 128, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 1, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.conv2d(net, 512, [3, 3], padding=fc_conv_padding, scope='6')
        net = slim.conv2d(net, 512, [3, 3], padding=fc_conv_padding, scope='7')
        net = slim.conv2d(net, 512, [3, 3], padding=fc_conv_padding, scope='8')
        net = slim.max_pool2d(net, [2, 2], scope='pool6')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if num_classes:
          # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
          #                    scope='dropout7')
          net = slim.conv2d(net, num_classes, [1, 1],
                            activation_fn=None,
                            normalizer_fn=None,
                            scope='logits')
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='logits/squeezed')
          end_points[sc.name + 'logits'] = net
          #net = slim.batch_norm(net, is_training=is_training)
          #end_points['BN']=net
        return net, end_points

test_net.default_image_size = 224

