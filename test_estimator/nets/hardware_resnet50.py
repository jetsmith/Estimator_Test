
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
def block_repeat(input, repeat_num, num_filters, scope):
    net = input
    #net = slim.max_pool2d(net, [2, 2], scope= scope + '/pool')
    #net = slim.conv2d(net, num_filters, [1, 1], scope=scope + '/conv1_1')
    #net = slim.conv2d(net, num_filters, [3, 3], scope=scope + '/conv1_2')
    #net = slim.conv2d(net, num_filters*4, [1, 1], scope=scope + '/conv1_3')
    for i in range(repeat_num):
        if i == 0:
            net = slim.max_pool2d(net, [2, 2], scope= scope + '/pool')
            net = slim.conv2d(net, num_filters, [1, 1], scope=scope + '/conv' + str(i+1) + '_1')
        else:
            net = slim.conv2d(net, num_filters, [1, 1], scope=scope + '/conv' + str(i+1) + '_1')

        net = slim.conv2d(net, num_filters, [3, 3], scope=scope + '/conv' + str(i+1) + '_2')
        net = slim.conv2d(net, num_filters*4, [1, 1], scope=scope + '/conv' + str(i+1) + '_3')
    return net

def resnet_arg_scope(weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(weight_decay),
        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

def resnet50(inputs,
        num_classes=None,
        is_training=True,
        dropout_keep_prob=0.5,
        spatial_squeeze=True,
        scope='resnet50',
        fc_conv_padding='VALID',
        global_pool=True):

    with tf.variable_scope(scope, 'resnet50', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
            outputs_collections=end_points_collection):
        
        net = slim.conv2d(inputs, 8, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.conv2d(net, 8, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        
        # block1
        net = slim.conv2d(net, 64, [1, 1], scope='block1/conv1')
        net = slim.conv2d(net, 64, [3, 3], scope='block1/conv2')
        net = slim.conv2d(net, 128, [1, 1], scope='block1/conv3')

        net = slim.conv2d(net, 64, [1, 1], scope='block1/conv4')
        net = slim.conv2d(net, 64, [3, 3], scope='block1/conv5')
        net = slim.conv2d(net, 128, [1, 1], scope='block1/conv6')

        net = slim.conv2d(net, 64, [1, 1], scope='block1/conv7')
        net = slim.conv2d(net, 64, [3, 3], scope='block1/conv8')
        net = slim.conv2d(net, 256, [1, 1], scope='block1/conv9')
        
        # block2
        net = block_repeat(net, 4, 128, 'block2')
        # block3
        net = block_repeat(net, 6, 256, 'block3')
        # block4
        net = block_repeat(net, 3, 512, 'block4')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if global_pool:
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name = "GAP")
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

resnet50.default_image_size = 256

