
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def slim_repeat(input_x, repeat_num, layer_type, layer_width, kernel_shape, scope, \
        clip_flag=True, min_value=-128.0, max_value=128.0):
    net = input_x
    for i in range(repeat_num):
        net = slim.conv2d(net, layer_width, kernel_shape, scope='{0}/{0}_{1}'.format(scope, i+1))
        if clip_flag:
            net = tf.clip_by_value(net, min_value, max_value)
    return net

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

def vgg16_clip(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim_repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim_repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim_repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim_repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim_repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # # Use conv2d instead of fully_connected layers.
      # net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
      #                    scope='dropout6')
      # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                    scope='dropout7')
        #net = slim.conv2d(net, num_classes, [1, 1],
        #                  activation_fn=None,
        #                  normalizer_fn=None,
        #                  scope='fc1')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc1/squeezed')
        end_points[sc.name + '/fc1'] = net
        net = slim.batch_norm(net, is_training=is_training)
        end_points['BN']=net
      return net, end_points

vgg16_clip.default_image_size = 224

def vgg16_face0(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
  """Oxford Net VGG 16-Layers version D Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output.
      Otherwise, the output prediction map will be (input / 32) - 6 in case of
      'VALID' padding.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original VGG architecture.)
  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the input to the logits layer (if num_classes is 0 or None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # # Use conv2d instead of fully_connected layers.
      # net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
      #                    scope='dropout6')
      # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if global_pool:
        net = tf.reduce_mean(net, [1, 2], keepdims=True, name='global_pool')
        end_points['global_pool'] = net
      if num_classes:
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
        #                    scope='dropout7')
        net = slim.conv2d(net, 512, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc1')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc1/squeezed')
        end_points[sc.name + '/fc1'] = net
        #net = slim.batch_norm(net, is_training=is_training)
        #end_points['BN']=net
      return net, end_points

vgg16_face0.default_image_size = 224

def vgg16_BN(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer is
          omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
          that is implemented as a convolutional layer. Use 'SAME' padding if you
          are applying the network in a fully convolutional manner and want to
          get a prediction map downsampled by a factor of 32 as an output.
          Otherwise, the output prediction map will be (input / 32) - 6 in case of
          'VALID' padding.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original VGG architecture.)
    Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the input to the logits layer (if num_classes is 0 or None).
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, center=True, scale=True, decay=0.99):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    #net = slim.flatten(net, scope='flatten')

                   # Use conv2d instead of fully_connected layers.
                    #net = slim.conv2d(net, 4096, [4, 4], padding=fc_conv_padding, scope='fc6')
                    #net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                    #                  scope='dropout6')
                    #net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                  # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                        end_points['global_pool'] = net
                    if num_classes:
                    # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                    #                    scope='dropout7')
                        net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc1')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc1/squeezed')
                        end_points[sc.name + '/fc1'] = net
                    print('net shape: ', net.get_shape())
                    return net, end_points
vgg16_BN.default_image_size = 224

def vgg16_MT(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer is
          omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
          that is implemented as a convolutional layer. Use 'SAME' padding if you
          are applying the network in a fully convolutional manner and want to
          get a prediction map downsampled by a factor of 32 as an output.
          Otherwise, the output prediction map will be (input / 32) - 6 in case of
          'VALID' padding.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original VGG architecture.)
    Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the input to the logits layer (if num_classes is 0 or None).
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, center=True, scale=True, decay=0.99):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    #net = slim.flatten(net, scope='flatten')

                    #Use conv2d instead of fully_connected layers.
                    #net = slim.conv2d(net, 4096, [4, 4], padding=fc_conv_padding, scope='fc6')
                    #net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                    #                  scope='dropout6')
                    #net = slim.conv2d(net, 4096, [1, 1], scope='fc7')

                  # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                        end_points['global_pool'] = net

                    logits = {}
                    num_classes = num_classes.split(',')
                    #for i, attribute in enumerate(['jc', 'pc', 'pt', 'age', 'gender']):                   
                    for i, attribute in enumerate(['age', 'gender']):                   
                        if num_classes:
                          # net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                          #                    scope='dropout7')
                            #tmp_net = slim.fully_connected(net,
                            #        int(num_classes[i]), activation_fn=None,
                            #        normalizer_fn=None, scope= attribute + '/fc1')
                            tmp_net = slim.conv2d(net, int(num_classes[i]), [1, 1],
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        scope= attribute + '/fc1')
                        if spatial_squeeze:
                            logits[attribute] = tf.squeeze(tmp_net, [1, 2], name='fc1/squeezed/' + attribute)
                            end_points[sc.name + '/fc1/' +  attribute] = logits[attribute]
                        #logits[attribute] = tmp_net
                    return logits, end_points
vgg16_MT.default_image_size = 224

def vgg16_mt_fc(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer is
          omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
          that is implemented as a convolutional layer. Use 'SAME' padding if you
          are applying the network in a fully convolutional manner and want to
          get a prediction map downsampled by a factor of 32 as an output.
          Otherwise, the output prediction map will be (input / 32) - 6 in case of
          'VALID' padding.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original VGG architecture.)
    Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the input to the logits layer (if num_classes is 0 or None).
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d, slim.fully_connected], normalizer_fn=slim.batch_norm):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, center=True, scale=True, decay=0.99):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')
                    #net = slim.flatten(net, scope='flatten')
                    #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                    #net = slim.max_pool2d(net, [2, 2], scope='pool4')
                    #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                    #net = slim.max_pool2d(net, [2, 2], scope='pool5')
                    # Convert end_points_collection into a end_point dict.
                    end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                    if global_pool:
                        net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                        end_points['global_pool'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='squeezed')
                    logits = {}
                    num_classes = num_classes.split(',')
                    for i, attribute in enumerate(['jc', 'pc', 'pt', 'age', 'gender']):                   
                      tmp_net = slim.fully_connected(net, 1024, scope=
                              attribute + '/fc1')
                      tmp_net = slim.dropout(tmp_net, dropout_keep_prob, is_training=is_training,
                                          scope= attribute + '/dropout1')
                      tmp_net = slim.fully_connected(tmp_net, 1024, scope=
                              attribute + '/fc2')
                      if num_classes:
                          tmp_net = slim.dropout(tmp_net, dropout_keep_prob, is_training=is_training,
                                          scope= attribute + '/dropout2')
                          tmp_net = slim.fully_connected(tmp_net,
                                  int(num_classes[i]), activation_fn=None,
                                  normalizer_fn=None, scope= attribute + '/fc3')
                      #if spatial_squeeze:
                      #    logits[attribute] = tf.squeeze(tmp_net, [1, 2], name='fc1/squeezed' + '/' + attribute)
                          logits[attribute] = tmp_net
                          end_points[sc.name + '/fc3/' +  attribute] = logits[attribute]
                    
                    return logits, end_points
vgg16_mt_fc.default_image_size = 224


def vgg16_mt_conv(inputs,
           num_classes=None,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=True):
    """Oxford Net VGG 16-Layers version D Example.
    Note: All the fully_connected layers have been transformed to conv2d layers.
            To use in classification mode, resize input to 224x224.
    Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        num_classes: number of predicted classes. If 0 or None, the logits layer is
          omitted and the input features to the logits layer are returned instead.
        is_training: whether or not the model is being trained.
        dropout_keep_prob: the probability that activations are kept in the dropout
          layers during training.
        spatial_squeeze: whether or not should squeeze the spatial dimensions of the
          outputs. Useful to remove unnecessary dimensions for classification.
        scope: Optional scope for the variables.
        fc_conv_padding: the type of padding to use for the fully connected layer
          that is implemented as a convolutional layer. Use 'SAME' padding if you
          are applying the network in a fully convolutional manner and want to
          get a prediction map downsampled by a factor of 32 as an output.
          Otherwise, the output prediction map will be (input / 32) - 6 in case of
          'VALID' padding.
        global_pool: Optional boolean flag. If True, the input to the classification
          layer is avgpooled to size 1x1, for any input size. (This is not part
          of the original VGG architecture.)
    Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the input to the logits layer (if num_classes is 0 or None).
        end_points: a dict of tensors with intermediate activations.
    """
    with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm):
                with slim.arg_scope([slim.batch_norm], is_training=is_training, center=True, scale=True, decay=0.99):
                    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                    net = slim.max_pool2d(net, [2, 2], scope='pool1')
                    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                    net = slim.max_pool2d(net, [2, 2], scope='pool2')
                    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                    net = slim.max_pool2d(net, [2, 2], scope='pool3')

                    logits = {}
                    num_classes = num_classes.split(',')
                    for i, attribute in enumerate(['jc', 'pc', 'pt', 'age', 'gender']):
                        tmp_net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                                scope= attribute + '/conv4')
                        tmp_net = slim.max_pool2d(tmp_net, [2, 2], scope= attribute
                                + '/pool4')
                        tmp_net = slim.repeat(tmp_net, 3, slim.conv2d, 512, [3, 3], scope= attribute + '/conv5')
                        tmp_net = slim.max_pool2d(tmp_net, [2, 2], scope=attribute + '/pool5')
                      # Convert end_points_collection into a end_point dict.
                        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                        if global_pool:
                            tmp_net = tf.reduce_mean(tmp_net, [1, 2],
                                    keep_dims=True, name=attribute + '/global_pool')
                            end_points[attribute + '/global_pool'] = tmp_net
                        if num_classes:
                            #net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                            #                scope='dropout7')
                            tmp_net = slim.conv2d(tmp_net, int(num_classes[i]), [1, 1],
                                          activation_fn=None,
                                          normalizer_fn=None,
                                          scope=attribute + '/fc1')
                        if spatial_squeeze:
                            tmp_net = tf.squeeze(tmp_net, [1, 2], name=attribute +
                                    '/fc1/squeezed')
                            end_points[sc.name + '/fc1'] = tmp_net
                            logits[attribute] = tmp_net
                    return logits, end_points
vgg16_mt_conv.default_image_size = 224


def test_net(inputs,
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
test_net.default_image_size = 64
