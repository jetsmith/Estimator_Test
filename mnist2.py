#  Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf  # pylint: disable=g-bad-import-order
import argparse

#def parse_args():
#  """
#  Parse input arguments
#  """
#  parser = argparse.ArgumentParser(description='test Estimator for classification')
#  parser.add_argument('--data_dir', dest='dataset directory',
#                      help='optional config file',
#                      default="/home/newhome/junjie/dataset/vggface2/record_10class/", type=str)
#  parser.add_argument('--model_dir', dest='path where ckpt is saved',
#                      help='initialize with pretrained model weights',
#                      default='/data/object_detection/test_cls',
#                      type=str)
#  parser.add_argument('--imdb', dest='imdb_name',
#                      help='dataset to train on',
#                      default='voc_2007_trainval+voc_2012_trainval', type=str)
#  args = parser.parse_args()
#  return args
#args = parse_args()

tf.app.flags.DEFINE_string(
    'data_dir', '/data/cls',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
    'model_dir', '/data/cls/train_dir',
    'gpu ids to train the network.'
    )
FLAGS = tf.app.flags.FLAGS

def parse_function(example_proto):
    features = {'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([], tf.string),
            'image/label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(example_proto, features)
    # You can do more image distortion here for training data
    img = tf.image.decode_jpeg(features['image/encoded'])
    img = tf.image.resize_images(img, [128, 128])
    img = tf.reshape(img, shape=(128, 128, 3))
    r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
    img = tf.concat([b, g, r], axis=-1)
    img = tf.cast(img, dtype=tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    img = tf.image.random_flip_left_right(img)
    label = tf.cast(features['image/label'], tf.int64)
    return img, label

def dataset_input_fn(params):
    dataset = tf.data.TFRecordDataset(params['filenames'])
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(params['shuffle_buff'])
    dataset = dataset.repeat()
    dataset = dataset.batch(params['batch'])
    dataset = dataset.prefetch(8*params['batch'])
    return dataset

def create_model(data_format):
  """Model to recognize digits in the MNIST dataset.
  Network structure is equivalent to:
  https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/examples/tutorials/mnist/mnist_deep.py
  and
  https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
  But uses the tf.keras API.
  Args:
    data_format: Either 'channels_first' or 'channels_last'. 'channels_first' is
      typically faster on GPUs while 'channels_last' is typically faster on
      CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
  Returns:
    A tf.keras.Model.
  """
  if data_format == 'channels_first':
    input_shape = [1, 28, 28]
  else:
    assert data_format == 'channels_last'
    input_shape = [28, 28, 1]

  l = tf.keras.layers
  max_pool = l.MaxPooling2D(
      (2, 2), (2, 2), padding='same', data_format=data_format)
  # The model consists of a sequential chain of layers, so tf.keras.Sequential
  # (a subclass of tf.keras.Model) makes for a compact description.
  return tf.keras.Sequential(
      [
          l.Conv2D(
              32,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Conv2D(
              64,
              5,
              padding='same',
              data_format=data_format,
              activation=tf.nn.relu),
          max_pool,
          l.Flatten(),
          l.Dense(1024, activation=tf.nn.relu),
          l.Dropout(0.4),
          l.Dense(10)
      ])



def model_fn(features, labels, mode, params):
  """The model_fn argument for creating an Estimator."""
  model = create_model(params['data_format'])
  image = features
  if isinstance(image, dict):
    image = features['image']

  if mode == tf.estimator.ModeKeys.PREDICT:
    logits = model(image, training=False)
    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits),
    }
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        export_outputs={
            'classify': tf.estimator.export.PredictOutput(predictions)
        })
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

    # If we are running multi-GPU, we need to wrap the optimizer.
    optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    logits = model(image, training=True)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=tf.argmax(logits, axis=1))

    # Name tensors to be logged with LoggingTensorHook.
    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='train_accuracy')

    # Save accuracy scalar to Tensorboard output.
    tf.summary.scalar('train_accuracy', accuracy[1])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.TRAIN,
        loss=loss,
        train_op=train_op)

  if mode == tf.estimator.ModeKeys.EVAL:
    logits = model(image, training=False)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    return tf.estimator.EstimatorSpec(
        mode=tf.estimator.ModeKeys.EVAL,
        loss=loss,
        eval_metric_ops={
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels, predictions=tf.argmax(logits, axis=1)),
        })


def run():
  """Run MNIST training and eval loop.
  Args:
    flags_obj: An object containing parsed flag values.
  """
  model_function = model_fn


  # Validate that the batch size can be split into devices.
  model_function = tf.contrib.estimator.replicate_model_fn(
      model_fn, loss_reduction=tf.losses.Reduction.MEAN)

  params={
      'data_format': 'channels_last',
	  'shuffle_buff': 1000,
	  'batch': 32,
      'learning_rate': 0.0005,
	  'filenames': FLAGS.data_dir + "/image_train_00000-of-00001.tfrecord"}
#'mode': tf.estimator.ModeKeys.TRAIN,
#'filenames': "/home/newhome/junjie/dataset/vggface2/record_10class/image_train_00000-of-00001.tfrecord"}
  session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
  session_config.gpu_options.allow_growth=True
  config = tf.estimator.RunConfig(
    keep_checkpoint_max=5,
    log_step_count_steps=20,
    save_checkpoints_steps=50,
    session_config=session_config)
 
  classifier = tf.estimator.Estimator(
      model_fn=model_function,
      model_dir=FLAGS.model_dir,
      params=params,
      config=config)

  # Train and evaluate model.
  train_spec = tf.estimator.TrainSpec(input_fn=dataset_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn=dataset_input_fn)
  tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

  # Export the model
  #if flags_obj.export_dir is not None:
  #  image = tf.placeholder(tf.float32, [None, 128, 128])
  #  input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
  #      'image': image,
  #  })
  #  mnist_classifier.export_savedmodel(flags_obj.export_dir, input_fn)



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #args = parse_args()
  run()
