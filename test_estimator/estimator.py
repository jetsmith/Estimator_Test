#coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import sys
import tempfile
import time
import tensorflow as tf

import test_estimator.nets.factory as nets_factory

slim = tf.contrib.slim

# 定义常量，用于创建数据流图
flags = tf.app.flags

# task_index从0开始。0代表用来初始化变量的第一个任务
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
# 每台机器GPU个数，机器没有GPU为0
flags.DEFINE_integer("num_gpus", 1,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
# 同步训练模型下，设置收集工作节点数量。默认工作节点总数
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
# 训练次数
flags.DEFINE_integer("train_steps", 1000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 32, "Training batch size")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
# 使用同步训练、异步训练
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
# 如果服务器已经存在，采用gRPC协议通信；如果不存在，采用进程间通信
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
# 参数服务器主机
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
# 工作节点主机
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
# 本作业是工作节点还是参数服务器
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string("record_path", "../dist_training/script/image_train_00000-of-00001.tfrecord", "Directory for storing mnist data")
flags.DEFINE_integer("image_size", 128, "input image height and width")
flags.DEFINE_integer("buffer_size", 500, "the size of buffer to shuffle")
flags.DEFINE_string("train_dir", '', "training directory")


FLAGS = flags.FLAGS

def parse_function_resnet(example_proto):
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
    dataset = dataset.map(parse_function_resnet)
    dataset = dataset.shuffle(params['shuffle_buff'])
    dataset = dataset.repeat()
    dataset = dataset.batch(params['batch'])
    dataset = dataset.prefetch(8*params['batch'])
    return dataset

#define pipeline
def _parse_image(filename, label):
    # 读取并解码图片
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    img = tf.image.resize_images(image_decoded, [128, 128])
    img = tf.reshape(img, shape=(128, 128, 3))
    # 一定要在这里转换类型！！！
    image_converted = tf.cast(img, tf.float32)
    # 缩放范围
    image_scaled = tf.divide(tf.subtract(image_converted, 127.5), 255)
    return image_scaled, label

def get_training_data():
    # 读取由path指定的文本文件，并返回由很多(图片路径,标签)组成的列表
    lists_and_labels = np.loadtxt(path, dtype=str).tolist()
    # 打乱下lists_and_labels
    np.random.shuffle(lists_and_labels)
    # 把图片路径和标签分开
    list_files, labels = zip(*[(l[0], int(l[1])) for l in lists_and_labels])
    # 如果使用keras构建模型，还需要对标签进行one_hot编码，如果使用tensorflow构建的模型，则不需要。
    #one_shot_labels = keras.utils.to_categorical(labels, NUM_CLASSES).astype(dtype=np.int32)
    # 定义数据集实例
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(list_files), tf.constant(one_shot_labels)))
    # 对每一对 (image, label)调用_parse_image，完成图像的预处理
    dataset = dataset.map(_parse_image, num_parallel_calls=mt.cpu_count())
    dataset = dataset.batch(FLAGS.batch_size)
    dataset = dataset.shuffle(FLAGS.buffer_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    # 计算遍历一遍数据集需要多少步
    steps_per_epoch = np.ceil(len(labels) / FLAGS.batch_size).astype(np.int32)
    return features, labels

def model_fn(features, labels, mode, params):
    
    network_fn = nets_factory.get_network_fn(
        "vgg16_face0", 10, is_training=params['is_training'],
        weight_decay = 0.00005)
    
    logit, _ = network_fn(features)    

    # predictions = {"logits": logits,
    #                "classes": tf.argmax(input=logits, axis=1),
    #                "probabilities": tf.nn.softmax(logits,name='softmax')}
    # export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    if (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels)
        loss = tf.reduce_mean(loss)
    else:
        loss = None
    
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits, axis=1))
        metrics = {'accuracy': accuracy}
        return tf.estimator.EstimatorSpec(mode=mode,loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = tf.argmax(input=logits, axis=1)
    else:
        predictions = None
    
    learning_rate = tf.train.exponential_decay(params['learning_rate'],
                                                   tf.train.get_global_step(),
                                                   decay_steps=100000,
                                                   decay_rate=0.96)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op )

def main():
    # print training log
    tf.logging.set_verbosity(tf.logging.INFO)
    model_dir = "/data/mnist/jj_model"
    config = tf.estimator.RunConfig(
    keep_checkpoint_max=5,
    log_step_count_steps=20,
    session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    model_params  = {'learning_rate': 1e-3, 'is_training': True}

    train_params = {'filenames': "/data/mnist/image_train_00000-of-00001.tfrecord",
                'mode': tf.estimator.ModeKeys.TRAIN,
                'threads': 8,
                'shuffle_buff': 1000,
                'batch': 32}
    estimator = tf.estimator.Estimator(
    model_fn=model_fn, model_dir=model_dir, params=model_params, config=config)
    estimator.train(input_fn=lambda: dataset_input_fn(train_params),
            max_steps=10000)




if __name__ == "__main__":
  main()

