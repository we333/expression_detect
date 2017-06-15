# -- coding: utf-8 --

import tensorflow as tf
import tools

def vgg13(x, batch_size, n_classes):
  x = tools.conv('conv1',x,16,kernel=[3,3],stride=[1,1,1,1])
  x = tools.pool('pool1',x,kernel=[1,3,3,1],stride=[1,2,2,1],max_pool=True)

  x = tools.conv('conv2',x,16,kernel=[3,3],stride=[1,1,1,1])
  x = tools.pool('pool2',x,kernel=[1,3,3,1],stride=[1,1,1,1],max_pool=True)

  x = tools.fc('fc3',x,128)
  #    x = tools.batch_norm(x)
  x = tools.fc('fc4',x,128)
  #    x = tools.batch_norm(x)
  # softmax
  with tf.variable_scope('softmax_linear') as scope:
    w = tf.get_variable('softmax_linear',
                              shape=[128, n_classes],
                              dtype=tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
    b = tf.get_variable('biases', 
                             shape=[n_classes],
                             dtype=tf.float32, 
                             initializer=tf.constant_initializer(0.1))
    x = tf.add(tf.matmul(x, w), b, name='softmax_linear')

  return x

def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

def trainning(loss, learning_rate):
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

def evaluation(logits, labels):
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy