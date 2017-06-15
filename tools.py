# -- coding: utf-8 --

import tensorflow as tf

def conv(layer_name,input,output_channels,kernel=[3,3],stride=[1,1,1,1]):
    input_channels = input.get_shape()[-1]

    with tf.variable_scope(layer_name) as scope:
        w = tf.get_variable('weights', 
                            shape = [kernel[0],kernel[1],input_channels,output_channels],
                            dtype = tf.float32, 
                            initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        b = tf.get_variable('biases',shape=output_channels,dtype=tf.float32,initializer=tf.constant_initializer(0.1))
        ret = tf.nn.conv2d(input,w,stride,padding='SAME')
        ret = tf.nn.bias_add(ret,b)
        ret = tf.nn.relu(ret, name=scope.name+'/relu')
        return ret

def pool(layer_name,input,kernel=[1,3,3,1],stride=[1,2,2,1],max_pool=True):
    with tf.variable_scope(layer_name) as scope:
        if max_pool==True:
          ret = tf.nn.max_pool(input,ksize=kernel,strides=stride,padding='SAME',name=scope.name)
        else:
          ret = tf.nn.avg_pool(input,ksize=kernel,strides=stride,padding='SAME',name=scope.name)

        ret = tf.nn.lrn(ret,depth_radius=4,bias=1.0,alpha=0.001/9.0,beta=0.75,name=scope.name+'/norm')
        return ret

def fc(layer_name,input,output_nodes):
    shape = input.get_shape()
    if len(shape) == 4:
      size = shape[1].value * shape[2].value * shape[3].value
    else:
      size = shape[-1].value

    with tf.variable_scope(layer_name) as scope:
        w = tf.get_variable('weights',shape=[size,output_nodes],dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        b = tf.get_variable('biases',shape=[output_nodes],dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        flat_x = tf.reshape(input, [-1, size])
        ret = tf.nn.relu(tf.matmul(flat_x, w) + b, name=scope.name)
        return ret

def dropout(layer_name,input,keep_prob):
    with tf.variable_scope(layer_name) as scope:
        return tf.nn.dropout(input,keep_prob,name=scope.name)

def batch_norm(input):
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(input, [0])
    input = tf.nn.batch_normalization(input,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return input