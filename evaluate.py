# -- coding: utf-8 --

import os
import numpy as np
import tensorflow as tf
import load_data
import model
 
import cv2

from PIL import Image
import matplotlib.pyplot as plt

def get_image(filelist):
    # random choose one image from file_list
    ind = np.random.randint(0, len(filelist))
    img_dir = filelist[ind]

    iii = cv2.imread(img_dir)
 #   iii = cv2.resize(iii,(208,208))

    image = Image.open(img_dir)
 #   image = image.resize([208, 208])
    image = np.array(image)

    return iii

def evaluate_image(image_array):
    image_array = cv2.resize(image_array,(208,208))

    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = './logs/train/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path) 
            #    print('Loading success, global_step is %s' % global_step)
            else:
                pass
            #    print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})

            max_index = np.argmax(prediction)
              # always return smile prediction
            #if max_index==0:
            #    print('This is smile with possibility %.6f' %prediction[:, 0])
            #    return True, prediction[:, 0]
            #else:
            #    print('This is non-smile with possibility %.6f' %prediction[:, 1])
            #    return False, prediction[:,1]
    return prediction[:,0]


#image = get_image(['./genki4k/files/file0007.jpg'])
#pred = evaluate_image(image)

#cv2.waitKey()
#cv2.destroyAllWindows()