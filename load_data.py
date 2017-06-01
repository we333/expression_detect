import tensorflow as tf
import numpy as np
import os

GENKI4K_DIR = "./genki4k/"

def get_file(dataset_dir):
	data = []
	labels = []

# read labels and save
	with open(dataset_dir+'labels.txt', 'r') as f:
		for line in filter(None,f):
			labels.append(line.split(' ')[0])

# read image files and save file name
	for file in os.listdir(dataset_dir+"files/"):
		data.append(dataset_dir+"files/"+file)

# create dataset as [data,label]
	tmp = np.array([data, labels])
	tmp = tmp.transpose()
	np.random.shuffle(tmp)

	print('There are %d image' %(len(data)))


	image_list = list(tmp[:,0])
	label_list = list(tmp[:,1])
	label_list = [int(i) for i in label_list]

	return image_list, label_list

def get_batch(image,label,w,h,batch_size,capacity):
	image = tf.cast(image, tf.string)
	label = tf.cast(label, tf.int32)

	input_queue = tf.train.slice_input_producer([image,label])
	
	image_contents = tf.read_file(input_queue[0])
	image = tf.image.decode_jpeg(image_contents, channels=3)
	label = input_queue[1]

	image = tf.image.resize_image_with_crop_or_pad(image,w,h)
	image = tf.image.per_image_standardization(image)

	image_batch,label_batch = tf.train.batch([image,label],
											batch_size = batch_size,
											num_threads=64,
											capacity=capacity)

	
	label_batch = tf.reshape(label_batch, [batch_size])
	image_batch = tf.cast(image_batch, tf.float32)

	return image_batch, label_batch

##########################################################
#simple test for load image & save as tensorflow batch
##########################################################

import matplotlib.pyplot as plt

image_list,label_list = get_file(GENKI4K_DIR)
image_batch,label_batch = get_batch(image_list,label_list,208,208,2,256)

with tf.Session() as sess:
	i = 0
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)

	while not coord.should_stop() and i < 1:
		img,label = sess.run([image_batch,label_batch])
		for j in np.arange(2):
			print('label: %d' %label[j])
			plt.imshow(img[j,:,:,:])
			plt.show()
		i += 1

