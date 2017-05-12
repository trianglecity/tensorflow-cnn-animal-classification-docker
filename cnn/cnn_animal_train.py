
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize

from os import listdir
from os.path import isfile, join

import fnmatch
import os
import random
from animal_classes import class_names


class cnn:
    def __init__(self, imgs, labels, sess=None):
	print "in __init__"
        self.imgs = imgs
        self.labels = labels
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
	self.logits = self.fc3l	
        self.cross_entropy_mean = self.loss()
	self.train_op = self.train()

    def convlayers(self):

	print "in convlayer"
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([125, 125, 125], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs - mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

	# conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')
	# conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=0.01), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')	

        
    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool3.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 1024],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[1024], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool3_flat = tf.reshape(self.pool3, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([1024, 4],
                                                         dtype=tf.float32,
                                                         stddev=0.01), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[4], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

	
	

    def loss(self):
		
		print "in loss"
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      					labels=self.labels, logits=self.logits, name='cross_entropy_per_example')
  		cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
		return cross_entropy_mean

    def train(self):

		print "in train"
		opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		grads = opt.compute_gradients(self.cross_entropy_mean)
		apply_gradient_op = opt.apply_gradients(grads, global_step=tf.contrib.framework.get_global_step())
		return apply_gradient_op
	

def read_animals():


	## class
	class ANIMAL4Record(object):
    		pass

	## scprit for loading images
	
  	result = ANIMAL4Record()	

	
  	result.height = 80
  	result.width  = 80
  	result.depth  = 3
  	
	
	## bear  -- label 0

	images_bear = []
	labels_bear = []

	for root, dirs, filenames in os.walk(top = './animals/bear', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_bear.append(os.path.join(root, filename))
			labels_bear.append(0)

	## deer -- label 1

	images_deer = []
	labels_deer = []

	for root, dirs, filenames in os.walk(top = './animals/deer', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_deer.append(os.path.join(root, filename))
			labels_deer.append(1)

	## duck -- label 2

	
	images_duck = []
	labels_duck = []

	for root, dirs, filenames in os.walk(top = './animals/duck', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_duck.append(os.path.join(root, filename))
			labels_duck.append(2)


	## turtle -- label 3

	images_turtle = []
	labels_turtle = []

	for root, dirs, filenames in os.walk(top = './animals/turtle', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_turtle.append(os.path.join(root, filename))
			labels_turtle.append(3)

	train_labels = labels_bear + labels_deer + labels_duck + labels_turtle
	train_images = images_bear + images_deer + images_duck + images_turtle
	
		
	result.images = train_images
	result.labels = train_labels
           
	result.n_images = 80

 	return result
 

if __name__ == '__main__':

    batch_size = 1
    
    
    
    ## Launch the graph in a session.
    with tf.Session() as sess:
	
    	## placeholder( dtype, shape=None, name=None)
    	## Its value must be fed using the feed_dict 
    	images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 80,80,3))
    	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

    	cnn = cnn(images_placeholder, labels_placeholder,  sess)

	saver = tf.train.Saver()
    	init = tf.global_variables_initializer()
    	sess.run(init)

    	result = read_animals()
    	imgs_info = result.images
    	labels = result.labels
     	

    	combined = list(zip(imgs_info, labels))
    	random.shuffle(combined)

    	imgs_info[:], labels[:] = zip(*combined)
    

    	for i in xrange(len(labels)):
		
		img =  imread(imgs_info[i], mode='RGB')
    		label = int(labels[i])
    		print "label = ", label		
    		_, loss_val = sess.run([cnn.train_op, cnn.cross_entropy_mean], feed_dict={cnn.imgs: [img], cnn.labels: [label] })
    		print "loss = ", loss_val
    
	
	
	saver.save(sess, "my_cnn_model.ckpt")

	##
	##	my_cnn_model.ckpt.data-00000-of-00001
 	##	my_cnn_model.ckpt.index
 	##	my_cnn_model.ckpt.meta
	##
	

    	print "prediction"
    	img =  imread(imgs_info[0], mode='RGB')
    	label = int(labels[0])
        print " true = ", class_names[0] 

    	prob = sess.run(cnn.probs, feed_dict={cnn.imgs: [img], cnn.labels: [label]})[0]
	preds = (np.argsort(prob)[::-1])[0:4]
    	for p in preds:
        	print class_names[p], prob[p]

