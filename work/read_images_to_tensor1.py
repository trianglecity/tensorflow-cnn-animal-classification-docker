
from PIL import Image

from os import listdir
from os.path import isfile, join

import fnmatch
import os
import math
import tensorflow as tf


def read_animals(filename_queue):

	class ANIMAL10Record(object):
    		pass

  	result = IMAL10Record()	

	label_bytes = 1  # 2 for CIFAR-100
  	result.height = 140
  	result.width = 160
  	result.depth = 3
  	image_bytes = result.height * result.width * result.depth

	record_bytes = label_bytes + image_bytes
	
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)
###
###

images_matches = []

for root, dirs, filenames in os.walk(top = './animals', topdown=True):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        images_matches.append(os.path.join(root, filename))


for f in images_matches:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

n_images = len(images_matches)
print "n_images = ", n_images

filename_queue = tf.train.string_input_producer(images_matches)

## The output of Read will be a filename (key) and the contents of that file (value)
reader = tf.WholeFileReader()
## Returns the next record (key, value pair) produced by a reader.
key_filename, value_image = reader.read(filename_queue)

## Decode a JPEG-encoded image to a uint8 tensor.
my_img = tf.image.decode_jpeg(value_image, channels=3)

with tf.Session() as sess:
	
	tf.global_variables_initializer().run()
	
	##This class implements a simple mechanism to coordinate the termination of a set of threads.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	
	##image_tensor = sess.run(my_img)
	##print image_tensor

	for i in range(n_images):
    		image = my_img.eval() 
		print(image.shape)
		
	
	coord.request_stop()
	coord.join(threads)
	
		
