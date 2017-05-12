
from PIL import Image

from os import listdir
from os.path import isfile, join

import fnmatch
import os
import math
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

IMAGE_HEIGHT = 140
IMAGE_WIDTH  = 160

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 4, """Number of images to process in a batch.""")



def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 2

  ## 	batch(
  ##  		tensors,
  ##  		batch_size,
  ##  		num_threads=1,
  ##  		capacity=32,
  ##  		enqueue_many=False,
  ##  		shapes=None,
  ##  		dynamic_pad=False,
  ##  		allow_smaller_final_batch=False,
  ##  		shared_name=None,
  ##  		name=None
  ##	)

  ## capacity: An integer. The maximum number of elements in the queue.
 
  ## tf.train.batch(..) will always load batch_size tensors
  ## You must call sess.run(...) again to load a new batch.

  


  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples)

  
  print "label_batch = ", label_batch
  return images, tf.reshape(label_batch, [batch_size])



def read_animals():

	

	## class
	class ANIMAL4Record(object):
    		pass

	## scprit for loading images
	
  	result = ANIMAL4Record()	

	
  	result.height = IMAGE_HEIGHT
  	result.width  = IMAGE_WIDTH
  	result.depth  = 3
  	
	
	## bear

	images_bear = []
	labels_bear = []

	for root, dirs, filenames in os.walk(top = './animals/bear', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_bear.append(os.path.join(root, filename))
			labels_bear.append(0)

	## deer 

	images_deer = []
	labels_deer = []

	for root, dirs, filenames in os.walk(top = './animals/deer', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_deer.append(os.path.join(root, filename))
			labels_deer.append(1)

	## duck

	
	images_duck = []
	labels_duck = []

	for root, dirs, filenames in os.walk(top = './animals/duck', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_duck.append(os.path.join(root, filename))
			labels_duck.append(2)


	## turtle

	images_turtle = []
	labels_turtle = []

	for root, dirs, filenames in os.walk(top = './animals/turtle', topdown=True):
    		for filename in fnmatch.filter(filenames, '*.jpg'):
        		images_turtle.append(os.path.join(root, filename))
			labels_turtle.append(3)

	labels_total = labels_bear + labels_deer + labels_duck + labels_turtle
	images_total = images_bear + images_deer + images_duck + images_turtle
	
	
	for f in images_total:
    		if not tf.gfile.Exists(f):
      			raise ValueError('Failed to find file: ' + f)	
	
	n_images = len(labels_total)
	print "n_images = ", n_images	

	
	# convert string into tensors
	all_images = ops.convert_to_tensor(images_total, dtype=dtypes.string)
	all_labels = ops.convert_to_tensor(labels_total, dtype=dtypes.int32)
	

	# create input queues
	train_input_queue = tf.train.slice_input_producer(
                                    [all_images, all_labels],
                                    shuffle=False)

	# process path and string tensor into an image and a label
	file_content = tf.read_file(train_input_queue[0])
	train_image = tf.image.decode_jpeg(file_content, channels=3)
	train_label = train_input_queue[1]


	##result.uint8image = tf.image.decode_jpeg(value_image, channels=3)
	result.uint8image = train_image
	
	# The first bytes represent the label, which we convert from uint8->int32.
        # tf.cast(a, tf.int32) 
	print labels_total
	
	print tf.constant(labels_total)
	result.label = train_label
           
	result.n_images = 80

 	return result
 	
###
###

def distorted_inputs( batch_size):

	
	read_input = read_animals()
	
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_HEIGHT
  	width = IMAGE_WIDTH

	print "height = ", height
	print "width = ", width

	## skip Randomly crop a [height, width] section of the image.

	## Randomly flip the image horizontally.
  	distorted_image = tf.image.random_flip_left_right(reshaped_image)

	## skip distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        ## skip distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

	
	# Set the shapes of tensors.
	print read_input.uint8image
	print distorted_image
	print read_input.label

	##To use tf.train_batch we need to define the shape of our image tensors before they can be combined into batches.

  	distorted_image.set_shape([height, width, 3])
  	

	train_image_batch, train_label_batch = tf.train.batch(
                                    [distorted_image, read_input.label],
                                    batch_size=batch_size,
				    capacity=80	
                                    #,num_threads=1
                                    )

	print "return batch"
	print train_image_batch
	print train_label_batch
	
	return train_image_batch, train_label_batch
##########

print "batch_size = ", FLAGS.batch_size
images, labels = distorted_inputs( batch_size = FLAGS.batch_size)

with tf.Session() as sess:
	
	tf.global_variables_initializer().run()
	
	##This class implements a simple mechanism to coordinate the termination of a set of threads.
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)
	
	image_tensor = sess.run(images)
	print image_tensor

			
	
	coord.request_stop()
	coord.join(threads)
	
		
