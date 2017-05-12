
##
##  Tensorflow Animal Classification on Docker
##


NOTICE 1: the animal dataset is from deeplearing4j (https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/convolution/AnimalsClassification.java).

NOTICE 2: the source code is based on
 
		[1] cifar10.py
	
		[2] Davi Frossard, 2016,   VGG16 implementation in Tensorflow , http://www.cs.toronto.edu/~frossard/post/vgg16/   

		[3] http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/



[1] download (or git clone ) this source code folder.

[2] cd downloaded-source-code-folder

[3] sudo make BIND_DIR=. shell

[4] wait... wait ... then a bash shell (root@1d1515666a6f:/#) will be ready.

[5]  root@1d1515666a6f:/# cd /home/deeplearning/

[6]  root@1d1515666a6f:/home/deeplearning# ldd --version

		ldd (Ubuntu GLIBC 2.23-0ubuntu5) 2.23


[7]  root@1d1515666a6f:/home/deeplearning# git clone https://github.com/tensorflow/tensorflow.git --branch r1.1

[8]  root@1d1515666a6f:/home/deeplearning# cd tensorflow/

[9]  root@1d1515666a6f:/home/deeplearning/tensorflow# ./configure
	
	
	Please specify the location of python. [Default is /usr/bin/python]: 
	Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 
	Do you wish to use jemalloc as the malloc implementation? [Y/n] y
	jemalloc enabled
	Do you wish to build TensorFlow with Google Cloud Platform support? [y/N] n
	No Google Cloud Platform support will be enabled for TensorFlow
	Do you wish to build TensorFlow with Hadoop File System support? [y/N] n
	No Hadoop File System support will be enabled for TensorFlow
	Do you wish to build TensorFlow with the XLA just-in-time compiler (experimental)? [y/N] n
	No XLA JIT support will be enabled for TensorFlow
	Do you wish to build TensorFlow with VERBS support? [y/N] y
	VERBS support will be enabled for TensorFlow
	Found possible Python library paths:
	  /usr/local/lib/python2.7/dist-packages
	  /usr/lib/python2.7/dist-packages
	Please input the desired Python library path to use.  Default is [/usr/local/lib/python2.7/dist-packages]
	
	Using python library path: /usr/local/lib/python2.7/dist-packages
	Do you wish to build TensorFlow with OpenCL support? [y/N] n
	No OpenCL support will be enabled for TensorFlow
	Do you wish to build TensorFlow with CUDA support? [y/N] n
	No CUDA support will be enabled for TensorFlow
	Extracting Bazel installation...
	...........
	INFO: Starting clean (this may take a while). Consider using --async if the clean takes more than several minutes.
	Configuration finished
	
	
[10]  root@1d1515666a6f:/home/deeplearning/tensorflow# gcc -v

	
	gcc version 5.4.0 20160609 (Ubuntu 5.4.0-6ubuntu1~16.04.4) 
	

[11]  root@1d1515666a6f:/home/deeplearning/tensorflow# bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package 

	Target //tensorflow/tools/pip_package:build_pip_package up-to-date:
  	bazel-bin/tensorflow/tools/pip_package/build_pip_package
	INFO: Elapsed time: 433.238s, Critical Path: 432.46s

[12] rroot@1d1515666a6f:/home/deeplearning/tensorflow# bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

	Output wheel file is in: /tmp/tensorflow_pkg


[13] root@1d1515666a6f:/home/deeplearning/tensorflow# cd /tmp/tensorflow_pkg/
[14] root@1d1515666a6f:/tmp/tensorflow_pkg# ls (copy the .whl file name)

		tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl

[15] root@1d1515666a6f:/tmp/tensorflow_pkg# pip install ./tensorflow-1.1.0-cp27-cp27mu-linux_x86_64.whl 

[16] root@1d1515666a6f:/tmp/tensorflow_pkg# cd /home/deeplearning/cnn

[17] root@1d1515666a6f:/home/deeplearning/cnn# python -c 'import tensorflow as tf; print(tf.__version__)'

	1.1.0

[18] root@1d1515666a6f:/home/deeplearning/cnn# python ./cnn_animal_train.py

	The output looks somthing like this:

	...
	loss =  2.29115
	loss =  1.9481
	loss =  1.44655
	loss =  0.744662
	loss =  0.426226
	loss =  2.88021
	loss =  1.99395
	

	prediction
 	true =  bear
	
	bear 0.646353
	duck 0.150156
	turtle 0.121589
	deer 0.0819018

[19] root@1d1515666a6f:/home/deeplearning/cnn# python ./cnn_animal_prediction.py

	The output may look like this:

	 true =  bear

	duck 0.358927
	bear 0.331918
	deer 0.191455
	turtle 0.1177

 	true =  duck

	duck 0.358922
	bear 0.33191
	deer 0.191489
	turtle 0.117679

	 true =  bear

	duck 0.358922
	bear 0.331911
	deer 0.191469
	turtle 0.117698

	 true =  bear

	duck 0.358923
	bear 0.331908
	deer 0.191487
	turtle 0.117682

	 true =  turtle

	duck 0.358904
	bear 0.331913
	deer 0.191486
	turtle 0.117697

	 true =  bear

	duck 0.358914
	bear 0.331905
	deer 0.191481
	turtle 0.1177

	 true =  duck

	duck 0.358933
	bear 0.331904
	deer 0.191468
	turtle 0.117695
	...
	...

[20 cleanup] root@1d1515666a6f:/home/deeplearning/cnn# rm ./my_cnn_model.ckpt.*
[21 cleanup] root@1d1515666a6f:/home/deeplearning/cnn# rm ./checkpoint
[22 cleanup] root@1d1515666a6f:/home/deeplearning/cnn# cd ..
[23 cleanup] root@1d1515666a6f:/home/deeplearning# rm -rf ./tensorflow/
