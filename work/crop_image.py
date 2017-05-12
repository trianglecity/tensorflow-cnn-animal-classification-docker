from PIL import Image

from os import listdir
from os.path import isfile, join

import fnmatch
import os
import math

matches = []

for root, dirs, filenames in os.walk(top = './animals', topdown=True):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        matches.append(os.path.join(root, filename))

min_width = 10000
min_height = 10000

for imagefile in matches:
	
	myimage = Image.open(imagefile)	
	width, height = myimage.size
	
	print (width, height), imagefile	
	if width < min_width:
		min_width = width

	if height < min_height:
		min_height = height


	
print min_width, min_height
crop_half_width = int(math.ceil(min_width/2))
crop_half_height = int(math.ceil(min_height/2))



for imagefile in matches:
	
	myimage = Image.open(imagefile)	
	width, height = myimage.size
		
	center_width = int(math.ceil(width/2))
	center_height = int(math.ceil(height/2))
	
	x0 = center_width - crop_half_width
	y0 = center_height - crop_half_height
	
	x1 = center_width + crop_half_width
	y1 = center_height + crop_half_height
	
	img2 = myimage.crop(( x0, y0, x1, y1))	
	
	img2.save(imagefile)
	
	


