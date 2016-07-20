##############################################################################
# Extract facial feature coordinates from the FERET text files
##############################################################################

def features_coordinates(txtDir):
	with open (txtDir) as file:
		for line in file:
			if 'left_eye_coordinates=' in line:
				s = line.split('=')
				left_eye_coords = map(int, s[1].split())
			if 'right_eye_coordinates' in line:
				s = line.split('=')
				right_eye_coords = map(int, s[1].split())
			if 'nose_coordinates=' in line:
				s = line.split('=')
				nose_coords = map(int, s[1].split())
			if 'mouth_coordinates=' in line:
				s = line.split('=')
				mouth_coords = map(int, s[1].split())
	return left_eye_coords, right_eye_coords, nose_coords, mouth_coords				
	

##############################################################################
# Find face window
##############################################################################

import numpy as np
import os, sys, scipy
import time
from PIL import Image, ImageDraw
from scipy import misc
from skimage.transform import pyramid_gaussian

			
SourceDir = ''
OutDir = ''

for root, dirs, files in os.walk(SourceDir):
    for name in files:
        ext = ['.jpg', '.gif']
        if name.endswith(tuple(ext)):
			path = os.path.join(root,name)
			foo = path.split('.')
			txtDir = foo[0] + '.txt'
			
			features = features_coordinates(txtDir)
			left_eye_x, left_eye_y = features[0]
			right_eye_x, right_eye_y = features[1]
			nose_x, nose_y = features[2]
			mouth_x, mouth_y = features[3]
			
			image = Image.open(path)
			max_height, max_width = np.array(image).shape[:2]
			
			# Gaussian pyramid scaling
			p = 3 # number of times scaled down
			
			scale_factor = 2
			image = tuple(pyramid_gaussian(image, downscale=scale_factor))
			
			# correction factor to bring face coordinates to scale
			correction = (scale_factor)**p
			image = Image.fromarray(np.uint8(image[p]*255))
			
			x_metric = left_eye_x-right_eye_x  # between eyes distance
			y_metric = mouth_y-(left_eye_y+right_eye_y)/2  # distance between mouth and eyes
			max_height /= correction
			max_width /= correction
			w, h = 2*x_metric/correction, 2*y_metric/correction
			
			# Make it a square
			#w = h = max(w,h)
			
			x = (right_eye_x-0.72*x_metric)/correction
			y = (right_eye_y-0.75*y_metric)/correction
			
			box = (int(max_width-w-x), int(y), int(max_width-x), int(y+h))  # (x, y, width, height)
			face = image.crop(box)
			
			# Rescale to "size"
			#size = 80, 80
			#face = face.resize(size)
			
			#face.show()
			
			draw = ImageDraw.Draw(image)
			draw.rectangle(box, outline = 'Chartreuse')
			del draw
			
			image.show()