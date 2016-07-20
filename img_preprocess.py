# Edge detection using the Sobel operator
def EdgeDetection(ImageDirectory): 
    original_image = scipy.misc.imread(ImageDirectory).astype('float')
    Gx = ndimage.sobel(original_image, 1)  # horizontal derivative 
    Gy = ndimage.sobel(original_image, 0)  # vertical derivative 
    theta = np.hypot(Gx, Gy)  # gradient direction
    edges = (theta-np.min(theta))/(np.max(theta)-np.min(theta))  # normalized
    return edges

##############################################################################
# Image Preprocessing - Edge detection, pyramid scaling, rotation
##############################################################################

import numpy as np
import os, sys, scipy
import time
from scipy import ndimage, misc
from skimage.transform import pyramid_gaussian
from pybrain.tools.xml.networkreader import NetworkReader
from skimage import exposure

t0 = time.time()
SourceDir = '...'
OutDir = '...'

net = NetworkReader.readFrom('/Users/.../trained_NN.xml')

for root, dirs, files in os.walk(SourceDir):
    for name in files:
        
        ext = ['.jpg', '.gif']
        if name.endswith(tuple(ext)):
            path = os.path.join(root,name)
            #img_E = EdgeDetection(path) 
            image = Image.open(path).convert('L')
			image = ImageOps.equalize(image)  # Histogram equalization
			max_height, max_width = np.array(image).shape
            scale_factor = 1.5  # downscaling factor 
            # Pyramid scaling
            pyramid = tuple(pyramid_gaussian(image, downscale=scale_factor))  
            correction = 1  # correction factor to adjust for pyramid scaling
            for p in xrange(15):  # (len(pyramid) would scale it down too much)                
                rotation = 90  # degrees of image rotation
                for r in xrange(-rotation/2, rotation/2 + 1, 36):
  
                    # Rotated image
                    img_ESR = scipy.misc.imrotate(pyramid[p], r)  
                    img_array = Image.fromarray(np.uint8(img_ESR[p]*255))
                    
                    # Image scanning
                    x, y = 0, 0  # Scanning coordinates
                    step = 5  # Pixel step
                    max_height /= correction;  max_width /= correction
                    
                    while y <= max_height - h:
                        while x <= max_width - w:
                            rectangle = (int(max_width-w-x), int(y), \
                                         int(max_width-x), int(y+h))  
                            scan_window = img_array.crop(rectangle)
                            
                            if net.activate(img_array[0])> 0.5:
                                face = scan_window
                                positive = True
                            x += step  # move to the left
                        x = 0  # reset x axis 
                        y += step  # and move down	
                    correction = scale_factor 
                    
print(time.time()-t0)