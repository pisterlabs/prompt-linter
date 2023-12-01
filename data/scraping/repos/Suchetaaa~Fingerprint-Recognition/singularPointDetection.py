import cv2
import numpy as np
from math import pi
from coherenceOrientationField import *
import os
from tqdm import tqdm

def singularPointDetection(path):	
	#returns the singular points
	image = cv2.imread(path)
	return Singular_point_detection(image)

i = 0
j = 0
k = 0
l = 0

# image_names1 = os.listdir('/home/neharika/Desktop/DIP/Project/DB1_B')
# image_names1.sort()
# image_names2 = os.listdir('/home/neharika/Desktop/DIP/Project/DB2_B')
# image_names1.sort()
# image_names3 = os.listdir('DB3_B/')
image_names4 = os.listdir('/home/neharika/Desktop/DIP/Project/DB4_B')
image_names4.sort()

singularPoint_1 = (-1) * np.ones((80, 2))
singularPoint_2 = (-1) * np.ones((80, 2))
singularPoint_3 = (-1) * np.ones((80, 2))
singularPoint_4 = (-1) * np.ones((80, 2))

# for images in tqdm(image_names1):
# 	print(images)
# 	(a, b) = singularPointDetection('/home/neharika/Desktop/DIP/Project/DB1_B/' + images)
# 	singularPoint_1[i,0] = a
# 	singularPoint_1[i, 1] = b
# 	print((a,b))
# 	i = i+1

	
# for images in tqdm(image_names2):
# 	# print(images)
# 	(a, b) = singularPointDetection('/home/neharika/Desktop/DIP/Project/DB2_B/' + images)
# 	singularPoint_2[i,0] = a
# 	singularPoint_2[i, 1] = b
# 	print((a,b))
# 	i = i+1

# for images in tqdm(image_names3):
# 	(a, b) = singularPointDetection('/home/neharika/Desktop/DIP/Project/DB3_B' + images)
# 	singularPoint_3[k,0] = a
# 	singularPoint_3[k, 1] = b
# 	k = k+1
	
for images in tqdm(image_names4):
	print(images)
	(a, b) = singularPointDetection('/home/neharika/Desktop/DIP/Project/DB4_B/' + images)
	singularPoint_4[i,0] = a
	singularPoint_4[i, 1] = b
	print((a,b))
	i = i+1
	

#np.save('singularPoint_1.npy', singularPoint_1)
# np.save('singularPoint_2.npy', singularPoint_2)
# np.save('singularPoint_3.npy', singularPoint_3)
np.save('singularPoint_4.npy', singularPoint_4)
	