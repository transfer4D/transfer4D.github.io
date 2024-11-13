# Generate binary mask using RGBD image + depth image
# Use cv2.GRABCUT for foreground selection for RGB Image 
# Check if already segmented in depth images  

import os
import sys
import cv2
import argparse
from skimage import io
from skimage import morphology
import numpy as np
import matplotlib.pyplot as  plt

def generate_mask(method,data):
	if method == "otsu":
		return generate_mask_depth(data[-1,...])
	elif method == "color":
		return generate_mask_color(np.moveaxis(data[:3,...],0,-1))
	else:
		NotImplementedError(f"Method:{method} not implemented")

def generate_mask_depth(depth_image,cropper=None):
	# depth_image = cv2.imread(depth_image_path,0)

	if cropper is not None:
		depth_image = cropper(depth_image)

	depth_image[depth_image == 0] = depth_image.max()
	ret2,th = cv2.threshold(depth_image,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	return (1 - th).astype(bool)


if __name__ == "__main__":
	deepdeform_dataset_path = sys.argv[1]
	depth_dir = os.path.join(deepdeform_dataset_path,"depth")
	color_dir = os.path.join(deepdeform_dataset_path,"color")
	mask_dir = os.path.join(deepdeform_dataset_path,"mask")
	bg_dir = os.path.join(deepdeform_dataset_path,"bg")

	# Use fig to visualize
	fig = plt.figure()
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)

	image_file_list = sorted(os.listdir(depth_dir),key=lambda x: int(x.split('.')[0]))


	if not os.path.isdir(mask_dir):
		os.mkdir(mask_dir)


	# Generate segmentation mask 
	for file in image_file_list:
		break
		im_depth= io.imread(os.path.join(depth_dir,file))

		if os.path.isfile(os.path.join(mask_dir,file)):
			mask = io.imread(os.path.join(mask_dir,file))
		else:		
			
			print("Generating mask",file)
			mask = im_depth > 0 # Generate mask based on thresholding the depth image

			# Use floodfill to merge holes inside segmented region
			mask = (255*mask).astype(np.uint8)
			flood_fill_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),dtype=mask.dtype) # Initialize floodFill mask (needs to have 2 size greater than image)
			cv2.floodFill(mask.copy(),flood_fill_mask,(0,0),255) # Run cv2 flood fill
			flood_fill_mask = flood_fill_mask[1:-1,1:-1] # Remove extra pixels

			mask = np.logical_not(flood_fill_mask) # Output after floodfill
			
			# Run Morpological opening 10 iterations (should be enough to converge)
			for _ in range(20):
				mask = morphology.binary_opening(mask,np.ones((3,3)))
			mask = (255*mask).astype(np.uint8)


			io.imsave(os.path.join(mask_dir,file),mask)


		plt.cla()
		ax1.imshow(im_depth)
		ax2.imshow(mask)
		ax1.set_title(file)
		plt.pause(0.0001)
		plt.draw()


	if not os.path.isdir(bg_dir):
		os.mkdir(bg_dir)	

	# Generate background image / frame for each timestep
	# Assume uniform lighting and merge all backgroun to single image 
	bg_image = None
	bg_weights = None
	weight_parameter = 1. # (Improtance parameter for each weight)

	for file in image_file_list:

		im_rgb = io.imread(os.path.join(color_dir,file))

		if os.path.isfile(os.path.join(bg_dir,file)):
			bg_image = io.imread(os.path.join(bg_dir,file))
		else:	
			mask = io.imread(os.path.join(mask_dir,file))

			# Erosion and dialiation not helpful in handling iamge boundary effectively 
			# for _ in range(3):
			# 	mask = morphology.binary_dilation(mask,np.ones((3,3)))
			# for _ in range(3):
			# 	mask = morphology.binary_erosion(mask,np.ones((3,3)))
			print("Generating background image",file)

			assert mask.shape == im_rgb.shape[:2], "RGB Image and Mask don't belong to the same shape" 

			# If not defined create a new image	
			if bg_image is None:
				bg_image = np.zeros_like(im_rgb)
				bg_weights = np.zeros_like(mask,dtype=np.float64)

			# Similiar to volume fusion (Curless 96) use a moving average to update background image.   	
			h,w = np.where(mask==0)		
			bg_image[h,w,:] = (bg_image[h,w,:]*bg_weights[h,w,None] + im_rgb[h,w,:]*weight_parameter)/(bg_weights[h,w,None] + weight_parameter)
			bg_weights[h,w] += weight_parameter
			
			io.imsave(os.path.join(bg_dir,file),bg_image)
	
		plt.cla()
		ax1.imshow(im_rgb)
		ax2.imshow(bg_image)
		ax1.set_title(file)
		plt.pause(0.0001)
		plt.draw()

	# Code to create new merged images
	for file_ind,file in enumerate(image_file_list):
		if file_ind == 0: 
			continue
		im_rgb = io.imread(os.path.join(color_dir,file))
		bg_image = io.imread(os.path.join(bg_dir,image_file_list[-1]))
		mask = io.imread(os.path.join(mask_dir,file))[...,None]
		mask = mask > 0 
		merged_image = mask*im_rgb + (1-mask)*bg_image

		plt.cla()
		ax1.imshow(im_rgb)
		ax2.imshow(merged_image)
		ax1.set_title(file)
		plt.pause(0.0001)
		plt.draw()




