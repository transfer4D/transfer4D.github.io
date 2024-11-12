# The code in this contains methods to load data and generate deformable graph for Fusion Experiments 

# Library imports	
import os
import json
import numpy as np
from scipy.io import loadmat 

# Modules from Neural Tracking 
from model import dataset
import options as opt
from utils import image_proc




class RGBDVideoLoader:
	def __init__(self, seq_dir):
		
		# Find the sequence split and name of the sequence 
		self.seq_dir = seq_dir
		self.split, self.seq_name = list(filter(lambda x: x != '', seq_dir.split('/')))[-2:]
		# Load internics matrix
		intrinsics_matrix = np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))
		self.intrinsics = {
			"fx": intrinsics_matrix[0, 0],
			"fy": intrinsics_matrix[1, 1],
			"cx": intrinsics_matrix[0, 2],
			"cy": intrinsics_matrix[1, 2]
		}

		# Check all elements in folder are #<number>.{png,jpg} files
		# assert 

		self.images_path = list(sorted(os.listdir(os.path.join(seq_dir, "depth")), key=lambda x: int(x.split('.')[0])  ))


	def get_frame_path(self,index):
		# Return the path to color, depth and mask
		return os.path.join(self.seq_dir,"color",self.images_path[index]),\
			os.path.join(self.seq_dir,"depth",self.images_path[index].replace('jpg','png')),\
			os.path.join(self.seq_dir,"mask",self.images_path[index].replace('jpg','png'))


	def get_source_data(self,source_frame):
		# Source color and depth
		src_color_image_path,src_depth_image_path,src_mask_image_path = self.get_frame_path(source_frame)
		source, _, cropper,intrinsics_cropped = dataset.DeformDataset.load_image(
			src_color_image_path, src_depth_image_path, self.intrinsics, opt.image_height, opt.image_width)

		# Load cropped mask image 
		mask = dataset.DeformDataset.load_mask(src_mask_image_path,cropper=cropper)

		source_data = {}
		source_data["id"]					= source_frame
		source_data["im"]					= source # Source Image (6xHxW)
		source_data["mask"]					= mask
		source_data["cropper"]				= cropper
		source_data["intrinsics"]			= intrinsics_cropped

		return source_data


	def get_target_data(self,target_frame,cropper):
		# Target color and depth (and boundary mask)
		tgt_color_image_path,tgt_depth_image_path,tgt_depth_mask_path = self.get_frame_path(target_frame)
		target, target_boundary_mask, _,intrinsics_cropped = dataset.DeformDataset.load_image(
			tgt_color_image_path, tgt_depth_image_path, self.intrinsics, opt.image_height, opt.image_width, cropper=cropper,
			max_boundary_dist=opt.max_boundary_dist, compute_boundary_mask=True)

		target_normals = dataset.DeformDataset.estimate_normals_from_depth(target[-1])

		target_data = {}
		target_data["id"]					= target_frame
		target_data["im"]					= target # Target Image, (6xHxW)
		target_data["target_boundary_mask"]	= target_boundary_mask
		target_data["target_mask"] 			= dataset.DeformDataset.load_mask(tgt_depth_mask_path) # Mask for target frame if avaible else None
		target_data["normal"]				= target_normals # Target Image, (3xHxW)
		return target_data 				

	def __len__(self):
		return len(self.images_path)

	