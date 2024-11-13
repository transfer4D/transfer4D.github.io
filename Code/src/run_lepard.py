# The code in the this file creates a class to run deformnet using torch
import os
import sys
import json
import torch
import numpy as np
import logging

# Modules (make sure modules are visible in sys.path)
from lepard.inference import Lepard	


import options as opt

class Lepard_runner():
	"""
		Runs deformnet to outputs result
	"""
	def __init__(self,vis,fopt):
		"""
			@params:
				vis: Visualizer  
				fopt: 
					parameters for fusion  	

		"""
		#####################################################################################################
		# Options
		#####################################################################################################
		self.fopt = fopt
		self.frame_id = fopt.source_frame
		self.lepard = Lepard(os.path.join(os.getcwd(),"../lepard/configs/test/4dmatch.yaml"))

		# Set logging level 
		self.log = logging.getLogger(__name__)

		self.vis = vis

	def __call__(self,target_data):
		"""
			Given Source and Target PC, estimate optical flow returns, 
				the displacement of each pixel from source image to target image in the for of a flow field pixel grid 
				and the target x,y,z position given input pixel position    
			
			@params: 
				source_data: 
					id: Frame number for source image 
					im: 6xHxw np.ndarray: RGBD + XYZ for source image
				target: 
					id: Frame number for target image 
					im: 6xHxw np.ndarray: RGBD + XYZ for target image

			@returns: 
				scene_flow_data: 
					target_matches: 3xHxW torch.Tensor, given x,y location, returns the predicted location in the target image, so returns xyz
					source_id: Source frame used in optical flow 
					target_id: Target frame used in optical flow 
	
		"""

		# assert self.frame_id + self.fopt.skip_rate == target_data["id"]

		# Move to device and unsqueeze in the batch dimension (to have batch size 1)
		source_vertices,source_faces,source_color,source_normals = self.tsdf.get_deformed_model()
		
		target_mask = target_data["im"][-1] > 0	
		target_pcd = target_data["im"][3:,target_mask].T

		# print("Mean position of source and target:")
		# print(np.mean(source_vertices,axis=0))
		# print(np.mean(target_pcd,axis=0))

		scene_flow,corresp,valid_verts = self.lepard(source_vertices,target_pcd)
		target_matches = source_vertices.copy()
		target_matches[valid_verts] += scene_flow[valid_verts]

		scene_flow_data = {'source':source_vertices,'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp}	
		return scene_flow_data