import os
import sys
import torch
import numpy as np 
from skimage import io, transform

sys.path.append(os.getcwd())

from lepard.main import load_config
from lepard.datasets.dataloader import collate_fn_4dmatch_inference
from lepard.lib.tester import blend_anchor_motion,blend_anchor_motion_confidence

from lepard.models.pipeline import Pipeline
from lepard.models.matching import Matching as CM

import open3d as o3d



class Lepard:

	def __init__(self, config_filepath):
		self.config = load_config(config_filepath)
		self.pipeline = Pipeline(self.config)
		print(f"[Warning]: Not loading lepard from:{os.path.join(os.path.dirname(__file__),self.config.pretrain)}")
		# state = torch.load(os.path.join(os.path.dirname(__file__),self.config.pretrain))
		# self.pipeline.load_state_dict(state['state_dict'])

		self.pipeline = self.pipeline.to(self.config.device)

		self.pipeline.eval()
		self.neighborhood_limits = [25, 27 ,29, 29] # Precalculated on the training set of DeformingThing4D


	@staticmethod    
	def get_pcd_from_depth_image(depth,intrinsic_params,extrinsics=None):
		"""
			Calculate the point cloud from depth image. 
			Note:- We are assuming the depth image is already segmented, i.e values outside silhouette is 0
		"""

		if type(intrinsic_params) == str:
			intrinsic_params = np.loadtxt(intrinsic_params)

		if len(intrinsic_params) == 4: # List form
			fx,fy,cx,cy = intrinsic_params
		elif len(intrinsic_params) == 3: 
			assert type(intrinsic_params) == np.ndarray, f"Intrinsic Matric must be a numpy array. Found:{intrinsic_params}" 		
			fx = intrinsic_params[0,0]
			fy = intrinsic_params[1,1]
			cx = intrinsic_params[2,0]
			cy = intrinsic_params[2,1]

		if extrinsics is None: 
			extrinsics = np.eye(4)
		elif type(extrinsics) == str:
			extrinsics = np.loadtxt(extrinsics)


		if type(depth) == str: 
			depth = io.imread(depth)

		mask = depth > 0

		pcd_y,pcd_x = np.where(mask)	
		pcd_z = depth[pcd_y,pcd_x]/1000
		pcd_x = pcd_z * (pcd_x - cx) / fx
		pcd_y = pcd_z * (pcd_y - cy) / fy

		pcd = np.array([pcd_x,pcd_y,pcd_z]).T

		pcd = extrinsics[:3,:3].T@(pcd.T - extrinsics[:3,3:4]) 

		return pcd.T


	def __call__(self, source_pcd,target_pcd,thr=0.1,gr_data=None):
		"""
			Run their pipeline to get result on random source and target point cloud
			This module should do all the dataloader preprocesing,
			pass through their pipeline,   
			
		"""
		source_features = np.ones_like(source_pcd[:,:1],dtype=np.float32)	
		target_features = np.ones_like(target_pcd[:,:1],dtype=np.float32)	

		inputs = collate_fn_4dmatch_inference([(source_pcd,target_pcd,source_features,target_features)],self.config['kpfcn_config'],neighborhood_limits=self.neighborhood_limits) # Pass a single batch size input

		# From cpu to gpu
		if self.config.gpu_mode:

			for k, v in inputs.items():
				# print(k)
				if type(v) == list:
					inputs[k] = [item.to(self.config.device) for item in v]
					# print([x.shape for x in v])

				elif type(v) in [ dict, float, type(None), np.ndarray]:
					pass
				else:
					inputs[k] = v.to(self.config.device)
					# print(inputs[k].shape)				
				# print(inputs[k])
		
		with torch.no_grad():
			output = self.pipeline(inputs,timers=self.config.timers)

		# for k in output:
		# 	print(k)
		# 	if type(output[k]) == list:
		# 		print([x.shape for x in output[k]])
		# 	elif type(output[k]) == dict:
		# 		try:
		# 			print([output[k][x].shape for x in output[k]])
		# 		except: 
		# 			continue
		# 	else:	
		# 		print(k,output[k].shape)
		# 	print(output[k])

		pcd = inputs['points'][-2].cpu().numpy()
		lenth = inputs['stack_lengths'][-2][0]
		src_pcd = pcd[:lenth]
		tgt_pcd = pcd[lenth:]

		# for thr in [  0.05, 0.1, 0.2]:
		match_pred, match_confidence, _ = CM.get_match(output['conf_matrix_pred'], thr=thr, mutual=True)
		match_pred = match_pred.cpu().data.numpy()
		match_confidence = match_confidence.cpu().data.numpy()




		# if gr_data is not None:
		# 	calulcate_losses(thr,match_pred,output)

		# print(match_pred.shape)	
		# self.plot_corresp(src_pcd,tgt_pcd,match_pred)

		assert len(np.unique(match_pred[:,1])) == match_pred.shape[0] 

		self.query_points = src_pcd[match_pred[:,1]]
		self.sceneflow_points = tgt_pcd[match_pred[:,2]] - src_pcd[match_pred[:,1]]
		self.target_points = tgt_pcd
		self.match_confidence_score = match_confidence

		scene_flow,corresp,mask,match_conf = self.find_scene_flow(source_pcd)

		# print(scene_flow.shape,match_conf.shape)



		return scene_flow,corresp,mask,match_conf

	
	def find_scene_flow(self,source_pcd):

		# scene_flow,corresp,mask,confidence_score = blend_anchor_motion(source_pcd,self.query_points,self.sceneflow_points,self.match_confidence_score,knn=3,search_radius=1e-2)
		scene_flow,corresp,mask,confidence_score = blend_anchor_motion(source_pcd,self.query_points,self.sceneflow_points,self.match_confidence_score,knn=3,search_radius=1e-1)
		# scene_flow,corresp,mask,confidence_score = blend_anchor_motion_confidence(source_pcd,self.query_points,self.sceneflow_points,self.match_confidence_score,knn=3,search_radius=1e-1)
		# self.plot_scene_flow(source_pcd[mask],self.target_points,scene_flow[mask],corresp)
		return scene_flow,corresp,mask,confidence_score	

	def plot_scene_flow(self,source_pcd,target_pcd,scene_flow,corresp):

		# scene_flow[:,0] -= 1
		# target_pcd[:,0] -= 1

		# source_pcd = source_pcd[::3]
		# scene_flow = scene_flow[::3]
		# target_pcd = target_pcd[::3]


		source_rendered_pcd = o3d.geometry.PointCloud()
		source_rendered_pcd.points = o3d.utility.Vector3dVector(source_pcd)
		source_rendered_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 255/255, 0/255] for i in range(len(source_pcd))]))

		target_rendered_pcd = o3d.geometry.PointCloud()
		target_rendered_pcd.points = o3d.utility.Vector3dVector(target_pcd)
		target_rendered_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 0/255, 255/255] for i in range(len(target_pcd))]))


		n_match_matches = source_pcd.shape[0]
		match_matches_points = np.concatenate([source_pcd, source_pcd+scene_flow], axis=0)
		match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]

		# --> Create match (unweighted) lines 
		match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
		match_matches_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(match_matches_points),
			lines=o3d.utility.Vector2iVector(match_matches_lines),
		)
		match_matches_set.colors = o3d.utility.Vector3dVector(match_matches_colors)


		o3d.visualization.draw_geometries([source_rendered_pcd,target_rendered_pcd,match_matches_set])


	def plot_corresp(self,source_pcd,target_pcd,match_pred):

		# source_pcd[:,0] -= 1

		source_rendered_pcd = o3d.geometry.PointCloud()
		source_rendered_pcd.points = o3d.utility.Vector3dVector(source_pcd)
		source_rendered_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 255/255, 0/255] for i in range(len(source_pcd))]))

		target_rendered_pcd = o3d.geometry.PointCloud()
		target_rendered_pcd.points = o3d.utility.Vector3dVector(target_pcd)
		target_rendered_pcd.colors = o3d.utility.Vector3dVector(np.array([[0/255, 0/255, 255/255] for i in range(len(target_pcd))]))

		n_match_matches = match_pred.shape[0]
		match_matches_points = np.concatenate([source_pcd[match_pred[:,1]], target_pcd[match_pred[:,2]]], axis=0)
		match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]

		# --> Create match (unweighted) lines 
		match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
		match_matches_set = o3d.geometry.LineSet(
			points=o3d.utility.Vector3dVector(match_matches_points),
			lines=o3d.utility.Vector2iVector(match_matches_lines),
		)

		o3d.visualization.draw_geometries([source_rendered_pcd,target_rendered_pcd,match_matches_set])



		# inlier_thr = recall_thr = 0.04

		# ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=inlier_thr, s2t_flow=data['coarse_flow'][0][None] )[0]

		# nrfmr = compute_nrfmr(match_pred, data, recall_thr=recall_thr)

		# IR += ir
		# NR_FMR += nrfmr

		# n_sample += match_pred.shape[0]

		# print( "conf_threshold", thr,  "NFMR:", NR_FMR, " Inlier rate:", IR, "Number sample:", n_sample)


if __name__ == "__main__":
	sample_path = sys.argv[1]
	lepard = Lepard("lepard/configs/test/4dmatch.yaml")

	# Test 1: With original data
	# data = np.load(sample_path)
	# for k in data:
	# 	print(k,data[k].shape)
	# corresp_data = lepard(data["s_pc"],data["t_pc"],gr_data=None)
	# print(data["correspondences"])

	# Test 2: With raw depth image
	sample_info = os.path.basename(sample_path)
	sample_name = os.path.basename(os.path.dirname(sample_path))
	raw_path = os.path.join("/media/srialien/Elements/AT-Datasets/4DMatch/4dmatch_raw_depth/raw/",sample_name)

	source_camera,source_index,target_camera,target_index = sample_info.split('.')[0].split('_')

	# Get depth image
	source_camera_intrinsics = os.path.join(raw_path,f"{source_camera}intr.txt")
	source_camera_extrinsics = os.path.join(raw_path,f"{source_camera}extr.txt")
	source_image_path = os.path.join(raw_path,"depth",f"{source_camera}_{source_index}.png")

	target_camera_intrinsics = os.path.join(raw_path,f"{target_camera}intr.txt")
	target_camera_extrinsics = os.path.join(raw_path,f"{target_camera}extr.txt")
	target_image_path = os.path.join(raw_path,"depth",f"{target_camera}_{target_index}.png")

	source_pcd = lepard.get_pcd_from_depth_image(source_image_path,source_camera_intrinsics,extrinsics=source_camera_extrinsics)
	target_pcd = lepard.get_pcd_from_depth_image(target_image_path,target_camera_intrinsics,extrinsics=target_camera_extrinsics)

	corresp_data = lepard(source_pcd,target_pcd,gr_data=None)

