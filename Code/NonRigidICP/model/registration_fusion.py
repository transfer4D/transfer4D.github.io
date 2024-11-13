import os
import cv2
import numpy as np
import logging
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage import io
from PIL import Image
from timeit import default_timer as timer
from easydict import EasyDict as edict
import datetime
import argparse
from .geometry import *

from scipy.spatial.transform import Rotation as R
import yaml
import matplotlib.pyplot as plt
import torch
from lietorch import SO3, SE3, LieGroupParameter
import torch.optim as optim
from .loss import *
from .point_render import PCDRender,MeshRender

def load_config(config_path):
	with open(config_path,'r') as f:
		config = yaml.load(f, Loader=yaml.Loader)
	config = edict(config)  

	if config.gpu_mode:
		config.device = torch.device("cuda:0")
	else:
		config.device = torch.device('cpu')

	return config


class Registration():


	def __init__(self, graph,warpfield, K,vis,opt,render_type="pcd"):

		self.opt = opt

		self.config = load_config(opt.nrrConfig)
		self.device = self.config.device

		self.deformation_model = self.config.deformation_model
		self.intrinsics = K

		self.graph = graph
		self.warpfield = warpfield 

		# Update/Create graph nodes in pytorch 
		self.update()

		"""define differentiable pcd renderer"""
		# self.renderer = PCDRender(K, img_size=image_size)
		if render_type == "pcd":
			self.renderer = PCDRender(K)
		elif render_type == "mesh":
			self.renderer = MeshRenderer(K)
		else: 
			raise NotImplementedError("Can only render point cloud of mesh. Pass:mesh,pcd as render_type")

		self.vis = vis

		self.prev_R = None
		self.prev_rot = None
		self.prev_trans = None

		self.log = logging.getLogger(__name__)

		os.makedirs(os.path.join(self.vis.opt.datadir,"results","optimization_convergence_info"),exist_ok=True)

	def update(self):

		"""initialize deformation graph"""
		self.graph_nodes = torch.from_numpy(self.graph.nodes).to(self.device)
		self.graph_edges = torch.from_numpy(self.graph.edges).long().to(self.device)
		self.graph_edges_weights = torch.from_numpy(self.graph.edges_weights).to(self.device)
		# self.graph_clusters = torch.from_numpy(self.graph.clusters).long() #.to(self.device)

		# print("Updating parameters for registration:",self.graph_nodes.shape,self.graph_edges.shape,self.graph_edges_weights.shape,self.graph_clusters.shape)

	def optimize(self, optical_flow_data,scene_flow_data,complete_node_motion_data,target_frame_data,target_tsdf=None, landmarks=None):
		"""
		:param tgt_depth_path:
		:return:
		"""

		# Only those matches for which source has valid skinning and valid scenflow based on calibration
		source_pcd = scene_flow_data["source"]	
		vert_anchors,vert_weights,valid_skinning,vert_uncertainity = self.warpfield.skin(source_pcd)
		self.log.debug(f"Source points:{source_pcd.shape}")


		if 'confidence' in scene_flow_data and self.config.iters > 0:
			valid_verts = np.logical_and(valid_skinning,scene_flow_data["valid_verts"])
		else:
			valid_verts = valid_skinning
		source_pcd = source_pcd[valid_verts]

		vert_anchors = vert_anchors[valid_verts]
		vert_weights = vert_weights[valid_verts]

		self.log.debug(f"Source points after skinning:{source_pcd.shape}")
		self.log.debug(f"Anchors points after skinning:{vert_anchors.shape}")
		self.log.debug(f"Weights points after skinning:{vert_weights.shape}")

		self.log.debug(f"Valid Skinning:{np.sum(valid_skinning)}/{valid_skinning.shape[0]}")

		self.log.debug(f"Valid Verts:{np.sum(valid_verts)}/{valid_verts.shape[0]}")


		source_pcd = torch.from_numpy(source_pcd).float().to(self.device)



		"""initialize point clouds"""
		point_anchors = torch.from_numpy(vert_anchors).long().to(self.device)
		anchor_weight = torch.from_numpy(vert_weights).to(self.device)
		anchor_loc = self.graph_nodes[point_anchors].to(self.device)
		frame_point_len = [ len(source_pcd)]

		"""pixel to pcd map"""
		# self.pix_2_pcd_map = [ self.map_pixel_to_pcd(valid_pixels).to(self.config.device) ] # TODO


		"""load target frame"""
		if "pcd" in target_frame_data:
			tgt_pcd = torch.from_numpy(target_frame_data['pcd']).float().to(self.device)
			tgt_depth = None
			depth_mask = None
		elif "im" in target_frame_data:	
			tgt_depth = target_frame_data["im"][-1]
			depth_mask = torch.from_numpy(tgt_depth > 0)
			tgt_pcd = depth_2_pc(tgt_depth, self.intrinsics).transpose(1,2,0)
			tgt_pcd = torch.from_numpy( tgt_pcd[ tgt_depth >0 ] ).float().to(self.device)
		else:
			raise KeyError(f"Unable to extract point cloud from target_frame_data, keys avialable:{target_frame_data.keys()}")


		# pix_2_pcd = self.map_pixel_to_pcd( depth_mask ).to(self.device)

		# load target matches
		target_matches = scene_flow_data["target_matches"]
		target_matches = target_matches[valid_verts] 
		target_matches = torch.from_numpy(target_matches).float().to(self.device)

		assert target_matches.shape[0] == source_pcd.shape[0], f"Valid Target Matches:{target_matches.shape[0]} does not match valid source points:{source_pcd.shape[0]}"
		# Only those matches which are not outside region of cyclic consistancy
		target_match_confidence = None
		if 'confidence' in scene_flow_data and self.config.iters > 0:
			if scene_flow_data['confidence'] is not None:
				target_match_confidence = torch.from_numpy(scene_flow_data['confidence'][valid_verts]).float().to(self.device)
				assert target_match_confidence.shape[0] == source_pcd.shape[0], f"Valid Target Matche confidence:{target_match_confidence.shape[0]} does not match valid source points:{source_pcd.shape[0]}"

		landmarks = np.array([(i,i) for i in range(source_pcd.shape[0])]).T
		# if landmarks is not None: # Marks the correspondence between source and the target pixel in depth images
		#     s_uv , t_uv = landmarks
		#     s_id = self.pix_2_pcd_map[-1][ s_uv[:,1], s_uv[:,0] ]
		#     t_id = pix_2_pcd [ t_uv[:,1], t_uv[:,0]]
		#     valid_id = (s_id>-1) * (t_id>-1)
		#     s_ldmk = s_id[valid_id]
		#     t_ldmk = t_id[valid_id]

		#     landmarks = (s_ldmk, t_ldmk)

		# self.visualize_results(self.tgt_pcd)

		target_graph_node_location, target_graph_node_confidence = complete_node_motion_data
		target_graph_node_location = torch.from_numpy(target_graph_node_location).to(self.device) if target_graph_node_location is not None else None
		target_graph_node_confidence = torch.from_numpy(target_graph_node_confidence).to(self.device) if target_graph_node_confidence is not None else None

		estimated_transforms = self.solve(source_pcd,tgt_pcd,target_matches,
			point_anchors,anchor_weight,anchor_loc,
			tgt_depth,depth_mask,
			landmarks=landmarks,
			target_match_confidence=target_match_confidence,
			target_graph_node_location=target_graph_node_location,
			target_graph_node_confidence=target_graph_node_confidence)
		# self.visualize_results( self.tgt_pcd, estimated_transforms["warped_verts"])

		# voxel misalignment error similar to fusion4D
		if target_tsdf is not None:
			per_vertex_usdf = target_tsdf.sdf_misalignment(estimated_transforms["warped_verts"])

			graph_sdf_alignment_cost = np.zeros(self.graph_nodes.shape[0])	
			graph_weight_sum = np.zeros(self.graph_nodes.shape[0])	
			# Graph error similiar to function4D
			for j in range(vert_anchors.shape[1]):

				anchor_mask = vert_anchors[:,j] != -1

				graph_sdf_alignment_cost[vert_anchors[:,j][anchor_mask]] += per_vertex_usdf*vert_weights[anchor_mask,j]
				graph_weight_sum[vert_anchors[:,j][anchor_mask]] += vert_weights[anchor_mask,j]

			self.graph_sdf_alignment = np.ones(self.graph_nodes.shape[0])
			self.graph_sdf_alignment[graph_weight_sum > 0] = graph_sdf_alignment_cost[graph_weight_sum > 0] / graph_weight_sum[graph_weight_sum > 0]	

			# print("Graph Error:",self.graph_sdf_alignment)


			estimated_transforms["graph_sdf_alignment"] = self.graph_sdf_alignment
		else: 
			estimated_transforms["graph_sdf_alignment"] = None


		warped_pcd = scene_flow_data["source"].copy()
		vert_anchors,vert_weights,valid_skinning,vert_uncertainity = self.warpfield.skin(warped_pcd)

		# Incase some verts were not assigned any graph
		warped_pcd[valid_skinning] = self.deform_ED(warped_pcd[valid_skinning],vert_anchors[valid_skinning],vert_weights[valid_skinning])[0]

		estimated_transforms['warped_verts'] = warped_pcd
		estimated_transforms["source_frame_id"] = optical_flow_data["source_id"]
		estimated_transforms["target_frame_id"] = optical_flow_data["target_id"]
		estimated_transforms["valid_verts"] = valid_skinning

		# print(estimated_transforms)
		return estimated_transforms



	def solve(self,*args, **kwargs ):


		if self.deformation_model == "ED":
			# Embeded_deformation, c.f. https://people.inf.ethz.ch/~sumnerb/research/embdef/Sumner2007EDF.pdf
			return self.optimize_ED(*args,**kwargs)

	
	def deform_ED(self,points,anchors,weights,batch_size=128**3):

		# convert to pytorch
		return_points = points.copy()
		return_graph_error = np.ones(points.shape[0])
		return_graph_sdf_alignment = np.ones(points.shape[0])
		with torch.no_grad():

			# Run in batches since above certain points leads to out of memory issue
			chunks = [ (i,min(points.shape[0],i + batch_size)) for i in range(0, points.shape[0], batch_size)]
			
			for chunk in chunks:  
				l,r = chunk
				points_batch = torch.from_numpy(points[l:r]).to(self.device)
				weights_batch = torch.from_numpy(weights[l:r]).to(self.device)
				anchors_batch = torch.from_numpy(anchors[l:r]).long().to(self.device)

				# print(self.t.shape)
				# print(anchors_batch.shape)

				# print(torch.sum(anchors_batch > 0))

				anchor_trn_batch = self.t[anchors_batch]
				# print(anchor_trn_batch.shape)
				anchor_rot_batch = self.R[anchors_batch]
				anchor_loc_batch = self.graph_nodes[anchors_batch]

				warped_points_batch = ED_warp(points_batch, anchor_loc_batch, anchor_rot_batch, anchor_trn_batch, weights_batch)        


				return_points[l:r] = warped_points_batch.detach().cpu().numpy() # Returns points padded with invalid points

				if hasattr(self,"graph_sdf_alignment"):
					return_graph_error[l:r] = torch.sum(weights_batch*self.graph_error[anchors_batch],dim=1).detach().cpu().numpy()

					return_graph_sdf_alignment[l:r] = np.sum(weights[l:r]*self.graph_sdf_alignment[anchors[l:r]],axis=1)			
					print("SDF alignment:",return_graph_sdf_alignment[l:r])		
			
			return return_points,return_graph_error,return_graph_sdf_alignment


	def deform_normals(self,normals,anchors,weights,batch_size=128**3):

		# convert to pytorch
		return_normals = normals.copy()
		with torch.no_grad():

			# Run in batches since above certain points leads to out of memory issue
			chunks = [ (i,min(normals.shape[0],i + batch_size)) for i in range(0, normals.shape[0], batch_size)]
			
			for chunk in chunks:  
				l,r = chunk
				normal_batch = torch.from_numpy(normals[l:r]).to(self.device)
				weights_batch = torch.from_numpy(weights[l:r]).to(self.device)
				anchors_batch = torch.from_numpy(anchors[l:r]).long().to(self.device)

				anchor_rot_batch = self.R[anchors_batch]
				warped_normal_batch = ED_warp_normal(normal_batch, anchor_rot_batch, weights_batch)        
				return_normals[l:r] = warped_normal_batch.detach().cpu().numpy() # Returns normals

			return return_normals


	def optimize_ED(self, source_pcd,tgt_pcd,target_matches,
			point_anchors,anchor_weight,anchor_loc,
			d_tgt=None,sil_tgt=None, 
			landmarks=None,target_match_confidence=None,
			target_graph_node_location=None,
			target_graph_node_confidence=None,use_prev=False):
		'''
			Optimize the source point cloud to matche the target point cloud. 
			We use embedded graph to deform the source point cloud 

			@params:
				- source_pcd: Sx3 : Torch.tensor, source point cloud or canonical model vertices which are required to be deformed
				- tgt_pcd: 	Tx3: Torch.tensor, Target frame point cloud
				- target_matches: Cx3 : Torch.tensor, target location of source pcd , predicted by lepard
				- point_anchors: NxK: Closest K graph nodes, for which skinning weights are computed
				- anchor_weight: NxK: Skinning weights for each graph node, based on euclian distance
				- anchor_loc: NxKx4: Location of each graph node for easier computation 
				- landmarks: Cx2: correspodence between source and target pcd predicted by Lepard
				- target_match_confidence: C, Confidence for each correspodence based on sceneflow  

		'''

		# rotations,translations = self.warpfield.get_transformation_wrt_origin(self.warpfield.rotations,self.warpfield.translations)

		# if target_matches is None:
		# 	target_matches = self.tgt_pcd.copy()	


		"""translations"""
		if self.prev_trans is not None:
			if self.prev_trans.shape[0] == self.graph_nodes.shape[0]: 
				node_translations = self.prev_trans.detach().clone()
			else:
				node_translations = torch.zeros_like(self.graph_nodes)
				# print("Trans:",node_translations.shape)
		else: 
			node_translations = torch.zeros_like(self.graph_nodes)
			# print("Trans:",node_translations.shape)
		# node_translations = torch.from_numpy(translations).float().to(self.device)
		# print(node_translations)
		self.t = torch.nn.Parameter(node_translations)
		self.t.requires_grad = True

		"""rotations"""
		if self.prev_rot is None:
			phi = torch.zeros_like(self.graph_nodes)
			# print("Rotations:",phi.shape)
			node_rotations = SO3.exp(phi)
		else:
			if self.prev_rot.shape[0] == self.graph_nodes.shape[0]: 
				node_rotations = SO3(self.prev_rot.detach().clone())
			else:
				phi = torch.zeros_like(self.graph_nodes)
				# print("Rotations:",phi.shape)
				node_rotations = SO3.exp(phi)

		# node_rotations = R.from_matrix(rotations).as_quat()
		# node_rotations = torch.from_numpy(node_rotations).float().to(self.device)
		# node_rotations = SO3(node_rotations)

		# print(node_rotations)

		self.R = LieGroupParameter(node_rotations)


		"""optimizer setup"""
		optimizer = optim.Adam([self.R, self.t], lr= self.config.lr )
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.config.gamma)


		"""render reference pcd"""
		sil_tgt, d_tgt, _ = self.render_pcd( tgt_pcd )

		convergence_info = {
				"lmdk":[],
				"arap":[],
				"total":[],
				"depth":[],
				"chamfer":[],
				"silh":[],
				"motion": [],
				"smooth_rot": [],
				"smooth_trans": [],
				"grad_trans":[],
				"grad_rot":[]}

		# Transform points
		for i in range(self.config.iters):

			anchor_trn = self.t [point_anchors]
			anchor_rot = self.R [point_anchors]
			# print(self.graph_nodes.shape,self.graph_edges.shape,self.graph_edges_weights.shape)
			# print(source_pcd.shape,)
			warped_pcd = ED_warp(source_pcd, anchor_loc, anchor_rot, anchor_trn, anchor_weight)


			err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)

			if target_match_confidence is None: 
				err_ldmk = landmark_cost(warped_pcd, target_matches, landmarks) if landmarks is not None else 0
			else:	
				# print(f"warped:",warped_pcd.shape)
				# print(f"target:",target_matches.shape)
				# print("Landmarks:",landmarks.shape)
				# print("target_match_confidence:",target_match_confidence.shape)
				err_ldmk = landmark_cost_with_conf(warped_pcd, target_matches, landmarks,target_match_confidence) if landmarks is not None else 0

			if self.config.w_silh > 0 or self.config.w_depth > 0: 
				sil_src, d_src, _ = self.render_pcd(warped_pcd) 
			err_silh = silhouette_cost(sil_src, sil_tgt) if self.config.w_silh > 0 else 0
			err_depth,depth_error_image = projective_depth_cost(d_src, d_tgt) if self.config.w_depth > 0 else 0,0

			# print(self.graph_nodes  + self.t)
			# print(target_graph_node_location)

			err_motion = occlusion_fusion_graph_motion_cost(self.graph_nodes,self.t,
					target_graph_node_location,
					target_graph_node_confidence) if self.config.w_motion > 0 and target_graph_node_location is not None and target_graph_node_confidence is not None else 0


			# print("Motion Error:",err_motion)
			cd,cd_corresp = chamfer_dist(warped_pcd, tgt_pcd) if self.config.w_chamfer > 0 else (0,[])
		
			err_smooth_trans = ((self.t - self.prev_trans)**2).mean() if self.prev_trans is not None and self.config.w_smooth_trans > 0 else 0     
			err_smooth_rot = ((self.R - self.prev_R)**2).mean() if self.prev_R is not None and self.config.w_smooth_rot > 0 else 0

			loss = \
				err_arap * self.config.w_arap + \
				err_ldmk * self.config.w_ldmk + \
				err_silh * self.config.w_silh + \
				err_depth * self.config.w_depth + \
				cd * self.config.w_chamfer + \
				err_motion * self.config.w_motion + \
				err_smooth_rot * self.config.w_smooth_rot + \
				err_smooth_trans * self.config.w_smooth_trans
			

			if loss.item() < 1e-7:
				break

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()    

			lr = optimizer.param_groups[0]["lr"]
			# print((warped_pcd[landmarks[0]]-target_matches[landmarks[1]]).max())
			if i %10 == 0 or i == self.config.iters-1: 
				self.log.info(f"Frame:{self.warpfield.frame_id}" + "\t-->Iteration: {0}. Lr:{1:.5f} Loss: arap = {2:.3f}, ldmk = {3:.6f}, chamher:{4:.6f} silh = {5:.3f} depth = {6:.7f} motion = {7:.7f} smooth rot:{8:.7f} smooth trans:{9:.7f} total = {10:.3f}".format(i, lr,err_arap, err_ldmk, cd, err_silh,err_depth,err_motion,err_smooth_rot,err_smooth_trans,loss.item()))


			convergence_info["lmdk"].append(np.sqrt(err_ldmk.item()/landmarks.shape[0]) if landmarks is not None else 0)
			convergence_info["arap"].append(err_arap.item())
			convergence_info["chamfer"].append(cd.item() if self.config.w_chamfer > 0 else 0 )
			convergence_info["silh"].append(err_silh.item() if self.config.w_silh > 0 else 0 )
			convergence_info["depth"].append(err_depth.item() if self.config.w_depth > 0 else 0 )
			convergence_info["motion"].append(err_motion.item() if self.config.w_motion > 0 and target_graph_node_location is not None else 0 )
			convergence_info["smooth_trans"].append(err_smooth_trans.item() if self.prev_trans is not None and self.config.w_smooth_trans > 0 else 0 )
			convergence_info["smooth_rot"].append(err_smooth_rot.item() if self.prev_R is not None and self.config.w_smooth_rot > 0 else 0 )
			convergence_info["total"].append(loss.item())



		# # Smooth graph using just arap
		# for i in range(100):

		#     err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)

		#     if err_arap.item() < 1e-7:
		#         break

		#     optimizer.zero_grad()
		#     err_arap.backward()
		#     optimizer.step()
		#     scheduler.step()    

		#     lr = optimizer.param_groups[0]["lr"]
		#     print("\t-->Smoothing Iteration: {0}. Lr:{1:.5f} Loss: arap = {2:.3f}".format(i, lr,err_arap))

		# only chamfer and arap loss
		if self.opt.finetuning:
			for i in range(self.opt.finetuning_iters):

				anchor_trn = self.t [point_anchors]
				anchor_rot = self.R [point_anchors]
				# print(self.source_pcd.shape, self.anchor_loc.shape, anchor_rot.shape, anchor_trn.shape, self.anchor_weight.shape)
				warped_pcd = ED_warp(source_pcd, anchor_loc, anchor_rot, anchor_trn, anchor_weight)

				err_arap = arap_cost(self.R, self.t, self.graph_nodes, self.graph_edges, self.graph_edges_weights)

				# print("Motion Error:",err_motion)
				cd,cd_corresp = chamfer_dist(warped_pcd, tgt_pcd) if self.opt.chamfer_weight > 0 else (0,[])

				err_smooth_trans = ((self.t - self.prev_trans)**2).mean() if self.prev_trans is not None and self.config.w_smooth_trans_finetune > 0 else 0     
				err_smooth_rot = ((self.R - self.prev_R)**2).mean() if self.prev_R is not None and self.config.w_smooth_rot_finetune > 0 else 0

				loss = \
					err_arap * self.config.w_arap + \
					cd * self.opt.chamfer_weight + \
					err_smooth_rot * self.config.w_smooth_rot_finetune + \
					err_smooth_trans * self.config.w_smooth_trans_finetune


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				scheduler.step()    

				lr = optimizer.param_groups[0]["lr"]
				if i %10 == 0 or i == 99: 
					self.log.info(f"Frame:{self.warpfield.frame_id}" + "\t-->Final Finetuning Iteration: {0}. Lr:{1:.5f} Loss: chamfer:{2:.7f} arap = {3:.7f} smooth rot:{4:.7f} smooth rot:{5:.7f} total: {6:.7f}".format(i, lr,err_arap,cd,err_smooth_rot,err_smooth_trans, loss.item()))


				# if err_arap.item() < 1e-7:
				# 	self.log.info("Arap Loss too low. Breaking")
				# 	break

				# if self.warpfield.frame_id > 20:
					# self.vis.plot_corresp(i,warped_pcd.detach().cpu().numpy(),self.tgt_pcd.detach().cpu().numpy(),cd_corresp,debug=False)


				# Plot results:
				# if self.warpfield.frame_id > 45:
				#     deformed_nodes = (self.graph_nodes+self.t).detach().cpu().numpy()
				#     targtarget_graph_node_location = target_graph_node_location.detach().cpu().numpy() if target_graph_node_location is not None else None 
				#     self.vis.plot_optimization_sceneflow(self.warpfield.frame_id,i,warped_pcd.detach().cpu().numpy(),self.tgt_pcd.detach().cpu().numpy(),target_matches.detach().cpu().numpy(),deformed_nodes,landmarks,target_graph_node_location, debug=False)

		# Vis plot optimization 

		# Calculate per-graph node error cost 
		with torch.no_grad():
			cd,cd_corresp = chamfer_dist(warped_pcd, tgt_pcd,samples=warped_pcd.shape[0])
			per_corresp_cost = torch.norm(warped_pcd[cd_corresp[:,0]] - tgt_pcd[cd_corresp[:,1]],dim=1)
			
			graph_cost_sum = torch.zeros(self.graph_nodes.shape[0],device=self.device)	
			graph_weight_sum = torch.zeros(self.graph_nodes.shape[0],device=self.device)	


			# print("asdasdsadasd")
			# print(torch.unique(point_anchors))
			# print(self.graph_nodes.shape)

			# Graph error similiar to function4D
			for j in range(point_anchors.shape[1]):

				anchor_mask = point_anchors[cd_corresp[:,0],j] != -1

				graph_cost_sum[point_anchors[cd_corresp[:,0],j][anchor_mask]] += per_corresp_cost[anchor_mask]
				graph_weight_sum[point_anchors[cd_corresp[:,0],j][anchor_mask]] += anchor_weight[cd_corresp[:,0],j][anchor_mask]

			graph_error = graph_cost_sum/(graph_weight_sum + 1e-6)

			self.log.info(f"Graph weights:{graph_weight_sum.min()} {graph_weight_sum.mean()} {graph_weight_sum.max()}")
			self.log.info(f"Graph graph_cost_sum:{graph_cost_sum.min()} {graph_cost_sum.mean()} {graph_cost_sum.max()}")
			self.log.info(f"Graph error:{graph_error.min()} {graph_error.mean()} {graph_error.max()}")




			sil_src, d_src, _ = self.render_pcd(warped_pcd)
			err_silh = iou(sil_src, sil_tgt)
			err_depth,depth_error_image = projective_depth_inverse_cost(d_src, d_tgt)         
			if self.vis.opt.vis == "matplotlib":
				self.vis.plot_depth_images([sil_src,d_src,sil_tgt,d_tgt,depth_error_image],savename=f"RenderedImages_{self.warpfield.frame_id}_{i}.png")

			convergence_info["silh"].append(err_silh.item())
			convergence_info["depth"].append(err_depth.item())
			self.log.info(f"Frame: {self.warpfield.frame_id} Depth Inverse Metric:{convergence_info['depth'][-1]} IOU:{convergence_info['silh'][-1]}")


		quat_data = self.R.retr().data

		self.prev_R = self.R.detach()
		self.prev_rot = quat_data.clone()
		self.prev_trans = self.t.detach().clone()

		quat_data_np = quat_data.detach().cpu().numpy()
		rotmats = R.from_quat(quat_data_np).as_matrix()
		t = self.t.cpu().data.numpy()

		self.graph_error = graph_error
		
		estimated_transforms = {'warped_verts':warped_pcd.cpu(),\
		"node_rotations":rotmats, "node_translations":self.t.cpu(),\
		"deformed_nodes_to_target": self.graph_nodes + self.t,
		'graph_error': graph_error.cpu(),
		"convergence_info":convergence_info}


		estimated_transforms = self.dict_to_numpy(estimated_transforms)

		# if self.warpfield.frame_id > 9:
		# 	self.vis.plot_optimization(estimated_transforms['warped_verts'],target_matches.detach().cpu().data.numpy(),target_match_confidence.detach().cpu().data.numpy())


		return estimated_transforms

	def render_pcd(self, x):
		# INF = 1e6 # Mark infinity/background as 1e6 depth 
		INF = 0
		px, dx = self.renderer(x)
		px, dx  = map(lambda feat: feat.squeeze(), [px, dx ])
		dx[dx < 0] = INF
		mask = px[..., 0] > 0
		return px, dx, mask

	def render_mesh(self, verts,faces):
		# INF = 1e6 # Mark infinity/background as 1e6 depth 
		INF = 0
		dx = self.renderer([verts],[faces])
		dx  = dx.squeeze()
		mask = dx >= 0
		dx[dx < 0] = INF
		return dx, mask


	def map_pixel_to_pcd(self, valid_pix_mask):
		''' establish pixel to point cloud mapping, with -1 filling for invalid pixels
		:param valid_pix_mask:
		:return:
		'''
		image_size = valid_pix_mask.shape
		pix_2_pcd_map = torch.cumsum(valid_pix_mask.view(-1), dim=0).view(image_size).long() - 1
		pix_2_pcd_map [~valid_pix_mask] = -1
		return pix_2_pcd_map


	def visualize_results(self, tgt_pcd, warped_pcd=None):

		# import mayavi.mlab as mlab
		import open3d as o3d
		c_red = (224. / 255., 0 / 255., 125 / 255.)
		c_pink = (224. / 255., 75. / 255., 232. / 255.)
		c_blue = (0. / 255., 0. / 255., 255. / 255.)
		scale_factor = 0.007
		source_pcd = self.source_pcd.cpu().numpy()
		tgt_pcd = tgt_pcd.cpu().numpy()

		# mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
		if warped_pcd is None:
			# mlab.points3d(source_pcd[ :, 0], source_pcd[ :, 1], source_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_red)
			warped_o3d = o3d.geometry.PointCloud()
			warped_o3d.points = o3d.utility.Vector3dVector(source_pcd)
			warped_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_red).reshape((1,3)),(source_pcd.shape[0],1)))
		else:
			warped_pcd = warped_pcd.detach().cpu().numpy()
			warped_o3d = o3d.geometry.PointCloud()
			warped_o3d.points = o3d.utility.Vector3dVector(warped_pcd)
			warped_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_pink).reshape((1,3)),(source_pcd.shape[0],1)))
			# mlab.points3d(warped_pcd[ :, 0], warped_pcd[ :, 1], warped_pcd[:,  2], resolution=4, scale_factor=scale_factor , color=c_pink)

		target_o3d = o3d.geometry.PointCloud()
		target_o3d.points = o3d.utility.Vector3dVector(tgt_pcd)
		target_o3d.colors = o3d.utility.Vector3dVector(np.tile(np.array(c_blue).reshape((1,3)),(source_pcd.shape[0],1)))
		# mlab.points3d(tgt_pcd[ :, 0] , tgt_pcd[ :, 1], tgt_pcd[:,  2],resolution=4, scale_factor=scale_factor , color=c_blue)
		# mlab.show()
		o3d.visualization.draw_geometries([warped_o3d,target_o3d])

	@staticmethod		
	def dict_to_numpy(model_data):	
		# Convert every torch tensor to np.array to save results 
		for k in model_data:
			if type(model_data[k]) == torch.Tensor:
				model_data[k] = model_data[k].cpu().data.numpy()
				# print(k,model_data[k].shape)
			elif type(model_data[k]) == list:
				for i,r in enumerate(model_data[k]): # Numpy does not handle variable length, this will produce error 
					if type(r) == torch.Tensor:
						model_data[k][i] = model_data[k][i].cpu().data.numpy()
						# print(k,i,model_data[k][i].shape)
			elif type(model_data[k]) == dict:
				for r in model_data[k]:
					if type(model_data[k][r]) == torch.Tensor:
						model_data[k][r] = model_data[k][r].cpu().data.numpy()
						# print(k,r,model_data[k][r].shape)

		return model_data