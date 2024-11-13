# function4D.py is the main file to connect everything complete non-rigid integration process. 
# Run python fusion.py --datadir <path-to-RGBD-frames>

import os
import json
import sys
import argparse # To parse arguments 
import logging # To log info 
sys.path.append("../")  # Making it easier to load modules

import numpy as np

from vis import get_visualizer # Visualizer 
from frame_loader import RGBDVideoLoader
from embedded_deformation_graph import EDGraph # Create ED graph from mesh, depth image, tsdf 
from run_lepard import Lepard_runner # SceneFlow module 
# from run_motion_model import MotionCompleteNet_Runner
from NonRigidICP.model.registration_fusion import Registration as PytorchRegistration
from lepard.inference import Lepard	
from warpfield import WarpField # Connects ED Graph and TSDF/Mesh/Whatever needs to be deformed  

from evaluation import CompleteMeshEvaluator

class AnimationTransfer4D:
	def __init__(self,fopt):
		"""
			Initialize using the first frame
			@params: 
				fopt: Fusion options: all attributes for fusion
		"""

		self.opt = fopt

		# Create visualizer 
		self.vis = get_visualizer(fopt) # To see data
	
		# For logging results
		self.log = logging.getLogger(__name__)

		self.frameLoader = RGBDVideoLoader(fopt.datadir) # Load images

		self.source_frame_data = self.frameLoader.get_source_data(fopt.source_frame) # Source RGBD Image
		self.target_frame = fopt.source_frame # Set target frame initially to source frame 

		self.lepard = Lepard(os.path.join(os.getcwd(),"../lepard/configs/test/4dmatch.yaml")) # Sceneflow estimator

		source_mask = self.source_frame_data["im"][-1] > 0	


		# intrinsics
		cam_intr = np.eye(3)
		cam_intr[0, 0] = self.source_frame_data["intrinsics"][0]
		cam_intr[1, 1] = self.source_frame_data["intrinsics"][1]
		cam_intr[0, 2] = self.source_frame_data["intrinsics"][2]
		cam_intr[1, 2] = self.source_frame_data["intrinsics"][3]


		self.graph = EDGraph(self.opt,self.vis,self.source_frame_data)

		self.source_pcd = self.graph.vertices
		self.source_pcd_prev = self.source_pcd.copy()

		self.vis.graph  = self.graph # Add graph to visualizer 

		self.warpfield = WarpField(self.graph,self.opt,self.vis)

		self.vis.warpfield = self.warpfield   # Add warpfield to visualizer

		if fopt.use_occlusionFusion:
			self.occlusionFusion = MotionCompleteNet_Runner(self.opt)
			self.occlusionFusion.graph = self.graph
			os.makedirs(os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation,f'visible_nodes'),exist_ok=True)
			np.save(os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation,f'visible_nodes/{fopt.source_frame}.npy'),np.ones(self.graph.nodes.shape[0],dtype=bool))

		else:
			self.occlusionFusion = None

		self.gradient_descent_optimizer = PytorchRegistration(self.graph,self.warpfield,cam_intr,self.vis,self.opt)		
		self.warpfield.optimizer = self.gradient_descent_optimizer
		self.vis.optimizer = self.gradient_descent_optimizer
		# self.gradient_descent_optimizer.config.iters = 150		
		# self.previous_sceneflow_lengths = np.zeros(self.source_pcd.shape[0],dtype=np.float32)

		# self.volume_evaluation = CompleteMeshEvaluator(self.opt,self.vis)

		# print("Initial point to point:",self.volume_evaluation.point_to_point_dist(self.tsdf.get_canonical_model()[0],visualize=False))
		# print("Initial point to plane:",self.volume_evaluation.point_to_plane_dist(self.tsdf.get_canonical_model()[0],visualize=False))


		self.savepath = os.path.join(self.opt.datadir,"results",self.opt.exp,"optimization_convergence_info",self.opt.ablation)
		os.makedirs(self.savepath,exist_ok=True)

		self.trajectory = [self.source_pcd]

		self.trajectory_path = os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation,"trajectory")
		os.makedirs(self.trajectory_path,exist_ok=True)


	def register_new_frame(self): 

		# Check next frame can be registered, currently only registering first 100 frames 
		if self.target_frame + self.opt.skip_rate >= len(self.frameLoader) or self.target_frame + self.opt.skip_rate >= self.opt.last_frame:
			return False,"Registration Completed"
		
		# If there out of memory error. Restart using this
		# if self.target_frame == 0: 
		# 	self.target_frame = 137

		self.log.info(f"==========================Registering {self.target_frame + self.opt.skip_rate}th frame==============================")

		success,msg = self.register_frame(self.opt.source_frame,self.target_frame + self.opt.skip_rate)

		# Update frame number 
		self.target_frame = self.target_frame + self.opt.skip_rate


		return success,msg

	def register_frame(self,source_frame,target_frame):
		"""
			Main step of the algorithm. Register new target frame 
			Args: 
				source_frame(int): which source frame id to integrate  
				target_frame(int): which target frame id to integrate 

			Returns:
				success(bool): Return whether sucess or failed in registering   
		"""


		target_frame_data = self.frameLoader.get_target_data(target_frame,self.source_frame_data["cropper"])
		target_mask = target_frame_data["im"][-1] > 0	
		target_pcd = target_frame_data["im"][3:,target_mask].T

		# TODO if depth cost too high or IOU too low use previous frame

		if self.opt.usePreviousFrame == False: 

			scene_flow,corresp,valid_verts,sceneflow_confidence = self.lepard(self.source_pcd,target_pcd)
			target_matches = self.source_pcd.copy()
			target_matches[valid_verts] += scene_flow[valid_verts]

			scene_flow_back,_,valid_verts_back,sceneflow_confidence_back = self.lepard(target_matches,self.source_pcd)
			returned_points = target_matches.copy()	
			returned_points[valid_verts_back] += scene_flow_back[valid_verts_back]
		else: 

			scene_flow,corresp,valid_verts,sceneflow_confidence = self.lepard(self.source_pcd_prev,target_pcd)
			target_matches = self.source_pcd.copy()
			target_matches[valid_verts] += self.source_pcd_prev[valid_verts] - self.source_pcd[valid_verts] \
							+ scene_flow[valid_verts] 

			scene_flow_back,_,valid_verts_back,sceneflow_confidence_back = self.lepard(target_matches,self.source_pcd_prev)
			returned_points = target_matches.copy()	
			returned_points[valid_verts_back] += self.source_pcd[valid_verts_back] - self.source_pcd_prev[valid_verts_back] \
							+ scene_flow_back[valid_verts_back] 

		# Comparing with length with original
		sceneflow_confidence_length = np.linalg.norm(self.source_pcd - returned_points,axis=1)
		sceneflow_confidence = (sceneflow_confidence_length)/np.linalg.norm(self.source_pcd.max(axis=0) - self.source_pcd.min(axis=0)) # Normalize by bounding box 
		confidence_length_threshold = self.opt.sceneflow_calibration_parameter

		# sceneflow_confidence_length = np.linalg.norm(self.source_pcd - target_matches,axis=1)
		# sceneflow_confidence = (sceneflow_confidence_length - self.previous_sceneflow_lengths)/np.linalg.norm(self.source_pcd.max(axis=0) - self.source_pcd.min(axis=0)) # Normalize by bounding box 
		# confidence_length_threshold = 0.01
		if self.opt.sceneflow_calibration_parameter > 0:

			sceneflow_confidence_length[~valid_verts] = 1.0
			

			sceneflow_confidence = np.exp(-0.5*(sceneflow_confidence**2)/(confidence_length_threshold**2))
			self.log.debug(f"SceneFlow confidence range:{sceneflow_confidence.min()} {sceneflow_confidence.max()}")
			self.log.debug(f"SceneFlow confidence vals:{sceneflow_confidence}")
			self.log.debug(f"Valid Sceneflow points:{np.sum(valid_verts)}/{valid_verts.shape[0]}")
			sceneflow_confidence[~valid_verts] = 0.0


			scene_flow_data = {'source':self.source_pcd.copy(),'target':target_pcd,'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp, 'confidence':sceneflow_confidence}	
		else: 	
			scene_flow_data = {'source':self.source_pcd.copy(),'target':target_pcd,'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp}	

		# scene_flow_data = {'source':self.source_pcd.copy(),'scene_flow': scene_flow,"valid_verts":valid_verts,"target_matches":target_matches,'landmarks':corresp, 'confidence':None}	
		optical_flow_data = {'source_id':self.opt.source_frame,'target_id':target_frame}


		# print(scene_flow_data["valid_verts"].shape)
		# print(scene_flow_data["confidence"].shape)
		if self.opt.use_occlusionFusion: 
			deformed_nodes_at_source = self.warpfield.get_deformed_nodes()

			predicted_node_location = target_matches[self.graph.node_indices]
			visible_node_mask = sceneflow_confidence[self.graph.node_indices] > 0.5
			
			np.save(os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation,f'visible_nodes/{target_frame}.npy'),visible_node_mask)

			print(sceneflow_confidence[self.graph.node_indices])
			# print(visible_node_mask)

			estimated_complete_nodes_motion_data = self.occlusionFusion(target_frame - opt.skip_rate,
				deformed_nodes_at_source,
				predicted_node_location,
				visible_node_mask)
		else:	
			estimated_complete_nodes_motion_data = (None,None)

		


		estimated_transformations = self.gradient_descent_optimizer.optimize(optical_flow_data,
											scene_flow_data,
											estimated_complete_nodes_motion_data,
											target_frame_data,
											None)

		estimated_transformations['warped_verts'][~estimated_transformations['valid_verts']] = self.source_pcd_prev[~estimated_transformations['valid_verts']]

		self.vis.plot_sceneflow_calibration(self.source_pcd,target_pcd,\
			target_matches,returned_points,\
			self.source_pcd_prev,
			estimated_transformations["warped_verts"],
			estimated_transformations["valid_verts"],
			debug=False)				
		# Update previous_sceneflow_lengths
		# self.previous_sceneflow_lengths = np.linalg.norm(self.source_pcd - estimated_transformations['warped_verts'],axis=1)

		assert estimated_transformations['warped_verts'].shape == self.source_pcd.shape, f"warped verts:{estimated_transformations['warped_verts'].shape} source shape:{self.source_pcd.shape}"

		self.trajectory.append(estimated_transformations['warped_verts'])
		self.source_pcd_prev = estimated_transformations['warped_verts']


		# Update warpfield parameters, warpfield maps to target frame  
		self.warpfield.update_transformations(estimated_transformations)


		# self.vis.plot_skinned_model()

		# Add new nodes to warpfield and graph if any


		
		# if source_frame > 0:		
		# vis.show_zoomed_in(scene_flow_data,fopt.source_frame,debug=False) # plot registration details 
		# vis.show(scene_flow_data,fopt.source_frame,debug=True if i > 20 and i < 30 else False) # plot registration details 

		# p2p_mean, p2p_max = self.volume_evaluation.point_to_point_dist(estimated_transformations["warped_verts"],target_frame)

		# estimated_transformations["convergence_info"]["point_to_point_mean"] = [p2p_mean]
		# estimated_transformations["convergence_info"]["point_to_point_max"] = [p2p_max]

		# p2pl_mean, p2pl_max = self.volume_evaluation.point_to_plane_dist(estimated_transformations["warped_verts"],target_frame)

		# estimated_transformations["convergence_info"]["point_to_plane_mean"] = [p2pl_mean]
		# estimated_transformations["convergence_info"]["point_to_plane_max"] = [p2pl_max]

		self.save_convergance_info(estimated_transformations["convergence_info"],optical_flow_data["source_id"],optical_flow_data["target_id"])
		self.vis.show(scene_flow_data,estimated_transformations,debug=False) # plot registration details 
		# self.vis.show(scene_flow_data,estimated_transformations,debug=target_frame == source_frame +1) # plot registration details 


		# update = self.warpfield.update_graph()
		# if update:
			# self.gradient_descent_optimizer.update() 
		# update = False


		# if source_frame > 8:
		# 	self.vis.plot_tsdf_volume(debug=True)	

		sucess  = estimated_transformations["convergence_info"]["silh"][-1] > self.opt.IOUThres




		# Return whether sucess or failed in registering 
		return sucess, f"Registered {source_frame}th frame to {target_frame}th frame."

	def clear_frame_data(self):
		if hasattr(self,'tsdf'):  self.tsdf.clear() # Clear image information, remove deformed model 
		# if hasattr(self,'warpfield'):  self.warpfield.clear() # Clear image information, remove deformed model 
		# if hasattr(self,'graph'): self.graph.clear()   # Remove estimated deformation from previous timesteps 	

	def save_convergance_info(self,convergence_info_dict,source_id,target_id):

		savepath = os.path.join(self.savepath, f"optimization_convergence_info_{source_id}_{target_id}.json")
		with open(savepath, "w") as f:
			json.dump(convergence_info_dict, f)

		self.log.info(f"Saved Convergance info to:{savepath}")	

		trajectory_path = os.path.join(self.trajectory_path, f"trajectory_{self.opt.source_frame}.npy")

		np.save(trajectory_path,np.array(self.trajectory))
			
	def __call__(self):

		# Run fusion 
		while True: 

			success, msg = self.register_new_frame()
			self.log.info(msg) # Print data

			if not success: 
				self.log.info("IOU too low. Hence stopping optimisation")
				break
			
			self.clear_frame_data() # Reset information 

		self.vis.create_video()
		return self.target_frame


if __name__ == "__main__":
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
	args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
	args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')

	# Arguments for tsdf
	args.add_argument('--voxel_size', default=0.01, type=float, help='length of each voxel cube in TSDF')

	# For GPU
	args.add_argument('--gpu', 	  dest='gpu', action="store_true",help='Try to use GPU for faster optimization')
	args.add_argument('--no-gpu', dest='gpu', action="store_false",help='Uses CPU')
	args.set_defaults(gpu=True)

	args.add_argument('--nrrConfig', required=True,type=str,help='path to config weight files')

	args.add_argument('--use_occlusionFusion', 	  dest='use_occlusionFusion', action="store_true",help='Try to use GPU for faster optimization')
	args.add_argument('--no-use_occlusionFusion', dest='use_occlusionFusion', action="store_false",help='Uses CPU')
	args.set_defaults(use_occlusionFusion=False)



	# Parameters for Sceneflow
	args.add_argument('--usePreviousFrame', dest='usePreviousFrame', action="store_true",help='Whether to pass previous frame as input to lepard')
	args.add_argument('--no-usePreviousFrame', dest='usePreviousFrame', action="store_false",help='Whether to pass previous frame as input to lepard')
	args.set_defaults(usePreviousFrame=True)	

	# Chamfer finetuning
	args.add_argument('--finetuning', dest='finetuning', action="store_true",help='Perform finetuning using Chamfer distance')
	args.add_argument('--no-finetuning', dest='finetuning', action="store_false",help='Dont perform finetuning')
	args.set_defaults(finetuning=True)

	args.add_argument('--finetuning_iters',default=100,type=int,help='number of finetuning iterations')

	# Graph error threshold 
	args.add_argument('--voxel_deformation_error', default=0.01, type=float, help='If loss above this, then dont integrate voxel')
	args.add_argument('--sceneflow_calibration_parameter', default=0.1, type=float, help='Calibration parameter for sceneflow')
	args.add_argument('--chamfer_weight', default=1000, type=float, help='Chamfer weight for finetuning')
	args.add_argument('--voxel_misalignment_thresh', default=0.1, type=float, help='Voxel sdf misalignment, if greater than threshold discard values')


	# Arguments for loading frames 
	args.add_argument('--source_frame', default=0, type=int, help='frame index to create the deformable model')
	args.add_argument('--skip_rate', default=1, type=int, help='frame rate while running code')
	args.add_argument('--last_frame', default=200, type=int, help='frame rate while running code')
	args.add_argument('--IOUThres', default=0.0, type=float, help='If IOU goes below this, then reload')



	# Arguments for debugging  
	args.add_argument('--debug', default=True, type=bool, help='Whether debbugging or not. True: Logging + Visualization, False: Only Critical information and plots')
	args.add_argument('--vis', default='polyscope', type=str, help='Visualizer to plot results')

	opt = args.parse_args()



	# stream_handler = logging.StreamHandler()
	# stream_handler.setLevel(logging.INFO)


	# Logging details 
	LOGFORMAT = "[%(filename)s:%(lineno)s - %(funcName)3s() ] %(message)s"
	logging.basicConfig(
						stream=sys.stdout,
						format=LOGFORMAT,
						level=logging.INFO if opt.debug is False else logging.DEBUG,
						# handlers=[stream_handler],
						# filename=os.path.join(opt.datadir,"results",opt.exp,opt.ablation,"my_log.log"),
						# filemode='w'
						) 
	logging.getLogger('numba').setLevel(logging.WARNING)
	logging.getLogger('PIL').setLevel(logging.WARNING)
	logging.getLogger('matplotlib').setLevel(logging.WARNING)


	while True:
		print(opt)
		animTransfer4D = AnimationTransfer4D(opt)
		last_registered_frame = animTransfer4D()
		if last_registered_frame + animTransfer4D.opt.skip_rate >= len(animTransfer4D.frameLoader) or last_registered_frame + animTransfer4D.opt.skip_rate >= animTransfer4D.opt.last_frame:
			break
		else: 
			opt.source_frame = last_registered_frame

	
	# opt.vis = "matplotlib"
	# visualizer_matplotlib = get_visualizer(opt)
	# visualizer_matplotlib.plot_input_sequence()

	opt.vis = "plotly"
	visulizer_plotly = get_visualizer(opt)
	visulizer_plotly.compare_convergance_info({'optimization':['silh','depth'],'volumetric_analysis':['point_to_plane_mean']})