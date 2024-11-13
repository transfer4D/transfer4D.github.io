import os
import json 
import numpy as np
import sys
from .visualizer import Visualizer

import polyscope as ps
import polyscope.imgui as psim
from scipy.spatial.transform import Rotation as sRotation

sys.path.append('..')
import utils.viz_utils as viz_utils # Neural tracking modules 


class VisualizerPolyScope(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)	

		ps.init()
		ps.set_ground_plane_mode('none')
		ps.set_view_projection_mode("orthographic")
		# ps.set_autoscale_structures(True)
		# self.canonical_mesh = ps.register_surface_mesh("canonical_mesh", np.zeros((0,3)), np.zeros((0,3),dtype=np.int32), smooth_shade=True,enabled=False)
		# self.canonical_mesh = ps.register_surface_mesh("deformed_mesh", np.zeros((0,3)), np.zeros((0,3),dtype=np.int32), smooth_shade=True,enabled=False)

	def get_pcd(self,rgbd_image,max_points=1e5):
		
		rgbd_mask = rgbd_image[-1] > 0
		# rgbd_flat = np.moveaxis(rgbd_image, 0, -1).reshape(-1, 6)
		rgbd_points = viz_utils.transform_pointcloud_to_opengl_coords(rgbd_image[3:,rgbd_mask].T)
		# rgbd_points = rgbd_image[..., 3:]
		rgbd_colors = rgbd_image[:3,rgbd_mask].T

		if len(rgbd_points) > max_points: # Downsample 
			print(f"Exceeded number of points:{len(rgbd_points)} reducing to:{max_points}")
			show_indices = np.random.choice(len(rgbd_points),size=max_points,replace=False)
			rgbd_points = rgbd_points[show_indices]
			rgbd_colors = rgbd_colors[show_indices]	

		return rgbd_points,rgbd_colors

	def update_pcd(self,source_pcd,source_confidence,target_matches,warped_verts,target_pcd):

		
		if hasattr(self,"object_position"):
			source_pcd +=   self.object_position[None] - source_pcd.mean(axis=0,keepdims=True)
			target_pcd +=   self.object_position[None] - target_pcd.mean(axis=0,keepdims=True)
			warped_verts +=   self.object_position[None] - warped_verts.mean(axis=0,keepdims=True)
			target_matches +=   self.object_position[None] - target_matches.mean(axis=0,keepdims=True)
			# target_pcd -= self.object_position[None]
			# warped_verts -= self.object_position


		if hasattr(self,"bbox"):
			source_pcd += np.array([-1.2, 0.5, 0]) * self.bbox
			target_matches += np.array([0, 0.5, 0]) * self.bbox
			target_pcd += np.array([+1.2, 0.5, 0]) * self.bbox
			warped_verts += np.array([1.2, -0.5, 0]) * self.bbox


		ps_source_pcd = ps.register_point_cloud("source_pcd", source_pcd, enabled=True)
		ps_source_pcd.add_color_quantity("confidence",self.get_viridis_color(source_confidence),enabled=True)


		ps_target_pcd = ps.register_point_cloud("target_matches", target_matches, enabled=True)
		ps_target_pcd = ps.register_point_cloud("target_pcd", target_pcd, enabled=True)
		ps_warped_verts = ps.register_point_cloud("warped_verts", warped_verts, enabled=True)

	def update_mesh(self):
		"""
			Create open3D object to visualize tsdf 
		"""
		assert hasattr(self,'tsdf'),  "TSDF not defined. Add tsdf as attribute to visualizer first." 
		verts, faces, normals, color = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
		deformed_vertices,deformed_normals,vert_anchors,vert_weights,valid_verts = self.warpfield.deform_mesh(verts,normals)
			

		verts = viz_utils.transform_pointcloud_to_opengl_coords(verts)
		deformed_vertices = viz_utils.transform_pointcloud_to_opengl_coords(deformed_vertices)

		# print("Verts:",verts.shape)

		if hasattr(self,"bbox"):
			verts += np.array([-1.2, -1, 0]) * self.bbox + self.object_position[None] - verts.mean(axis=0,keepdims=True)
			deformed_vertices += np.array([1.2, -1, 0]) * self.bbox + self.object_position[None] - deformed_vertices.mean(axis=0,keepdims=True)

		ps_canonical_mesh = ps.register_surface_mesh("canonical_mesh", verts, faces, smooth_shade=True,enabled=True)		
		ps_deformed_mesh = ps.register_surface_mesh("deformed_mesh", deformed_vertices, faces, smooth_shade=True,enabled=True)		


	def update_graph(self,graph_error,graph_sdf_alignment):
		assert hasattr(self,'warpfield'),  "warpfield not defined. Add warpfield as attribute to visualizer first." 


		nodes = viz_utils.transform_pointcloud_to_opengl_coords(self.warpfield.graph.nodes)
		deformed_nodes = viz_utils.transform_pointcloud_to_opengl_coords(self.warpfield.get_deformed_nodes())

		nodes -= nodes.mean(0,keepdims=True)
		deformed_nodes -= deformed_nodes.mean(0,keepdims=True)

		edges = self.warpfield.graph.edges

		node_ids = np.tile(np.arange(len(nodes), dtype=np.int32)[:,None],(1,edges.shape[1])) # (opt_num_nodes_i, num_neighbors)
		# print(node_ids)
		graph_edge_pairs = np.concatenate([node_ids.reshape((-1, edges.shape[1], 1)), edges.reshape(-1, edges.shape[1], 1)], axis=2) # (opt_num_nodes_i, num_neighbors, 2)

		# print(graph_edge_pairs)

		valid_edges = edges >= 0
		valid_edge_idxs = np.where(valid_edges)
		# print(valid_edge_idxs)
		graph_edge_pairs_filtered = graph_edge_pairs[valid_edge_idxs[0], valid_edge_idxs[1], :]

		# print(graph_edge_pairs_filtered)
		if hasattr(self,"object_position"):
			nodes += self.object_position[None] - nodes.mean(axis=0,keepdims=True)
			deformed_nodes += self.object_position[None] - deformed_nodes.mean(axis=0,keepdims=True)

		if hasattr(self,"bbox"):
			nodes += np.array([0, -0.5, 0]) * self.bbox
			deformed_nodes += np.array([0, -0.5, 0]) * self.bbox


		ps_ed_graph = ps.register_point_cloud("ed_graph_nodes", nodes,enabled=False,radius=0.01),\
				ps.register_curve_network("ed_graph_edges", nodes, graph_edge_pairs_filtered,enabled=False)		
	
		ps_deformed_graph = ps.register_point_cloud("deformed_graph_nodes", deformed_nodes,enabled=True,radius=0.01),\
			ps.register_curve_network("deformed_graph_edges", deformed_nodes, graph_edge_pairs_filtered,enabled=True)		
	
		ed_graph_color = np.ones((nodes.shape[0],3))	 
		ed_graph_color[:graph_error.shape[0]] = self.get_viridis_color(graph_error/graph_error.max())
		ps_ed_graph[0].add_color_quantity('graph error',ed_graph_color,enabled=True)

		# print(graph_error)
		# print(graph_sdf_alignment)
		if graph_sdf_alignment is not None:
			ed_graph_sdf_color = np.ones((nodes.shape[0],3))	 
			ed_graph_sdf_color[:graph_sdf_alignment.shape[0]] = self.get_viridis_color(graph_sdf_alignment/graph_sdf_alignment.max())
			ps_deformed_graph[0].add_color_quantity('graph sdf misalignment',ed_graph_sdf_color,enabled=True)


	def plot_corresp(self,iteration,source_pcd,target_pcd,corresp,debug=False,savename="chamfer_distance_optimization"):

		ps.remove_all_structures()

		source_pcd = viz_utils.transform_pointcloud_to_opengl_coords(source_pcd)
		target_pcd = viz_utils.transform_pointcloud_to_opengl_coords(target_pcd)

		ps_source_pcd = ps.register_point_cloud("source_pcd", source_pcd, enabled=True,radius=0.0005)		


		ps_target_pcd = ps.register_point_cloud("target_pcd", target_pcd, enabled=True,radius=0.0005)		


		num_corresp = corresp.shape[0]
		nodes = np.concatenate([source_pcd[corresp[:,0]],target_pcd[corresp[:,1]]])
		matches = np.array([[i,i+num_corresp] for i in range(num_corresp)])

		# if hasattr(self,"corresp"): 
			# self.corresp.remove()

		ps_corresp = ps.register_curve_network("corresp_graph_edges", nodes, matches,enabled=True,radius=0.001)

		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.max(axis=0) - source_pcd.min(axis=0))

		ps.look_at(np.array([0,0,0.01*self.bbox[2]]),np.array([0,0,0]))

		image_path = os.path.join(self.savepath,"images",f"{savename}_{iteration}.png")
		print("Saving results to:",image_path)

		if debug:
			ps.show()

		print(f"Saving plot to :{image_path}")	
		ps.set_screenshot_extension(".jpg");
		ps.screenshot(image_path,transparent_bg=False)


		ps_source_pcd.remove()
		ps_target_pcd.remove()
		ps_corresp.remove()

		ps.reset_camera_to_home_view()

	
	# Does not considers invalid points
	def plot_optimization(self,source_pcd,target_pcd,valid_verts):
		source_pcd = viz_utils.transform_pointcloud_to_opengl_coords(source_pcd)			
		target_pcd = viz_utils.transform_pointcloud_to_opengl_coords(target_pcd)

		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.max(axis=0) - source_pcd.min(axis=0))

		if not hasattr(self,'object_position'):
			self.object_position = source_pcd.mean(axis=0)

		# Result of sceneflow 
		source_pcd += np.array([-0.6, 0, 0]) * self.bbox	
		target_pcd += np.array([+0.6, 0, 0]) * self.bbox	

		ps_source_pcd = ps.register_point_cloud("source_pcd", source_pcd, enabled=True,radius=0.005)		
		ps_target_pcd = ps.register_point_cloud("target_pcd", target_pcd, enabled=True,radius=0.005)

		valid_color = self.get_viridis_color(valid_verts)
		ps_source_pcd.add_color_quantity("valid",valid_color,enabled=True)
		ps_target_pcd.add_color_quantity("valid",valid_color,enabled=True)

		ps.show()

		ps_source_pcd.remove()
		ps_target_pcd.remove()

	def plot_sceneflow_calibration(self,source_pcd,target_pcd,target_matches,returned_points,source_pcd_prev,nrr_output,valid_verts,debug=False):

		if self.opt.usePreviousFrame:
			source_pcd = viz_utils.transform_pointcloud_to_opengl_coords(source_pcd_prev)
		else: 
			source_pcd = viz_utils.transform_pointcloud_to_opengl_coords(source_pcd)

		target_pcd = viz_utils.transform_pointcloud_to_opengl_coords(target_pcd)
		target_matches = viz_utils.transform_pointcloud_to_opengl_coords(target_matches)
		returned_points = viz_utils.transform_pointcloud_to_opengl_coords(returned_points)
		nrr_output = viz_utils.transform_pointcloud_to_opengl_coords(nrr_output)


		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.max(axis=0) - source_pcd.min(axis=0))

		if not hasattr(self,'object_position'):
			self.object_position = source_pcd.mean(axis=0)

		# Result of sceneflow 
		target_matches += 	np.array([+0.6, 0, 0]) * self.bbox	
		target_pcd += 		np.array([-0.6, -1.2, 0]) * self.bbox	
		nrr_output += 		np.array([+0.6, -1.2, 0]) * self.bbox

		# Result of sceneflow returned 
		source_pcd += 		np.array([-0.6, 0, 0]) * self.bbox	
		returned_points += 	np.array([0.6, 0, 0]) * self.bbox	

		ps_source_pcd = ps.register_point_cloud("source_pcd", source_pcd, enabled=True,radius=0.005)		
		ps_target_pcd = ps.register_point_cloud("target_pcd", target_pcd, enabled=True,radius=0.005)
		ps_target_matches = ps.register_point_cloud("target_matches", target_matches, enabled=True,radius=0.005)
		ps_returned_points = ps.register_point_cloud("returned_points", returned_points, enabled=False,radius=0.005)
		ps_nrr_output = ps.register_point_cloud("nrr_output", nrr_output, enabled=True,radius=0.005)

		valid_color = self.get_viridis_color(valid_verts)
		ps_nrr_output.add_color_quantity("valid",valid_color,enabled=True)


		image_path = os.path.join(self.savepath,"images",f"sceneflow_{self.warpfield.frame_id}.png")
		print("Saving results to:",image_path)


		if debug:
			ps.show()

		print(f"Saving plot to :{image_path}")	
		ps.set_screenshot_extension(".jpg");
		ps.screenshot(image_path,transparent_bg=False)

		ps_source_pcd.remove()
		ps_target_pcd.remove()
		ps_target_matches.remove()
		ps_returned_points.remove()
		ps_nrr_output.remove()


		ps.reset_camera_to_home_view()




	def plot_skinning(self,trajectory,face_data,weights,debug=True,framerate=5,gr_skeleton=None):

		object_mean = trajectory[0].mean(axis=0,keepdims=True)

		trajectory -= trajectory.mean(axis=1,keepdims=True)


		ps.reset_camera_to_home_view()
		assert trajectory.shape[1] == weights.shape[1], "trajectory:{} and weights:{} should be for same number of points"

		color = self.get_color_from_matrix(weights.T)

		print(color)
		print(color.shape)
		print(trajectory.shape)

		# Create bounding box for later use 
		if not hasattr(self,'bbox'):
			self.bbox = (trajectory.max(axis=(0,1)) - trajectory.min(axis=(0,1)))
		if not hasattr(self,'object_position'):
			self.object_position = trajectory.mean(axis=(0,1))


		for timestep,traj in enumerate(trajectory):
			
			print(traj.shape)
			print(color.shape)
			if timestep == 0:
				if len(face_data) > 0:
					ps_trajectory = ps.register_surface_mesh("trajectory",viz_utils.transform_pointcloud_to_opengl_coords(traj),faces=face_data,smooth_shade=True)	
				else:
					ps_trajectory = ps.register_point_cloud("trajectory",viz_utils.transform_pointcloud_to_opengl_coords(traj))	

				ps_trajectory.add_color_quantity("skinning",color,enabled=True)

					


				if debug: 
					if gr_skeleton is not None:
						gr_skeleton = (gr_skeleton[0] - object_mean, gr_skeleton[1])
						# 8 Reduce skeleton 
						J = gr_skeleton[0].shape[0]
						print(gr_skeleton[1])
						edges_array = np.array([[j1,j2] for j1 in range(J) for j2 in range(J) if gr_skeleton[1][j1,j2]])
						ps_gr_joints = ps.register_point_cloud("gr_joints",viz_utils.transform_pointcloud_to_opengl_coords(gr_skeleton[0]))	
						ps_gr_skeleton = ps.register_curve_network("gr_skeleton", viz_utils.transform_pointcloud_to_opengl_coords(gr_skeleton[0]), edges_array,enabled=True)


					ps.show()
					return 
			else:
				if len(face_data) > 0:
					ps_trajectory.update_vertex_positions(viz_utils.transform_pointcloud_to_opengl_coords(traj))
				else:
					ps_trajectory.update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(traj))

			image_path = os.path.join(self.savepath,"images",f"skinning_{timestep}.png")
			print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".jpg");
			ps.screenshot(image_path,transparent_bg=False)

		image_path = os.path.join(self.savepath,"images",f"skinning_\%d.png")
		video_path = os.path.join(self.savepath,"video","skinning.mp4")
		palette_path = os.path.join(self.savepath,"video","palette2.png")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		ps_trajectory.remove()	


	def plot_skeleton(self,trajectory,skeleton_motion,parent_array,weights,gr_skeleton=None,debug=True,framerate=5):	

		object_mean = trajectory[0].mean(axis=0,keepdims=True)


		# trajectory -= np.mean(skeleton_motion,axis=1,keepdims=True)
		# skeleton_motion -= np.mean(skeleton_motion,axis=1,keepdims=True)
		# skeleton_motion[:,:,2] -= 0.1
		color = self.get_color_from_matrix(weights)

		ps.reset_camera_to_home_view()
		assert skeleton_motion.shape[1] == len(parent_array), "trajectory:{} and weights:{} should be for same number of points"

		# Create bounding box for later use 
		if not hasattr(self,'bbox'):
			self.bbox = (skeleton_motion.max(axis=(0,1)) - skeleton_motion.min(axis=(0,1)))
		if not hasattr(self,'object_position'):
			self.object_position = skeleton_motion.mean(axis=(0,1))

				



		parent_array[0] = 0 			
		bone_array = [[i,p] for i,p in enumerate(parent_array)]
		bone_array = np.array(bone_array)

		for timestep,skel in enumerate(skeleton_motion):
			
			if timestep == 0:

				ps_skeleton = ps.register_point_cloud("skeleton", viz_utils.transform_pointcloud_to_opengl_coords(skel),enabled=True,radius=0.01),\
						ps.register_curve_network("bones", viz_utils.transform_pointcloud_to_opengl_coords(skel), bone_array,enabled=True)

				ps_trajectory = ps.register_point_cloud("trajectory",viz_utils.transform_pointcloud_to_opengl_coords(trajectory[timestep]),enabled=True,radius=0.005)
				ps_trajectory.add_color_quantity("skinning",color,enabled=True)

				if debug:

					if gr_skeleton is not None:
						gr_skeleton = (gr_skeleton[0] - object_mean, gr_skeleton[1])
						# 8 Reduce skeleton 
						J = gr_skeleton[0].shape[0]
						edges_array = np.array([[j1,j2] for j1 in range(J) for j2 in range(J) if gr_skeleton[1][j1,j2]])
						ps_gr_joints = ps.register_point_cloud("gr_joints",viz_utils.transform_pointcloud_to_opengl_coords(gr_skeleton[0]))	
						ps_gr_skeleton = ps.register_curve_network("gr_skeleton", viz_utils.transform_pointcloud_to_opengl_coords(gr_skeleton[0]), edges_array,enabled=True)

					ps.show()
					return 
			else:
				ps_skeleton[0].update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(skel))
				ps_skeleton[1].update_node_positions(viz_utils.transform_pointcloud_to_opengl_coords(skel))
				ps_trajectory.update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(trajectory[timestep]))

	
			image_path = os.path.join(self.savepath,"images",f"skeleton_{timestep}.png")
			print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".jpg");
			ps.screenshot(image_path,transparent_bg=False)

		image_path = os.path.join(self.savepath,"images",f"skeleton_\%d.png")
		video_path = os.path.join(self.savepath,"video","skeleton.mp4")
		palette_path = os.path.join(self.savepath,"video","palette2.png")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		ps_skeleton[0].remove()	
		ps_skeleton[1].remove()	
		ps_trajectory.remove()





	def plot_retarget_motion(self,source_frame_id,target_verts_motion,faces,
						trajectory,
						target_skeleton_motion,skeleton_motion,parent_array,
						weights,
						boneVis=None,boneDists=None,proj=None,
						debug=True,framerate=5):	


		# proj[:,2] += 1

		ps.reset_camera_to_home_view()
		# assert skeleton_motion.shape[1] == len(parent_array), "trajectory:{} and weights:{} should be for same number of points"

		# Create bounding box for later use 
		if not hasattr(self,'bbox'):
			self.bbox = (skeleton_motion.max(axis=(0,1)) - skeleton_motion.min(axis=(0,1)))
		if not hasattr(self,'object_position'):
			self.object_position = skeleton_motion.mean(axis=(0,1))



		parent_array[0] = 0 			
		bone_array = [[i,p] for i,p in enumerate(parent_array)]
		bone_array = np.array(bone_array)

		vert_array = [[i,i+1] for i in range(1)]
		vert_array = np.array(vert_array)

		mesh_orgin = np.array([0.54035, 0.141903, 0.571507])
		mesh_scale = 0.875714

		if proj is not None:
			print("Difference: between projected and computed distance")
			print(np.linalg.norm(target_verts_motion[0]*0.875714 - proj*0.875714,axis=1)**2,np.min(boneDists,axis=1))
			print(np.linalg.norm(target_verts_motion[0]*0.875714 - proj*0.875714,axis=1)**2 - np.min(boneDists,axis=1))


		labels = np.argmax(weights,axis=1)

		print(labels)


		vert_colors = self.get_color_from_labels(labels)

		print(vert_colors)


		# print(bone_vis_colors.shape)
		# print(bone_dists_colors.shape)

		# See if the label is visisble 


		bone_colors = self.get_color_from_labels(list(range(weights.shape[1])))
		bone_colors = np.concatenate([bone_colors[0:1,:],bone_colors],axis=0)
		
		for timestep,skel in enumerate(skeleton_motion):
			
			if timestep == 0:




				ps_target_mesh = ps.register_surface_mesh("target",viz_utils.transform_pointcloud_to_opengl_coords(target_verts_motion[timestep]),faces=faces,enabled=True,smooth_shade=True,transparency=0.9)
				ps_target_mesh.add_color_quantity("skinning_weights",vert_colors,enabled=True)



				if boneVis is not None:
					for i in range(weights.shape[1]):
						boneVis_i = boneVis[:,i]
						boneValid_i = np.logical_and(boneVis[:,i], boneDists[:,i] <= np.min(boneDists,axis=1)*1.0001) 
						boneDists_i = np.min(boneDists,axis=1)/boneDists[:,i]
						bone_vis_colors = self.get_viridis_color(boneVis_i)
						bone_dists_colors = self.get_viridis_color(boneDists_i)
						bone_valid_colors = self.get_viridis_color(boneValid_i)

						ps_target_mesh.add_color_quantity(f"boneVis_{i}",bone_vis_colors,enabled=False)
						ps_target_mesh.add_color_quantity(f"boneDists_{i}",bone_dists_colors,enabled=False)
						ps_target_mesh.add_color_quantity(f"boneValid_{i}",bone_valid_colors,enabled=False)



				ps_skeleton = ps.register_point_cloud("skeleton", viz_utils.transform_pointcloud_to_opengl_coords(skel),enabled=True,radius=0.01),\
						ps.register_curve_network("bones", viz_utils.transform_pointcloud_to_opengl_coords(skel), bone_array,enabled=True)

				ps_skeleton[1].add_color_quantity("skinning", bone_colors, defined_on='edges',enabled=True)



				# Vert curve edges 
				if proj is not None:
					print(vert_colors.shape, np.concatenate([target_verts_motion[0],proj],axis=0).shape)
					ps_proj = ps.register_point_cloud("boneProjection", viz_utils.transform_pointcloud_to_opengl_coords(proj),enabled=True,radius=0.01)
					ps_proj.add_color_quantity("labels",vert_colors,enabled=True)
		
					ps_vert_proj = ps.register_curve_network("verts2projection", viz_utils.transform_pointcloud_to_opengl_coords(np.concatenate([target_verts_motion[0,0:1],proj[0:1]],axis=0)), vert_array,enabled=True)
					ps_vert_proj.add_color_quantity("skinning_weights",vert_colors[0:1],enabled=True,defined_on='edges')


				ps_target_skeleton = ps.register_point_cloud("target_skeleton", viz_utils.transform_pointcloud_to_opengl_coords(target_skeleton_motion[timestep]),enabled=True,radius=0.01),\
						ps.register_curve_network("target_bones", viz_utils.transform_pointcloud_to_opengl_coords(target_skeleton_motion[timestep]), bone_array,enabled=True)		

				ps_target_skeleton[1].add_color_quantity("skinning", bone_colors, defined_on='edges',enabled=True)


				ps_trajectory = ps.register_point_cloud("source",viz_utils.transform_pointcloud_to_opengl_coords(trajectory[timestep]),enabled=True,radius=0.001)

				if debug: 
					ps.show()
					return 
			else:
				ps_skeleton[0].update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(skel))
				ps_skeleton[1].update_node_positions(viz_utils.transform_pointcloud_to_opengl_coords(skel))
				
				ps_target_skeleton[0].update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(target_skeleton_motion[timestep]))
				ps_target_skeleton[1].update_node_positions(viz_utils.transform_pointcloud_to_opengl_coords(target_skeleton_motion[timestep]))




				# ps_trajectory.update_point_positions(viz_utils.transform_pointcloud_to_opengl_coords(trajectory[timestep]))
				ps_target_mesh.update_vertex_positions(viz_utils.transform_pointcloud_to_opengl_coords(target_verts_motion[timestep]))



	
			image_path = os.path.join(self.savepath,"images",f"retarget_{timestep+source_frame_id}.png")
			print(f"Saving plot to :{image_path}")	
			ps.set_screenshot_extension(".jpg");
			ps.screenshot(image_path,transparent_bg=False)

		image_path = os.path.join(self.savepath,"images",f"retarget_\%d.png")
		video_path = os.path.join(self.savepath,"video","retarget.mp4")
		palette_path = os.path.join(self.savepath,"video","palette4.png")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")	

		# ps_skeleton[0].remove()	
		# ps_skeleton[1].remove()	
		# ps_trajectory.remove()
		ps_target_mesh.remove()

		ps_target_skeleton[0].remove()
		ps_target_skeleton[1].remove()

	
	def plot_boundary(self,trajectory,boundary_dict,centroids,boundary_joints):

		boundary_points = []
		for k in boundary_dict: 
			boundary_points.extend(boundary_dict[k])

		color = 0.5*np.ones((trajectory.shape[1],3))

		color[:,0] = 0.5
		color[:,2] = 1
		color[boundary_points,0] = 1
		color[boundary_points,1] = 0
		color[boundary_points,2] = 0

		all_joints = np.concatenate([centroids,boundary_joints],axis=1)

		joints_color = np.zeros(all_joints.shape[1])
		joints_color[:centroids.shape[1]] = 1
		joints_color = self.get_viridis_color(joints_color) 
		print(joints_color.shape)

		ps_trajectory = ps.register_point_cloud("trajectory",viz_utils.transform_pointcloud_to_opengl_coords(trajectory[0]),enabled=True,radius=0.01)
		ps_trajectory.add_color_quantity("boundary",color,enabled=True)

		ps_joints = ps.register_point_cloud("joints",viz_utils.transform_pointcloud_to_opengl_coords(all_joints[0]),enabled=True,radius=0.01)
		print(joints_color.shape,all_joints.shape)
		ps_joints.add_color_quantity("centroid",joints_color,enabled=True)


		ps.show()

		ps_trajectory.remove()
		ps_joints.remove()

	def show(self,scene_flow_data,estimated_transformations,matches=None,debug=True):
		"""
			For visualizing the tsdf integration: 
			1. Source RGBD + Graph(visible nodes)   2. Target RGBD as Point Cloud 
			3. Canonical Model + Graph   			3. Deformed Model   
		"""

		ps.reset_camera_to_home_view()


		# Top left
		# source_pcd = self.get_source_RGBD()
		source_pcd = viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['source'])
		target_pcd = viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['target'])
		lepard_sceneflow = viz_utils.transform_pointcloud_to_opengl_coords(scene_flow_data['target_matches'])
		warped_verts = viz_utils.transform_pointcloud_to_opengl_coords(estimated_transformations['warped_verts'])
		
		source_confidence = np.ones(source_pcd.shape[0],dtype=np.float32)
		if 'confidence' in scene_flow_data: 
			if scene_flow_data['confidence'] is not None:
				source_confidence = scene_flow_data['confidence']
				source_confidence[~scene_flow_data['valid_verts']] = 0

		# Create bounding box for later use 
		# if not hasattr(self,'bbox'):
		self.bbox = (source_pcd.max(axis=0) - source_pcd.min(axis=0))

		print(f"Bounding box length:",np.linalg.norm(self.bbox))

		# if not hasattr(self,'object_position'):
		self.object_position = source_pcd.mean(axis=0)

		self.update_pcd(source_pcd,source_confidence,lepard_sceneflow,warped_verts,target_pcd)	
		# self.update_mesh()
		self.update_graph(estimated_transformations['graph_error'],estimated_transformations["graph_sdf_alignment"])	



		ps.look_at(np.array([0,0,6*self.bbox[2]]) + self.object_position,self.object_position + np.array([0,0,0])) # For human cam1

		# scene_flow = scene_flow_data['scene_flow'] 
		# scene_flow[:,0] += bbox[0]

		# rendered_deformed_nodes,rendered_deformed_edges = self.get_rendered_graph(self.warpfield.get_deformed_nodes(),self.graph.edges,color=color,trans=np.array([0.0, -1.0, 0.00]) * bbox)


		# # Bottom left
		# canonical_mesh = self.get_model_from_tsdf(trans=np.array([0, -1.0, 0]) * bbox)
		# # rendered_graph_nodes,rendered_graph_edges = self.get_rendered_graph(self.graph.nodes,self.graph.edges,color=color,trans=np.array([0, -1.0, 0.01]) * bbox)

		# # Bottom right
		# deformed_mesh = self.get_deformed_model_from_tsdf(trans=np.array([1.0, -1.0, 0]) * bbox)

		image_path = os.path.join(self.savepath,"images",f"subplot_{self.warpfield.frame_id}.png")
		print("Saving results to:",image_path)

		if debug:
			ps.show()

		print(f"Saving plot to :{image_path}")	
		ps.set_screenshot_extension(".jpg");
		ps.screenshot(image_path,transparent_bg=False)

		ps.remove_all_structures()



	def get_rigid_transforms_from_user(self,source_pcd,verts,faces,R=np.eye(3),T=np.zeros(3),scale=1): 

		ps.remove_all_structures()


		if not hasattr(self,'bbox'):
			self.bbox = (source_pcd.max(axis=0) - source_pcd.min(axis=0))
		if not hasattr(self,'object_position'):
			self.object_position = source_pcd.mean(axis=0)	



		ps_source_pcd = ps.register_point_cloud("source", viz_utils.transform_pointcloud_to_opengl_coords(source_pcd),enabled=True,radius=0.01)

		R_inv = R.T
		T_inv = -R_inv@T

		R_inv *= np.array([scale,scale,scale]).reshape(3,1)

		ps_mesh = ps.register_surface_mesh("target", viz_utils.transform_pointcloud_to_opengl_coords(verts@R_inv.T + T_inv), faces, smooth_shade=True,enabled=True)

		verts_mean = verts.mean(0)
		rot_x,rot_y,rot_z = sRotation.from_matrix(R).as_euler('xyz',degrees=True)
		trans_x,trans_y,trans_z = T


		print("Before callback transformation:",rot_x,rot_y,rot_z)


		def callback():
			psim.PushItemWidth(150)


			# == Show text in the UI

			psim.TextUnformatted("Align Target Mesh to source")
			psim.Separator()
			psim.TextUnformatted("Rotation")

			nonlocal rot_x,rot_y,rot_z,trans_x,trans_y,trans_z,scale,R,T,verts,ps_mesh

			changed_rot_x, rot_x = psim.SliderFloat("Rx", rot_x,v_min=-180, v_max=180) 
			changed_rot_y, rot_y = psim.SliderFloat("Ry", rot_y,v_min=-180, v_max=180) 
			changed_rot_z, rot_z = psim.SliderFloat("Rz", rot_z,v_min=-180, v_max=180) 

			psim.Separator()
			psim.TextUnformatted("Translation")

			changed_trans_x, trans_x = psim.SliderFloat("Tx", trans_x,v_min=-self.bbox[0],v_max=self.bbox[0]) 
			changed_trans_y, trans_y = psim.SliderFloat("Ty", trans_y,v_min=-self.bbox[1],v_max=self.bbox[1]) 
			changed_trans_z, trans_z = psim.SliderFloat("Tz", trans_z,v_min=-self.bbox[2],v_max=self.bbox[2]) 

			psim.Separator()

			changed_scale, scale = psim.InputFloat("Scale", scale) 


			if changed_rot_x or changed_rot_y or changed_rot_z: 
				R = sRotation.from_euler('xyz',[rot_x,rot_y,rot_z],degrees=True).as_matrix()

			if changed_trans_x or changed_trans_y or changed_trans_z: 
				T = np.array([trans_x,trans_y,trans_z])


			if changed_scale or changed_rot_x or changed_rot_y or changed_rot_z or changed_trans_x or changed_trans_y or changed_trans_z:
				transformed_verts = apply_transform(verts,R,T,scale)		
				ps_mesh.update_vertex_positions(viz_utils.transform_pointcloud_to_opengl_coords(transformed_verts))



		ps.set_user_callback(callback)
		# ps.show()

		ps_source_pcd.remove()
		ps_mesh.remove()

		transformed_verts = apply_transform(verts,R,T,scale)		


		return transformed_verts,R,T,scale


def apply_transform(verts,R,T,scale):
	R_inv = R.T.copy()
	T_inv = -R_inv@T

	R_inv *= np.array([scale,scale,scale]).reshape(3,1) 

	# R_inv[[0,1,2],[0,1,2]] *= scale 

	verts_mean = verts.mean(0)
	transformed_verts = (verts - verts_mean)@R_inv.T  + verts_mean + T_inv

	return transformed_verts