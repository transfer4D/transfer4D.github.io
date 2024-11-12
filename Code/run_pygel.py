# Module to run gel on skeleton 
import os 
import sys
import argparse
import logging	
import trimesh
import numpy as np 
import subprocess
import polyscope as ps
import sys
import scipy.io as sio 
sys.path.append("../")
from utils.viz_utils import transform_pointcloud_to_opengl_coords as reflect_opengl


# Neural Tracking Modules
from model import dataset
from utils import utils

from NeuralNRT._C import erode_mesh as erode_mesh_c
import find_connected_components2 as find_connected_components

from pygel3d import hmesh,graph


def load_anime_file(filename):
	f = open(filename, 'rb')
	nf = np.fromfile(f, dtype=np.int32, count=1)[0]
	nv = np.fromfile(f, dtype=np.int32, count=1)[0]
	nt = np.fromfile(f, dtype=np.int32, count=1)[0]
	vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
	face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
	offset_data = np.fromfile(f, dtype=np.float32, count=-1)
	vert_data = vert_data.reshape((-1, 3))
	face_data = face_data.reshape((-1, 3))
	offset_data = offset_data.reshape((nf - 1, nv, 3))

	trajectory = np.tile(vert_data.reshape((1,-1,3)),(nf,1,1))
	trajectory[1:] += offset_data

	return trajectory,face_data


def load_mesh(datadir,source_frame_id,align_to_depth_map=True):
		
		mesh_path = os.path.join(datadir,os.path.basename(datadir) + '.anime')

		ext = mesh_path.split('.')[-1]

		if ext == "anime":
			verts,faces = load_anime_file(mesh_path)
			verts = verts[self.opt.anime_index]

		elif ext in ['obj','off','ply']:
			# mesh = o3d.io.read_triangle_mesh(mesh_path)
			tmesh = trimesh.load_mesh(mesh_path,process=False)
			verts = np.asarray(tmesh.vertices)
			faces = np.asarray(tmesh.faces)

		else:
			raise NotImplementedError(f"Unable to load:{mesh_path}. Only .anime, .off, .obj, .ply supported") 

		if align_to_depth_map:	
			extrinsic_matrix = np.loadtxt(os.path.join(self.opt.datadir,"extrinsics.txt"))
			R = extrinsic_matrix[:3,:3]
			T = extrinsic_matrix[:3,3]

			R_inv = R.T
			T_inv = -R_inv@T

			R_inv *= np.array([scale,scale,scale]).reshape(3,1) 

			verts = verts@R_inv.T + T_inv

		print(verts,faces)	

		return verts,faces


def get_sampled_mesh(vertices,faces,non_eroded_vertices):
	V = vertices.shape[0]
	new_vert_indices = -1*np.ones(V,dtype=np.int64)

	for i,n in enumerate(non_eroded_vertices):
		new_vert_indices[n] = i

	new_verts = vertices[non_eroded_vertices]	
	new_faces = new_vert_indices[faces.reshape(-1)].reshape(-1,3)

	keep_faces = np.all(new_faces != -1,axis=1)
	new_faces = new_faces[keep_faces]

	return new_verts,new_faces


def load_trajectory(data_path,source_frame_id):

	trajectory_path = os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
	trajectory = np.load(trajectory_path)


	# TODO fix for multiple source frame
	face_filename = os.listdir(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path"))[0] 

	face_data = np.load(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path",face_filename))


	return trajectory,face_data	


def run_gel(opt):
	data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
	trajectory_path = os.path.join(data_path,"trajectory")

	save_path = os.path.join(data_path,"gel")
	os.makedirs(save_path,exist_ok=True)

	
	traj_files = sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1]))
	for file in traj_files:
		print(f"=====Loading trajectory:{file}========")
		source_frame_id = int(file.split('.')[0].split('_')[-1])
		trajectory,face_data = load_trajectory(data_path,source_frame_id)


		# ROSA Scripts
		matlab_script_input_path = os.path.join(save_path,f"gel_{source_frame_id}_{opt.num_points}") # First element is only required

		if not os.path.isfile(matlab_script_input_path + '_corresp.npy'):
			# Erode mesh 
			non_eroded_vertices = erode_mesh_c(trajectory[0], face_data,3,4)
			non_eroded_vertices = np.where(non_eroded_vertices==True)[0]

			new_vertices, new_faces = get_sampled_mesh(trajectory[0],face_data,non_eroded_vertices) 

			# ps.init()
			# ps_eroded = ps.register_surface_mesh("OG",trajectory[0],face_data)
			# ps_eroded = ps.register_surface_mesh("Eroded",new_vertices,new_faces)
			# ps.show()

			vert_components, vert_compontent_cnt = find_connected_components.find_components(new_vertices,new_faces)

			all_sampled_verts = np.zeros((0,3))  
			all_corresp = np.array([],dtype=np.int64)
			
			all_joints = np.zeros((0,3)) # Stores position of each joint
			all_edges = []
			current_joint_cnt = 0
			all_skel_components = []
			
			for comp in range(vert_compontent_cnt):
				component_inds = np.where(vert_components == comp)[0]

				# print(f"component_length:",component_inds.shape)

				comp_verts,comp_faces = get_sampled_mesh(new_vertices,new_faces,component_inds)	
				
				# print(f"Comp verts:{comp_verts.shape} faces:{comp_faces.shape}")
				quadric_simplify_fraction = min(1,opt.num_points/len(component_inds))
				gel_mesh = hmesh.Manifold.from_triangles(comp_verts,comp_faces)
				
				# print(f"Simplification Fraction:{quadric_simplify_fraction}")
				if quadric_simplify_fraction < 1:
					hmesh.quadric_simplify(gel_mesh, quadric_simplify_fraction) # 20 percent or 8000 vertices

				gel_mesh.cleanup()

				# print("vertices after simplification :", gel_mesh.no_allocated_vertices())

				graph_input = graph.from_mesh(gel_mesh)
				gel_sampled_verts = graph_input.positions()
				curve_skeleton,gel_corresp = graph.LS_skeleton_and_map(graph_input)

				if len(curve_skeleton.positions()) <= 6: 
					continue 

				gel_corresp = np.array(gel_corresp)

				# print(f"Before:{np.unique(gel_corresp)}")
				gel_corresp[gel_corresp >= 0] = current_joint_cnt + gel_corresp[gel_corresp >= 0]
				# print(f"After:{np.unique(gel_corresp)}")

				all_corresp = np.concatenate([all_corresp,gel_corresp],axis=0)
				all_sampled_verts = np.concatenate([all_sampled_verts,gel_sampled_verts],axis=0)


				gel_joints = curve_skeleton.positions()
				J = gel_joints.shape[0]

				all_skel_components.extend([comp]*J)

				# print("Before:",end='')
				# print(gel_joints.shape,all_joints.shape,np.unique(gel_corresp).shape,np.unique(all_corresp).shape,current_joint_cnt)

				all_joints = np.concatenate([all_joints,gel_joints],axis=0)
				for j1 in range(J):
					edges = curve_skeleton.neighbors(j1)
					for j2 in edges:
						all_edges.append([current_joint_cnt+j1,current_joint_cnt+j2])

				current_joint_cnt = all_joints.shape[0]
				# print("After:",end='')
				# print(gel_joints.shape,all_joints.shape,np.unique(gel_corresp).shape,np.unique(all_corresp).shape,current_joint_cnt)



			# print(f"All joints:{all_joints.shape} edges:{len(all_edges)} corresp:{all_corresp.shape} verts:{all_sampled_verts.shape}")


			J = all_joints.shape[0]
			all_skel_adj = np.zeros((J,J),dtype=bool)
			for j1,j2 in all_edges:
				all_skel_adj[j1,j2] = True
				all_skel_adj[j2,j1] = True

			all_corresp[all_corresp == -1] = J
			all_skel_components = np.array(all_skel_components) 
			# 	Save results
			np.save(matlab_script_input_path+'_skel_components.npy',all_skel_components)
			np.save(matlab_script_input_path+'_sampled_verts.npy',all_sampled_verts)
			np.save(matlab_script_input_path+'_joints.npy',all_joints)
			np.save(matlab_script_input_path+'_adj.npy',all_skel_adj)
			np.save(matlab_script_input_path+'_corresp.npy',all_corresp)


			# print(np.unique(all_corresp))
			

			
		# Plot results 
		gel_skel_components = np.load(matlab_script_input_path+'_skel_components.npy')
		print(gel_skel_components)
		gel_sampled_vertex = np.load(matlab_script_input_path+'_sampled_verts.npy')
		gel_joints = np.load(matlab_script_input_path+'_joints.npy')
		gel_skel_adj = np.load(matlab_script_input_path+'_adj.npy')
		gel_corresp = np.load(matlab_script_input_path+'_corresp.npy')
		gel_skeleton = (gel_joints, None, gel_skel_adj)


		# Load intrinsic matrix 
		intrinsics_matrix = np.loadtxt(os.path.join(opt.datadir,'intrinsics.txt'))
		fx = intrinsics_matrix[0, 0]
		fy = intrinsics_matrix[1, 1]
		cx = intrinsics_matrix[0, 2]
		cy = intrinsics_matrix[1, 2]

		project_depth_x = fx *trajectory[0,:,0]/trajectory[0,:,2] + cx
		project_depth_y = fy *trajectory[0,:,1]/trajectory[0,:,2] + cy
		project_depth = np.concatenate([project_depth_x[:,None],project_depth_y[:,None]],axis=1)

		J = gel_skeleton[0].shape[0]
		gel_joints_orig = gel_joints.copy()
		gel_edges_orig = np.array([[j1,j2] for j1 in range(J) for j2 in range(j1+1,J) if gel_skel_adj[j1,j2]])


		tmesh = trimesh.Trimesh(vertices=trajectory[0],faces=face_data,process=False)
		tmesh.fix_normals()

		vertex_normals = tmesh.vertex_normals

		labels_coarse,labels,gel_joints,gel_skel_adj = find_connected_components.find_coarse_labels_from_gel(trajectory,vertex_normals,gel_sampled_vertex,gel_corresp,gel_skeleton,project_depth,gel_skel_components.copy())
		J = gel_joints.shape[0]
		gel_edges = np.array([[j1,j2] for j1 in range(J) for j2 in range(j1+1,J) if gel_skel_adj[j1,j2]])
		gel_skeleton = gel_joints,gel_edges,gel_skel_adj # Changing connectivity

		np.save(matlab_script_input_path+'_filtered_joints.npy',gel_joints)
		np.save(matlab_script_input_path+'_filtered_adj.npy',gel_skel_adj)
		np.save(matlab_script_input_path+'_filtered_corresp.npy',labels)
		np.save(matlab_script_input_path+'_filtered_coarse_corresp.npy',labels_coarse)


		# boundary_indices = []
		# for x in segmented_boundaries:
		# 	boundary_indices.extend(segmented_boundaries[x])

		# boundary_indices = np.unique(np.array(boundary_indices))
		# print("Boundary Indices:",boundary_indices)


		# if opt.debug: 
		# 	ps.init()
		# 	ps.set_navigation_style("free")
		# 	ps.set_view_projection_mode("orthographic")
		# 	ps.set_ground_plane_mode('none')


		# 	ps_verts = ps.register_surface_mesh("PC",reflect_opengl(trajectory[0]),face_data,transparency=0.5)
		# 	# ps_verts = ps.register_surface_mesh("proj",np.concatenate([project_depth,np.ones((project_depth.shape[0],1))],axis=1),face_data)


		# 	colors = np.random.random((gel_joints_orig.shape[0]+1,3))
		# 	colors[-1,:] = 0

		# 	joint_degree = np.sum(gel_skel_adj,axis=0).astype(np.int32)

		# 	vert_colors = colors[labels]
		# 	ps_verts.add_color_quantity("coarse labels",vert_colors,enabled=True)

			
		# 	ps_skeleton = ps.register_curve_network("Skeleton Old",reflect_opengl(gel_joints_orig),gel_edges_orig,enabled=False)
		# 	ps_skeleton.add_color_quantity("gel weights",colors[gel_skel_components],defined_on='nodes',enabled=True)

		# 	ps_skeleton = ps.register_curve_network("Skeleton",reflect_opengl(gel_skeleton[0]),gel_edges)
		# 	ps_skeleton.add_color_quantity("gel weights",colors[:gel_skeleton[0].shape[0]],defined_on='nodes',enabled=True)

		# 	# ps_skeleton_joints = ps.register_point_cloud("sampled verts",gel_sampled_vertex[:,:3],enabled=True)
		# 	# ps_skeleton_joints.add_color_quantity("color",colors[gel_corresp],enabled=True)
		# 	# ps_skeleton_joints = ps.register_point_cloud("boundary",trajectory[0][boundary_indices],enabled=True)

		# 	ps.show()
 




if __name__ == "__main__": 
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
	args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
	args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')
	args.add_argument('--num_points', required=False,type=float,default=8000,help='Max Number of points to sample for skeletonisation')
	args.add_argument('--debug', required=False,type=bool,default=True,help='Plot figures')

	opt = args.parse_args()

	# Logging details 
	LOGFORMAT = "[%(filename)s:%(lineno)s - %(funcName)3s() ] %(message)s"
	logging.basicConfig(
						stream=sys.stdout,
						format=LOGFORMAT,
						level=logging.DEBUG,
						# handlers=[stream_handler],
						# filename=os.path.join(opt.datadir,"results",opt.exp,opt.ablation,"my_log.log"),
						# filemode='w'
						) 
	logging.getLogger('numba').setLevel(logging.WARNING)
	logging.getLogger('PIL').setLevel(logging.WARNING)
	logging.getLogger('matplotlib').setLevel(logging.WARNING)


	run_gel(opt)
