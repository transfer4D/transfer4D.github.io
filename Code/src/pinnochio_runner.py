# Given data dir and input mesh/anime file embed skeleton and perform motion retargetting  

import os
import sys
import logging 
import argparse 
import subprocess
import numpy as np 
import numpy as np
from scipy.spatial import cKDTree
import json
import torch
import trimesh
import open3d as o3d

from vis import get_visualizer # Visualizer 
from motion_retargeting import get_mesh_motion
from utils.geometry_utils import save_obj, simplify_mesh, clean_mesh, preprocess_mesh_np
from pathlib import Path


def calculate_bounding_box(vertices):
    min_coords, _ = torch.min(vertices, dim=0)
    max_coords, _ = torch.max(vertices, dim=0)

    center = (min_coords + max_coords) / 2
    max_dim = torch.max(max_coords - min_coords).item()
    return center, max_dim


def convert_obj_to_ply(obj_file, ply_file):
    vertices = []
    faces = []

    with open(obj_file, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                faces.append(face)

    with open(ply_file, 'w') as ply:
        ply.write("ply\nformat ascii 1.0\n")
        ply.write(f"element vertex {len(vertices)}\n")
        ply.write("property float x\nproperty float y\nproperty float z\n")
        ply.write(f"element face {len(faces)}\n")
        ply.write("property list uchar int vertex_indices\n")
        ply.write("end_header\n")

        for v in vertices:
            ply.write(f"{v[0]} {v[1]} {v[2]}\n")

        for face in faces:
            ply.write(f"3 {face[0]} {face[1]} {face[2]}\n")



class SkeletonEmbedding: 
	def __init__(self,opt,vis):
		workspace_folder = os.getcwd()
		self.pinoccio_path = os.path.join(workspace_folder, "Code/src/Pinocchio/DemoUI/DemoUI")

		self.opt = opt
		self.vis = vis

		self.log = logging.getLogger(__name__)
		self.embedding_list = []
		self.penalty_name = ["DistPF" ,"GlobalDotPF" ,"SymPF" ,"DoublePF" ,"FootPF" ,"DupPF" ,"DotPF" ,"ExtremPF" ,"DisjointPF"]
		self.non_zero_penalty = set()


	# Read skel file 
	def read_skel(self,filepath):
		with open(filepath,'r') as f:
			lines = f.read().split('\n')

		parents = []
		skeleton = []

		for line in lines:  
			if len(line) == 0:
				break
			ind,x,y,z,parent = line.split()
			skeleton.append([float(x),float(y),float(z)])
			parents.append(int(parent))

		skeleton = np.array(skeleton)
		return skeleton,parents	

	def write_skel(self,pinocchio_skel_path,joint_positions,new_parent_array):
		with open(pinocchio_skel_path,'w') as f:
			for i,v in enumerate(joint_positions):
				f.write(f"{i} {v[0]} {v[1]} {v[2]} {new_parent_array[i]}\n")


	def parse_embedding(self,s):
		# Parse string to get output
		idx,candidate,matches, penalty, total_penalty = s.strip().split(';')


		idx = int(idx.split(':')[-1])
		candidate = int(candidate.split(':')[-1])
		matches = [ int(x) for x in matches.strip().split()]
		penalty = [ float(x) for x in penalty.strip().split()]
		total_penalty = float(total_penalty.split(':')[-1])


		non_zero_penalty = set([i for i,x in enumerate(penalty) if x > 0])
		self.non_zero_penalty = self.non_zero_penalty.union(non_zero_penalty)


		embedding = {'idx': int(idx),
					'matches': matches + [int(candidate)],
					'penalty': penalty,
					'total_penalty': total_penalty}
		self.embedding_list.append(embedding)


	def load_mesh(self, mesh_path, align_to_depth_map=False):
		
		ext = mesh_path.split('.')[-1]

		if ext == "anime":
			verts,faces = load_anime_file(mesh_path)
			verts = verts[self.opt.anime_index]

		elif ext in ['obj','off','ply',"OFF"]:
			tmesh = trimesh.load_mesh(mesh_path, process=False, maintain_order=True)
			verts = np.asarray(tmesh.vertices)
			faces = np.asarray(tmesh.faces)
		else:
			raise NotImplementedError(f"Unable to load:{mesh_path}. Only .anime, .off, .obj, .ply supported") 


		if align_to_depth_map:
			extrinsic_matrix = np.loadtxt(os.path.join(self.opt.datadir, "extrinsics.txt"))
			R = extrinsic_matrix[:3, :3]
			T = extrinsic_matrix[:3, 3]

			bbox = (verts.max(axis=0) - verts.min(axis=0))
			orig_scale = np.linalg.norm(bbox)

			print("Orig Scale:",orig_scale)
			if orig_scale > 2 and orig_scale < 3:
				scale = 1
			else: 
				scale = 2/np.linalg.norm(bbox)

			R_inv = R.T
			T_inv = -R_inv@T

			R_inv *= np.array([scale,scale,scale]).reshape(3,1) 

			verts = verts@R_inv.T + T_inv

		print(verts,faces)	

		return verts,faces

	def load_trajectory(self,data_path,source_frame_id):

		trajectory_path = os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
		trajectory = np.load(trajectory_path)


		# TODO fix for multiple source frame
		face_filename = os.listdir(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path"))[0] 

		face_data = np.load(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path",face_filename))


		return trajectory,face_data	

	def save_transforms(self,filename,scale,R,T):
	
		with open(filename,'w') as f:
			f.write(f'{scale}\n')		
			f.write(" ".join([str(r) for r in R.reshape(-1)]))		
			f.write(" ".join([str(t) for t in T]))		


	def save_mesh(self, filename, verts, faces, largest_component=True, align_mesh=False, to_pinnochio=False, target_pcd=None):
		if not to_pinnochio:
			mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False, maintain_order=True)
			rot_inv_y = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])
			mesh.apply_transform(rot_inv_y)

			rot_inv_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
			mesh.apply_transform(rot_inv_x)
			save_obj(Path(filename), mesh.vertices, mesh.faces)
			return

		print("before processed: ", verts.shape, faces.shape)

		# Mesh processing for pinocchio
		mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
		if largest_component:
			components = mesh.split(only_watertight=False)
			if len(components) > 1:
				mesh = max(components, key=lambda comp: len(comp.faces))
		verts, faces = mesh.vertices, mesh.faces

		vertices, faces = preprocess_mesh_np(np.array(verts).copy(), np.array(faces).copy())
		vertices, faces = simplify_mesh(vertices, faces)
		mesh = clean_mesh(vertices, faces)

		mesh.update_faces(mesh.unique_faces())
		mesh.remove_unreferenced_vertices()
		mesh.remove_degenerate_faces()
		
		if not mesh.is_winding_consistent:
			mesh.fix_normals()
		if not mesh.is_watertight:
			mesh.fill_holes()

		mesh.remove_unreferenced_vertices()
		print("after processed:", mesh.vertices.shape, mesh.faces.shape)
  
		if align_mesh:
			# Align mesh to the target point cloud
			if target_pcd is not None:
				mesh_icp = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices), triangles=o3d.utility.Vector3iVector(mesh.faces))
				target_pc = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target_pcd))

				mesh_bbox = mesh_icp.get_axis_aligned_bounding_box()
				mesh_center = mesh_bbox.get_center()
				mesh_extent = np.max(mesh_bbox.get_extent())

				pcd_bbox    = target_pc.get_axis_aligned_bounding_box()
				pcd_center  = pcd_bbox.get_center()
				pcd_extent  = np.max(pcd_bbox.get_extent())

				scale_factor = pcd_extent / mesh_extent
				mesh_icp.scale(scale_factor, center=mesh_center)
				print(f"Rescaled mesh by factor: {scale_factor:.4f}")

				translation = pcd_center - mesh_center
				mesh_icp.translate(translation)
				print(f"Translated mesh by: {translation}")

				# Update original mesh data
				mesh.vertices = np.asarray(mesh_icp.vertices)
				mesh.faces = np.asarray(mesh_icp.triangles)

			# normalize for rotation
			centroid = mesh.centroid
			mesh.vertices -= centroid

			# Camear azimuth and elevation angles
			azim_base = 60
			elev_base = 10
			# Comment out if the mesh and camera are already aligned
			inv_az = trimesh.transformations.rotation_matrix(
				np.radians(-azim_base), [0, 1, 0])
			inv_el = trimesh.transformations.rotation_matrix(
				np.radians(-elev_base), [0, 0, 1])
			mesh.apply_transform(inv_el)
			mesh.apply_transform(inv_az)
			
			# Rotate mesh to align with the pcd
			rot = trimesh.transformations.rotation_matrix(
				np.radians(180), [1,0,0])
			mesh.apply_transform(rot)

			mesh.vertices += centroid
   
			mesh_pc = o3d.geometry.TriangleMesh(
					vertices=o3d.utility.Vector3dVector(mesh.vertices), 
					triangles=o3d.utility.Vector3iVector(mesh.faces)
				)
			mesh_pc.compute_vertex_normals()
			if target_pcd is not None:
				o3d.visualization.draw_geometries([mesh_pc, target_pc])
   
			rot = trimesh.transformations.rotation_matrix(
				np.radians(90), [1, 0, 0])
			mesh.apply_transform(rot)
	
			rot = trimesh.transformations.rotation_matrix(
				np.radians(90), [0, 1, 0])
			mesh.apply_transform(rot)
  
		save_obj(Path(filename), mesh.vertices, mesh.faces)
  

	def get_retargeted_motion(self,source_frame_id,source_path,target_path):

		# Read mesh 
		mesh_path = os.path.join(target_path,f'{source_frame_id}_mesh.obj')
		verts, faces = self.load_mesh(mesh_path, align_to_depth_map=False)

		# Get Target skel file 
		target_skel_path = os.path.join(target_path, f'{source_frame_id}_target.skel') 
		target_skeleton, parent_array = self.read_skel(target_skel_path)


		# Get attachment file 
		target_attachment_path = os.path.join(target_path,f'{source_frame_id}_attachment.out')
		target_attachment = np.loadtxt(target_attachment_path)

		
		assert verts.shape[0] == target_attachment.shape[0], f"Attachment file:{target_attachment.shape} doesn't match the target mesh:{verts.shape}"
		assert len(parent_array) == target_attachment.shape[1] +1, f"Attachment file:{target_attachment.shape} should have N-1 bones. Joints:{len(parent_array)}"

		# Get source skel file
		skeleton_path =  os.path.join(source_path,f"skeleton/{source_frame_id}.skel")
		source_skeleton, source_parent_array = self.read_skel(skeleton_path)
		
		reduce2full_skeleton_indices = np.load(os.path.join(target_path,f"reduced2full_skeleton_{source_frame_id}.npy"))

		source_skeleton = source_skeleton[reduce2full_skeleton_indices]

		# Skeleton motion file 
		skeleton_motion_path = os.path.join(source_path,"skeleton",f"skeletonMotion_{source_frame_id}.npy")
		skeleton_motion = np.load(skeleton_motion_path)

		skeleton_motion = skeleton_motion[:,reduce2full_skeleton_indices]

		# Get bone visibility 

		boneVis = np.loadtxt("./boneVis.out") if os.path.isfile("./boneVis.out") else None
		boneDists = np.loadtxt("./boneDists.out") if os.path.isfile("./boneDists.out") else None
		projected_Location = np.loadtxt('./projToSeg.out') if os.path.isfile("./projected_Location.out") else None


		# print(boneVis.shape)
		# print(boneDists.shape)

		# Get skeleton motion
			# Assert first frame is equal 

		# target_attachment_label = np.argmax(target_attachment,axis=1)
		# target_attachment[:,:] = 0
		# target_attachment[np.where(np.sum(target_attachment[:,0:2]) < 0.1)[0],0] = 1

		# target_attachment /= target_attachment.sum(axis=1,keepdims=True)

		# print(target_attachment)

		mesh_motion,target_skeleton_motion = get_mesh_motion(verts, parent_array, target_skeleton, skeleton_motion, target_attachment)

		# mesh_motion = np.tile(verts.reshape(1,-1,3),(len(skeleton_motion),1,1))
		# target_skeleton_motion = np.tile(target_skeleton.reshape(1,-1,3),(len(skeleton_motion),1,1))

		trajectory_path = os.path.join(source_path,"trajectory",f"trajectory_{source_frame_id}.npy")
		trajectory = np.load(trajectory_path)

		o3d.visualization.draw_geometries([
			o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(trajectory[0])).paint_uniform_color([0,0,1]),
			o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(verts)).paint_uniform_color([1,0,0]),
			o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target_skeleton)).paint_uniform_color([0,1,0]),
		])

		self.vis.plot_retarget_motion(
			source_frame_id, mesh_motion, faces, trajectory, target_skeleton_motion,skeleton_motion,parent_array,target_attachment,boneVis=boneVis,boneDists=boneDists,proj=projected_Location)

		return mesh_motion,faces,target_skeleton_motion 



	def __call__(self,mesh_path):	


		# Load mesh or anime file  
		target_verts, target_faces = self.load_mesh(mesh_path, align_to_depth_map=self.opt.align_to_depth_map)

		data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
		trajectory_path = os.path.join(data_path,"trajectory")

		target_path = "_".join(os.path.basename(mesh_path).split('.')) + '_' + str(self.opt.anime_index)
		target_path = os.path.join(data_path,target_path)
		os.makedirs(target_path, exist_ok=True)


		traj_files = sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1]))
		for file in traj_files:
			self.log.info(f"=====Loading trajectory:{file}========")
			source_frame_id = int(file.split('.')[0].split('_')[-1])
			trajectory,face_data = self.load_trajectory(data_path,source_frame_id)

			transforms_path = os.path.join(target_path,f"{source_frame_id}_transforms.txt")
			mesh_path = os.path.join(target_path,f"{source_frame_id}_mesh.obj")
			print(mesh_path)
   
			if not self.opt.align_mesh_orientation:
				# Get transformatin matrix to align source and target as input 
				transformed_target_verts,R,T,scale = self.vis.get_rigid_transforms_from_user(trajectory[0],target_verts,target_faces)
				self.save_transforms(transforms_path,scale,R,T)	
				self.save_mesh(
        			mesh_path,transformed_target_verts,target_faces,largest_component=self.opt.select_largest_component) 
				ply_filepath = mesh_path
			else:
				# Fix problems in mesh and save it
				self.save_mesh(mesh_path, target_verts, target_faces, largest_component=self.opt.select_largest_component, 
					align_mesh=self.opt.align_mesh_orientation, to_pinnochio=True, target_pcd=trajectory[0]) 

				ply_filepath = str(mesh_path)[:-4] + ".ply"
				convert_obj_to_ply(mesh_path, ply_filepath)

			# if not os.path.isfile(os.path.join(target_path,f'{source_frame_id}_target.skel')):
			skeleton_path =  os.path.join(data_path,f"skeleton/{source_frame_id}.skel")
			joint_positions, original_parent_array = self.read_skel(skeleton_path)

			# Keep running until no nodes are can be removed
			reduced2full_skeleton = list(range(len(joint_positions)))
			full2reduce_skeleton = dict([(x,i) for i,x in enumerate(reduced2full_skeleton)])
			prev_num_joints = len(joint_positions)
			updated_pinocchio_skel_path = skeleton_path

			while True:
				
				print("Running Command:", f"{self.pinoccio_path} {ply_filepath} -skel {updated_pinocchio_skel_path}")
				process = subprocess.Popen(
					f"{self.pinoccio_path} {ply_filepath} -skel {updated_pinocchio_skel_path}",
					shell=True,stdout = subprocess.PIPE, 
					stderr = subprocess.PIPE, 
					encoding='utf8'
				)	# Reference https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
				
				while True:
					output = process.stdout.readline()
					if output == '' and process.poll() is not None:
						break
					elif "Idx" in output: # Discrete embedding details
						self.parse_embedding(output)
					elif output :
						print(output.strip())


				self.optimised_skeleton = self.read_skel('./skeleton.out') # skeleton_out
				if len(self.optimised_skeleton[0]) == prev_num_joints:
					break
				else: 
					# Update Pinnocio skeleton
					remove_node = reduced2full_skeleton[len(self.optimised_skeleton[0])]
					remove_joints_list = [full2reduce_skeleton[remove_node]]
					stack = [remove_node]
					while len(stack) > 0: 
						print("Stack:",stack)
						x = stack[-1]
						stack = stack[:-1]
						# Find nodes for which are desecdants of x but are not counted in children
						children = []
						for i,p in enumerate(original_parent_array):
							if i not in full2reduce_skeleton:
								continue
							elif p == x and full2reduce_skeleton[i] not in remove_joints_list:
								children.append(i)
						remove_joints_list.extend([full2reduce_skeleton[c] for c in children])
						stack.extend(children)
						print(f"Removing Joints:{remove_joints_list}")

					# Update reduced2full_skeleton
					new_reduced2full_skeleton = []
					new_parent_array = []
					mew_full2reduce_skeleton = {}
					new_joint_positions = []
					for i in range(prev_num_joints): # Iterating over all previous
						if i in remove_joints_list: 
							 continue
						
						mew_full2reduce_skeleton[reduced2full_skeleton[i]] = len(new_reduced2full_skeleton)
						new_reduced2full_skeleton.append(reduced2full_skeleton[i])
						new_parent_array.append(mew_full2reduce_skeleton[original_parent_array[reduced2full_skeleton[i]]])
						new_joint_positions.append(joint_positions[reduced2full_skeleton[i]])


					prev_num_joints = len(new_joint_positions)
					updated_pinocchio_skel_path = ".".join(skeleton_path.split('.')[:-1]) + f'_{prev_num_joints}.skel'
					new_joint_positions = np.array(new_joint_positions)
					self.write_skel(updated_pinocchio_skel_path,new_joint_positions,new_parent_array)
					reduced2full_skeleton = new_reduced2full_skeleton
					full2reduce_skeleton = mew_full2reduce_skeleton

			print("Reduce to full skeleton:",reduced2full_skeleton)			
			np.save(os.path.join(target_path,f"reduced2full_skeleton_{source_frame_id}.npy"), reduced2full_skeleton)		
					
			os.system(f"mv ./skeleton.out {os.path.join(target_path,f'{source_frame_id}_target.skel')}")
			os.system(f"mv ./attachment.out {os.path.join(target_path,f'{source_frame_id}_attachment.out')}")
			if not self.opt.align_mesh_orientation:
				os.system(f"mv ./transforms.txt {os.path.join(target_path,f'{source_frame_id}_transforms.txt')}")
			os.system(f"mv ./vertices.obj {os.path.join(target_path,f'{source_frame_id}_final_vertices.obj')}")
			self.log.info("Saved Pinnochio Results!")

			if os.path.isfile(os.path.join(target_path,f'{source_frame_id}_target.skel')):
       
				if self.opt.align_mesh_orientation:
					target_verts, target_faces = self.load_mesh(mesh_path, align_to_depth_map=False)
					self.save_mesh(mesh_path, target_verts, target_faces, largest_component=self.opt.select_largest_component, target_pcd=trajectory[0])
					self.log.info("Saved Final Mesh!")
		
				# Perform motion retargetting 
				retargeted_motion,faces,target_skeleton_motion = self.get_retargeted_motion(source_frame_id,data_path,target_path)

				mesh_motion_path = os.path.join(target_path,f"{source_frame_id}_retargeted_motion.npy")
				faces_path = os.path.join(target_path,f"{source_frame_id}_faces.npy")
				target_skeleton_motion_path = os.path.join(target_path,f"{source_frame_id}_retargeted_skeleton_motion.npy")
				rendered_retargeted_video_path = os.path.join(target_path,f"{source_frame_id}_retargeted_video.mp4")

				np.save(mesh_motion_path, retargeted_motion)
				np.save(faces_path, faces)
				np.save(target_skeleton_motion_path ,target_skeleton_motion)

				# Plot and show output 
				import matplotlib.pyplot as plt
				from matplotlib.animation import FFMpegWriter
				def render_mesh_video(npy_file_path, output_video_path, fps=10):
					data = np.load(npy_file_path)  # Shape: (Frames, N, 3)
					frames, num_points, _ = data.shape

					dpi = 100
					fig = plt.figure(figsize=(8, 8), dpi=dpi)
					ax = fig.add_subplot(111, projection='3d')
					metadata = {'title': 'Mesh Trajectory', 'artist': 'Matplotlib'}
					writer = FFMpegWriter(fps=fps, metadata=metadata)

					with writer.saving(fig, output_video_path, dpi):
						for frame_idx in range(frames):
							ax.clear()
							ax.scatter(data[frame_idx, :, 0], data[frame_idx, :, 1], data[frame_idx, :, 2], c='b', marker='o')
							ax.set_xlim(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
							ax.set_ylim(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
							ax.set_zlim(np.min(data[:, :, 2]), np.max(data[:, :, 2]))
							ax.set_xlabel("X Axis")
							ax.set_ylabel("Y Axis")
							ax.set_zlabel("Z Axis")
							ax.set_title(f"Frame {frame_idx + 1}/{frames}")
							writer.grab_frame()

					print(f"Video saved to {output_video_path}")
					
				render_mesh_video(mesh_motion_path,rendered_retargeted_video_path)




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




if __name__ == "__main__": 
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
	args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
	args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')

	# Mesh details 
	args.add_argument('--mesh', required=True,type=str,help='path to mesh to transfer motion')
	args.add_argument('--anime_index', required=False,default=0,type=int,help='Since anime files contains trajectory. If using anime file, provide frame index to be used')
	args.add_argument('--align_to_depth_map', action='store_true', help='If provided, aligns using extrinsics matrix')
	args.add_argument('--select_largest_component', action='store_true', help='If provided, selects largest component of mesh')	
	args.add_argument('--align_mesh_orientation', action='store_true', help='If provided, aligns mesh orientation to world coordinate system')

	# Arguments for debugging  
	args.add_argument('--debug', default=True, type=bool, help='Whether debbugging or not. True: Logging + Visualization, False: Only Critical information and plots')
	args.add_argument('--vis', default='polyscope', type=str, help='Visualizer to plot results')

	opt = args.parse_args()

	# stream_handler = logging.StreamHandler()
	# stream_handler.setLevel(logging.INFO)

	opt.config_file = os.path.join(opt.datadir, "camera_config.txt")

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

	print(opt)

	# Create visualizer 
	vis = get_visualizer(opt) # To see data
	skel_embedding = SkeletonEmbedding(opt,vis)

	skel_embedding(opt.mesh)	
