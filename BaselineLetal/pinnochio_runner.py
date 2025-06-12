# Given data dir and input mesh/anime file embed skeleton and perform motion retargetting  

import os
import sys
import logging 
import argparse 
import subprocess
import numpy as np 
import open3d as o3d


import trimesh # Fix mesh


sys.path.append("../")  # Making it easier to load modules
sys.path.append("../../")  # Making it easier to load modules
import find_connected_components
from vis import get_visualizer # Visualizer 
from motion_retargeting import get_mesh_motion

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

	def rearrange_skeleton(self,joint_positions,parent_array):
		"""
			Rearange by running dfs from root 
		"""

		new_indices = []
		old2new_indices = {}
		new_parent_array = []

		root_ind = 0

		stack = [[root_ind,root_ind]]

		while len(stack) > 0:
			x,p = stack[-1]
			stack.pop()

			old2new_indices[x] = len(new_indices)
			new_indices.append(x)
			new_parent_array.append(old2new_indices[p])

			children = [c for c,p in enumerate(parent_array) if p == x]	

			for c in children:
				if c not in new_indices:
					stack.append([c,x])
		

		print(new_indices)
		print(parent_array)
		print(new_parent_array)

		new_indices = np.array(new_indices)

		joint_positions = joint_positions[:,new_indices,:]
		# labels = new_indices[labels]
		return joint_positions,new_parent_array


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


	def load_mesh(self,mesh_path,align_to_depth_map=False):
		
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

			# scale = 1
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


	def save_mesh(self,filename,verts,faces,largest_component=True):
		
		# import find_connected_components
		if largest_component:
			colors,c_cnt = find_connected_components.find_components(verts,faces)
			verts,faces = find_connected_components.save_largest_component(colors,verts,faces,filename)

		# mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts),triangles=o3d.utility.Vector3iVector(faces))
		# o3d.io.write_triangle_mesh(filename,mesh)
		# o3d.visualization.draw_geometries([mesh])

		tmesh = trimesh.Trimesh(vertices=verts,faces=faces,process=True)
		# tmesh.process()

		# tmesh.merge_vertices()

		if not tmesh.is_watertight:
			print("Not watertight Fixing holes:",tmesh.fill_holes())
			print(tmesh)

		if not tmesh.is_winding_consistent:
			print("Not winding consistent Fixing normals:",tmesh.fix_normals())
			print(tmesh)

		# print("Remove unreferenced vertices:",tmesh.remove_unreferenced_vertices())
		tmesh.fix_normals()
		# print(tmesh)

		tmesh.export(filename)


		# mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(sub_vertices),triangles=o3d.utility.Vector3iVector(sub_faces))	
		# o3d.io.write_triangle_mesh(savepath,mesh)
		# o3d.visualization.draw_geometries([mesh])

	
		os.system(f"meshlabserver -i {filename} -o {filename} -s ../closeholes.mlx")
		print("Running Command:",f"meshlabserver -i {filename} -o {filename} -s ../closeholes.mlx")

	def get_retargeted_motion(self,source_frame_id,source_path,target_path,orig_data_path):


		trajectory_path = os.path.join(orig_data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
		trajectory = np.load(trajectory_path)


		# Read mesh 
		mesh_path = os.path.join(target_path,f'{source_frame_id}_final_vertices.obj')
		verts,faces = self.load_mesh(mesh_path,align_to_depth_map=False)

		# Get Target skel file 
		target_skel_path = os.path.join(target_path,f'{source_frame_id}_target.skel')
		target_skeleton,parent_array = self.read_skel(target_skel_path)


		# Get attachment file 
		target_attachment_path = os.path.join(target_path,f'{source_frame_id}_attachment.out')
		target_attachment = np.loadtxt(target_attachment_path)

		
		assert verts.shape[0] == target_attachment.shape[0], f"Attachment file:{target_attachment.shape} doesn't match the target mesh:{verts.shape}"
		if len(parent_array) != target_attachment.shape[1] +1:
			if len(parent_array) == 1:
				return np.tile(verts.reshape((1,-1,3)),(trajectory.shape[0],1,1)),faces,np.tile(target_skeleton.reshape((1,-1,3)),(trajectory.shape[0],1,1)) 
			raise NotImplementedError(f"Error: Attachment file:{target_attachment.shape} should have N-1 bones. Joints:{len(parent_array)}. If Joints==1 basically that means no match found")

		# Get source skel file
		skeleton_path =  os.path.join(source_path,f"rearranged_{source_frame_id}.skel")
		source_skeleton,source_parent_array = self.read_skel(skeleton_path)
		
		reduce2full_skeleton_indices = np.load(os.path.join(target_path,f"reduced2full_skeleton_{source_frame_id}.npy"))
	
		# Skeleton motion file 
		skeleton_motion_path = os.path.join(source_path,f"skelMotion__rearranged_{source_frame_id}.npy")
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

		mesh_motion,target_skeleton_motion = get_mesh_motion(verts,parent_array,target_skeleton,skeleton_motion,target_attachment)

		# mesh_motion = np.tile(verts.reshape(1,-1,3),(len(skeleton_motion),1,1))
		# target_skeleton_motion = np.tile(target_skeleton.reshape(1,-1,3),(len(skeleton_motion),1,1))

		
		print(f"Plotting Parent:",parent_array)



		# self.vis.plot_retarget_motion(source_frame_id,mesh_motion,faces,trajectory,target_skeleton_motion,skeleton_motion,parent_array,target_attachment,boneVis=boneVis,boneDists=boneDists,proj=projected_Location)

		return mesh_motion,faces,target_skeleton_motion 



	def __call__(self,mesh_path):	


		# Load mesh or anime file  
		target_verts, target_faces = self.load_mesh(mesh_path,align_to_depth_map=self.opt.align_to_depth_map)

		data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
		trajectory_path = os.path.join(data_path,"trajectory")

		target_path = "SSDR_".join(os.path.basename(mesh_path).split('.')) + '_' + str(self.opt.anime_index)
		target_path = os.path.join(data_path,target_path)
		os.makedirs(target_path,exist_ok=True)


		traj_files = sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1]))
		for file in traj_files:
			self.log.info(f"=====Loading trajectory:{file}========")
			source_frame_id = int(file.split('.')[0].split('_')[-1])
			trajectory,face_data = self.load_trajectory(data_path,source_frame_id)


			transforms_path = os.path.join(target_path,f"{source_frame_id}_transforms.txt")
			mesh_path = os.path.join(target_path,f"{source_frame_id}_mesh.obj")


			if not os.path.isfile(mesh_path): # If already computed skip

				# Get transformatin matrix to align source and target as input 
				transformed_target_verts,R,T,scale = self.vis.get_rigid_transforms_from_user(trajectory[0],target_verts,target_faces)


				# preprocess for pinnochio 
					# Save transformation 
				self.save_transforms(transforms_path,scale,R,T)	
				
				largest_component = True
				print(mesh_path)
				# Fix problems in mesh and save it
				self.save_mesh(mesh_path,transformed_target_verts,target_faces,largest_component=largest_component) 


			skeleton_path =  os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}/{source_frame_id}.skel")
			joint_positions,original_parent_array = self.read_skel(skeleton_path)

			skeleton_motion_path = os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}",f"skelMotion_{source_frame_id}.npy")
			skeleton_motion = np.load(skeleton_motion_path)

			skeleton_motion,original_parent_array = self.rearrange_skeleton(skeleton_motion,original_parent_array)


			self.write_skel(os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}/rearranged_{source_frame_id}.skel"),skeleton_motion[0],original_parent_array)

			np.save(os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}",f"skelMotion__rearranged_{source_frame_id}.npy"),skeleton_motion)

			skeleton_path = os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}/rearranged_{source_frame_id}.skel")


			# Keep running until no nodes are can be removed
			reduced2full_skeleton = list(range(len(joint_positions)))
			full2reduce_skeleton = dict([(x,i) for i,x in enumerate(reduced2full_skeleton)])
			prev_num_joints = len(joint_positions)
			updated_pinocchio_skel_path = skeleton_path
			while True:
				
				print("Running Command:", f"{self.pinoccio_path} {mesh_path} -skel {updated_pinocchio_skel_path}")
				process = subprocess.Popen(f"{self.pinoccio_path} {mesh_path} -skel {updated_pinocchio_skel_path}",shell=True,stdout = subprocess.PIPE, stderr = subprocess.PIPE, encoding='utf8')# Reference https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
				while True:
					output = process.stdout.readline()
					if output == '' and process.poll() is not None:
						break
					elif "Idx" in output: # Discrete embedding details
						self.parse_embedding(output)
					elif output :
						print(output.strip())


				self.optimised_skeleton = self.read_skel('./skeleton.out')
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
			np.save(os.path.join(target_path,f"reduced2full_skeleton_{source_frame_id}.npy"),reduced2full_skeleton)		
			


			os.system(f"mv ./skeleton.out {os.path.join(target_path,f'{source_frame_id}_target.skel')}")
			os.system(f"mv ./attachment.out {os.path.join(target_path,f'{source_frame_id}_attachment.out')}")
			os.system(f"mv ./vertices.obj {os.path.join(target_path,f'{source_frame_id}_final_vertices.obj')}")
			self.log.info("Saved Pinnochio Results!")

			if os.path.isfile(os.path.join(target_path,f'{source_frame_id}_target.skel')):
				# Perform motion retargetting 
				retargeted_motion,faces,target_skeleton_motion = self.get_retargeted_motion(source_frame_id,os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}"),target_path,data_path)

				np.save(os.path.join(target_path,f"{source_frame_id}_retargeted_motion.npy"),retargeted_motion)
				np.save(os.path.join(target_path,f"{source_frame_id}_retargeted_skeleton_motion.npy"),target_skeleton_motion)
				# Plot and show output 



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
	args.add_argument('--align_to_depth_map', required=False,default=True,type=bool,help='If same anime file is being used. You can align using the extrinsics matrix')



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

	print(opt)

	# Create visualizer 
	vis = get_visualizer(opt) # To see data
	skel_embedding = SkeletonEmbedding(opt,vis)

	skel_embedding(opt.mesh)	
