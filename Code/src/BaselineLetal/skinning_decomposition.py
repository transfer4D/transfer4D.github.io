# Goal is to use the trajectroy information to cluster articulated regions
# Also ask user for right number of clusters 
# Run global update to remove redundant clusters
import os
import sys
import time
import logging
import argparse
import numpy as np 
from scipy import sparse
import heapq
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
sys.path.append("../")  # Making it easier to load modules
sys.path.append("../../")  # Making it easier to load modules
from fusion_tests import pyssdr
from vis import get_visualizer # Visualizer 


class Edge:
	def __init__(self, i,j,dist):
		self.i = i
		self.j = j
		self.d = dist

	def __repr__(self):
		return f'Edge {self.i}-{self.j} dist: {self.d}'

	def __lt__(self, other):
		return self.d < other.d


def minimum_spanning_tree(root_ind,joint_matrix,adj_matrix,trajectory,labels):

	visited = [root_ind]
	num_clusters = adj_matrix.shape[0]


	root_joint_motion = np.mean(trajectory[:,labels==root_ind,:],axis=1)
	joint_positions = [root_joint_motion]

	joint2cluster = [root_ind] # Source and target cluster
	cluster2joint = {root_ind:0}

	parent_array = [0] # Parent and child


	root_children = np.where(adj_matrix[root_ind] < np.inf)[0]
	heap = [Edge(root_ind,c,adj_matrix[root_ind,c]) for c in root_children]		
	heapq.heapify(heap)

	while len(visited) < num_clusters and len(heap) > 0:
		# print(visited,num_clusters)
		e = heapq.heappop(heap)
		print(f"Removing:{e}")
		if e.i in visited and e.j in visited:
			continue
		elif e.i in visited and e.j not in visited:
			parent = e.i 
			child = e.j 
		elif e.i not in visited and e.j in visited:
			parent = e.j 
			child = e.i
		else: 
			raise Error("Edge not in visisted.") 

		visited.append(child)


		cluster2joint[child] = len(joint_positions) 	
		joint_positions.append(joint_matrix[:,parent,child])	
		joint2cluster.append(child)
		parent_array.append(cluster2joint[parent]) 

		neigbouring_clusters = np.where(adj_matrix[child] < np.inf)[0]
		for n in neigbouring_clusters:
			heapq.heappush(heap,Edge(child,n,adj_matrix[child,n]))


	# Adding virtual joints 
	# Degree 1 joints 
	degree1_joints = [ i for i in range(len(parent_array)) if i not in parent_array] 

	for j in degree1_joints:
		child_cluster = joint2cluster[j]
		cluster_verts = np.where(labels == child_cluster)[0]

		virtual_joint = np.argmax(np.linalg.norm(trajectory[0,cluster_verts,:] - joint_positions[j][0],axis=1))
		virtual_joint = cluster_verts[virtual_joint]
		joint_positions.append(trajectory[:,virtual_joint])
		parent_array.append(j)



	

	joint_positions = np.array(joint_positions)
	joint_positions = joint_positions.transpose((1,0,2))

	return joint_positions, parent_array		


def clean_labels(labels):
	unique_labels,label_count = np.unique(labels,return_counts=True)
	new_labels = -1*np.ones_like(labels)

	old2new = {}
	new2old = []
	for i,x in enumerate(unique_labels):
		new_labels[labels==x] = i 
		old2new[x] = i
		new2old.append(x)
	assert np.sum(new_labels==-1) == 0, "Some labels didn't update"

	return new_labels,old2new,new2old

# Read skel file 
def read_skel(filepath):
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

def find_boundaries(points,labels,k=12):
	# Calulate the leaf nodes and boundary points as joints of a skeleton
	points = points
	tree = KDTree(points)
	N = points.shape[0]
	num_labels = np.max(np.unique(labels))+1
	dist_matrix,neighbours_matrix = tree.query(points,k) 

	segment_boundaries = {}

	for i in range(N):
		neighbours = neighbours_matrix[i,:]
		neighbours_labels = labels[neighbours]
		

		not_same_labels = np.where(neighbours_labels != labels[i])[0] 
		for n in not_same_labels:
			idx = labels[i]*num_labels + neighbours_labels[n]
			if idx not in segment_boundaries:
				segment_boundaries[idx] = []

			segment_boundaries[idx].extend(neighbours)

	already_computed = []
	for x  in segment_boundaries:
		if x not in already_computed:
			already_computed.append(x)
			a = x//num_labels
			b = x%num_labels
			y = b*num_labels + a
			if y in segment_boundaries:
				already_computed.append(y)
				segment_boundaries[x].extend(segment_boundaries[y])
			segment_boundaries[x] = np.unique(segment_boundaries[x])		
	return segment_boundaries







class SSDR(pyssdr.MyDemBones):
	def __init__(self,vis):
		super().__init__()

		self.log = logging.getLogger(__name__)
		self.vis = vis 


	def load_trajectory(self,data_path,source_frame_id):

		trajectory_path = os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
		trajectory = np.load(trajectory_path)

		# TODO fix for multiple source frame
		face_filename = os.listdir(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path"))[0] 

		face_data = np.load(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path",face_filename))

		return trajectory,face_data




	def check_sparse_matrix(self,skinning_anchors,skinning_weights):
		# w = sparse.load_npz(os.path.join(savepath,f"weights_{0}.npz"))
		w = self.w
		w_dense = self.w.todense()
		print(w.indices.shape,w.indptr.shape,w.data.shape,w.shape,w.nnz)
		for col in range(w.shape[1]): # Loop over all the cols
			anchors = skinning_anchors[col]
			for nz_id in range(w.indptr[col],w.indptr[col+1]): # Loop over all ids for that row
				row = w.indices[nz_id]
				val = w.data[nz_id]

				anchor = np.where(anchors==row)[0]
				assert len(anchor) == 1,f"W(skinning vector is wrong), For {col}, graph node:{row} not present in {anchors}"         
				anchor = anchor[0]
				assert val == skinning_weights[col,anchor], f"W[{row,col}] = {val} != {skinning_weights[col,anchor]}"

		return True
	
	def plot_reconstruction(self,deformed_vertices,target_vertices,faces):
		import open3d as o3d
		vis = o3d.visualization.Visualizer()
		vis.create_window(width=1280, height=960,window_name="Fusion Pipeline")

		for t in range(deformed_vertices.shape[0]):
			if t == 0:

				deformed_mesh = o3d.geometry.TriangleMesh(
					o3d.utility.Vector3dVector(deformed_vertices[t]),
					o3d.utility.Vector3iVector(faces))

				vis.add_geometry(deformed_mesh)

				# target_mesh = o3d.geometry.TriangleMesh(
				#     o3d.utility.Vector3dVector(target_vertices[t]),
				#     o3d.utility.Vector3iVector(faces))
				# vis.add_geometry(target_mesh)
			else: 
				deformed_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(deformed_vertices[t]))
				vis.update_geometry(deformed_mesh)
				# target_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(target_vertices[t]))
				# vis.update_geometry(target_mesh)

			time.sleep(1)

			vis.poll_events()
			vis.update_renderer()

		
	def get_transforms_global_update(self,trajectory,faces,savepath, num_clusters=10):

		movement = trajectory.copy()
		# Preprocess trajectory to pass as input
		U = trajectory[0]
		trajectory = trajectory.transpose((0,2,1))
		T,D,N = trajectory.shape
		trajectory = trajectory.reshape((D*T,N))

		# Set parameters to initize SSDR
		self.nInitIters = 10
		self.tolerance = 1e-2
		self.bindUpdate=2
		self.nIters = 5
		self.min_bones = 5
		self.patience = 1
		self.global_update_threshold = np.inf


		self.load_data(trajectory,faces) # Load data into pyssdr 

		W,M,rmse_err = self.run_ssdr(num_clusters,"here.fbx")

		print(self.mTm.shape)

		# Bone difference:
		bone_diff = np.zeros((self.nB,self.nB)) 
		for i in range(self.nB):
			for j in range(i+1,self.nB):
				bone_diff[i,j] = bone_diff[j,i] = np.sum(self.mTm[4*i:4*i+4,4*i:4*i+4] + self.mTm[4*j:4*j+4,4*j:4*j+4] - self.mTm[4*j:4*j+4,4*i:4*i+4] - self.mTm[4*i:4*i+4,4*j:4*j+4])


		print(bone_diff)

		skin_list = [];
		relative_change_list = [0]
		relative_change_list_rsme = [rmse_err]
		removed_bones_list = []

		np.save(os.path.join(savepath,"relative_change_list.npy"),np.array(relative_change_list))
		np.save(os.path.join(savepath,"relative_change_list_rsme.npy"),np.array(relative_change_list_rsme))
		sparse.save_npz(os.path.join(savepath,f"weights_{len(removed_bones_list)}.npz"),self.w)
		np.save(os.path.join(savepath,f"reconstruction_{len(removed_bones_list)}.npy"),self.compute_reconstruction(list(range(N))))

		self.compute_errorVtxBoneALL() # Precompute error of each vertex w.r.t every bone

		# Loop until minimum bones is reached. Break if threshold is reached 
		while len(self.keep_bones) > self.min_bones:

			# Create a vector of vertices for which reconstruction error needs to be calculated
			all_verts_inds=list(range(self.nV))
			# Get reconstruction
			orig_recon_err = self.vertex_rmse(all_verts_inds);
				
			# self.prev_nB = self.nB;
			self.prev_w_csc = self.w.copy();
			# self.prev_w = self.w.todense(); # Create a dense matrix for easier manupulation
			# self.prev_m = self.m.copy();


			# print(self.prev_w.shape,self.prev_m.shape)
			old_keepbones = self.keep_bones.copy()
			reconstruction_change_list = np.inf*np.ones(self.nB)
			for b in old_keepbones:
				# Try removing bone b and check reconstruction error on the affected vertices
				# affected_vert_inds = self.prev_w_csc.getrow(b).indices;
				affected_vert_inds = self.w.getrow(b).indices;
				# print("Trying to calculate affected vertices for bone:",b,affected_vert_inds)

				llll = len(affected_vert_inds)
				print(llll)
				if llll > 3: 

					self.keep_bones = np.delete(self.keep_bones,np.where(self.keep_bones==b))

					# self.remove_bone(b);
					# print(self.keep_bones)


					# self.computeTranformations();
					# print(m.m[:4,:4])
					self.w = self.prev_w_csc.copy()
					for i in range(1):
						self.computeWeights()
						# self.computeTranformations()        
						# if self.cbIterEnd():
							# break

					new_recon_err = self.vertex_max_rmse(affected_vert_inds);               
					# Compute relative change in reconstruction error 
					relative_rec_error_change = -np.inf;
					orig_recon_err_sum = 0;

					
					test_w_dense = self.w.todense().T
					test_w_prev_dense = self.prev_w_csc.todense().T

					for i in range(llll):
						vert_ind = affected_vert_inds[i];
						# relative_rec_error_change += new_recon_err[i] - orig_recon_err[vert_ind];
						relative_rec_error_change = max(relative_rec_error_change,new_recon_err[i] - orig_recon_err[vert_ind]);
						orig_recon_err_sum += orig_recon_err[vert_ind];

						# std::cout << "Ind:" << vert_ind << " Orig error:" << orig_recon_err[vert_ind] << " New error:" << new_recon_err[i] << " Change:" << relative_rec_error_change << std::endl;
						new_idxs = np.where(test_w_dense[vert_ind,:])[1]
						old_idxs = np.where(test_w_prev_dense[vert_ind,:])[1]
						# if len(new_idxs) == 1 and len(old_idxs) == 1: 
						#     print(vert_ind,new_idxs,old_idxs,bone_diff[new_idxs,old_idxs])
					
					# relative_rec_error_change /= orig_recon_err_sum;
				
					new_rmse = self.rmse()
					print("Testing bone:", b, " Relative change:", relative_rec_error_change,self.w.shape,self.m.shape,new_rmse,(new_rmse-rmse_err)/rmse_err);
					reconstruction_change_list[b] = relative_rec_error_change   
					self.keep_bones = np.insert(self.keep_bones,self.keep_bones.shape[0],b)
				else:
					# reconstruction_change_list[b] = 0   
					reconstruction_change_list[b] = -np.inf   
					break

			remove_bone = np.argmin(reconstruction_change_list[self.keep_bones])
			remove_bone = self.keep_bones[remove_bone] 
			print("Can remove:",remove_bone,reconstruction_change_list)
			if reconstruction_change_list[remove_bone] < self.global_update_threshold:
			# if self.prev_nB > self.min_bones:
				# self.remove_bone(remove_bone)
				removed_bones_list.append(remove_bone)
				self.keep_bones = np.delete(self.keep_bones,np.where(self.keep_bones==remove_bone))
				self.w = self.prev_w_csc.copy()
				for i in range(10):
					self.computeWeights()
					# self.computeTranformations()        
					if self.cbIterEnd():
						break

				relative_change_list.append(reconstruction_change_list[remove_bone])
				relative_change_list_rsme.append(self.rmse())

				print("Removing Bone:",remove_bone," Recontruction change:",relative_change_list[-1], " New  RSME:",relative_change_list_rsme[-1])
				print("Removed Bones:",removed_bones_list)
				# Save reconstruction result                 
				# Save results
				np.save(os.path.join(savepath,"relative_change_list.npy"),np.array(relative_change_list))
				np.save(os.path.join(savepath,"relative_change_list_rsme.npy"),np.array(relative_change_list_rsme))
				sparse.save_npz(os.path.join(savepath,f"weights_{len(removed_bones_list)}.npz"),self.w)
				np.save(os.path.join(savepath,f"reconstruction_{len(removed_bones_list)}.npy"),self.compute_reconstruction(list(range(N))))


			else: 
				break
		
		for i in range(10):
			self.computeTranformations()        
			self.computeWeights()
			if self.cbIterEnd():
				break


		deformed_vertices_ssdr = self.compute_reconstruction(list(range(N)))
		deformed_vertices = np.zeros((T,N,3))
		deformed_vertices[:,:,0] = deformed_vertices_ssdr[list(range(0,3*T,3))]
		deformed_vertices[:,:,1] = deformed_vertices_ssdr[list(range(1,3*T,3))]
		deformed_vertices[:,:,2] = deformed_vertices_ssdr[list(range(2,3*T,3))]

		# self.plot_reconstruction(deformed_vertices,movement,faces)

		return deformed_vertices,self.w.todense(),self.m,self.rmse()

	def initialize_ssdr(self,trajectory,faces,num_clusters=30):

		movement = trajectory.copy()
		# Preprocess trajectory to pass as input
		U = trajectory[0]
		trajectory = trajectory.transpose((0,2,1))
		T,D,N = trajectory.shape
		trajectory = trajectory.reshape((D*T,N))

		# Set parameters to initize SSDR
		self.nInitIters = 10
		self.tolerance = 1e-2
		self.bindUpdate=2
		self.nIters = 5
		self.nItersFinetuning = 10
		self.min_bones = 5
		self.patience = 1
		self.global_update_threshold = np.inf


		self.load_data(trajectory,faces) # Load data into pyssdr 

		self.nB = num_clusters

		self.init()


	def get_different_cluster(self,trajectory,faces, num_clusters=10,minimum_label_size_percentage=1e-3):

		movement = trajectory.copy()
		# Preprocess trajectory to pass as input
		U = trajectory[0]
		trajectory = trajectory.transpose((0,2,1))
		T,D,N = trajectory.shape
		trajectory = trajectory.reshape((D*T,N))

		# Set parameters to initize SSDR
		self.nInitIters = 10
		self.tolerance = 1e-2
		self.bindUpdate=2
		self.nIters = 5
		self.nItersFinetuning = 10
		self.min_bones = 5
		self.patience = 1
		self.global_update_threshold = np.inf


		self.load_data(trajectory,faces) # Load data into pyssdr 

		W,M,rmse_err = self.run_ssdr(num_clusters,"here.fbx")

		reconstruction_list = []
		weights_list = []
		rmse_list = []

		while len(self.keep_bones) > 1: 
			print(f"Keep Bones:{self.keep_bones}")
			# Clusters 
			weight_influence = np.sum(self.w,axis=1)
			weight_influence = weight_influence[self.keep_bones]
			
			least_influence_bone = np.argmin(weight_influence)
			print("Least influcnce bone:",least_influence_bone)
			self.keep_bones = np.delete(self.keep_bones,least_influence_bone)

			for i in range(10):
				self.computeWeights()
				self.computeTranformations()        
				if self.cbIterEnd():
					break

			deformed_vertices_ssdr = self.compute_reconstruction(list(range(N)))
			deformed_vertices = np.zeros((T,N,3))
			deformed_vertices[:,:,0] = deformed_vertices_ssdr[list(range(0,3*T,3))]
			deformed_vertices[:,:,1] = deformed_vertices_ssdr[list(range(1,3*T,3))]
			deformed_vertices[:,:,2] = deformed_vertices_ssdr[list(range(2,3*T,3))]

			reconstruction_list.append(deformed_vertices)
			weights_list.append(self.w.todense())
			rmse_list.append(self.rmse())

		reconstruction_list = np.array(reconstruction_list[::-1])    
		weights_list = np.array(weights_list[::-1])    
		rmse_list = np.array(rmse_list[::-1])    

		return reconstruction_list,weights_list,rmse_list

	def get_joint_position(self,bone_i,bone_j,joint_centroid,lambda_reg=1e-1):

		transforms_i = self.m[:,4*bone_i : 4*bone_i+4]
		transforms_j = self.m[:,4*bone_j: 4*bone_j+4]

		A = []
		B = []
		for f in range(self.nF):

			a = (transforms_i[4*f:4*f+3,:3] - transforms_j[4*f:4*f+3,:3]) + 0.5*lambda_reg*(transforms_i[4*f:4*f+3,:3] + transforms_j[4*f:4*f+3,:3])
			b = (transforms_j[4*f:4*f+3,3] - transforms_i[4*f:4*f+3,3]) + lambda_reg*( joint_centroid[f] - 0.5*(transforms_i[4*f:4*f+3,3] + transforms_j[4*f:4*f+3,3]) )
			A.append(a)
			B.append(b)

		A = np.concatenate(A,axis=0)
		B = np.concatenate(B,axis=0)


		U,S,Vt = np.linalg.svd(A,full_matrices=False)
		jp = Vt.T@np.linalg.inv(np.diag(S)) @ U.T @ B

		joint_motion = []
		cost = 0
		for f in range(self.nF):
			cost += np.linalg.norm((transforms_i[4*f:4*f+3,:3] - transforms_j[4*f:4*f+3,:3])@jp + transforms_i[4*f:4*f+3,3] - transforms_j[4*f:4*f+3,3])**2
			joint_position = 0.5*(transforms_i[4*f:4*f+3,:3] + transforms_j[4*f:4*f+3,:3])@jp + 0.5*(transforms_i[4*f:4*f+3,3] + transforms_j[4*f:4*f+3,3])
			joint_motion.append(joint_position)
		cost /= np.array((self.w[bone_i]@self.w[bone_i].T).todense()).reshape(1)

		print(jp,lambda_reg,joint_centroid[0],cost,joint_motion[0])

		return joint_motion,cost

	def get_graph(self,trajectory):
		bones = self.keep_bones.copy()
		num_bones = len(bones)

		adj_matrix = np.inf*np.ones((num_bones,num_bones))
		joint_positions = np.zeros((self.nF,num_bones,num_bones,3))

		labels = np.array(np.argmax(self.w,axis=0)).reshape(-1)
		uq_labels = np.unique(labels)
		for l in uq_labels:
			if l not in self.keep_bones:
				labels[labels==l] = -1

		if np.sum(labels==-1) > 0: # Some vertices might have labels after being pruned out
			kdtree = KDTree(trajectory[0,labels!=-1])
			_, nn = kdtree.query(trajectory[0,labels==-1], 1)  
			nn = nn.reshape(-1)
			labels[labels==-1] = labels[labels!=-1][nn]
		

		labels,old2new,new2old = clean_labels(labels)
		segment_boundaries = find_boundaries(trajectory[0,:,:2],labels)

		for bind in segment_boundaries: 
			i = bind // num_bones
			k = bind % num_bones
			joint_centroid = np.mean(trajectory[:,segment_boundaries[bind],:],axis=1)

			joint_positions[:,i,k],adj_matrix[i,k] = self.get_joint_position(new2old[i],new2old[k],joint_centroid)
			joint_positions[:,k,i] = joint_positions[:,i,k]
			adj_matrix[k,i] = adj_matrix[i,k]



		trajectory_center = np.mean(trajectory[0],axis=0)
		cluster_centers = []
		uq_clusters = np.unique(labels)
		for l in uq_clusters:
			cluster_ind = np.where(labels == l)[0]
			cluster_centers.append(np.mean(trajectory[0,cluster_ind],axis=0))
		cluster_centers = np.array(cluster_centers)

		root_ind = uq_clusters[np.argmin(np.linalg.norm(cluster_centers - trajectory_center[None],axis=1))]
		print(f"Rootind:",root_ind)

		joint_positions, parent_array = minimum_spanning_tree(root_ind,joint_positions,adj_matrix.copy(),trajectory,labels)

		return joint_positions,parent_array

	def __call__(self,opt):    
		data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
		trajectory_path = os.path.join(data_path,"trajectory")


		for file in sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1])):
			self.log.info(f"=====Loading trajectory:{file}========")

			source_frame_id = int(file.split('.')[0].split('_')[-1])
			trajectory,face_data = self.load_trajectory(data_path,source_frame_id)

			skinning_path = os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}")
			print(os.listdir(skinning_path))
			weights_path =  os.path.join(data_path,f"reconstructionSSDR_{source_frame_id}/weights_{source_frame_id}.npy")
			if not os.path.isfile(weights_path):
				os.makedirs(skinning_path,exist_ok=True)
	 
				reconstructions,weights,rmse_list =  self.get_different_cluster(trajectory,face_data,num_clusters=opt.clusters,minimum_label_size_percentage=opt.minimum_label_size_percentage)
	 
				self.log.info(f"Reconsturction Error:{rmse_list}")

				np.save(os.path.join(skinning_path,f"reconstruction_{source_frame_id}.npy"),reconstructions)
				np.save(os.path.join(skinning_path,f"weights_{source_frame_id}.npy"),weights)
				with open(os.path.join(skinning_path,'rmse.txt'),'w') as f: 
					f.write(','.join([str(r) for r in rmse_list]))
			else: 
				self.initialize_ssdr(trajectory,face_data,num_clusters=opt.clusters)

			skeleton_path =  os.path.join(data_path,f"skeleton/{source_frame_id}.skel")
			joint_positions,original_parent_array = read_skel(skeleton_path)
			# Num joints same as ours 
			num_joints = joint_positions.shape[0]


			weights = np.load(weights_path)[num_joints-1]


			self.keep_bones = np.where( np.sum(weights**2, axis=1) > 1e-1*np.max(np.sum(weights**2,axis=1)))[0]

			print(self.keep_bones)
			self.w = sparse.csc_matrix(weights)
			self.computeTranformations()

			print("RMSE:",self.rmse())


			# Compute Adj matrix 
			joint_positions, parent_array = self.get_graph(trajectory)

			# self.vis.plot_skeleton(trajectory,joint_positions,parent_array,weights.T,gr_skeleton=None, debug=True,framerate=5)

			# reconstruction = get_reconstruction(trajectory,Rot,Trans,labels)
			write_skel_file(joint_positions,parent_array,os.path.join(skinning_path,f"{source_frame_id}.skel"))
			np.save(os.path.join(skinning_path,f"skelMotion_{source_frame_id}.npy"),joint_positions)

def write_skel_file(skeleton,parent_array,datapath):
	N = skeleton.shape[1]
	with open(datapath,'w') as f:
		for i in range(N):
			f.write("{} {} {} {} {}\n".format(i,skeleton[0,i,0],skeleton[0,i,1],skeleton[0,i,2],int(parent_array[i])))

if __name__ == "__main__":
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
	args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
	args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')

	# Clusters 
	args.add_argument('--clusters', default=30, type=int, help='Number of clusters for skinning decomposition')
	args.add_argument('--minimum_label_size_percentage', required=False, type=float, default=1e-3,help="Minimum size of cluster to be valid")

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
	ssdr = SSDR(vis)
	ssdr(opt)

	# opt.vis = "plotly"
	# visulizer_plotly = get_visualizer(opt)
	# visulizer_plotly.compare_convergance_info({'optimization':['silh','depth'],'volumetric_analysis':['point_to_plane_mean']})

