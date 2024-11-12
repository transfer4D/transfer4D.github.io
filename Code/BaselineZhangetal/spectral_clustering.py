import os
import sys 
import argparse
import logging 
import numpy as np

from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.stats import mode

sys.path.append('../../')
sys.path.append('../')

from vis import get_visualizer
from numba import njit,prange


def union_find(psu,a):
	if psu[a] == -1:
		return a
	else:
		psu[a] = union_find(psu,psu[a])
		return psu[a]

def update_union_find(psu,a,val):
	if psu[a] != -1:
		update_union_find(psu,psu[a],val)
	
	psu[a] = val	
	return

def joint_similarity(joint_points,alpha=1000):
	# Krushkal's algorithm 
	T,N,D = joint_points.shape
	XX = np.reshape(joint_points, (T,1, N, D))
	YY = np.reshape(joint_points, (T,N, 1, D))
	XX = np.tile(XX, (1,N, 1, 1))
	YY = np.tile(YY, (1,1, N, 1))
	diff = XX-YY
	diff = np.multiply(diff, diff)
	diff = np.sqrt(np.sum(diff, 3))
	diff1 = np.mean(diff,axis=0)
	diff2 = np.var(diff,axis=0)
	diff = diff1 + alpha*diff2 
	diff[np.arange(N),np.arange(N)] = np.inf
	return diff

def minimum_spanning_tree(adj_matrix):
	N = adj_matrix.shape[0]
	psu = -1*np.ones(N,dtype=np.int32)
	skeleton_adj = np.zeros((N,N))
	while len(np.where(psu==-1)[0]) > 1:
		a,b = np.unravel_index(np.argmin(adj_matrix),adj_matrix.shape)
		adj_matrix[a,b] = np.inf
		adj_matrix[b,a] = np.inf

		if union_find(psu,a) != union_find(psu,b):
			update_union_find(psu,a,union_find(psu,b))
			skeleton_adj[a,b] = 1
			skeleton_adj[b,a] = 1
				
	return skeleton_adj


def spectral_clustering(movement,normals,samples=200,k=2,lmbda=1,alpha=0.7):
	
	if samples >= movement.shape[1]:
		ind = np.arange(movement.shape[1])
		samples = ind
	else:
		# ind = np.random.choice(movement.shape[1],samples,reduce=False)
		ind = np.random.permutation(movement.shape[1])[:samples]
	samples = ind.shape[0]
	# print(np.max(np.linalg.norm(movement[0:-1,:,:] - movement[1:,:,:],axis=2)))
	# dt = int(np.round(20*np.max(np.linalg.norm(movement[0:-1,:,:] - movement[1:,:,:],axis=2))))
	trajectory = movement[:,ind,:]
	T,N,D = trajectory.shape
	XX = np.reshape(trajectory, (T,1, N, D))
	YY = np.reshape(trajectory, (T,N, 1, D))
	XX = np.tile(XX, (1,N, 1, 1))
	YY = np.tile(YY, (1,1, N, 1))

	dx = XX-YY
	dx = np.linalg.norm(dx,axis=3,ord=2)

	dx = np.std(dx,axis=0)




	phi = np.exp(-lmbda*dx)
	phi[np.arange(samples),np.arange(samples)] = 0


	D = np.diag(np.sum(phi,axis=1))
	D_inv_half = np.diag(1/np.sqrt(np.sum(phi,axis=1)))

	L = D_inv_half.dot(D - phi).dot(D_inv_half)
	U,D,Vh = np.linalg.svd(L)

	return ind,D,Vh

def KMeans_partiton(D,Vh,partition):
	samples = D.shape[0]
	# allowed_vectors = np.where(D <= eigen_thresh)[0]
	allowed_vectors = np.arange(samples-partition,samples)
	k = allowed_vectors.shape[0]
	vk = Vh.T[:,allowed_vectors].reshape((samples,k))

	# Kmeans on vk	
	kmeans = KMeans(n_clusters=k).fit(vk)
	# print((L/(np.linalg.norm(L[:,allowed_vectors],axis=1)[:,None])).shape)
	labels = kmeans.predict(vk)

	return labels

# @njit(parallel=True)	
def bone_trajectory(trajectory,labels):
	"""
		Given the set of lable for the cluster
		Extract the bone movement of each cluster using methods similiar to linear blend skinning
	"""
	num_labels = np.unique(labels)	
	T,N,D = trajectory.shape

	reconstruction = np.zeros_like(trajectory)

	Rot_list = []
	Trans_list = []
	for label in num_labels:
		ind = np.where(labels == label)[0]
		cluster_trajectory = trajectory[:,ind,:]
		label_rot = []
		label_trans = []

		for t in range(T):
			N = len(ind)
			current_pose = np.concatenate([cluster_trajectory[t],np.ones((N,1))],axis=1).T
			init_pose = np.concatenate([cluster_trajectory[0],np.ones((N,1))],axis=1).T
			H = current_pose.dot(init_pose.T)
			H = H/H[3,3]
			A = H[:3,:3] - H[:3,3:4]@H[3:4,:3]
			U,S,Vh = np.linalg.svd(A)
			d = np.linalg.det(U@Vh)

			D = np.eye(3,3)
			D[2,2] = d
			R = U@D@Vh 

			Trans = H[:3,3:4] - R@H[3:4,:3].T

			Trans = Trans.reshape(-1)

			label_rot.append(R)
			label_trans.append(Trans)
			reconstruction[t,ind,:] = cluster_trajectory[0]@R.T + Trans


		Rot_list.append(label_rot)	
		Trans_list.append(label_trans)	

	Rot = np.array(Rot_list)
	Trans = np.array(Trans_list)

	return Rot,Trans,reconstruction		




def get_reconstruction(trajectory,Rot,Trans,labels,iterations=10):
	
	T,N,_ = trajectory.shape


	uq_labels = np.unique(labels)
	num_labels = len(uq_labels)

	reconstruction = np.zeros((T,N,3))

	for l in range(num_labels):
		ind = np.where(labels == uq_labels[l])[0]
		for t in range(T):
			init_mean = np.mean(trajectory[0,ind,:],axis=0,keepdims=True)
			reconstruction[t,ind] = (trajectory[0,ind,:]-init_mean).dot(Rot[l,t,:,:].T) + Trans[l,t,:]

	return reconstruction		

		

def find_boundaries(points,labels,k=12):
	# Calulate the leaf nodes and boundary points as joints of a skeleton
	points = points
	tree = KDTree(points)
	N = points.shape[0]
	num_labels = np.unique(labels).shape[0]
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


def find_connected_components(labels,closest_neighbours,min_cluster_percentage=0.01):
	'''
		One component could be seperted into different regions
	'''

	visited = -1*np.ones_like(labels)
	label_cnt = 0
	for i in range(len(labels)):
		if visited[i] == -1: 
			stack = [i]

			while len(stack) > 0:
				x = stack[0]
				stack = stack[1:]
				visited[x] = label_cnt  


				# print(x,closest_neighbours[x],labels[x],labels[closest_neighbours[x]],stack)
				for c in closest_neighbours[x]:
					if visited[c] == -1 and labels[c] == labels[x] and c not in stack:
						stack.append(c)

			label_cnt += 1


	# CLean labels 
	min_cluster_cnt = min_cluster_percentage*len(labels)
	uq_clusters,cluster_size = np.unique(visited,return_counts=True)
	small_clusters = np.where(cluster_size < min_cluster_cnt)[0]

	# print(f"Small Cluster Size:{cluster_size[small_clusters]}")

	for i in small_clusters:
		label_name = uq_clusters[i]
		stack = np.where(visited==label_name)[0]
		while len(stack) > 0:
			x = stack[0]
			stack = stack[1:]

			diff_cluster = np.where(visited[closest_neighbours[x]] != label_name)[0]
			if len(diff_cluster) > 0: 
				diff_cluster = closest_neighbours[x,diff_cluster[0]]

				visited[np.where(visited==label_name)[0]] = visited[diff_cluster]
				break

	print("Labels and cnt:",np.unique(visited,return_counts=True))		



	return visited, label_cnt



def load_trajectory(data_path,source_frame_id):

	trajectory_path = os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
	trajectory = np.load(trajectory_path)

	return trajectory


def load_luetal_gr(datadir):
	if not os.path.isfile(os.path.join(datadir,'lu_Coord.txt')):
		return None
	main_joints = np.loadtxt(os.path.join(datadir,'lu_Coord.txt'))
	leaf_joints = np.loadtxt(os.path.join(datadir,'lu_virtualJ.txt'))

	# joints = np.concatenate([main_joints,leaf_joints],axis=1)
	joints = main_joints


	skel_adj = np.loadtxt(os.path.join(datadir,'lu_Graph.txt'))

	return joints,skel_adj

def zhang_method(trajectory,labels):
	uq_labels = np.unique(labels)
	num_labels = uq_labels.shape[0]
	T,N,D = trajectory.shape

	adj_matrix = np.zeros((num_labels,num_labels))
	for i in range(num_labels):
		for j in range(num_labels):
			if j <= i:
				continue

			ind_i = np.where(labels==uq_labels[i])[0]
			ind_j = np.where(labels==uq_labels[j])[0]

			XX = np.reshape(trajectory[:,ind_i,:], (T,1, ind_i.shape[0], D))
			YY = np.reshape(trajectory[:,ind_j,:], (T,ind_j.shape[0],1, D))
			XX = np.tile(XX, (1,ind_j.shape[0], 1, 1))
			YY = np.tile(YY, (1,1, ind_i.shape[0], 1))



			diff = np.linalg.norm(XX-YY,ord=2,axis=3)
			# print(diff[0,:,:])
			adj_matrix[i,j] = np.max(np.min(diff,axis=(1,2)))
			adj_matrix[j,i] = adj_matrix[i,j]
	return adj_matrix


def cluster_centroid(trajectory,labels):	

	num_label = np.unique(labels)
	# For every label 
	T,N,D = trajectory.shape
	points = np.zeros((T,0,D))
	for label in num_label:
		ind = np.where(labels == label)[0]

		joint_point = np.mean(trajectory[0,ind,:],axis=0)
		centroid_trajetory = trajectory[:,ind,:]
		centroid_trajetory = np.mean(centroid_trajetory - centroid_trajetory[0,:,:],axis=1)

		joint_trajectory = np.tile(joint_point,(T,1))
		joint_trajectory = joint_trajectory + centroid_trajetory
		joint_trajectory = joint_trajectory.reshape(T,1,D)
		points = np.concatenate([points,joint_trajectory],axis=1)

	return points

def rearrange_skeleton(joint_motion,skel_adj):
		"""
			2. Connect disconnected components 
			1. Closest functional node to the center becomones the root node
			3. Run DFS and get Parent array 
			4. rearrange joints
		"""    

		# skeleton_component = get_connected_components_from_adj_matrix(skel_adj)
		# components = np.unique(skeleton_component)

		# assert len(components) == 1 , "More than 1 component. Can't find parent array"


		# Get root (>2 degree node closest to the mean of the surface)
		skeleton_centroid = np.mean(joint_motion[0],axis=0,keepdims=True)

		# Cloest non-funcational node
		# joint_degree = np.sum(skel_adj, axis=0)
		# functional_node_indices = np.where(joint_degree > 2)[0]

		root_ind = np.argmin(np.linalg.norm(joint_motion[0] - skeleton_centroid,axis=1))

		# DFS
		# Rearrange points for Pinnochio using dfs 
		new_indices = []
		old2new_indices = {}
		parent_array = []
		stack = [[root_ind,root_ind]]

		while len(stack) > 0:
			x,p = stack[-1]
			stack.pop()

			old2new_indices[x] = len(new_indices)
			new_indices.append(x)
			parent_array.append(old2new_indices[p])

			for c in np.where(skel_adj[x,:] > 0)[0]:
				if c not in new_indices:
					stack.append([c,x])
		

		new_indices = np.array(new_indices)
		joint_motion = joint_motion[:,new_indices,:]
		# labels = new_indices[labels]
		return joint_motion,parent_array


def write_skel_file(skeleton,parent_array,datapath):	
	N = skeleton.shape[1]
	with open(datapath,'w') as f:
		for i in range(N):
			f.write("{} {} {} {} {}\n".format(i,skeleton[0,i,0],skeleton[0,i,1],skeleton[0,i,2],int(parent_array[i])))


# Get numpy vertex arrays from selected objects. Rest pose is most recently selected.
if __name__ == "__main__":
	args = argparse.ArgumentParser() 
	# Path locations
	args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
	args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
	args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')

	# Clusters 
	args.add_argument('--clusters', default=30, type=int, help='Maximum number of clusters for skinning decomposition')

	# Arguments for debugging  
	args.add_argument('--debug', default=True, type=bool, help='Whether debbugging or not. True: Logging + Visualization, False: Only Critical information and plots')
	args.add_argument('--vis', default='polyscope', type=str, help='Visualizer to plot results')

	opt = args.parse_args()

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

	gr_skeleton = load_luetal_gr(opt.datadir)

	data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
	trajectory_path = os.path.join(data_path,"trajectory")

	log = logging.getLogger(__name__)


	for file in sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1])):
		log.info(f"=====Loading trajectory:{file}========")
		source_frame_id = int(file.split('.')[0].split('_')[-1])
		trajectory = load_trajectory(data_path,source_frame_id)

		reconstruction_path = os.path.join(data_path, f"reconstructionZhang_{source_frame_id}")
		os.makedirs(reconstruction_path,exist_ok=True)
		rmse_list = []

		ind,D,Vh = spectral_clustering(trajectory,None,samples=2000)
		for partition in range(1,opt.clusters):
			labels = KMeans_partiton(D,Vh,partition)

			kdtree = KDTree(trajectory[0,ind,0:2])
			_,closest_neighbours = kdtree.query(trajectory[0,:,0:2],k=10)
			neighbours_labels = labels[closest_neighbours]
			labels = mode(neighbours_labels,axis=1)[0][:,0]

			Rot, Trans,reconstruction = bone_trajectory(trajectory,labels)


			kdtree = KDTree(trajectory[0,:,:2])
			_,closest_neighbours = kdtree.query(trajectory[0,:,:2],k=10)
			labels,component_cnt = find_connected_components(labels,closest_neighbours[:,1:]) # Exclude same element from closest_neighbours

			skeleton = cluster_centroid(trajectory,labels)
			
			adj_matrix = zhang_method(trajectory[:,ind],labels[ind])
			skeleton_adj = minimum_spanning_tree(adj_matrix)



			skeleton_motion,parent_array = rearrange_skeleton(skeleton,skeleton_adj)

			rmse = np.sqrt(np.mean(np.linalg.norm(trajectory - reconstruction,axis=2)**2))

			rmse_list.append(rmse)



			# reconstruction = get_reconstruction(trajectory,Rot,Trans,labels)
			write_skel_file(skeleton,parent_array,os.path.join(reconstruction_path,f"{partition}.skel"))
			np.save(os.path.join(reconstruction_path,f"label_{partition}.npy"),labels)
			np.save(os.path.join(reconstruction_path,f"skelMotion_{partition}.npy"),skeleton_motion)
			np.save(os.path.join(reconstruction_path,f"reconstruction_{partition}.npy"),reconstruction)


			# if partition > 4:
				# vis.plot_skinning(trajectory.copy(),np.zeros((0,3)),weights_old.T,gr_skeleton=gr_skeleton)
				# vis.plot_skinning(trajectory.copy(),np.zeros((0,3)),weights.T,gr_skeleton=gr_skeleton)
				# vis.plot_skeleton(reconstruction,skeleton_motion,parent_array,weights,gr_skeleton=None, debug=True,framerate=5)

		with open(os.path.join(reconstruction_path,'rmse.txt'),'w') as f: 
			f.write(",".join([str(x) for x in rmse_list]))



