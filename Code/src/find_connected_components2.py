import os
import sys
import numpy as np 
import open3d as o3d
import scipy.io as sio
from sklearn.neighbors import KDTree
from estimate_joint_location import FindJointLocation
import heapq



# Create a priority queue based on the distance of edges
class Node:
	def __init__(self,i,position,neighbours):
		self.i = i
		self.p = position
		self.n = list(neighbours)
		self.removed = False
	def __repr__(self):
		return f'Node {self.i} neighbours: {self.n} removed:{self.removed}'

class Edge:
	def __init__(self, i,j,dist):
		self.i = i
		self.j = j
		self.d = dist

	def __repr__(self):
		return f'Edge {self.i}-{self.j} dist: {self.d}'

	def __lt__(self, other):
		return self.d < other.d






def clean_labels(labels):
	unique_labels,label_count = np.unique(labels,return_counts=True)
	new_labels = -1*np.ones_like(labels)

	for i,x in enumerate(unique_labels):
		new_labels[labels==x] = i 

	assert np.sum(new_labels==-1) == 0, "Some labels didn't update"

	return new_labels,label_count


def find_boundaries_from_point_cloud(points,labels,k,num_labels,consider_3d=False):
	"""
		Calulate the boundary points of the of the clustering.

		@param: points => 3D point set
		@param:	labels => cluster label for each point
		@param: k => number of neighbours to consider 
	"""
	if not consider_3d:
		points = points[:,0:2]
	tree = KDTree(points)
	N = points.shape[0]
	# num_labels = np.unique(labels).shape[0]
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
			a = x//num_labels
			b = x%num_labels
			y = b*num_labels + a

			if y in segment_boundaries:
				already_computed.append(y)
				segment_boundaries[x].extend(segment_boundaries[y])
			segment_boundaries[x] = np.unique(segment_boundaries[x])		

	# print(sorted(segment_boundaries.keys()))
	print(sorted(already_computed))

	# Remove Duplicate neighbours
	for y in already_computed:
		segment_boundaries.pop(y)

	return segment_boundaries


def find_overlapping_components(segment_boundaries,components,joints):
	"""
		Goal is to check if 2 components are overlapped by a third component

	"""

	# For each joint check is the segments connects others components 

		# If overlapping mark those components as and share the overlapping component

	return {}	


def anime_read( filename):
	"""
	filename: path of .anime file
	return:
		nf: number of frames in the animation
		nv: number of vertices in the mesh (mesh topology fixed through frames)
		nt: number of triangle face in the mesh
		vert_data: vertice data of the 1st frame (3D positions in x-y-z-order)
		face_data: riangle face data of the 1st frame
		offset_data: 3D offset data from the 2nd to the last frame
	"""
	f = open(filename, 'rb')
	nf = np.fromfile(f, dtype=np.int32, count=1)[0]
	nv = np.fromfile(f, dtype=np.int32, count=1)[0]
	nt = np.fromfile(f, dtype=np.int32, count=1)[0]
	vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
	face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
	offset_data = np.fromfile(f, dtype=np.float32, count=-1)
	'''check data consistency'''
	if len(offset_data) != (nf - 1) * nv * 3:
		raise ("data inconsistent error!", filename)
	vert_data = vert_data.reshape((-1, 3))
	face_data = face_data.reshape((-1, 3))
	offset_data = offset_data.reshape((nf - 1, nv, 3))
	return nf, nv, nt, vert_data, face_data, offset_data

def find_components(verts,faces):
	cluster_list = []
	visited = -1*np.ones(verts.shape[0],dtype=np.int32)

	adj_list = {}
	for face in faces:

		for f in face: 
			if f not in adj_list:
				adj_list[f] = []
		
		adj_list[face[0]].append(face[1])
		adj_list[face[0]].append(face[2])
		adj_list[face[1]].append(face[0])
		adj_list[face[1]].append(face[2])
		adj_list[face[2]].append(face[0])
		adj_list[face[2]].append(face[1])

	for x in adj_list: 
		adj_list[x] = np.unique(adj_list[x])	

	c_cnt = 0
	for v in adj_list:
		if visited[v] == -1:
			stack = [v]
			visited[v] = c_cnt 
			while len(stack) > 0:
				x = stack[0]
				stack = stack[1:]
				for c in adj_list[x]:
					if visited[c] == -1:
						visited[c] = c_cnt
						stack.append(c)
			# print(f"Cnt:{c_cnt} Verts:{np.where(visited == c_cnt)}")
			c_cnt += 1	

	# print(visited)
	return visited,c_cnt

def save_largest_component(components,verts, faces,savepath):
	print(verts.shape,faces.shape)
	diff_components = np.unique(components)
	# print("Components Size:",[ np.sum(components==x) for x in diff_components])
	largest_component = np.argmax([ np.sum(components==x) for x in diff_components])
	sub_vert_indices = np.where(components==largest_component)[0]
	inverse_indices_dict = dict([(x,i) for i,x in enumerate(sub_vert_indices)])
	# print(inverse_indices_dict)
	sub_vertices = verts[sub_vert_indices,:]
	sub_faces = []
	for face in faces:
		if sum([f in sub_vert_indices for f in face]) == len(face):
			sub_faces.append([ inverse_indices_dict[f] for f in face])
	sub_faces = np.array(sub_faces)
	# print(sub_faces[sub_faces > len(sub_vert_indices)])		
	# mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(sub_vertices),triangles=o3d.utility.Vector3iVector(sub_faces))	
	# o3d.io.write_triangle_mesh(savepath,mesh)
	# o3d.visualization.draw_geometries([mesh])

	return sub_vertices,sub_faces

def get_connected_components_from_adj_matrix(skel_adj):
	J = skel_adj.shape[0]
	visisted = [-1]*J


	component_cnt = 0
	for j in range(J):
		# print(j,visisted[j])
		if visisted[j] == -1: 
			visisted[j] = component_cnt
			stack = [j]
			while len(stack) > 0:	
				x = stack[0]
				stack = stack[1:]
				edges = np.where(skel_adj[x,:])[0]
				
				# print(x,visisted[x],edges)

				for c in edges:
					if visisted[c] == -1: 
						visisted[c] = component_cnt
						stack.append(c)


			component_cnt += 1
	print("Components:",visisted)
	return np.array(visisted)			


def closest_point_projection(vertices,vertex_normals,joint_position_orig):


	N = vertices.shape[0];

	
	Lix2 = vertex_normals[:,0]**2;
	Liy2 = vertex_normals[:,1]**2;
	Liz2 = vertex_normals[:,2]**2;
	
	M = np.zeros((3,3));
	B = np.zeros((3,1));
	
	# Pox equations
	M[0,0] += np.sum( Liy2 + Liz2 );
	M[0,1] -= np.sum(vertex_normals[:,0]*vertex_normals[:,1]);
	M[0,2] -= np.sum(vertex_normals[:,0]*vertex_normals[:,2]);
	B[0]   = np.sum( vertices[:,0]*(Liy2 + Liz2) ) -np.sum(vertex_normals[:,0]*vertex_normals[:,1]*vertices[:,1]) -np.sum(vertex_normals[:,0]*vertex_normals[:,2]*vertices[:,2]);

	# Poy equations
	M[1,0] -= np.sum( vertex_normals[:,1]*vertex_normals[:,0] );
	M[1,1] += np.sum( Lix2 + Liz2 );
	M[1,2] -= np.sum( vertex_normals[:,1]*vertex_normals[:,2] );
	B[1]   = np.sum( vertices[:,1]*(Lix2 + Liz2) ) -np.sum(vertex_normals[:,1]*vertex_normals[:,0]*vertices[:,0]) -np.sum(vertex_normals[:,1]*vertex_normals[:,2]*vertices[:,2]);
	# Poz equations
	M[2,0] -= np.sum( vertex_normals[:,2]*vertex_normals[:,0] );
	M[2,1] -= np.sum( vertex_normals[:,2]*vertex_normals[:,1] );
	M[2,2] += np.sum( Lix2 + Liy2 );
	B[2]   = np.sum( vertices[:,2]*(Lix2 + Liy2) ) -np.sum(vertex_normals[:,2]*vertex_normals[:,0]*vertices[:,0]) -np.sum(vertex_normals[:,2]*vertex_normals[:,1]*vertices[:,1]);
	
	# % compute solution
	if np.abs(np.linalg.det(M)) < 1e-10:
		P0 = joint_position_orig;
	else:
		P0 = np.linalg.inv(M)@B;
	
	return P0.reshape(-1)


def remove_cycles(joints,skel_adj,components,labels):
	J = joints.shape[0]

	joint_degree = np.sum(skel_adj, axis=0)
	print("New joint Degrees:", joint_degree)

	def detect_cycle(node,parent,visited):
		children = np.where(skel_adj[node,:] > 0)[0]
		
		visited.append(node)
		# print("Detecting cycle:",node,parent,visited)
		for c in children:
			if c!=parent and c in visited:
				return [node],c

			elif c!=parent and c not in visited:
				r = detect_cycle(c,node,visited)
				# print("Current Detected Cycle:",r)
				if r == None:
					continue
				else:
					r[0].append(node)
					return r
			else:
				continue
	
	# Remove cyles 
	cycle_present = True 
	while cycle_present: 
		cycle_present = False 
		for j in range(J):
			if joint_degree[j] < 2: continue

			# Detect cycle 
			r = detect_cycle(j,-1,[])
			if r is None:
				continue

			cycle_path, common_node = r

			cycle_path_filtered = [common_node]
			for x in cycle_path:
				if x != common_node:
					cycle_path_filtered.append(x)
				else: 
					break

			cycle_path = cycle_path_filtered		
			# print(f"Detected cycle:",cycle_path,common_node)
			# Merge cycle into a single node 
			joint_position = np.mean(joints[cycle_path],axis=0,keepdims=True)

			# print("New joint position:",joint_position)
			joints = np.concatenate([joints,joint_position],axis=0)

			components = np.append(components,[components[cycle_path[0]]],axis=0)


			J = joints.shape[0]

			new_skel_adj = np.zeros((J,J),dtype=bool)
			new_skel_adj[:-1,:-1] = skel_adj
			skel_adj = new_skel_adj
			# print(np.where(new_skel_adj[J-1,:])[0])

			# Connect all cycle path nodes to new node 
			for c in cycle_path: 
				# print(f"Removing joint:{c}")
				edges = np.where(skel_adj[c,:])[0]

				skel_adj[J-1,edges] = True
				skel_adj[edges,J-1] = True

				# Disconnect all edges from the cycle node
				skel_adj[c,:] = False
				skel_adj[:,c] = False				
				skel_adj[J-1,J-1] = False # Might end up reconnecting to old joint
			
				labels[labels == c] = J-1 
				# print(np.where(skel_adj[J-1,:])[0])

			joint_degree = np.sum(skel_adj,axis=0)			
			# print(np.where(skel_adj[J-1,:])[0])

	return joints,skel_adj,components,labels		

def remove_dangling_nodes(vertices,labels,joints,skel_adj,components,dangling_path_cnt=3,dangling_path_min_corresp_prercentage=5e-3):

	J = joints.shape[0]

	# Get degree of each joint node
	joint_degree = np.sum(skel_adj,axis=0)
	# print("Joint Degree:",joint_degree)
	# Remove dangling nodes (1-degree node connected to 3-degree node or having less number of intermediary 2 degree nodes)
	N = vertices.shape[0] 
	print("Num vertices:",N)
	while True: 
		print("Removing Dangling nodes")
		dangling_path_dict = {}
		dangling_path_corresp_cnt_dict = {}
		for j,deg in enumerate(joint_degree):
			if deg == 1:
				dangling_path = [j]
				c = np.where(skel_adj[j,:])[0] 
				c = c[0] # Single degree node so only one edge
				while True: 
					dangling_path.append(c)
					if joint_degree[c] != 2:
						break 

					# If encountered 2 degree node
					edges = np.where(skel_adj[c,:])[0] # Single degree nd
					edges = [ e for e in edges if e not in dangling_path]
					assert len(edges) == 1, f"Found more than 1 node not in path:{dangling_path}, edges:{edges}"	
					c = edges[0]
						


				if len(dangling_path) > dangling_path_cnt: 
					dangling_path = dangling_path[:dangling_path_cnt]
					# continue
				dangling_path_corresp_cnt = sum([np.sum(labels==d) for i,d in enumerate(dangling_path) if i < dangling_path_cnt]) 	
				print(f"From node:{j} deg:{deg} path:{dangling_path} parent degree:{joint_degree[dangling_path[-1]]} Cnt: {dangling_path_corresp_cnt} Percentage:{dangling_path_corresp_cnt/N}")
				if joint_degree[dangling_path[-1]] > 2:
					# else:
					dangling_path_corresp_cnt_dict[j] = dangling_path_corresp_cnt	
					dangling_path_dict[j] = dangling_path	
		
		if len(dangling_path_dict) == 0: 
			break 

		for j in dangling_path_dict:				
			dangling_path = dangling_path_dict[j]
			if joint_degree[dangling_path[-1]] > 2 or dangling_path_corresp_cnt_dict[j]/N < dangling_path_min_corresp_prercentage: # End of dangling_path is a funcational node
				parent = dangling_path[-1]
				for p in dangling_path[:-1]:
					skel_adj[p,:] = False 
					skel_adj[:,p] = False 
					joint_degree[p] = 0

					# Check if boundary of p and parent are connected. Else merge to cloest neighboring node of parent
					# if p*J + parent in segmented_boundaries or parent*J + p in segmented_boundaries:
					labels[labels==p] = parent
					# else: 
					# labels[labels==p] = -1


				joint_degree[parent] -= 1


	print("Old Joint Degree:",joint_degree)				

	joint_degree = np.sum(skel_adj, axis=0)
	print("New joint Degrees:", joint_degree)


	# Remove degree 0 joints 
	new_joints = []
	new_components = []
	new2old_joint_ind = []
	new_labels = -1*np.ones_like(labels)
	for i in range(J):
		if joint_degree[i] > 0:

			vertex_inds = np.where(labels==i)[0]

			new_labels[vertex_inds] = len(new_joints) 
			new2old_joint_ind.append(i)

			new_joints.append(joints[i])
			new_components.append(components[i])


	joints = np.array(new_joints)
	labels = new_labels
	components = new_components


	if np.sum(labels==-1) > 0: # Some vertices might be unattahed after removing dangling nodes
		kdtree = KDTree(vertices[labels!=-1])
		_, nn = kdtree.query(vertices[labels==-1], 1)  # Find closest vertex used in ROSA
		nn = nn.reshape(-1)
		labels[labels==-1] = labels[labels!=-1][nn]


	J = joints.shape[0]
	new_skel_adj = np.zeros((J,J))
	for i in range(J):
		for j in range(J):
			new_skel_adj[i,j] = skel_adj[new2old_joint_ind[i],new2old_joint_ind[j]]
	skel_adj = new_skel_adj		


	joint_degree = np.sum(skel_adj,axis=0)		

	# print("New joints:",joints.shape)
	# print("Joint degree:",joint_degree)


	return joints,skel_adj,labels,components



def create_single_connected_component(joints,skel_adj,components,trajectory,projected_depth,labels):

	J = joints.shape[0]
	joint_degree = np.sum(skel_adj,axis=0)

	# Get connected components from skeleton adj matrix
	adj_components = get_connected_components_from_adj_matrix(skel_adj)

	# Connect disjoint components sharing boundary in 2D space
	segmented_boundaries = find_boundaries_from_point_cloud(projected_depth, labels, 25,J, consider_3d=False)
	overlapping_components = {}

	# Connecting disjoint segments
	# Sort keys by the number of boundary points
	sorted_keys = sorted(list(segmented_boundaries.keys()), key=lambda x: len(segmented_boundaries[x]), reverse=True)

	# # Connent skeleton disjoint but extracted from same component
	for x in segmented_boundaries:
		j1 = x // J
		j2 = x % J
		if skel_adj[j1, j2]:
			continue

		if components[j1] == components[j2] and adj_components[j1] != adj_components[j2]:
			skel_adj[j1, j2] = True
			skel_adj[j2, j1] = True
			print(f"Connecting: j1:{j1},{j2} adj_component:{adj_components[j1]},{adj_components[j2]} main component:{components[j1]} {components[j2]}")

			adj_components[adj_components == adj_components[j2]] = adj_components[j1]


	assert len(np.unique(adj_components)) == len(np.unique(components))

	joint_motion = np.tile(joints.reshape(1,-1,3),(trajectory.shape[0],1,1))
	for j in range(J):
		inds = np.where(labels == j)[0]
		if len(inds) == 0: continue
		joint_motion[:,j] += np.mean(trajectory[:,inds,:] - trajectory[0:1,inds,:],axis=1)

	T, _, D = trajectory.shape

	XX = np.reshape(joint_motion, (T, J, 1, D))
	YY = np.reshape(joint_motion, (T, 1, J, D))
	XX = np.tile(XX, (1, 1, J, 1))
	YY = np.tile(YY, (1, J, 1, 1))

	bone_variations = np.linalg.norm(XX - YY, ord=2, axis=3)
	bone_variations = np.max(bone_variations, axis=0) /(1e-6 + np.min(bone_variations, axis=0))



	uq_components = np.unique(adj_components)
	num_components = len(uq_components)


	adj_matrix = np.inf*np.ones((num_components,num_components))
	for comp1 in range(num_components):
		for comp2 in range(comp1+1,num_components):
			j1 = np.where(adj_components == uq_components[comp1])[0]
			j2 = np.where(adj_components == uq_components[comp2])[0]

			ind1 = []
			for j in j1:
				ind1.extend(list(np.where(labels == j)[0]))

			ind2 = []
			for j in j2:
				ind2.extend(list(np.where(labels == j)[0]))

			ind1 = np.random.permutation(ind1)[:1000]
			ind2 = np.random.permutation(ind2)[:1000]

			XX = np.reshape(trajectory[:,ind1,:], (T, 1, len(ind1), D))
			YY = np.reshape(trajectory[:,ind2,:], (T, len(ind2), 1, D))
			XX = np.tile(XX, (1, len(ind2), 1, 1))
			YY = np.tile(YY, (1, 1, len(ind1), 1))

			diff = np.linalg.norm(XX - YY, ord=2, axis=3)
			diff = np.max(np.min(diff,axis=(1,2)))
			adj_matrix[comp1,comp2] = diff
			adj_matrix[comp2,comp1] = diff


	while len(np.unique(adj_components)) > 1:
		print(adj_components)
		a,b = np.unravel_index(np.argmin(adj_matrix),adj_matrix.shape)
		adj_matrix[a,b] = np.inf
		adj_matrix[b, a] = np.inf
		if uq_components[a] != uq_components[b]:

			j1_list = np.where(adj_components == uq_components[a])[0]
			j2_list = np.where(adj_components == uq_components[b])[0]

			bone_variation = 100
			for x in sorted_keys:
				j1 = x // J
				j2 = x % J
				# print("J1 & J2:",j1,j2)
				if j1 in j1_list and j2 in j2_list:
					bone_variation = bone_variations[j1,j2]
					break
				elif j2 in j1_list and j1 in j2_list:
					j1,j2 = j2,j1
					bone_variation = bone_variations[j1,j2]
					break

			print(f"Bone length variation:{bone_variation}")

			if bone_variation >= 4.0: # For bear betweem 1.23 and 1.83
				if bone_variation == 100:
					j1 = j1_list[0]
					j2 = j2_list[0]
				heap = [Edge(j1, j2, bone_variations[j1, j2])]
				heapq.heapify(heap)
				visited = []
				while True:
					e = heapq.heappop(heap)
					print(f"Checking Edge:{e}")
					if e.d < 4.0:
						break

					visited.append(e.i*J+e.j)	
					visited.append(e.j*J+e.i)	

					nn1 = np.where(skel_adj[e.i])[0]
					nn2 = np.where(skel_adj[e.j])[0]

					for n1 in nn1: 
						if n1*J + e.j not in visited:
							heapq.heappush(heap,Edge(n1,e.j,bone_variations[n1,e.j]))

					for n2 in nn2: 
						if e.i*J + n2 not in visited:
							heapq.heappush(heap,Edge(e.i,n2,bone_variations[e.i,n2]))


				j1 = e.i 
				j2 = e.j			


				print(f"Updating joints:{j1} {j2} since bone length variation was too high. New variation:{bone_variation}  ")
				# component_bone_lengths = bone_variations[np.ix_(j1_list,j2_list)]
				# while True:

				# j1,j2 = np.unravel_index(np.argmin(component_bone_lengths),component_bone_lengths.shape)
				# j1 = j1_list[j1]
				# j2 = j2_list[j2]
				bone_variation = bone_variations[j1,j2]


				print(f"Updating joints since bone length variation was too high. New variation:{bone_variation} ")

			print("J1 & J2:",j1,j2,adj_components[j1],adj_components[j2])

			skel_adj[j1, j2] = True
			skel_adj[j2, j1] = True
			adj_components[adj_components == adj_components[j2]] = adj_components[j1]
			uq_components[uq_components == uq_components[b]] = uq_components[a]

	return skel_adj


def merge_close_nodes(joints,skel_adj,labels,min_dist=0.05):


	J = joints.shape[0]
	heap = []		
	# Edge
	for i in range(J):
		for j in range(i+1,J):
			if skel_adj[i,j]:        
				dist = np.linalg.norm(joints[i] - joints[j])
				heap.append(Edge(i,j,dist))
	joints_dict = dict([ (i,Node(i,joints[i],np.where(skel_adj[i] > 0)[0])) for i in range(J)])

	heapq.heapify(heap)

	while True:
		e = heapq.heappop(heap)
		# print(f"Removing:{e}")
		if joints_dict[e.i].removed or joints_dict[e.i].removed:
			continue
		if e.d > min_dist:
			break

		# Update old nodes
		joints_dict[e.i].removed = True
		joints_dict[e.j].removed = True

		new_position = (joints_dict[e.i].p + joints_dict[e.j].p)/2
		new_neighbours = [n for j in [joints_dict[e.i],joints_dict[e.j]] for n in j.n if not joints_dict[n].removed  ]

		ind = len(joints_dict)
		# print(ind)
		joints_dict[ind] = Node(ind, new_position, new_neighbours)
		labels[labels == e.i] = ind
		labels[labels == e.j] = ind

		for n in new_neighbours:
			edge_dist = np.linalg.norm(new_position - joints_dict[n].p)
			heapq.heappush(heap,Edge(ind,n,edge_dist))
			joints_dict[n].n.append(ind)

	# Create joints and skeleton dict again
	old2new_mapping = {}
	new_joints = []
	new_labels = -1*np.ones_like(labels)
	for j in joints_dict:
		if joints_dict[j].removed:
			continue
		new_labels[labels == j] = len(new_joints)
		old2new_mapping[j] = len(new_joints)
		new_joints.append(joints_dict[j].p)

	new_joints = np.array(new_joints)
	J = len(new_joints)
	new_skel_adj = np.zeros((J,J),dtype=bool)
	for j in joints_dict:
		if joints_dict[j].removed:
			continue
		for n in joints_dict[j].n:
			if joints_dict[n].removed: continue
			new_skel_adj[old2new_mapping[j],old2new_mapping[n]] = True
			new_skel_adj[old2new_mapping[n],old2new_mapping[j]] = True

	assert np.sum(new_labels==-1) == 0 and np.sum(new_labels>=J) == 0

	return np.array(new_joints),new_skel_adj,new_labels

def find_coarse_labels_from_gel(trajectory,vertex_normals, sampled_vertex, corresp, skeleton, projected_depth,components):
	"""
		@params:
			vertices: Vertices for which we want to calculate coarse labels
			sampled_vertex: points used for ROSA
			corresp: correspondence between sampled_vertex and rosa_joints
			skeleton: <joints, edges>: Skeleton output from ROSA
			projected_depth: Projected depth image to pixel space
			components: component used to create skeleton segment
	"""

	vertices = trajectory[0]


	joints, edges,skel_adj = skeleton
	J = joints.shape[0]

	# Extrapolate labels from samples vertices used in pygel
	valid_sampled_verts = np.where(corresp < J)[0]  # Non-corresp vertices are marked as max(corresp) + 1
	kdtree = KDTree(sampled_vertex[valid_sampled_verts])
	_, nn = kdtree.query(vertices, 1)  # Find closest vertex used in ROSA
	nn = nn.reshape(-1)
	labels = corresp[valid_sampled_verts][nn]

	joints,skel_adj,components,labels = remove_cycles(joints,skel_adj,components,labels)

	joints,skel_adj,labels,components = remove_dangling_nodes(vertices,labels,joints,skel_adj,components)		

	# import matplotlib.pyplot as plt
	# J = joints.shape[0]
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
	# for j1 in range(J):
	# 	ax.text(joints[j1, 0], joints[j1, 1], joints[j1, 2], str(j1))
	# 	for j2 in range(j1 + 1, J):
	# 		if skel_adj[j1, j2]:
	# 			ax.plot([joints[j1, 0], joints[j2, 0]], [joints[j1, 1], joints[j2, 1]], [joints[j1, 2], joints[j2, 2]])

	# plt.show()

	# Get joint position using ROSA
	joint_location_finder = FindJointLocation()
	joints = joint_location_finder.run(vertices,vertex_normals,labels,joints,skel_adj)

	skel_adj = create_single_connected_component(joints,skel_adj,components,trajectory,projected_depth,labels)

	# joints,skel_adj,labels,components = remove_dangling_nodes(vertices,labels,joints,skel_adj,components)

	# Due to smoothing 1 degree nodes could get merged really close to one another hence merge
	joints,skel_adj,labels = merge_close_nodes(joints,skel_adj,labels)



	J = joints.shape[0]
	# Get connected components from skeleton adj matrix 
	adj_components = get_connected_components_from_adj_matrix(skel_adj)
	print("Adjcanet components:",adj_components)


	joint_degree = np.sum(skel_adj, axis=0)



	# Disperse 3D labels close to its nearest 2 degree labels 
	while np.sum(joint_degree[np.unique(labels)] > 2):
		for j in range(J):
			if joint_degree[j] > 2:
				junction_verts = np.where(labels == j)[0]

				if len(junction_verts) == 0:
					continue

				labels[junction_verts] = -1

		junction_verts = np.where(labels == -1)[0]				
		non_junction_verts = np.where(labels != -1)[0]

		print(junction_verts,non_junction_verts)

		kdtree = KDTree(vertices[non_junction_verts])
		_, nn = kdtree.query(vertices[junction_verts], 1)  # Find closest vertex non junction node
		nn = nn.reshape(-1)
		nn = non_junction_verts[nn]

		print(junction_verts,nn)
		print(np.unique(labels[junction_verts]),np.unique(labels[nn]))

		labels[junction_verts] = labels[nn]





	# Cloest non-funcational node
	non_functional_node_indices = np.where(joint_degree != 2)[0]

	# Merge labels of all 2 degree nodes
	degree2_paths = {}
	for i1, j1 in enumerate(non_functional_node_indices):
		for i2, j2 in enumerate(non_functional_node_indices[i1 + 1:]):
			ind = j1 * J + j2
			degree2_paths[ind] = []
			# print(f"Checking: {j1, j2}")
			# Run dfs from source(j1) to target(j2)
			children = np.where(skel_adj[j1, :])[0]
			for c in children:
				if joint_degree[c] != 2: continue

				visited = set()
				path = [c]
				stack = [c]
				# print(f"For node:{c}, deg:{joint_degree[c]}")
				while len(stack) > 0:
					x = stack[-1]
					stack = stack[:-1]
					if x in visited:
						continue
					visited.add(x)
					path_children = np.where(skel_adj[x, :])[0]
					if j2 in path_children:
						# print("Path:", path)
						degree2_paths[ind].extend(path)
						break
					for cc in path_children:
						if cc not in visited and joint_degree[cc] == 2:
							path.append(cc)
							stack.append(cc)


	# Only keep the possible paths 
	degree2_paths_copy = {}
	for ind in degree2_paths:
		if len(degree2_paths[ind]) != 0:
			degree2_paths_copy[ind] = degree2_paths[ind]
	degree2_paths = degree2_paths_copy




	labels_coarse = labels.copy()

	# Merge Labels
	for ind in degree2_paths:
		j1 = ind // J
		j2 = ind % J
		# print(j1, j2, degree2_paths[ind])
		base_joint = degree2_paths[ind][0]
		for x in degree2_paths[ind][1:]:
			labels_coarse[labels == x] = base_joint

		if joint_degree[j1] == 1: 
			labels_coarse[labels == j1] = base_joint
		if joint_degree[j2] == 1: 
			labels_coarse[labels == j2] = base_joint			







	# import matplotlib.pyplot as plt
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
	# for j1 in range(J):
	# 	ax.text(joints[j1, 0], joints[j1, 1], joints[j1, 2], str(j1))
	# 	for j2 in range(j1 + 1, J):
	# 		if skel_adj[j1, j2]:
	# 			ax.plot([joints[j1, 0], joints[j2, 0]], [joints[j1, 1], joints[j2, 1]], [joints[j1, 2], joints[j2, 2]])

	# plt.show()

	return labels_coarse,labels, joints,skel_adj




if __name__ == "__main__":
	filename = sys.argv[1]
	_,_,_, verts, faces,offset_data = anime_read(filename)
