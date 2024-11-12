import os
import sys
import numpy as np 
import open3d as o3d
import scipy.io as sio
from sklearn.neighbors import KDTree

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
	# print(sorted(already_computed))		

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


def find_coarse_labels_from_rosa(vertices,sampled_vertex,corresp,rosa_skeleton,projected_depth):
	"""
		@params: 
			vertices: Vertices for which we want to calculate coarse labels 
			sampled_vertex: points used for ROSA
			corresp: correspondence between sampled_vertex and rosa_joints 
			rosa_skeleton: <joints, edges>: Skeleton output from ROSA
			projected_depth: Projected depth image to pixel space
	"""



	joints,edges = rosa_skeleton
	J = joints.shape[0]

	# Extrapolate labels from samples vertices used in rosa
	valid_sampled_verts = np.where(corresp < J)[0] # Non-corresp vertices are marked as max(corresp) + 1
	kdtree = KDTree(sampled_vertex[valid_sampled_verts][:,:3])
	_,nn = kdtree.query(vertices,1) # Find closest vertex used in ROSA
	nn = nn.reshape(-1)
	labels = corresp[valid_sampled_verts][nn] 
	
	# Create adj matrix 
	skel_adj = np.zeros((J,J),dtype=np.bool)
	for e in edges:
		skel_adj[e[0],e[1]] = True
		skel_adj[e[1],e[0]] = True


	# Get connected components
	components = get_connected_components_from_adj_matrix(skel_adj)

	# Connect disjoint components sharing boundary in 2D space  
	segmented_boundaries = find_boundaries_from_point_cloud(projected_depth,labels,10,consider_3d=False)
	overlapping_components = find_overlapping_components(segmented_boundaries,components,joints)


	# Connect overlapping components
	# for x in overlapping_components:
	# 	j1,j2 = overlapping_components[x]
	# 	skel_adj[j1,j2] = True 
	# 	skel_adj[j2,j1] = True 					

	# 	components[components == components[j1]] = components[j2] # Merge componenets
	# 	print(f"Connecting: {j1} and {j2}")

	# Connecting disjoint segments
	# Sort keys by the number of boundary points 
	sorted_keys = sorted(list(segmented_boundaries.keys()), key=lambda x: len(segmented_boundaries[x]),reverse=True)
	for x in segmented_boundaries:
		j1 = x//J
		j2 = x%J

		if skel_adj[j1,j2]:
			continue 


		if components[j1] != components[j2]:
			print(f"J1:{j1} J2:{j2} component:{components[j1]} != component:{components[j2]} j1")
			if j1 in overlapping_components:
				if j2 not in overlapping_components[j1]: 
					skel_adj[j1,j2] = True 
					skel_adj[j2,j1] = True 			

					print(f"Connecting: {j1} and {j2}")

					components[components == components[j1]] = components[j2] # Merge componenets


			elif j2 in overlapping_components:
				if j1 not in overlapping_components[j2]: 
					skel_adj[j1,j2] = True 
					skel_adj[j2,j1] = True 		
					
					components[components == components[j1]] = components[j2] # Merge componenets

					print(f"Connecting: {j1} and {j2}")
			else: 
				skel_adj[j1,j2] = True 
				skel_adj[j2,j1] = True 							

				components[components == components[j1]] = components[j2] # Merge componenets

			print("Components:",components)

	# Get degree of each joint node
	joint_degree = np.sum(skel_adj,axis=0)
	print("Joint Degree:",joint_degree)


	# Remove dangling nodes (1-degree node directly connected to 3-degree node)
	for j,deg in enumerate(joint_degree):
		if deg == 1:
			parent = np.where(skel_adj[j,:])[0]
			assert len(parent) == 1, "Multiple parents for degree 1 node?" 
			parent = parent[0]

			if joint_degree[parent] > 2:
				print(f"Redundant node:{j},parent:{parent}")
				labels[labels==j] = parent
				skel_adj[j,parent] = False
				skel_adj[parent,j] = False

	# Get root (>2 degree node closest to the mean of the surface)
	surface_centroid = np.mean(vertices,axis=0,keepdims=True)
	# Cloest non-funcational node
	joint_degree = np.sum(skel_adj, axis=0)
	print("New joint Degrees:", joint_degree)
	non_functional_node_indices = np.where(joint_degree != 2)[0]
	# cloest_non_funcational_node = np.argmin(np.linalg.norm(joints[non_functional_node_indices] - surface_centroid,axis=1))
	# print(non_functional_node_indices,joint_degree[non_functional_node_indices])
	# print("Closest functional node to centroid:",np.linalg.norm(joints[non_functional_node_indices] - surface_centroid,axis=1))
	# root_node = non_functional_node_indices[cloest_non_funcational_node]


	# Merge labels of all 2 degree nodes
	degree2_paths = {}
	for i1,j1 in enumerate(non_functional_node_indices):
		for i2,j2 in enumerate(non_functional_node_indices[i1+1:]):
			ind = j1*J + j2
			degree2_paths[ind] = []
			# print(f"Checking: {j1,j2}")
			# Run dfs from source(j1) to target(j2)
			children = np.where(skel_adj[j1,:])[0]
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
					path_children = np.where(skel_adj[x,:])[0]
					if j2 in path_children: 
						print("Path:",path)			
						degree2_paths[ind].extend(path)	
						break
					for cc in path_children:
						if cc not in visited and joint_degree[cc] == 2:
							path.append(cc)
							stack.append(cc)

	degree2_paths_copy = {}						
	for ind in degree2_paths:
		if len(degree2_paths[ind]) != 0:
			degree2_paths_copy[ind] = degree2_paths[ind]
	degree2_paths = degree2_paths_copy
	
	# Merge Labels 
	for ind in degree2_paths:
		j1 = ind//J
		j2 = ind%J	
		
		base_joint = degree2_paths[ind][0]
		for x in degree2_paths[ind][1:]:
			labels[labels==x] = base_joint

		


	# queue = [root_node]
	# visisted = [False]*J
	# while len(queue) > 0:
	# 	x = queue[0]
	# 	queue = queue[1:]
	# 	visisted[x] = True
	# 	children = np.where(skel_adj[x,:])[0]
	# 	print(f"For functional node:{x}, deg:{joint_degree[x]}, children:{children}")
		
	# 	degree2_paths[x] = []

	# 	for c in children:
	# 		if ~visisted[c] and joint_degree[c] == 2:
	# 			stack = [c]
	# 			while len(stack) > 0:
	# 				y = stack[0]
	# 				stack = stack[1:]
	# 				visisted[y] = True
	# 				path_children = np.where(skel_adj[y,:])[0]
	# 				print(f"Checking node:{y}, deg:{joint_degree[y]}, children:{path_children}")
					
	# 				if joint_degree[y] == 2:
	# 					degree2_paths[x].append(y)


	# 				for cc in path_children: 
	# 					if visisted[cc]: 
	# 						continue
	# 					if joint_degree[cc] != 2: 
	# 						queue.append(cc)
	# 						print(f"Adding:{cc}, deg:{joint_degree[cc]} to queue")	

	# 					else: 
	# 						degree2_paths[x].append(cc)
	# 						stack.append(cc)
	# 						print(f"Adding:{cc},deg:{joint_degree[cc]} to stack")	

	# print(degree2_paths)						


	# import matplotlib.pyplot as plt
	# fig = plt.figure()
	# ax = fig.add_subplot(111,projection='3d')
	# ax.scatter(joints[:,0],joints[:,1],joints[:,2])
	# for j1 in range(J):
	# 	ax.text(joints[j1,0],joints[j1,1],joints[j1,2],str(j1))
	# 	for j2 in range(j1+1,J):
	# 		if skel_adj[j1,j2]:
	# 			ax.plot([joints[j1,0],joints[j2,0]],[joints[j1,1],joints[j2,1]],[joints[j1,2],joints[j2,2]])

	# plt.show()


	return labels,skel_adj,segmented_boundaries

def find_coarse_labels_from_gel(vertices,vertex_normals, sampled_vertex, corresp, rosa_skeleton, projected_depth,components,dangling_path_cnt=6):
	"""
		@params:
			vertices: Vertices for which we want to calculate coarse labels
			sampled_vertex: points used for ROSA
			corresp: correspondence between sampled_vertex and rosa_joints
			rosa_skeleton: <joints, edges>: Skeleton output from ROSA
			projected_depth: Projected depth image to pixel space
			components: component used to create skeleton segment
	"""

	joints, edges,skel_adj = rosa_skeleton
	J = joints.shape[0]

	# Extrapolate labels from samples vertices used in rosa
	valid_sampled_verts = np.where(corresp < J)[0]  # Non-corresp vertices are marked as max(corresp) + 1
	kdtree = KDTree(sampled_vertex[valid_sampled_verts])
	_, nn = kdtree.query(vertices, 1)  # Find closest vertex used in ROSA
	nn = nn.reshape(-1)
	labels = corresp[valid_sampled_verts][nn]

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

	# Get connected components from skeleton adj matrix 
	adj_components = get_connected_components_from_adj_matrix(skel_adj)


	# Connect disjoint components sharing boundary in 2D space
	segmented_boundaries = find_boundaries_from_point_cloud(projected_depth, labels, 10,J, consider_3d=False)
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


	for x in segmented_boundaries:	
		j1 = x // J
		j2 = x % J

		if skel_adj[j1, j2]:
			continue

		if components[j1] != components[j2]:
			# print(f"J1:{j1} J2:{j2} component:{components[j1]} != component:{components[j2]} j1")
			if j1 in overlapping_components:
				if j2 not in overlapping_components[j1]:
					skel_adj[j1, j2] = True
					skel_adj[j2, j1] = True

					# print(f"Connecting: {j1} and {j2}")

					components[components == components[j1]] = components[j2]  # Merge componenets


			elif j2 in overlapping_components:
				if j1 not in overlapping_components[j2]:
					skel_adj[j1, j2] = True
					skel_adj[j2, j1] = True

					components[components == components[j1]] = components[j2]  # Merge componenets

					# print(f"Connecting: {j1} and {j2}")
			else:
				skel_adj[j1, j2] = True
				skel_adj[j2, j1] = True

				components[components == components[j1]] = components[j2]  # Merge componenets

			print("Components:", components)

	# import matplotlib.pyplot as plt
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
	# for j1 in range(J):
	# 	# if joint_degree[j1] == 0: 
	# 		# continue
	# 	ax.text(joints[j1, 0], joints[j1, 1], joints[j1, 2], str(j1))
	# 	for j2 in range(j1 + 1, J):
	# 		if skel_adj[j1, j2]:
	# 			ax.plot([joints[j1, 0], joints[j2, 0]], [joints[j1, 1], joints[j2, 1]], [joints[j1, 2], joints[j2, 2]])

	# plt.show()

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



	# Get degree of each joint node
	joint_degree = np.sum(skel_adj,axis=0)
	print("Joint Degree:",joint_degree)
	# Remove dangling nodes (1-degree node connected to 3-degree node or having less number of intermediary 2 degree nodes)
	removing_dangling_nodes = True
	while removing_dangling_nodes: 
		removing_dangling_nodes = False
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
						

				# print(f"From node:{j} deg:{deg} path:{dangling_path} parent degree:{joint_degree[dangling_path[-1]]}")

				if len(dangling_path) > dangling_path_cnt: 
					continue

				if joint_degree[dangling_path[-1]] > 2: # End of dangling_path is a funcational node
					removing_dangling_nodes = True	
					parent = dangling_path[-1]
					for p in dangling_path[:-1]:
						skel_adj[p,:] = False 
						skel_adj[:,p] = False 
						joint_degree[p] = 0

						# Check if boundary of p and parent are connected. Else merge to cloest neighboring node of parent
						if p*J + parent in segmented_boundaries or parent*J + p in segmented_boundaries:
							labels[labels==p] = parent
						else: 
							labels[labels==p] = -1


					joint_degree[parent] -= 1

	if np.sum(labels==-1) > 0: # Some vertices might be unattahed after removing dangling nodes	
		kdtree = KDTree(vertices[labels!=-1])
		_, nn = kdtree.query(vertices[labels==-1], 1)  # Find closest vertex used in ROSA
		nn = nn.reshape(-1)
		labels[labels==-1] = labels[labels!=-1][nn]

	print("Old Joint Degree:",joint_degree)				

	joint_degree = np.sum(skel_adj, axis=0)
	print("New joint Degrees:", joint_degree)


	# Remove degree 0 joints 
	new_joints = []
	new2old_joint_ind = []
	new_labels = -1*np.ones_like(labels)
	for i in range(J):
		if joint_degree[i] > 0:
			new_labels[labels==i] = len(new_joints) 
			new2old_joint_ind.append(i)
			new_joints.append(joints[i])

	joints = np.array(new_joints)
	labels = new_labels
	J = joints.shape[0]
	new_skel_adj = np.zeros((J,J))
	for i in range(J):
		for j in range(J):
			new_skel_adj[i,j] = skel_adj[new2old_joint_ind[i],new2old_joint_ind[j]]
	skel_adj = new_skel_adj		
			
	joint_degree = np.sum(skel_adj,axis=0)		

	print("New joints:",joints.shape)
	print("Joint degree:",joint_degree)


	# Get root (>2 degree node closest to the mean of the surface)
	surface_centroid = np.mean(vertices, axis=0, keepdims=True)
	# Cloest non-funcational node
	non_functional_node_indices = np.where(joint_degree != 2)[0]
	# cloest_non_funcational_node = np.argmin(np.linalg.norm(joints[non_functional_node_indices] - surface_centroid,axis=1))
	# print(non_functional_node_indices,joint_degree[non_functional_node_indices])
	# print("Closest functional node to centroid:",np.linalg.norm(joints[non_functional_node_indices] - surface_centroid,axis=1))
	# root_node = non_functional_node_indices[cloest_non_funcational_node]

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

	return labels_coarse,labels, joints,skel_adj, segmented_boundaries




if __name__ == "__main__":
	filename = sys.argv[1]
	_,_,_, verts, faces,offset_data = anime_read(filename)



	# find_components(np.random.random((6,3)),[[0,1,2],[3,4,5]])
	colors,c_cnt = find_components(verts,faces)
	save_largest_component(colors,verts+offset_data[20],faces,f"{filename.split('/')[-1]}_20.off")

	# mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(verts),triangles=o3d.utility.Vector3iVector(faces))
	# color_list = np.random.random((c_cnt,3))
	# color_list = color_list[colors,:]
	# mesh.vertex_colors = o3d.utility.Vector3dVector(color_list)
	# o3d.visualization.draw_geometries([mesh])