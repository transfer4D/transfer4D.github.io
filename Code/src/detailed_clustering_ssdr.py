# Goal is to use the trajectroy information to cluster articulated regions
# Also ask user for right number of clusters 
# Run global update to remove redundant clusters
import os
import sys
import time
import logging
import argparse
import pyssdr
import numpy as np 
from scipy import sparse
from sklearn.cluster import KMeans
from vis import get_visualizer # Visualizer 
import polyscope as ps

sys.path.append("../")  # Making it easier to load modules

from sklearn.neighbors import KDTree
from utils.viz_utils import transform_pointcloud_to_opengl_coords as reflect_opengl
from find_connected_components import get_connected_components_from_adj_matrix


def write_skel(pinocchio_skel_path,joint_positions,new_parent_array):
		with open(pinocchio_skel_path,'w') as f:
			for i,v in enumerate(joint_positions):
				f.write(f"{i} {v[0]} {v[1]} {v[2]} {new_parent_array[i]}\n")

class SSDR(pyssdr.MyDemBones):
    def __init__(self,opt):
        super().__init__()

        self.log = logging.getLogger(__name__)
        self.opt = opt 
        self.max_bones = 30

        # Location to save data
        if hasattr(opt,"exp",) and hasattr(opt,"ablation"):
            self.savepath = os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation)
            os.makedirs(self.savepath,exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"images"),exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"video"),exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"target_pcd"),exist_ok=True)


    def load_trajectory(self,data_path,source_frame_id):

        trajectory_path = os.path.join(data_path,"trajectory",f"trajectory_{source_frame_id}.npy")
        trajectory = np.load(trajectory_path)

        # TODO fix for multiple source frame
        face_filename = os.listdir(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path"))[0] 

        face_data = np.load(os.path.join(data_path,"updated_graph",str(source_frame_id),"face_path",face_filename))

        return trajectory,face_data

    def load_gel_coarse_skeleton(self,data_path,source_frame_id):
        gel_path = os.path.join(data_path,"gel")

        assert os.path.isdir(gel_path),"Curve skeleton not computed. Run python3 run_pygel.py --datapath"

        files = [file for file in os.listdir(gel_path) if f"gel_{source_frame_id}" in file and "filtered" in file]

        gel_data = {}
        for file in files:
            data = np.load(os.path.join(gel_path,file))
            k = file.split('filtered_')[-1].split('.')[0]
            gel_data[k] = data 

        for r in ["joints","adj","corresp", "coarse_corresp"]:
            assert r in gel_data, f"{r} not present in gel data. Delete previous saved results and Re-run python3 run_pygel.py" 

        return gel_data    


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
           
    def get_cluster_influence_list(self,savepath, max_clusters=30):


        skin_list = [];
        relative_change_list_rsme = [rmse_err]
        removed_bones_list = []

        # Loop until minimum bones is reached. Break if threshold is reached 
        while len(self.keep_bones) > self.min_bones:


            cluster_influence = np.sum(self.w,axis=1)
                

            if reconstruction_change_list[remove_bone] < self.global_update_threshold:
            # if self.prev_nB > self.min_bones:
                # self.remove_bone(remove_bone)
                removed_bones_list.append(remove_bone)
                self.keep_bones = np.delete(self.keep_bones,np.where(self.keep_bones==remove_bone))
                self.w = self.prev_w_csc.copy()
                for i in range(10):
                    self.computeWeights()
                    self.computeTranformations()        
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

                deformed_vertices_ssdr = self.compute_reconstruction(list(range(N)))
                deformed_vertices = np.zeros((T,N,3))
                deformed_vertices[:,:,0] = deformed_vertices_ssdr[list(range(0,3*T,3))]
                deformed_vertices[:,:,1] = deformed_vertices_ssdr[list(range(1,3*T,3))]
                deformed_vertices[:,:,2] = deformed_vertices_ssdr[list(range(2,3*T,3))]

        # self.plot_reconstruction(deformed_vertices,movement,faces)

        return deformed_vertices,self.w.todense(),self.m,self.rmse()




    def initialize_ssdr(self,trajectory,faces,labels):    

        trajectory = trajectory.transpose((0,2,1))
        T,D,N = trajectory.shape
        trajectory = trajectory.reshape((D*T,N))

        # Set parameters to initize SSDR 
        self.tolerance = 1e-2
        self.bindUpdate=2
        self.nIters = 5
        self.patience = 1
        self.load_data(trajectory,faces) # Load data into pyssdr 

        # Clean labels 
        new2old_label_mapping = np.unique(labels)
        new_labels = -1*np.ones_like(labels)
        for i,l in enumerate(new2old_label_mapping):
            new_labels[labels==l] = i

        self.nB = len(new2old_label_mapping)
        self.label = new_labels
        self.labelToWeights();

        self.init()
        self.computeTransFromLabel()


        return new2old_label_mapping


    def get_reduced_skeleton(self,skel_adj):

        joint_degree = np.sum(skel_adj,axis=0)
        degree2indices = np.where(joint_degree==2)[0]

        # Remove 2 degree index and connect it's neigbours
        for x in degree2indices:

            edges = np.where(skel_adj[x,:])[0]
            skel_adj[edges[0],edges[1]] = True
            skel_adj[edges[1],edges[0]] = True


            skel_adj[x,:] = False
            skel_adj[:,x] = False 
        return degree2indices,skel_adj

    @staticmethod
    def get_2degreepath(x,parent,joint_degree,skel_adj):
        path = []
        while True:
            path.append(x)
            if joint_degree[x] != 0:
                break 

            children = np.where(skel_adj[x,:])[0]                        
            children = [c for c in children if c != parent]

            assert len(children) == 1, f"Reached a node greater than degree 2:{children}"
            parent = x
            x = children[0]
        return path

    def subdivide_LS_coarse_labels(self,trajectory,face_data,gel_data,min_reconstruction_change_percentage=0.2,min_component_size=10,num_split=3):



        joints = gel_data['joints'] # Jx3   
        skel_adj = gel_data['adj']  # JxJ 

        self.colors = np.random.random((joints.shape[0],3))
        self.colors[-1,:] = 0

        # get reduced skeleton 
        degree2indices,reduced_skel_adj = self.get_reduced_skeleton(gel_data['adj'].copy()) 
        corresp = gel_data['corresp']
        new2old_label_mapping = self.initialize_ssdr(trajectory,face_data,gel_data['coarse_corresp'])

        verts = trajectory[0]

        degree2joint_to_newlabel = {} # Stores after splitting degree 2 joint belongs to which label

        new_rmse = self.rmse()
        for split in range(num_split):


            # print(f"RMSE:{new_rmse}")
            joint_degree = np.sum(reduced_skel_adj,axis=0,dtype=np.int32)
            degree2indices = np.where(joint_degree==0)[0] 

            print("New joint degrees:",joint_degree)


            num_labels = self.nB
            new_labels = self.label.copy()

            if num_labels > self.max_bones:
                break


            # self.plot(gel_data['joints'][None],reduced_skel_adj,trajectory,face_data,new_labels)


            # print("Joints showing possible split locations:",new2old_label_mapping[np.unique(new_labels)])
            # print("Degree2 joint label",degree2joint_to_newlabel)
            for l in np.unique(new_labels):
                cluster_node_joint = new2old_label_mapping[l] # Should always represent 0 degree node else is should not be split
                label_degree = joint_degree[cluster_node_joint]
                # print(f"Found label:{l} joint:{cluster_node_joint} degree:{label_degree}")
                if label_degree > 0:
                    continue

                cluster_verts_indices = np.where(new_labels == l)[0]
                cluster_cnt = len(cluster_verts_indices)


                #     Test 1: Check if cluster_verts_indices is equal to the 0 degree nodes it consistues
                edges = np.where(skel_adj[cluster_node_joint])[0]
                split_path = self.get_2degreepath(edges[0],cluster_node_joint,joint_degree,skel_adj)[::-1] + [cluster_node_joint] + self.get_2degreepath(edges[1],cluster_node_joint,joint_degree,skel_adj)
                
                # print("Split path:",split_path)
                # Delete degree2 indices from split path if not associated wit the label
                if joint_degree[split_path[0]] == 2:
                    assert split_path[0] in degree2joint_to_newlabel, f"Degree 2 joint:{split_path[0]} not in degree2joint_to_newlabel:{degree2joint_to_newlabel.keys()}"
                    if degree2joint_to_newlabel[split_path[0]] != l:
                        del split_path[0]
                if joint_degree[split_path[-1]] == 2:
                    assert split_path[-1] in degree2joint_to_newlabel, f"Degree 2 joint:{split_path[-1]} not in degree2joint_to_newlabel:{degree2joint_to_newlabel.keys()}"
                    if degree2joint_to_newlabel[split_path[-1]] != l:
                        del split_path[-1] 



                # other_verts = []
                # for s in split_path:
                #     if joint_degree[s] < 3:
                #         other_verts.extend(list(np.where(corresp==s)[0]))

                # verts = trajectory[0]
                # kdtree = KDTree(verts)        
                # _, vert_closest = kdtree.query(joints, 1)  # Find closest vertex used in ROSA
                # vert_closest = vert_closest.reshape(-1)
                # joint2label = new_labels[vert_closest]


                # assert len(set(list(cluster_verts_indices)).difference(set(other_verts))) == 0, f"Cluster:{l} closest joints:{np.where(joint2label==l)} len:{len(cluster_verts_indices)} verts does not represent its corresponding joints:{sorted(split_path)} len:{len(other_verts)}"


                cluster_error = np.sqrt(self.rmse_from_cluster(cluster_verts_indices,True)/cluster_cnt)

                # Condition to not update
                if num_labels > self.max_bones or cluster_cnt < 2*min_component_size:
                    # print(f"For label:{l}, num_clusters:{num_labels} > max bones:{self.max_bones} cluster_cnt:{cluster_cnt} < 2*{min_component_size} Cluster Error:{cluster_error} < {new_rmse}")
                    continue

                # Get split path 
                start_node = new2old_label_mapping[l]
                # Propogate labels
                edges = np.where(skel_adj[start_node,:])[0]
                split_path = self.get_2degreepath(edges[0],start_node,joint_degree,skel_adj)[::-1] + [start_node] + self.get_2degreepath(edges[1],start_node,joint_degree,skel_adj)
                # Delete degree2 indices from split path if not associated wit the label
                if joint_degree[split_path[0]] == 2:
                    assert split_path[0] in degree2joint_to_newlabel, f"Degree 2 joint:{split_path[0]} not in degree2joint_to_newlabel::{degree2joint_to_newlabel.keys()}"
                    if degree2joint_to_newlabel[split_path[0]] != l:
                        del split_path[0]
                if joint_degree[split_path[-1]] == 2:
                    assert split_path[-1] in degree2joint_to_newlabel, f"Degree 2 joint:{split_path[-1]} not in degree2joint_to_newlabel:{degree2joint_to_newlabel.keys()}"
                    if degree2joint_to_newlabel[split_path[-1]] != l:
                        del split_path[-1] 


                if len(split_path) < 3:
                    # print(f"Split path:{split_path}")
                    continue
                else:
                    split_path_vertices = []  # Vertices associated with each joint in the path 
                    for x in split_path:
                        if joint_degree[x] < 3:
                            inds = list(np.where(corresp==x)[0])
                        else:
                            inds = []

                        split_path_vertices.append(inds)

                    # print(f"Cluster Error before splitting:{cluster_error}")
                    seed_error = cluster_error
                    seed_joint = -1
                    for i,sj in enumerate(split_path):

                        # Can't split at 1 degree node or node funcational nodes
                        if joint_degree[sj] != 0:
                            continue

                        setA = []
                        for a in range(0,i+1):
                            setA.extend(split_path_vertices[a])
                        setB = []
                        for b in range(i+1,len(split_path)):
                            setB.extend(split_path_vertices[b])


                        if len(setA) < min_component_size or len(setB) < min_component_size:
                            # print(f"i={i} split joint={sj} setA={len(setA)}  setB={len(setB)} one of the components is too small")
                            continue


                        # Test 2 setA and setB should make the cluster_verts_indices
                        assert len(set(setA).union(setB)) == len(cluster_verts_indices), f"SetA:{len(setA)} and SetB:{len(setB)} dont make cluster:{len(cluster_verts_indices)}"

                        errorA = self.rmse_from_cluster(setA,True)
                        errorB = self.rmse_from_cluster(setB,True)
                        total_error = np.sqrt((errorA+errorB)/(len(setA) + len(setB)))
                        # print(f"i={i} split joint={sj} setA={len(setA)} errorA:{errorA/len(setA)} setB={len(setB)} errorB:{errorB/len(setB)} total_error:{total_error} min:{seed_error}")

                        if seed_error > total_error:
                            seed_joint = sj
                            seed_error = total_error


                reconstruction_error_change = 1-(seed_error/cluster_error)
                print(f"Seed Error:{seed_error} < cluster error:{cluster_error}, %change :{reconstruction_error_change} split_path:{split_path}")
                if reconstruction_error_change < min_reconstruction_change_percentage or seed_joint == -1:
                    continue


                print(f"Splitting at joint:{seed_joint} split path:{split_path}")


                assert len(split_path) >=3, "Split path should have 2 degree node and 2 neighbors"
                
                for i,sj in enumerate(split_path):
                    if joint_degree[sj] < 3:
                        new_labels[corresp == sj] = num_labels

                    if joint_degree[sj] == 2: # Update label of degree 2 index if in path
                        degree2joint_to_newlabel[sj] = num_labels     

                    if sj == seed_joint:
                        new2old_label_mapping = np.concatenate([new2old_label_mapping,[split_path[i-1]]],axis=0) # Updated path should be assigned a new label and new joint  
                        new2old_label_mapping[l] = split_path[i+1] # Remaining nodes should also be assigned to new nodes
                        degree2joint_to_newlabel[seed_joint] = num_labels # Every degree 2 node added should be assigned to its new label   

                        num_labels += 1
                        break 

                        
                # Update graph (since 2 degree nodes can not be present during cluster update. Need to add them again for updating the graph)
                split_path = self.get_2degreepath(edges[0],start_node,joint_degree,skel_adj)[::-1] + [start_node] + self.get_2degreepath(edges[1],start_node,joint_degree,skel_adj)
                reduced_skel_adj[seed_joint,split_path[0]] = True
                reduced_skel_adj[split_path[0],seed_joint] = True

                reduced_skel_adj[seed_joint,split_path[-1]] = True
                reduced_skel_adj[split_path[-1],seed_joint] = True

                reduced_skel_adj[split_path[0],split_path[-1]] = False
                reduced_skel_adj[split_path[-1],split_path[0]] = False

                joint_degree[seed_joint] = 2 # Joint degree of seed joint gets updated        

            self.nB = num_labels
            self.label = new_labels
            self.computeTransFromLabel()    
            self.labelToWeights()

            old_rmse = new_rmse            
            new_rmse = self.rmse()

            # import matplotlib.pyplot as plt
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2])
            # for j1 in range(joints.shape[0]):
            #   ax.text(joints[j1, 0], joints[j1, 1], joints[j1, 2], str(j1))
            #   for j2 in range(j1 + 1, joints.shape[0]):
            #       if skel_adj[j1, j2]:
            #           ax.plot([joints[j1, 0], joints[j2, 0]], [joints[j1, 1], joints[j2, 1]], [joints[j1, 2], joints[j2, 2]])

            # plt.show()


            # self.plot(gel_data['joints'][None].copy(),reduced_skel_adj,trajectory.copy(),face_data,new_labels,debug=True)


            print(f"Old rmse:{old_rmse} New rmse:{new_rmse}")            
            if new_rmse > old_rmse:
                break

        return reduced_skel_adj,new2old_label_mapping    

    def get_joint_motion(self,trajectory,labels,joints,skel_adj):
        
        N = joints.shape[0]    
        T = trajectory.shape[0]
        joint_trajectory = np.tile(joints,(T,1,1))
        
        print(joints.shape,trajectory.shape)


        for j in range(N):
            ind = np.where(labels==j)[0]
            if len(ind) == 0: # If no index found take the mean of neigbouring joints  (Hack for now)
                edges = np.where(skel_adj[j,:])[0]
                ind = []
                for e in edges:
                    ind.extend(list(np.where(labels==e)[0]))
                ind = np.array(ind)

            assert len(ind) > 0, f"No verts found to take mean for joint:{j}"

            cluster_trajectory = trajectory[:,ind,:]
            trajectory_sceneflow = np.mean(cluster_trajectory - cluster_trajectory[0:1],axis=1)


            joint_trajectory[:,j,:] = joint_trajectory[:,j,:] + trajectory_sceneflow

        return joint_trajectory    

    @staticmethod    
    def remove_redundant_joints(joint_motion, skel_adj, label_to_joint_mapping):
        joint_degree = np.sum(skel_adj,axis=0).astype(np.int32)
        keep_joints = [i for i,x in enumerate(joint_degree) if x > 0]

        print("Keep Joints:",keep_joints)

        joint_motion = joint_motion[:,keep_joints,:]

        J = joint_motion.shape[1]
        new_skel_adj = np.zeros((J,J),dtype=bool)
        for i in range(J):
            for j in range(J):
                if skel_adj[keep_joints[i],keep_joints[j]]: 
                    new_skel_adj[i,j] = True

        # for i,x in enumerate(label_to_joint_mapping):
        #     if x in keep_bones: 
        #         label_to_joint_mapping[x] = 

        return joint_motion,new_skel_adj

    def get_reconstruction(self,inds=[]):
        if len(inds) == 0: 
            inds = list(range(self.nV))
        deformed_vertices_ssdr = self.compute_reconstruction(inds)
        deformed_vertices = np.zeros((self.nF,self.nV,3))
        deformed_vertices[:,:,0] = deformed_vertices_ssdr[list(range(0,3*self.nF,3))]
        deformed_vertices[:,:,1] = deformed_vertices_ssdr[list(range(1,3*self.nF,3))]
        deformed_vertices[:,:,2] = deformed_vertices_ssdr[list(range(2,3*self.nF,3))]    

        return deformed_vertices

    def test_cluster_error(self):
        pointsA = np.random.random((1000,3))
        pointsB = np.random.random((1000,3))
        pointsB[1] -= 1
        points = np.concatenate([pointsA,pointsB],axis=0)
        from scipy.spatial.transform import Rotation
        small_rotationA = Rotation.from_euler('xyz',[np.pi/10,0,0]).as_matrix()
        small_rotationB = Rotation.from_euler('xyz',[np.pi/20,0,0]).as_matrix()
        small_translationA = 0.001*np.ones((1,3))
        small_translationB = 0.001*np.ones((1,3))

        trajectory = [points]
        for i in range(10):

            v_prev = trajectory[-1].copy()
            v_prev_meanA = np.mean(v_prev[:1000],axis=0,keepdims=True)
            v_prev_meanB = np.mean(v_prev[1000:],axis=0,keepdims=True)
            v_newA = (v_prev[:1000] - v_prev_meanA)@small_rotationA + v_prev_meanA + small_translationA
            v_newB = (v_prev[1000:] - v_prev_meanB)@small_rotationB + v_prev_meanB + small_translationB
            v_new = np.concatenate([v_newA,v_newB],axis=0)
            
            trajectory.append(v_new) # Rotation + Translation 
            # trajectory.append(points + i*small_translation) # Only translation

        trajectory = np.array(trajectory)

        kdtree = KDTree(points)
        _,faces = kdtree.query(points,3)


        labels = np.zeros((points.shape[0]))
        self.initialize_ssdr(trajectory,faces,labels)


        point_indsA = list(range(points.shape[0]//2))
        point_indsB = list(range(points.shape[0]//2,points.shape[0]))
        error = self.rmse_from_cluster(point_indsA,False)
        print(f"Error:",error)
        
        error = self.rmse_from_cluster(point_indsB,False)

        print(f"Error:",error)

    def connect_disjoint_bones(self,joint_motion,skel_adj):
        # Connect components by connecting joints with least bone length variation

        T,J,D = joint_motion.shape
        skeleton_component = get_connected_components_from_adj_matrix(skel_adj)

        components = np.unique(skeleton_component)
        num_components = len(components)

        adj_matrix = np.inf*np.ones((num_components,num_components))
        for i in range(num_components):
            for j in range(i+1,num_components):
                j1 = np.where(skeleton_component == components[i])[0]
                j2 = np.where(skeleton_component == components[j])[0]

                XX = np.reshape(joint_motion[:,j1,:], (T, 1, len(j1), D))
                YY = np.reshape(joint_motion[:,j2,:], (T, len(j2), 1, D))
                XX = np.tile(XX, (1, len(j2), 1, 1))
                YY = np.tile(YY, (1, 1, len(j1), 1))

                diff = np.linalg.norm(XX - YY, ord=2, axis=3)
                diff = np.max(np.min(diff,axis=(1,2)))
                adj_matrix[i,j] = diff
                adj_matrix[j,i] = diff


        while len(np.unique(components)) > 1:
            a,b = np.unravel_index(np.argmin(adj_matrix),adj_matrix.shape)
            print(a,b,components[a] , components[b])
            adj_matrix[a,b] = np.inf
            adj_matrix[b, a] = np.inf
            if components[a] != components[b]:
                
                j1,j2 = find_joints_sharing_boundary(trajectory[0],joint_motion[0],)
                skel_adj[a,b] = True
                skel_adj[b, a] = True

                components[ components == components[b] ] = components[a]
        return skel_adj

    def rearrange_skeleton(self,joint_motion,skel_adj,labels):
        """
            2. Connect disconnected components 
            1. Closest functional node to the center becomones the root node
            3. Run DFS and get Parent array 
            4. rearrange joints
        """

        # Get root (>2 degree node closest to the mean of the surface)
        skeleton_centroid = np.mean(joint_motion[0],axis=0,keepdims=True)

        # Cloest non-funcational node
        joint_degree = np.sum(skel_adj, axis=0)
        functional_node_indices = np.where(joint_degree > 2)[0]

        closest_funcational_node = np.argmin(np.linalg.norm(joint_motion[0,functional_node_indices] - skeleton_centroid,axis=1))
        root_ind = functional_node_indices[closest_funcational_node]

        # DFS
        # Rearrange points for Pinnochio using dfs 
        new_indices = []
        old2new_indices = {}
        parent_array = []
        stack = [[root_ind,root_ind]]

        while len(stack) > 0:
            x,p = stack[-1]
            print(x,p,stack,new_indices,parent_array)
            stack.pop()

            old2new_indices[x] = len(new_indices)
            new_indices.append(x)
            parent_array.append(old2new_indices[p])

            for c in np.where(skel_adj[x,:] > 0)[0]:
                if c not in new_indices:
                    stack.append([c,x])
        
        print(old2new_indices)
        print(joint_motion.shape)                    
        print(skel_adj.shape)
        print(np.unique(labels))
        print(len(new_indices))

        new_indices = np.array(new_indices)
        joint_motion = joint_motion[:,new_indices,:]
        # labels = new_indices[labels]
        return joint_motion,parent_array,labels           


    def get_different_cluster(self):

        N = self.nV
        T = self.nF

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





    def __call__(self,opt):    
        data_path = os.path.join(opt.datadir,"results",opt.exp,opt.ablation)
        trajectory_path = os.path.join(data_path,"trajectory")
        # Save skinning 
        skinning_path = os.path.join(data_path,"skinning")
        os.makedirs(skinning_path,exist_ok=True)
        skeleton_path = os.path.join(data_path,"skeleton")
        os.makedirs(skeleton_path,exist_ok=True)


        for file in sorted(os.listdir(trajectory_path),key=lambda x: int(x.split('.')[0].split('_')[-1])):
            self.log.info(f"=====Loading trajectory:{file}========")
            source_frame_id = int(file.split('.')[0].split('_')[-1])
            trajectory,face_data = self.load_trajectory(data_path,source_frame_id)

            # if not os.path.isfile(os.path.join(skeleton_path,f"skeletonMotion_{source_frame_id}.npy")):
            if True:
                gel_data = self.load_gel_coarse_skeleton(data_path,source_frame_id)

                skel_adj,label_to_joint_mapping = self.subdivide_LS_coarse_labels(trajectory,face_data,gel_data,min_reconstruction_change_percentage=opt.min_reconstruction_change_percentage)

                np.save(os.path.join(skinning_path,f"detailed_skeleton_labels_{source_frame_id}.npy"),self.label)


                self.run_ssdr(0,"here.fbx")
                reconstruction = self.get_reconstruction()
                np.save(os.path.join(skinning_path,f"reconstruction_{source_frame_id}.npy"),reconstruction)

                # Get skeleton motion 
                joint_motion = self.get_joint_motion(reconstruction,gel_data['corresp'],gel_data['joints'],gel_data['adj'])
                joint_motion,skel_adj = self.remove_redundant_joints(joint_motion,skel_adj,label_to_joint_mapping)

                # skel_adj = self.connect_disjoint_bones(joint_motion,skel_adj,trajectory)

                # Remove degree joints 

                labels = np.array(self.w.argmax(axis=0)).squeeze()


                # joint_motion = np.load(os.path.join(skeleton_path,f"skeletonMotion_{source_frame_id}.npy"))
                # skel_adj = np.load(os.path.join(skeleton_path,f"skelAdj_{source_frame_id}.npy"))

        
                # Get skel file and parent array 
                joint_motion,parent_array,labels = self.rearrange_skeleton(joint_motion,skel_adj,labels)

                np.save(os.path.join(skeleton_path,f"skeletonMotion_{source_frame_id}.npy"),joint_motion)
                np.save(os.path.join(skeleton_path,f"skelAdj_{source_frame_id}.npy"),skel_adj)
                np.save(os.path.join(skeleton_path,f"labels_{source_frame_id}.npy"),labels)
                
                write_skel(os.path.join(skeleton_path,f"{source_frame_id}.skel"), joint_motion[0],parent_array)
                np.save(os.path.join(skinning_path,f"weights_{source_frame_id}.npy"),self.w.todense())
                np.save(os.path.join(skinning_path,f"boneTransformations_{source_frame_id}.npy"),self.m)



                self.plot(joint_motion,parent_array,trajectory,face_data,labels,source_frame_id,debug=False)

                # Get reconstruction for every cluster size 
                # reconstructions,weights,rmse_list =  self.get_different_cluster()
                # rec_path = os.path.join(data_path,f"reconstructionOurs_{source_frame_id}")
                # os.makedirs(rec_path,exist_ok=True)    
                # self.log.info(f"Reconsturction Error:{rmse_list}")

                # np.save(os.path.join(rec_path,f"reconstruction_{source_frame_id}.npy"),reconstructions)
                # np.save(os.path.join(rec_path,f"weights_{source_frame_id}.npy"),weights)
                # with open(os.path.join(rec_path,'rmse.txt'),'w') as f: 
                #     f.write(','.join([str(r) for r in rmse_list]))


    def plot(self,joints,parent_array,trajectory,face_data,labels,source_frame_id=0,framerate=5,debug=False):

        if not hasattr(self,"trajectory_mean"):
            self.trajectory_mean = np.mean(trajectory,axis=1,keepdims=True)
        
        if joints.shape[0] == 1: 
            joints = np.tile(joints,(trajectory.shape[0],1,1))

        trajectory -= self.trajectory_mean
        joints -= self.trajectory_mean


        if type(parent_array) == np.ndarray: 
            if len(parent_array.shape) == 1:
                source_bone_array = np.array([[i,p] for i,p in enumerate(parent_array)])
            elif len(parent_array.shape) == 2:
                J = parent_array.shape[0]
                source_bone_array = np.array([[i,j] for i in range(J) for j in range(J) if parent_array[i,j]])
                
        elif type(parent_array) == list:
            source_bone_array = np.array([[i,p] for i,p in enumerate(parent_array)])
        else: 
            raise Error("Cannot parse parent array")    

        ps.init()
        self.bbox = trajectory.max(axis=(0,1)) - trajectory.min(axis=(0,1))
        self.bbox[0] *= 1.2
        # self.bbox[1] *= 1.2

        self.object_position = trajectory[0].mean(axis=0)


        # Set camera 
        # ps.set_automatically_compute_scene_extents(False)
        ps.set_navigation_style("free")
        ps.set_view_projection_mode("orthographic")
        ps.set_ground_plane_mode('none')

        camera_position = np.array([0,0,7*self.bbox[0]]) + self.object_position
        look_at_position = np.array([0,0,0]) + self.object_position
        ps.look_at(camera_position,look_at_position)

        # trajectory += np.array([0,0,+0.5])*self.bbox    
        # joints += np.array([0,0,-0.5])*self.bbox

        T = trajectory.shape[0]    
        J = joints.shape[1]

        if hasattr(self,"colors"):
            colors = self.colors
        else:
            colors = np.random.random((J+1,3))
            colors[-1] = 0

        for t in range(T):
            if t == 0:
                ps_verts = ps.register_surface_mesh("PC",reflect_opengl(trajectory[0]),face_data,enabled=True,smooth_shade=True,transparency=0.5)
                vert_colors = colors[labels]
                ps_verts.add_color_quantity("skinning",vert_colors,enabled=True)

                ps_joints = ps.register_point_cloud("Joints",reflect_opengl(joints[0]),enabled=True,radius=0.01)
                
                ps_skeleton = ps.register_curve_network("Skeleton",reflect_opengl(joints[0]),source_bone_array,enabled=True)
                ps_skeleton.add_color_quantity("joint",colors[:J],defined_on='nodes',enabled=True)
                
                
            else: 
                ps_verts.update_vertex_positions(reflect_opengl(trajectory[t]))
                ps_joints.update_point_positions(reflect_opengl(joints[t]))
                ps_skeleton.update_node_positions(reflect_opengl(joints[t]))
            
            if t + source_frame_id == 0:
                ps.show()
            if debug:
                return

            image_path = os.path.join(self.savepath,"images",f"skeleton_{t+source_frame_id}.png")
            print(f"Saving plot to :{image_path}")  
            ps.set_screenshot_extension(".jpg");
            ps.screenshot(image_path,transparent_bg=False)


        image_path = os.path.join(self.savepath,"images",f"skeleton_\%d.png")
        video_path = os.path.join(self.savepath,"video","skeleton.mp4")
        palette_path = os.path.join(self.savepath,"video","palette2.png")
        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")    
        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")    



if __name__ == "__main__":
    args = argparse.ArgumentParser() 
    # Path locations
    args.add_argument('--datadir', required=True,type=str,help='path to folder containing RGBD video data')
    args.add_argument('--exp', required=True,type=str,help='path to folder containing experimental name')
    args.add_argument('--ablation', required=True,type=str,help='path to folder experimental details')

    # Clusters 

    args.add_argument('--min_reconstruction_change_percentage', default=0.0, type=int, 
                      help='minimum change in reconstruction error required for each split')


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
    # vis = get_visualizer(opt) # To see data
    ssdr = SSDR(opt)
    ssdr(opt)
    # ssdr.test_cluster_error()

