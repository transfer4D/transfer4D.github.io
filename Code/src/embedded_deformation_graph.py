# Imports
import os
import logging # For logging info
import numpy as np 
import json # For loading graph config
from skimage import io
from timeit import default_timer as timer
import re

from queue import PriorityQueue # For finding geodesic distance 



# Neural Tracking Modules
from model import dataset
from utils import utils

# Neural Tracking C compiled modules
from NeuralNRT._C import erode_mesh as erode_mesh_c
from NeuralNRT._C import sample_nodes as sample_nodes_c
from NeuralNRT._C import compute_edges_geodesic as compute_edges_geodesic_c
from NeuralNRT._C import node_and_edge_clean_up as node_and_edge_clean_up_c
from NeuralNRT._C import compute_clusters as compute_clusters_c
from NeuralNRT._C import compute_mesh_from_depth as compute_mesh_from_depth_c

class EDGraph: 
    def __init__(self,opt,visualizer,source_frame_data):

        # Log path 
        self.log = logging.getLogger(__name__)


        self.opt = opt
        self.load_graph_config()        
    
        # Save path
        self.savepath = os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation,"updated_graph",str(source_frame_data['id']))
        os.makedirs(self.savepath,exist_ok=True)

        # create_graph_from_tsdf
        self.tsdf = None
        self.vis = visualizer
        # self.tsdf.graph = self

        
        # self.create_graph_from_tsdf()
        self.create_graph_from_depth(source_frame_data)




    def load_graph_config(self):

        # Parameters for generating new graph
        if os.path.isfile(os.path.join(self.opt.datadir,'graph_config.json')):
            with open(os.path.join(self.opt.datadir,'graph_config.json')) as f:
                self.graph_generation_parameters = json.load(f) 

        else: # Defualt case
            self.graph_generation_parameters = {
                # Given a depth image, its mesh is constructed by joining using its adjecent pixels, 
                # based on the depth values, a perticular vertex could be far away from its adjacent pairs in for a particular face/triangle
                # `max_triangle_distance` controls this. For a sperical surfaces  this should be set high. But for clothes this can be small      
                'max_triangle_distance' : 0.05, 
                'erosion_num_iterations': 1,   # Will reduce outliers points (simliar to erosion in images)
                'erosion_min_neighbours' : 3,    # Will reduce outlier clusters
                'node_coverage'         : 0.05, # Sampling parameter which defines the number of nodes
                'min_neighbours'        : 2,    # Find minimum nunber of neighbours that must be present for ech node     
                'num_neighbours'        : 8,    # Maximum number of neighbours
                'require_mask'          : True,
                }
            with open(os.path.join(self.opt.datadir,'graph_config.json'), "w") as f:
                json.dump(self.graph_generation_parameters, f)    

    

    def create_graph_from_tsdf(self):
        
        assert hasattr(self,'tsdf'),  "TSDF not defined in graph. Add tsdf as attribute to EDGraph first." 
        vertices, faces, normals, colors = self.tsdf.get_mesh()  # Extract the new canonical pose using marching cubes
        self.log.debug(f"Creating graph from mesh. Vertices:{vertices.shape} Faces:{faces.shape}")
        self.create_graph_from_mesh(vertices,faces)

    def create_graph_from_depth(self,source_frame_data):
        
        vertices,faces = self.create_mesh_from_depth(source_frame_data)
        print(vertices.shape,faces.shape)
        # self.vis.plot([self.vis.get_mesh(vertices,faces)],"",True)
        self.create_graph_from_mesh(vertices,faces)


    # If RGBD Image run other steps
    def create_mesh_from_depth(self,im_data,depth_normalizer = 1000.0):
        """
            im_data: np.ndarray: 6xHxW RGB+PointImage 
        """

        #########################################################################
        # Load data.
        #########################################################################
        # Load intrinsics.

        # fx = intrinsics[0, 0]
        # fy = intrinsics[1, 1]
        # cx = intrinsics[0, 2]
        # cy = intrinsics[1, 2]
        fx,fy,cx,cy = im_data["intrinsics"]
        print(im_data["intrinsics"])

        # Load depth image.
        depth_image = im_data["im"][-1] 

        # Load mask image.
        mask_image = im_data["im"] > 0
        # if mask_image is None and self.graph_generation_parameters["require_mask"] 
        #########################################################################
        # Convert depth to mesh.
        #########################################################################
        width = depth_image.shape[1]
        height = depth_image.shape[0]

        # Invalidate depth values outside object mask.
        # We only define graph over dynamic object (inside the object mask).
        mask_image[mask_image > 0] = 1
        depth_image = depth_image * mask_image

        # Backproject depth images into 3D.
        # point_image = depth_image.astype(np.float32)
        point_image = im_data["im"][3:].astype(np.float32)

        # Convert depth image into mesh, using pixelwise connectivity.
        # We also compute flow values, and invalidate any vertex with non-finite
        # flow values.
        vertices = np.zeros((0), dtype=np.float32)
        vertex_flows = np.zeros((0), dtype=np.float32)
        vertex_pixels = np.zeros((0), dtype=np.int32)
        faces = np.zeros((0), dtype=np.int32)

        compute_mesh_from_depth_c(
        point_image,
        self.graph_generation_parameters["max_triangle_distance"],
        vertices, faces)
        
        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]

        assert num_vertices > 0 and num_faces > 0

        return vertices,faces

    def erode_mesh(self,vertices,faces,num_iterations=-1):
        """
            Erode the graph based on the graph strucuture. 
            Basically return vertices with greater than min_neighbours after x iterations 

            @params:
                vertices: Nx3 np.ndarray 
                faces: Mx3 np.ndarray(int)

            @returns: 
                non_eroded_vertices indices
        """    

        if num_iterations: 
            num_iterations = self.graph_generation_parameters["erosion_num_iterations"]
        non_eroded_vertices = erode_mesh_c(
            vertices, faces,\
            num_iterations,\
            self.graph_generation_parameters["erosion_min_neighbours"])
        return non_eroded_vertices

    def create_graph_from_mesh(self,vertices,faces):

        # Node sampling and edges computation
        USE_ONLY_VALID_VERTICES = True
        NODE_COVERAGE = self.graph_generation_parameters["node_coverage"]

        num_vertices = vertices.shape[0]
        num_faces = faces.shape[0]

        assert num_vertices > 0 and num_faces > 0

        # Erode mesh, to not sample unstable nodes on the mesh boundary.
        non_eroded_vertices = self.erode_mesh(vertices,faces)

        #########################################################################
        # Sample graph nodes.
        #########################################################################
        valid_vertices = non_eroded_vertices

        
        SAMPLE_RANDOM_SHUFFLE = False
        # Sample graph nodes.
        node_coords = np.zeros((0), dtype=np.float32)
        node_indices = np.zeros((0), dtype=np.int32)
        
        num_nodes = sample_nodes_c(
            vertices, valid_vertices,
            node_coords, node_indices, 
            NODE_COVERAGE, 
            USE_ONLY_VALID_VERTICES,
            SAMPLE_RANDOM_SHUFFLE
        )

        node_coords = node_coords[:num_nodes, :]
        node_indices = node_indices[:num_nodes, :]

        # Just for debugging
        # pcd_nodes = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(node_coords))
        # o3d.visualization.draw_geometries([pcd_nodes], mesh_show_back_face=True)

        ENFORCE_TOTAL_NUM_NEIGHBORS = True
        NUM_NEIGHBORS = self.graph_generation_parameters["num_neighbours"]
        #########################################################################
        # Compute graph edges.
        #########################################################################
        # Compute edges between nodes.
        graph_edges              = -np.ones((num_nodes, NUM_NEIGHBORS), dtype=np.int32)
        graph_edges_weights      =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
        graph_edges_distances    =  np.zeros((num_nodes, NUM_NEIGHBORS), dtype=np.float32)
        node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

        visible_vertices = np.ones_like(valid_vertices)

        compute_edges_geodesic_c(
            vertices, visible_vertices, faces, node_indices, 
            NUM_NEIGHBORS, NODE_COVERAGE, 
            graph_edges, graph_edges_weights, graph_edges_distances,
            node_to_vertex_distances,
            USE_ONLY_VALID_VERTICES,
            ENFORCE_TOTAL_NUM_NEIGHBORS
        )

        # Save mesh details 
        self.vertices = vertices
        self.faces = faces

        # Save current results 
        self.nodes           = node_coords
        self.node_indices    = node_indices # Which vertex correspondes to graph node in the original mesh
        self.edges           = graph_edges 
        self.edges_weights   = graph_edges_weights 
        self.edges_distances = graph_edges_distances
        self.clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32) # Will be calculated later 

        self.node_to_vertex_distances = node_to_vertex_distances # Not required except for edge calculatation    
        self.remove_nodes_with_not_enough_neighbours()

        #########################################################################
        # Compute clusters.
        #########################################################################
        self.compute_clusters()

        self.create_graph_pyramid() # Graph pyramid used for motion completetion

        self.save()
        

    def create_graph_pyramid(self):
        """
            Create graph pyramid used for motion completion
        """
        node_coverage = self.graph_generation_parameters["node_coverage"]
        num_neighbours_list = [8,6,4,3]

        pyd = {}
        for level in range(4):
            if level == 0:
                pyd['nn_index_l0'] = self.edges
                old_nodes = self.nodes
                new_node_indices = self.node_indices

            else:    
                # Get Downsample index 
                node_coverage = node_coverage*2

                old_num_nodes = old_nodes.shape[0]
                down_sample_idx = []
                up_sample_idx = []
                for i in range(old_num_nodes):
                    if len(down_sample_idx) == 0:
                        up_sample_idx.append(i)        
                        down_sample_idx.append(i)
                        continue

                    # If node already covered
                    nodes_dist = np.linalg.norm(old_nodes[down_sample_idx] - old_nodes[i], axis=1)
                    min_node_ind = np.argmin(nodes_dist)
                    min_nodes_dist = nodes_dist[min_node_ind]

                    up_sample_idx.append(min_node_ind)        

                    if min_nodes_dist < node_coverage: # Not sure if correct
                        continue

                    down_sample_idx.append(i)
                    # self.log.debug(f"Level:{level} Nodes:{len(down_sample_idx)},Node index:{i},Min dists:{min_nodes_dist}")

                num_neighbours = num_neighbours_list[level]    
                new_node_indices = new_node_indices[down_sample_idx]
                new_num_nodes = new_node_indices.shape[0]

                # Compute edges between nodes.
                graph_edges              = -np.ones((new_num_nodes, num_neighbours), dtype=np.int32)
                graph_edges_weights      =  np.zeros((new_num_nodes, num_neighbours), dtype=np.float32)
                graph_edges_distances    =  np.zeros((new_num_nodes, num_neighbours), dtype=np.float32)
                node_to_vertex_distances = -np.ones((new_num_nodes, self.vertices.shape[0]), dtype=np.float32)

                visible_vertices = np.ones(self.vertices.shape[0],dtype=bool)    
                USE_ONLY_VALID_VERTICES = False 
                ENFORCE_TOTAL_NUM_NEIGHBORS = True 
                compute_edges_geodesic_c(
                            self.vertices, visible_vertices, self.faces, new_node_indices, 
                            num_neighbours, node_coverage, 
                            graph_edges, graph_edges_weights, graph_edges_distances,
                            node_to_vertex_distances,
                            USE_ONLY_VALID_VERTICES,
                            ENFORCE_TOTAL_NUM_NEIGHBORS
                        )

                pyd['down_sample_idx' + str(level)] = down_sample_idx    
                pyd['up_sample_idx' + str(level)] = up_sample_idx    
                pyd['nn_index_l' + str(level)] = graph_edges
                old_nodes = old_nodes[down_sample_idx]    

        
        self.pyd = pyd

        return         

    def remove_nodes_with_not_enough_neighbours(self):    

        # Cluster nodes in graph
        NEIGHBORHOOD_DEPTH = 2
        MIN_CLUSTER_SIZE = 10

        MIN_NUM_NEIGHBORS = self.graph_generation_parameters["min_neighbours"]
        REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True

        num_nodes = self.nodes.shape[0]

        # Remove nodes 
        valid_nodes_mask = np.ones((num_nodes, 1), dtype=bool)
        node_id_black_list = []

        if REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS:
            # Mark nodes with not enough neighbours
            node_and_edge_clean_up_c(self.edges, valid_nodes_mask)
        else:
            self.log.info("You're allowing nodes with not enough neighbours!")

        # Remove black listed nodes
        reduced_graph_dict = self.get_reduced_graph(valid_nodes_mask)

        # Remove invalid graph
        self.nodes           = reduced_graph_dict["valid_nodes_at_source"]
        self.edges           = reduced_graph_dict["graph_edges"] 
        self.edges_weights   = reduced_graph_dict["graph_edges_weights"] 
        self.edges_distances = reduced_graph_dict["graph_edges_distances"]                  
        self.clusters  = reduced_graph_dict["graph_clusters"] 
        self.num_nodes = reduced_graph_dict["num_nodes"]
        # self.node_to_vertex_distances = reduced_graph_dict["node_to_vertex_distances"] # Not required 


        # Done manually only at the start 
        self.node_indices    = self.node_indices[reduced_graph_dict["valid_nodes_mask"]] # Which vertex correspondes to graph node


    def compute_clusters(self):
        """
            Based on graph traversal find connected components and put them in seperate clusters. 
        """    
        clusters_size_list = compute_clusters_c(self.edges, self.clusters)

        for i, cluster_size in enumerate(clusters_size_list):
            if cluster_size <= 2:
                self.log.error(f"Cluster is too small {clusters_size_list}")
                self.log.error(f"It only has nodes:{np.where(self.clusters == i)[0]}")

    def get_reduced_graph(self,valid_nodes_mask):
            

        # Get the list of invalid nodes
        node_id_black_list = np.where(valid_nodes_mask == False)[0].tolist()

        # Get only graph corresponding info
        node_coords           = self.nodes[valid_nodes_mask.squeeze()]
        graph_edges           = self.edges[valid_nodes_mask.squeeze()] 
        graph_edges_weights   = self.edges_weights[valid_nodes_mask.squeeze()] 
        graph_edges_distances = self.edges_distances[valid_nodes_mask.squeeze()] 
        graph_clusters        = self.clusters[valid_nodes_mask.squeeze()]
        # node_to_vertex_distances = self.node_to_vertex_distances[valid_nodes_mask.squeeze()]
        #########################################################################
        # Graph checks.
        #########################################################################
        num_nodes = node_coords.shape[0]

        # Check that we have enough nodes
        if (num_nodes == 0):
            print("No nodes! Exiting ...")
            exit()

        self.log.info(f"Node filtering: initial num nodes: {num_nodes} | invalid nodes: {len(node_id_black_list)}:{node_id_black_list}")

        # Update node ids only if we actually removed nodes
        if len(node_id_black_list) > 0:
            # 1. Mapping old indices to new indices
            count = 0
            node_id_mapping = {}
            for i, is_node_valid in enumerate(valid_nodes_mask):
                if not is_node_valid:
                    node_id_mapping[i] = -1
                else:
                    node_id_mapping[i] = count
                    count += 1

            # 2. Update graph_edges using the id mapping
            for node_id, graph_edge in enumerate(graph_edges):
                # compute mask of valid neighbours
                valid_neighboring_nodes = np.invert(np.isin(graph_edge, node_id_black_list))

                # make a copy of the current neighbours' ids
                graph_edge_copy           = np.copy(graph_edge)
                graph_edge_weights_copy   = np.copy(graph_edges_weights[node_id])
                graph_edge_distances_copy = np.copy(graph_edges_distances[node_id])

                # set the neighbours' ids to -1
                graph_edges[node_id]           = -np.ones_like(graph_edge_copy)
                graph_edges_weights[node_id]   =  np.zeros_like(graph_edge_weights_copy)
                graph_edges_distances[node_id] =  np.zeros_like(graph_edge_distances_copy)

                count_valid_neighbours = 0
                for neighbor_idx, is_valid_neighbor in enumerate(valid_neighboring_nodes):
                    if is_valid_neighbor:
                        # current neighbor id
                        current_neighbor_id = graph_edge_copy[neighbor_idx]    

                        # get mapped neighbor id       
                        if current_neighbor_id == -1: mapped_neighbor_id = -1
                        else:                         mapped_neighbor_id = node_id_mapping[current_neighbor_id]    

                        graph_edges[node_id, count_valid_neighbours]           = mapped_neighbor_id
                        graph_edges_weights[node_id, count_valid_neighbours]   = graph_edge_weights_copy[neighbor_idx]
                        graph_edges_distances[node_id, count_valid_neighbours] = graph_edge_distances_copy[neighbor_idx]

                        count_valid_neighbours += 1

                # normalize edges' weights
                sum_weights = np.sum(graph_edges_weights[node_id])
                if sum_weights > 0:
                    graph_edges_weights[node_id] /= (sum_weights + 1e-6)

                # TODO: FIX if sum_weights == 0 
                # else:
                #     print("Hmmmmm", graph_edges_weights[node_id])
                #     raise Exception("Not good")



        # TODO: Check if some cluster is not present in reduced nodes. Then exit or raise error  

        reduced_graph_dict = {} # Store graph dict 

        reduced_graph_dict["all_nodes_at_source"]   = self.nodes.copy() # Not nessacary, copying just for safety
        reduced_graph_dict["valid_nodes_at_source"] = node_coords
        reduced_graph_dict["graph_edges"]           = graph_edges 
        reduced_graph_dict["graph_edges_weights"]   = graph_edges_weights 
        reduced_graph_dict["graph_edges_distances"] = graph_edges_distances                  
        reduced_graph_dict["graph_clusters"]        = graph_clusters                  

        reduced_graph_dict["num_nodes"]             = np.array(num_nodes, dtype=np.int64) # Number of nodes in this graph
        reduced_graph_dict["valid_nodes_mask"]         = valid_nodes_mask 
        # reduced_graph_dict["node_to_vertex_distances"] = node_to_vertex_distances # NxV geodesic distance between vertices of mesh and graph nodes

        return reduced_graph_dict

    def get_graph_pyramid(self):

        # Add asserts if any 

        return self.pyd

    def calculate_distance_matrix(self,X, Y):
        (N, D) = X.shape
        (M, D) = Y.shape
        XX = np.reshape(X, (N, 1, D))
        YY = np.reshape(Y, (1, M, D))
        XX = np.tile(XX, (1, M, 1))
        YY = np.tile(YY, (N, 1, 1))
        diff = XX - YY
        diff = np.linalg.norm(diff, axis=2)
        return diff

    def update(self,canonical_model_vertices,canonical_model_faces,new_verts_indices,plot_update=False):
        """
            Given vertices which are have outside the coverage of the graph. 
            Sample nodes and create edges betweem them and old graph

            @params: 
                canonical_model_vertices: (Vx3) np.ndarray (float32): Vertices of the updated canonical model
                canonical_model_faces:    (Fx3) np.ndarray (int32)  : Faces of the model     
                new_verts_indices: (Nx3) np.ndarry: New vertices indices from which nodes are sampled

            @returns: 
                update: bool: Whether new nodes were added to graph     
        """

        if len(new_verts_indices) == 0: 
            return False


        node_coverage = self.graph_generation_parameters["node_coverage"]

        # Sample nodes such that they are node_coverage apart from one another 
        new_node_indices = []
        for x in new_verts_indices:

            # Some vertex might erroneously get added if after tsdf update it goes outside node coverage. 
            # Hence make sure newly added node not already present in graph
            if np.min(np.linalg.norm(self.nodes - canonical_model_vertices[x], axis=1)) < node_coverage:
                continue

            if len(new_node_indices) == 0:
                new_node_indices.append(x)
                continue

            # If node already covered
            min_nodes_dist = np.min(np.linalg.norm(canonical_model_vertices[new_node_indices] - canonical_model_vertices[x], axis=1))
            if min_nodes_dist < node_coverage: # Not sure if correct
                continue

            self.log.debug(f"Node:{len(new_node_indices)},Vertex index:{x},Min dists:{min_nodes_dist}")

            new_node_indices.append(x)

        self.log.debug(f"New Node vertex indexes:{new_node_indices,len(new_node_indices)}")
        if len(new_node_indices) == 0: # No new nodes were added
            return False

        self.nodes = np.concatenate([self.nodes, canonical_model_vertices[new_node_indices]], axis=0)

        # Update self.node_indices using the new canonical model and old graph nodes 
        old_nodes_distance_matrix = self.calculate_distance_matrix(self.nodes,canonical_model_vertices)
        self.log.debug("New Node distance")
        self.node_indices = np.argmin(old_nodes_distance_matrix,axis=1)
        self.log.debug(f"Node Indices:{len(self.node_indices)}")
        # self.log.debug(f"{self.node_indices}")                  
        self.node_indices = self.node_indices[:,None] # Required in Nx1 format for getting reduced graph

        # Compute new graph edges 
        # (TODO very computationally expensive. 
        # Takes O(F + V*N + V*N), 
        #   calculating vertex neighbours, 
        #   finding new node indices for the updated canonical model, 
        #   using priority que for adding new graph edges)
        num_nodes = self.nodes.shape[0]
        num_vertices = canonical_model_vertices.shape[0]
        min_neighbours = self.graph_generation_parameters["min_neighbours"]
        num_neighbours = self.graph_generation_parameters["num_neighbours"]
        USE_ONLY_VALID_VERTICES = True
        ENFORCE_TOTAL_NUM_NEIGHBORS = True
        
        #########################################################################
        # Compute graph edges.
        #########################################################################
        # Compute edges between nodes.
        graph_edges              = -np.ones((num_nodes, num_neighbours), dtype=np.int32)
        graph_edges_weights      =  np.zeros((num_nodes, num_neighbours), dtype=np.float32)
        graph_edges_distances    =  np.zeros((num_nodes, num_neighbours), dtype=np.float32)
        node_to_vertex_distances = -np.ones((num_nodes, num_vertices), dtype=np.float32)

        visible_vertices = np.ones_like(canonical_model_vertices)

        compute_edges_geodesic_c(
            canonical_model_vertices, visible_vertices, canonical_model_faces, self.node_indices, 
            num_neighbours, node_coverage, 
            graph_edges, graph_edges_weights, graph_edges_distances,
            node_to_vertex_distances,
            USE_ONLY_VALID_VERTICES,
            ENFORCE_TOTAL_NUM_NEIGHBORS
        )

        self.vertices = canonical_model_vertices
        self.faces = canonical_model_faces

        # Save current results 
        self.edges           = graph_edges 
        self.edges_weights   = graph_edges_weights 
        self.edges_distances = graph_edges_distances
        self.clusters = -np.ones((graph_edges.shape[0], 1), dtype=np.int32) # Will be calculated later 

        # self.node_to_vertex_distances = node_to_vertex_distances # Not required except for edge calculatation    
        self.log.debug("Removing nodes without not enough neighbours")
        self.remove_nodes_with_not_enough_neighbours()

        #########################################################################
        # Compute clusters.
        #########################################################################
        self.compute_clusters()

        self.create_graph_pyramid()


        # I tried implementing the whole update thing. But the results are basically the same. 
        # Its better to use neural tracking setup. :) (This smile represents the time wasted on the whole thing)

        return True

    def save(self):
        #########################################################################
        # Save data.
        #########################################################################

        dst_graph_nodes_dir = os.path.join(self.savepath, "graph_nodes")
        if not os.path.exists(dst_graph_nodes_dir): os.makedirs(dst_graph_nodes_dir)

        dst_graph_edges_dir = os.path.join(self.savepath, "graph_edges")
        if not os.path.exists(dst_graph_edges_dir): os.makedirs(dst_graph_edges_dir)

        dst_graph_edges_weights_dir = os.path.join(self.savepath, "graph_edges_weights")
        if not os.path.exists(dst_graph_edges_weights_dir): os.makedirs(dst_graph_edges_weights_dir)

        dst_node_deformations_dir = os.path.join(self.savepath, "graph_node_deformations")
        if not os.path.exists(dst_node_deformations_dir): os.makedirs(dst_node_deformations_dir)

        dst_graph_clusters_dir = os.path.join(self.savepath, "graph_clusters")
        if not os.path.exists(dst_graph_clusters_dir): os.makedirs(dst_graph_clusters_dir)

        dst_pixel_anchors_dir = os.path.join(self.savepath, "pixel_anchors")
        if not os.path.exists(dst_pixel_anchors_dir): os.makedirs(dst_pixel_anchors_dir)

        dst_pixel_weights_dir = os.path.join(self.savepath, "pixel_weights")
        if not os.path.exists(dst_pixel_weights_dir): os.makedirs(dst_pixel_weights_dir)

        face_path_dir = os.path.join(self.savepath, "face_path")
        if not os.path.exists(face_path_dir): os.makedirs(face_path_dir)


        output_graph_nodes_path           = os.path.join(dst_graph_nodes_dir,        "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_graph_edges_path           = os.path.join(dst_graph_edges_dir,        "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_graph_edges_weights_path   = os.path.join(dst_graph_edges_weights_dir,"{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_graph_clusters_path        = os.path.join(dst_graph_clusters_dir,     "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_pixel_anchors_path         = os.path.join(dst_pixel_anchors_dir,      "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_pixel_weights_path         = os.path.join(dst_pixel_weights_dir,      "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))
        output_face_path                  = os.path.join(face_path_dir,              "{}_{:.2f}.bin".format("geodesic", self.graph_generation_parameters['node_coverage']))


        utils.save_graph_nodes(output_graph_nodes_path, self.nodes)
        utils.save_graph_edges(output_graph_edges_path, self.edges)
        utils.save_graph_edges_weights(output_graph_edges_weights_path, self.edges_weights)
        utils.save_graph_clusters(output_graph_clusters_path, self.clusters)

        # Not saved here. Look at Warpfield code
        # utils.save_int_image(output_pixel_anchors_path, self.pixel_anchors) 
        # utils.save_float_image(output_pixel_weights_path, self.pixel_weights)

        # Save faces 
        np.save(output_face_path,self.faces)

        self.log.info(f"Saving Graph data at:{self.savepath}")



    def load_graph_savepaths(self):    
        

        # Load all types of data avaible
        graph_dict = {}
        if os.path.isdir(os.path.join(self.opt.datadir,"graph_nodes")):
            for file in os.listdir(os.path.join(self.opt.datadir,"graph_nodes")):
            
                file_data = file[:-4].split('_')
                if len(file_data) == 4: # Using our setting frame_<frame_index>_geodesic_<node_coverage>.bin
                    frame_index = int(file_data[1])
                    node_coverage = float(file_data[-1])
                elif len(file_data) == 6: # Using name setting used by authors <random_str>_<Obj-Name>_<Source-Frame-Index>_<Target-Frame-Index>_geodesic_<Node-Coverage>.bin
                    frame_index = int(file_data[2])
                    node_coverage = float(file_data[-1])
                else:
                    raise NotImplementedError(f"Unable to understand file:{file} to get graph data")

                graph_dicts[frame_index] = {}
                graph_dicts[frame_index]["graph_nodes_path"]             = os.path.join(self.opt.datadir, "graph_nodes",        file)
                graph_dicts[frame_index]["graph_edges_path"]             = os.path.join(self.opt.datadir, "graph_edges",        file)
                graph_dicts[frame_index]["graph_edges_weights_path"]     = os.path.join(self.opt.datadir, "graph_edges_weights",file)
                graph_dicts[frame_index]["graph_clusters_path"]          = os.path.join(self.opt.datadir, "graph_clusters",     file)
                graph_dicts[frame_index]["pixel_anchors_path"]           = os.path.join(self.opt.datadir, "pixel_anchors",      file)
                graph_dicts[frame_index]["pixel_weights_path"]           = os.path.join(self.opt.datadir, "pixel_weights",      file)
                graph_dicts[frame_index]["node_coverage"]                = node_coverage

        self.graph_save_path = graph_dicts

    def load_graph(self,frame_index=0):

        self.load_graph_savepaths()

        self.node           =          utils.load_graph_nodes(self.graph_save_path[frame_index]["graph_nodes_path"])
        self.edges          =          utils.load_graph_edges(self.graph_save_path[frame_index]["graph_edges_path"])
        self.edges_weights  =  utils.load_graph_edges_weights(self.graph_save_path[frame_index]["graph_edges_weights_path"])
        self.clusters       =       utils.load_graph_clusters(self.graph_save_path[frame_index]["graph_clusters_path"])

    def get_graph_path(self,index):
        """
            This function returns the paths to the graph generated for a particular frame, and geodesic distance (required for sampling nodes, estimating edge weights etc.)
        """

        if index not in self.graph_dicts:
            self.graph_dicts[index] = create_graph_data_using_depth(\
                os.path.join(self.seq_dir,"depth",self.images_path[index].replace('jpg','png')),\
                max_triangle_distance=self.graph_generation_parameters['max_triangle_distance'],\
                erosion_num_iterations=self.graph_generation_parameters['erosion_num_iterations'],\
                erosion_min_neighbours=self.graph_generation_parameters['erosion_min_neighbours'],\
                node_coverage=self.graph_generation_parameters['node_coverage'],\
                require_mask=self.graph_generation_parameters['require_mask']
                )

        return self.graph_dicts[index]

    def get_graph(self,index,cropper):
        # Graph
        graph_path_dict = self.get_graph_path(index)

        graph_nodes, graph_edges, graph_edges_weights, _, graph_clusters, pixel_anchors, pixel_weights = dataset.DeformDataset.load_graph_data(
            graph_path_dict["graph_nodes_path"], graph_path_dict["graph_edges_path"], graph_path_dict["graph_edges_weights_path"], None, 
            graph_path_dict["graph_clusters_path"], graph_path_dict["pixel_anchors_path"], graph_path_dict["pixel_weights_path"], cropper
        )

        num_nodes = np.array(graph_nodes.shape[0], dtype=np.int64)  

        graph_dict = {}
        graph_dict["graph_nodes"]               = graph_nodes
        graph_dict["graph_edges"]               = graph_edges
        graph_dict["graph_edges_weights"]       = graph_edges_weights
        graph_dict["graph_clusters"]            = graph_clusters
        graph_dict["pixel_weights"]             = pixel_weights
        graph_dict["pixel_anchors"]             = pixel_anchors
        graph_dict["num_nodes"]                 = num_nodes
        graph_dict["node_coverage"]             = graph_path_dict["node_coverage"]
        graph_dict["graph_neighbours"]          = min(len(graph_nodes),4) 
        return graph_dict


    def load_dict(self,path):
        data = loadmat(path)
        model_data = {}
        for k in data: 
            if '__' in k:
                continue
            else:
                model_data[k] = np.array(data[k])
                if model_data[k].shape == (1,1):
                    model_data[k] = np.array(model_data[k][0,0])
        return model_data


if __name__ == "__main__":
    #########################################################################
    # Options
    #########################################################################
    # Depth-to-mesh conversion

    MAX_TRIANGLE_DISTANCE = 0.05

    # Erosion of vertices in the boundaries
    EROSION_NUM_ITERATIONS = 10
    EROSION_MIN_NEIGHBORS = 4

    # Node sampling and edges computation
    NODE_COVERAGE = 0.05 # in meters
    USE_ONLY_VALID_VERTICES = True
    NUM_NEIGHBORS = 8
    ENFORCE_TOTAL_NUM_NEIGHBORS = False
    SAMPLE_RANDOM_SHUFFLE = False

    # Pixel anchors
    NEIGHBORHOOD_DEPTH = 2

    MIN_CLUSTER_SIZE = 3
    MIN_NUM_NEIGHBORS = 2 

    # Node clean-up
    REMOVE_NODES_WITH_NOT_ENOUGH_NEIGHBORS = True






    

