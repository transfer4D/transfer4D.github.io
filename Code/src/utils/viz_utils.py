import os
import time
import copy
import open3d as o3d
import numpy as np

import utils.line_mesh as line_mesh_utils

def get_pcd(rgbd_image,max_points=1000000):
    #####################################################################################################
    # Prepare data
    # @pramas: rgbd_image => 6xHxW np.array (containing the rgb and point image)
    #          max_points => int (maximum number of points allowed)
    #####################################################################################################
    
    rgbd_flat = np.moveaxis(rgbd_image, 0, -1).reshape(-1, 6)
    # rgbd_points = transform_pointcloud_to_opengl_coords(rgbd_flat[..., 3:])
    rgbd_points = rgbd_flat[..., 3:]
    rgbd_colors = rgbd_flat[..., :3]

    if len(rgbd_points) > max_points: # Downsample 
        print(f"Exceeded number of points:{len(rgbd_points)} reducing to:{max_points}")
        show_indices = np.random.choice(len(rgbd_points),size=max_points,replace=False)
        rgbd_points = rgbd_points[show_indices]
        rgbd_colors = rgbd_colors[show_indices]

    rgbd_pcd = o3d.geometry.PointCloud()
    rgbd_pcd.points = o3d.utility.Vector3dVector(rgbd_points)
    rgbd_pcd.colors = o3d.utility.Vector3dVector(rgbd_colors)
    return rgbd_pcd


def create_open3d_graph(graph_nodes,graph_edges,color=None):
    # Transform to OpenGL coords
    # graph_nodes = transform_pointcloud_to_opengl_coords(graph_nodes)

    size = max(1,np.linalg.norm(np.max(graph_nodes)-np.min(graph_nodes)))
    
    # Graph nodes
    rendered_graph_nodes = []
    for i,node in enumerate(graph_nodes):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004*size)
        mesh_sphere.compute_vertex_normals()

        

        # Color Nodes  
        if color is None: 
            ncolor = [1.0,0.0,0.0]
        else: 
            if i >= len(color):
                ncolor = [1.0,0.0,0.0]
            else:    
                ncolor = color[i]
        mesh_sphere.paint_uniform_color(ncolor)

        mesh_sphere.translate(node)
        rendered_graph_nodes.append(mesh_sphere)
    
    # Merge all different sphere meshes
    rendered_graph_nodes = merge_meshes(rendered_graph_nodes)

    # Graph edges
    if len(graph_edges) == 1: 
        line_mesh_geoms = o3d.geometry.TriangleMesh()  
    else:
        edges_pairs = []
        for node_id, edges in enumerate(graph_edges):
            for neighbor_id in edges:
                if neighbor_id == -1:
                    break
                edges_pairs.append([node_id, neighbor_id])    

        colors = [[0.0, 0.0, 0.0] for i in range(len(edges_pairs))]
        line_mesh = line_mesh_utils.LineMesh(graph_nodes, edges_pairs, colors, radius=0.0006*size)
        line_mesh_geoms = line_mesh.cylinder_segments
        # Merge all different line meshes
        line_mesh_geoms = merge_meshes(line_mesh_geoms)


    # Combined nodes & edges
    rendered_graph = [rendered_graph_nodes, line_mesh_geoms]
    return rendered_graph

def create_matches_lines(match_mask,high_color,low_color,source_pcd,target_matches,mask_pred_flat,weight_thr,weight_scale):

    match_source_points_corresp  = np.asarray(source_pcd.points)[match_mask]
    match_target_matches_corresp = target_matches[match_mask]
    match_mask_pred              = mask_pred_flat[match_mask]
    # number of match matches
    n_match_matches = match_source_points_corresp.shape[0]
    
    # Subsample if too many lines
    N = 2000
    subsample = N < n_match_matches
    if subsample:
        sampled_idxs = np.random.permutation(n_match_matches)[:N]
        match_source_points_corresp  = match_source_points_corresp[sampled_idxs]
        match_target_matches_corresp = match_target_matches_corresp[sampled_idxs]
        match_mask_pred              = match_mask_pred[sampled_idxs]
        n_match_matches = N
    # both match_source and match_target points together into one vector
    match_matches_points = np.concatenate([match_source_points_corresp, match_target_matches_corresp], axis=0)
    match_matches_lines = [[i, i + n_match_matches] for i in range(0, n_match_matches, 1)]

    # --> Create match (unweighted) lines 
    match_matches_colors = [[201/255, 177/255, 14/255] for i in range(len(match_matches_lines))]
    match_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(match_matches_points),
        lines=o3d.utility.Vector2iVector(match_matches_lines),
    )
    match_matches_set.colors = o3d.utility.Vector3dVector(match_matches_colors)

    # --> Create match weighted lines 
    match_weighted_matches_colors = np.ones_like(match_source_points_corresp)

    weights_normalized = np.maximum(np.minimum(0.5 + (match_mask_pred - weight_thr) / weight_scale, 1.0), 0.0)
    weights_normalized_opposite = 1 - weights_normalized

    match_weighted_matches_colors[:, 0] = weights_normalized * high_color[0] + weights_normalized_opposite * low_color[0]
    match_weighted_matches_colors[:, 1] = weights_normalized * high_color[1] + weights_normalized_opposite * low_color[1]
    match_weighted_matches_colors[:, 2] = weights_normalized * high_color[2] + weights_normalized_opposite * low_color[2]

    match_weighted_matches_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(match_matches_points),
        lines=o3d.utility.Vector2iVector(match_matches_lines),
    )
    match_weighted_matches_set.colors = o3d.utility.Vector3dVector(match_weighted_matches_colors)

    return match_matches_set,match_weighted_matches_set



def transform_pointcloud_to_opengl_coords(points_cv):
    assert len(points_cv.shape) == 2 and points_cv.shape[1] == 3

    T_opengl_cv = np.array(
        [[1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0]]
    )

    # apply 180deg rotation around 'x' axis to transform the mesh into OpenGL coordinates
    point_opengl = np.matmul(points_cv, T_opengl_cv.transpose())

    return point_opengl


class CustomDrawGeometryWithKeyCallback():
    def __init__(self, geometry_dict, alignment_dict, corresp_set):
        self.added_source_pcd = True
        self.added_source_obj = False
        self.added_target_pcd = True
        self.added_graph = True

        self.added_both = False

        self.added_corresp = False
        self.added_weighted_corresp = True

        self.aligned = False

        self.rotating = False
        self.stop_rotating = False
        
        self.source_pcd = geometry_dict["source_pcd"] 
        self.source_obj = geometry_dict["source_obj"] 
        self.target_pcd = geometry_dict["target_pcd"] 
        self.graph      = geometry_dict["graph"] 

        # align source to target
        self.valid_source_points_cached = alignment_dict["valid_source_points"]
        self.valid_source_colors_cached = copy.deepcopy(self.source_obj.colors)
        self.line_segments_unit_cached  = alignment_dict["line_segments_unit"]
        self.line_lengths_cached        = alignment_dict["line_lengths"]

        self.good_matches_set          = corresp_set["good_matches_set"]
        self.good_weighted_matches_set = corresp_set["good_weighted_matches_set"]
        self.bad_matches_set           = corresp_set["bad_matches_set"]
        self.bad_weighted_matches_set  = corresp_set["bad_weighted_matches_set"]

        # correspondences lines
        self.corresp_set = corresp_set

    def remove_both_pcd_and_object(self, vis, ref):
        if ref == "source":
            if self.added_source_pcd:
                vis.remove_geometry(self.source_pcd)
                self.added_source_pcd = False

            if self.added_source_obj:
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False
        elif ref == "target":
            if self.added_target_pcd:
                vis.remove_geometry(self.target_pcd)
                self.added_target_pcd = False

    def clear_for(self, vis, ref):
        if self.added_both:
            if ref == "target":
                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False
            
            self.added_both = False

    def get_name_of_object_to_record(self):
        if self.added_both:
            return "both"

        if self.added_graph:
            if self.added_source_pcd:
                return "source_pcd_with_graph"
            if self.added_source_obj:
                return "source_obj_with_graph"
        else:
            if self.added_source_pcd:
                return "source_pcd"
            if self.added_source_obj:
                return "source_obj"
                
        if self.added_target_pcd:
            return "target_pcd"
    
    def custom_draw_geometry_with_key_callback(self):

        def toggle_graph(vis):
            if self.graph is None:
                print("You didn't pass me a graph!")
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_graph:
                for g in self.graph:
                    vis.remove_geometry(g)
                self.added_graph = False
            else:
                for g in self.graph:
                    vis.add_geometry(g)
                self.added_graph = True
            
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_obj(vis):
            print("::toggle_obj")

            if self.added_both:
                print("-- will not toggle obj. First, press either S or T, for source or target pcd")
                return False
            
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            # Add source obj
            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False
            # Add source pcd
            elif self.added_source_obj:
                vis.add_geometry(self.source_pcd)
                vis.remove_geometry(self.source_obj)
                self.added_source_pcd = True
                self.added_source_obj = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_source(vis):

            self.clear_for(vis, "source")
            
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if not self.added_source_pcd:
                print("::Added Source")
                vis.add_geometry(self.source_pcd)
                self.added_source_pcd = True

            if self.added_source_pcd:
                print("::Removed Source")
                vis.remove_geometry(self.source_pcd)
                self.added_source_pcd = False


            if self.added_source_obj:

                vis.remove_geometry(self.source_obj)
                self.added_source_obj = False

            self.remove_both_pcd_and_object(vis, "target")

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_target(vis):
            print("::view_target")

            self.clear_for(vis, "target")
            
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_target_pcd:
                vis.remove_geometry(self.target_pcd)
                self.added_target_pcd = False

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True


            self.remove_both_pcd_and_object(vis, "source")
            
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def view_both(vis):
            print("::view_both")
            
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            
            if self.added_source_pcd:
                vis.add_geometry(self.source_obj)
                vis.remove_geometry(self.source_pcd)
                self.added_source_obj = True
                self.added_source_pcd = False

            if not self.added_source_obj:
                vis.add_geometry(self.source_obj)
                self.added_source_obj = True

            if not self.added_target_pcd:
                vis.add_geometry(self.target_pcd)
                self.added_target_pcd = True

            self.added_both = True

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

        def rotate(vis):
            print("::rotate")

            self.rotating = True
            self.stop_rotating = False

            total = 2094
            speed = 5.0
            n = int(total / speed)
            for _ in range(n):
                ctr = vis.get_view_control()

                if self.stop_rotating:
                    return False

                ctr.rotate(speed, 0.0)
                vis.poll_events()
                vis.update_renderer()

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()
            
            return False

        def rotate_slightly_left_and_right(vis):
            print("::rotate_slightly_left_and_right")

            self.rotating = True
            self.stop_rotating = False

            if self.added_both and not self.aligned:
                moves = ['lo', 'lo', 'pitch_f', 'pitch_b', 'ri', 'ro']
                totals = [(2094/4)/2, (2094/4)/2, 2094/4, 2094/4, 2094/4, 2094/4]
                abs_speed = 5.0
                abs_zoom = 0.15
                abs_pitch = 5.0
                iters_to_move = [range(int(t / abs_speed)) for t in totals]
                stop_at = [True, True, True, True, False, False]
            else:
                moves = ['ro', 'li']
                total = 2094 / 4
                abs_speed = 5.0
                abs_zoom = 0.03
                iters_to_move = [range(int(total / abs_speed)), range(int(total / abs_speed))]
                stop_at = [True, False]

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed
                
                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom / 4.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom
                    
                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)

                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        toggle_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    if move_idx == 1:
                        toggle_weighted_correspondences(vis)
                        vis.poll_events()
                        vis.update_renderer()

                    time.sleep(0.5)

            ctr.rotate(0.4, 0.0)
            vis.poll_events()
            vis.update_renderer()
            
            return False

        def align(vis):
            if not self.added_both:
                return False

            moves = ['lo', 'li', 'ro']
            totals = [(2094/4)/2, (2094/4), 2094/4 + (2094/4)/2]
            abs_speed = 5.0
            abs_zoom = 0.15
            abs_pitch = 5.0
            iters_to_move = [range(int(t / abs_speed)) for t in totals]
            stop_at = [True, True, True]

            # Paint source object yellow, to differentiate target and source better
            # self.source_obj.paint_uniform_color([0, 0.8, 0.506])
            self.source_obj.paint_uniform_color([1, 0.706, 0])
            vis.update_geometry(self.source_obj)
            vis.poll_events()
            vis.update_renderer()

            for move_idx, move in enumerate(moves):
                if move == 'l' or move == 'lo' or move == 'li':
                    h_speed = abs_speed
                elif move == 'r' or move == 'ro' or move == 'ri':
                    h_speed = -abs_speed
                
                if move == 'lo':
                    zoom_speed = abs_zoom
                elif move == 'ro':
                    zoom_speed = abs_zoom/50.0
                elif move == 'li' or move == 'ri':
                    zoom_speed = -abs_zoom/5.0
                    
                for _ in iters_to_move[move_idx]:
                    ctr = vis.get_view_control()

                    if self.stop_rotating:
                        return False

                    if move == "pitch_f":
                        ctr.rotate(0.0, abs_pitch)
                        ctr.scale(-zoom_speed/2.0)
                    elif move == "pitch_b":
                        ctr.rotate(0.0, -abs_pitch)
                        ctr.scale(zoom_speed/2.0)
                    else:
                        ctr.rotate(h_speed, 0.0)
                        if move == 'lo' or move == 'ro' or move == 'li' or move == 'ri':
                            ctr.scale(zoom_speed)
                    
                    vis.poll_events()
                    vis.update_renderer()

                # Minor halt before changing direction
                if stop_at[move_idx]:
                    time.sleep(0.5)

                    if move_idx == 0:
                        n_iter = 125
                        for align_iter in range(n_iter+1):
                            p = float(align_iter) / n_iter
                            self.source_obj.points = o3d.utility.Vector3dVector( self.valid_source_points_cached + self.line_segments_unit_cached * self.line_lengths_cached * p )
                            vis.update_geometry(self.source_obj)
                            vis.poll_events()
                            vis.update_renderer()

                    time.sleep(0.5)

            self.aligned = True

            return False

        def toggle_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            
            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False
            else:
                vis.add_geometry(self.good_matches_set)
                vis.add_geometry(self.bad_matches_set)
                self.added_corresp = True

            # also remove weighted corresp
            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def toggle_weighted_correspondences(vis):
            if not self.added_both:
                return False

            param = vis.get_view_control().convert_to_pinhole_camera_parameters()

            if self.added_weighted_corresp:
                vis.remove_geometry(self.good_weighted_matches_set)
                vis.remove_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = False
            else:
                vis.add_geometry(self.good_weighted_matches_set)
                vis.add_geometry(self.bad_weighted_matches_set)
                self.added_weighted_corresp = True

            # also remove (unweighted) corresp
            if self.added_corresp:
                vis.remove_geometry(self.good_matches_set)
                vis.remove_geometry(self.bad_matches_set)
                self.added_corresp = False

            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(param)

            return False

        def reload_source_object(vis):
            self.source_obj.points = o3d.utility.Vector3dVector(self.valid_source_points_cached)
            self.source_obj.colors = self.valid_source_colors_cached

            if self.added_both:
                vis.update_geometry(self.source_obj)
                vis.poll_events()
                vis.update_renderer()

        key_to_callback = {}
        key_to_callback[ord("G")] = toggle_graph
        key_to_callback[ord("S")] = view_source
        key_to_callback[ord("T")] = view_target
        key_to_callback[ord("O")] = toggle_obj
        key_to_callback[ord("B")] = view_both
        key_to_callback[ord("C")] = toggle_correspondences
        key_to_callback[ord("W")] = toggle_weighted_correspondences
        key_to_callback[ord(",")] = rotate
        key_to_callback[ord(";")] = rotate_slightly_left_and_right
        key_to_callback[ord("A")] = align
        key_to_callback[ord("Z")] = reload_source_object

        # vis = o3d.visualization.VisualizerWithKeyCallback()
        # vis.create_window()



        o3d.visualization.draw_geometries_with_key_callbacks([self.source_pcd,self.target_pcd,self.good_weighted_matches_set,self.bad_weighted_matches_set] + [g for g in self.graph], key_to_callback)


def merge_meshes(meshes):
    # Compute total number of vertices and faces.
    num_vertices = 0
    num_triangles = 0
    num_vertex_colors = 0
    for i in range(len(meshes)):
        num_vertices += np.asarray(meshes[i].vertices).shape[0]
        num_triangles += np.asarray(meshes[i].triangles).shape[0]
        num_vertex_colors += np.asarray(meshes[i].vertex_colors).shape[0]

    # Merge vertices and faces.
    vertices = np.zeros((num_vertices, 3), dtype=np.float64)
    triangles = np.zeros((num_triangles, 3), dtype=np.int32)
    vertex_colors = np.zeros((num_vertex_colors, 3), dtype=np.float64)

    vertex_offset = 0
    triangle_offset = 0
    vertex_color_offset = 0
    for i in range(len(meshes)):
        current_vertices = np.asarray(meshes[i].vertices)
        current_triangles = np.asarray(meshes[i].triangles)
        current_vertex_colors = np.asarray(meshes[i].vertex_colors)

        vertices[vertex_offset:vertex_offset + current_vertices.shape[0]] = current_vertices
        triangles[triangle_offset:triangle_offset + current_triangles.shape[0]] = current_triangles + vertex_offset
        vertex_colors[vertex_color_offset:vertex_color_offset + current_vertex_colors.shape[0]] = current_vertex_colors

        vertex_offset += current_vertices.shape[0]
        triangle_offset += current_triangles.shape[0]
        vertex_color_offset += current_vertex_colors.shape[0]

    # Create a merged mesh object.
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    mesh.paint_uniform_color([1, 0, 0])
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh