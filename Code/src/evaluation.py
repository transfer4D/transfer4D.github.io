# Metrics to evaluate with the ground truth (complete mesh provided by DeformingThings4D): 
import os
import numpy as np
import open3d as o3d
from fusion_tests.ssdr import SSDR
# CPU module imports
from pykdtree.kdtree import KDTree as KDTreeCPU

class Volumetric_Evaluation:

	def __init__(self,opt,vis):

		self.opt = opt
		self.vis = vis

		# Find if ground truth exist
		seq_dir = opt.datadir

		self.split, self.seq_name = list(filter(lambda x: x != '', seq_dir.split('/')))[-2:]
		# Check if anime file exists 
		gr_file_name = "_".join(self.seq_name.split('_')[:-1]) + '.anime'
		gr_file_name = os.path.join(seq_dir,gr_file_name)
		print("Ground truth filepath:",gr_file_name)

		self.can_evaluate = os.path.isfile(gr_file_name)

		if self.can_evaluate:
			# Use external intrisics matrix to transform here
			extrinsic_matrix = np.loadtxt(os.path.join(opt.datadir,"extrinsics.txt"))
			R = extrinsic_matrix[:3,:3]
			R_inv = R.T


			T = extrinsic_matrix[:3,3]


			T_inv = -R_inv@T

			# Manually set scale  
			if "dinoET" in gr_file_name:
				R_inv /= np.array([4.8,4.8,4.8]).reshape(3,1) # For Dino 
			

			ssdr = SSDR()	
			trajectory,faces = ssdr.load_anime_file(gr_file_name)
			print("Trajectory scale:",np.linalg.norm(trajectory[0],axis=1).mean())
			trajectory = trajectory@R_inv.T + T_inv
			print("Trajectory scale:",np.linalg.norm(trajectory[0],axis=1).mean())

			self.trajectory = trajectory

			# self.trajectory[frame_id] = trajectory[opt.source_frame]
			# print(np.linalg.norm(self.trajectory[frame_id],axis=1).mean())

			compete_mesh = o3d.geometry.TriangleMesh(
			o3d.utility.Vector3dVector(self.trajectory[0]),
			o3d.utility.Vector3iVector(faces))

			compete_mesh.compute_vertex_normals(normalized=True)
			self.vertex_normals = np.asarray(compete_mesh.vertex_normals)	

	# Currently we want to evaluate using canonical model 
	def point_to_point_dist(self,canocanical_verts,frame_id,visualize=False):	
		if not self.can_evaluate: return 0,0

		# print("canonical scale:",np.linalg.norm(canocanical_verts,axis=1).mean())
		ground_truth_kdtree = KDTreeCPU(self.trajectory[frame_id], leafsize=16)


		dist, anchors = ground_truth_kdtree.query(canocanical_verts, k=3)

		# print(dist)
		if visualize or dist.mean() > 1: 
			corresp = np.concatenate([ np.arange(anchors.shape[0],dtype=np.int).reshape(-1,1), anchors[:,0:1]],axis=1)
			# self.vis.plot_corresp(self.opt.source_frame,canocanical_verts,self.trajectory[frame_id],corresp,debug=True,savename="point-to-point-metric")

		return dist[:,0].mean(),dist[:,0].max()

	def point_to_plane_dist(self,canocanical_verts,frame_id,visualize=False):
		if not self.can_evaluate: return 0,0

		ground_truth_kdtree = KDTreeCPU(self.trajectory[frame_id], leafsize=16)


		dist, anchors = ground_truth_kdtree.query(canocanical_verts, k=3)
		
		if visualize: 
			corresp = np.concatenate([ np.arange(anchors.shape[0],dtype=np.int).reshape(-1,1), anchors[:,0:1]],axis=1)
			# self.vis.plot_corresp(self.opt.source_frame,canocanical_verts,self.trajectory[frame_id],corresp,debug=True,savename="point-to-point-metric")

		normals = self.vertex_normals[anchors[:,0]]

		point_to_plane = np.einsum('BN,BN->B',normals,canocanical_verts - self.trajectory[frame_id][anchors[:,0]])
		point_to_plane = np.abs(point_to_plane)
		# print("Point to plane:",point_to_plane.shape)
		# print(point_to_plane)

		return point_to_plane.mean(),point_to_plane.max()
