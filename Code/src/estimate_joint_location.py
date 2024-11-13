import os 
import numpy as np
import torch 
import torch.optim as optim
import logging


class FindJointLocation: 
	def __init__(self):
		pass

		self.lambda_ccp = 0.8
		self.lambda_eucld = 0.2 
		self.lamda_smooth = 200
		self.lr = 0.1
		self.num_iter = 100
		self.device = torch.device("cuda:0")

		self.adam_gamma = 0.99 # Learning rate decay

		self.log = logging.getLogger(__name__)

	def run(self,vertices,vertex_normals,labels,euclidian_centers,skel_adj):

		joints = euclidian_centers.copy()

		J = euclidian_centers.shape[0]
		joint_degree = np.sum(skel_adj,axis=0)

		edges = np.array([ [i,j] for i in range(J) for j in range(J) if skel_adj[i,j]]) # Edges 

		# print(f"Edges:",edges)

		N = vertices.shape[0]
		per_joint_weight = np.ones(J)
		for j in range(J):
			if joint_degree[j] == 1:
			# inds = np.where(labels==j)[0]
			# per_joint_weight[j] = 2**joint_degree[j]
				per_joint_weight[j] = 2

		print(per_joint_weight)



		# Convert to torch 
		per_joint_weight = torch.from_numpy(per_joint_weight).to(self.device)
		vertices = torch.from_numpy(vertices).to(self.device)
		vertex_normals = torch.from_numpy(vertex_normals).to(self.device)
		labels = torch.from_numpy(labels).to(self.device)

		euclidian_centers = torch.from_numpy(euclidian_centers).to(self.device)
		edges = torch.from_numpy(edges).to(self.device)

		joints = torch.from_numpy(joints).to(self.device)
		joints = torch.nn.Parameter(joints)
		joints.requires_grad = True


		# self.lbfgs = optim.LBFGS([joints],
		# 	history_size=10, 
		# 	max_iter=4, 
		# 	line_search_fn="strong_wolfe")

		optimizer = optim.Adam([joints], lr= self.lr )
		scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.adam_gamma)

		for i in range(self.num_iter):

			# Cloest point projection cost 
			cross_product = torch.cross(vertices - joints[labels],vertex_normals,dim=1) 
		
			# print("Error Cpp:",error_cpp)
			error_cpp = torch.mean(per_joint_weight[labels]*cross_product.norm(dim=1))


			# print("Euclidian Cpp:",(joints - euclidian_centers).norm(dim=1))
			error_eucld = torch.mean( per_joint_weight[labels]*(joints[labels] - vertices).norm(dim=1) )	

			# print((joints[edges[:,0]] - joints[edges[:,1]]).norm(dim=1))

			error_smooth = torch.mean( (joints[edges[:,0]] - joints[edges[:,1]]).norm(dim=1) )*edges.shape[0]/N	



			loss = self.lambda_ccp*error_cpp + self.lambda_eucld*error_eucld + self.lamda_smooth*error_smooth

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()    



			lr = optimizer.param_groups[0]["lr"]
			# print((warped_pcd[landmarks[0]]-target_matches[landmarks[1]]).max())
			if i %10 == 0 or i == self.num_iter-1: 
				# print(joints.grad[32])
				self.log.info("\t-->Iteration: {0}. Lr:{1:.5f} Loss: ccp = {2:.3f}, eucl = {3:.6f}, smooth:{4:.6f} total = {5:.3f}".format(i, lr,error_cpp, error_eucld, error_smooth, loss.item()))
				# print(joints[32])


		return joints.cpu().data.numpy()