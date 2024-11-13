import os
import torch
import numpy as np
from motion_model import MotionCompleteNet

class MotionCompleteNet_Runner:
    def __init__(self, fopt):

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

        checkpoint_path = '../checkpoints/model_noise_all.tar'

        model = MotionCompleteNet().to(self.device)
        torch_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(torch_checkpoint['model_state_dict'])
        model.eval()


        self.model = model

        self.historical_motion = None
        self.historical_max_len = 16
        self.std_curr = None
        self.std_prev = None
        self.rigid_motion_curr = None

        self.fopt = fopt

        self.savepath = os.path.join(self.fopt.datadir,"results",self.fopt.exp,self.fopt.ablation)

        os.makedirs(self.savepath,exist_ok=True)

    @staticmethod    
    def rigid_icp(pc0, pc1):
        c0 = np.mean(pc0, axis=0)
        c1 = np.mean(pc1, axis=0)
        H = (pc0 - c0).transpose() @ (pc1 - c1)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        t = c1 - R @ c0
        return R, t



    def preprocess(self, node_pos,node_motion,visible):
        """
           Preprocess data as input for MotionCompleteNet

           @params: 
                node_pos: Positions of nodes at source 
                node_motion: Displacement to move visible nodes from source to target
                visible: Boolean mask to show nodes visisble during registration      
        """

        node_num_l0 = node_pos.shape[0]

        # extract rigid motion
        rigid_R, rigid_t = self.rigid_icp(node_pos[visible, :], node_pos[visible, :] + node_motion[visible, :])
        self.rigid_motion_curr = np.dot(node_pos, rigid_R.transpose()) + rigid_t - node_pos
        nonrigid_motion = node_motion - self.rigid_motion_curr

        curr_motion = np.zeros(shape=(node_num_l0, 4))
        # motion in centimeter
        curr_motion[visible, :3] = nonrigid_motion[visible, :] * 100.0

        # normalize the motion
        self.curr_std = np.mean(np.std(curr_motion[visible, :3], axis=0)) + 0.1
        curr_motion[visible, :3] = curr_motion[visible, :3] / self.curr_std
        curr_motion[:, -1] = visible

        # init the mu of new nodes as 0.0, and the sigma of new nodes as a larger value (1.0)
        prev_motion = np.zeros(shape=(node_num_l0, 4))
        prev_motion[:, -1] = 1.0

        # for the first frame, set historical motion
        # using node position change between consequent frames as historical motion
        if self.frame_id > self.fopt.source_frame:
            node_pos_prev = np.load(os.path.join(self.savepath,"deformed_nodes", f'{self.frame_id}.npy'))
            visible_prev = np.load(os.path.join(self.savepath,"visible_nodes", f'{self.frame_id - self.fopt.skip_rate}.npy'))
            prev_node_num = node_pos_prev.shape[0]

            # node num of current frame could be larger than the previous frame, and new nodes will be add to the end of the node array
            node_motion_prev = node_pos[:node_pos_prev.shape[0]] - node_pos_prev

            print(f"Prev Frame id:{self.frame_id - self.fopt.skip_rate}",node_motion_prev.shape,visible_prev.shape)

            rigid_R, rigid_t = self.rigid_icp(node_pos_prev[visible_prev, :], node_pos_prev[visible_prev, :] + node_motion_prev[visible_prev, :])
            rigid_motion_prev = np.dot(node_pos_prev, rigid_R.transpose()) + rigid_t - node_pos_prev
            prev_motion[:prev_node_num, :3] = (node_motion_prev - rigid_motion_prev) * 100.0

        if self.historical_motion is None:
            self.historical_motion = np.zeros(shape=(1, node_num_l0, 4))
        else:
            seq_len = self.historical_motion.shape[0]
            prev_node_num = self.historical_motion.shape[1]
            drop = (seq_len == self.historical_max_len) * 1
            seq_len = min(seq_len + 1, self.historical_max_len)
            temp = np.zeros(shape=(seq_len, node_num_l0, 4))
            temp[:-1, :prev_node_num, :] = self.historical_motion[drop:, :, :] * self.std_prev / self.curr_std
            temp[-1, :prev_node_num, :] = prev_motion[:prev_node_num, :] / self.curr_std
            self.historical_motion = temp

        self.std_prev = self.curr_std

        node_pos = node_pos - np.mean(node_pos, axis=0)

        node_pos_torch = torch.from_numpy(node_pos.astype(np.float32)).to(self.device)
        curr_motion_torch = torch.from_numpy(curr_motion.astype(np.float32)).to(self.device)
        historical_motion_torch = torch.from_numpy(self.historical_motion.astype(np.float32)).to(self.device)

        return node_pos_torch, curr_motion_torch,historical_motion_torch
    
    def get_graph_data(self):    

        assert hasattr(self,"graph"), "Add graph to MotionCompleteNet_Runner first"

        pyd = self.graph.get_graph_pyramid()

        down_sample_idx1 = pyd['down_sample_idx1']
        down_sample_idx2 = pyd['down_sample_idx2']
        down_sample_idx3 = pyd['down_sample_idx3']

        down_sample_idx1 = torch.from_numpy(np.array(down_sample_idx1).astype(np.int64)).to(self.device)
        down_sample_idx2 = torch.from_numpy(np.array(down_sample_idx2).astype(np.int64)).to(self.device)
        down_sample_idx3 = torch.from_numpy(np.array(down_sample_idx3).astype(np.int64)).to(self.device)

        up_sample_idx1 = pyd['up_sample_idx1']
        up_sample_idx2 = pyd['up_sample_idx2']
        up_sample_idx3 = pyd['up_sample_idx3']
        up_sample_idx1 = torch.from_numpy(np.array(up_sample_idx1).astype(np.int64)).to(self.device)
        up_sample_idx2 = torch.from_numpy(np.array(up_sample_idx2).astype(np.int64)).to(self.device)
        up_sample_idx3 = torch.from_numpy(np.array(up_sample_idx3).astype(np.int64)).to(self.device)

        edge_index_list = []
        for nn_index in ['nn_index_l0','nn_index_l1','nn_index_l2','nn_index_l3']:
            
            graph_edges = pyd[nn_index]
            node_num, num_neighbors = graph_edges.shape
            node_ids = np.tile(np.arange(node_num, dtype=np.int32)[:,None],(1, num_neighbors)) # (opt_num_nodes_i, num_neighbors)
            graph_edge_pairs = np.concatenate([node_ids.reshape(1,-1,num_neighbors), graph_edges.reshape(1,-1,num_neighbors)], 0) # (opt_num_nodes_i, num_neighbors, 2)

            valid_edges = graph_edges >= 0
            valid_edge_idxs = np.where(valid_edges)
            edge_index = graph_edge_pairs[:,valid_edge_idxs[0], valid_edge_idxs[1]]
                
            edge_index = torch.from_numpy(edge_index).type(torch.LongTensor).to(self.device)
            
            edge_index_list.append(edge_index)

        return edge_index_list, \
               [down_sample_idx1, down_sample_idx2, down_sample_idx3], \
               [up_sample_idx1, up_sample_idx2, up_sample_idx3]

    def postprocess(self,outputs):
        """
            Get the output from MotionCompleteNet and post process for future use
        """               
        outputs = outputs.detach().cpu().numpy()
        mu = outputs[:, :3]
        sigma = outputs[:, -1]

        # eq.7 in the paper
        motion_scale = np.sqrt(np.sum(np.square(mu), axis=1))
        confidence = np.exp(-0.5*4 * np.square(sigma / (motion_scale + 1.0)))

        mu = mu * self.curr_std
        sigma = sigma * self.curr_std

        pred_motion = mu / 100.0
        node_motion = pred_motion + self.rigid_motion_curr

        return node_motion,confidence

    def __call__(self,frame_id,source_node_positions,deformed_nodes_position,visible_node_mask):
        """
            Runs MotionCompleteNet from OcclusionFusion 

            @params:
                frame_id: Source frame id used during optical flow estimation
                source_node_positions: Positions of nodes at source frame 
                deformed_nodes_position: Deformed position of nodes at the target frame
                visible_node_mask: Nodes visible during registration  

        """
        self.frame_id = frame_id
        node_motion = deformed_nodes_position - source_node_positions

        node_pos, curr_motion, historical_motion = self.preprocess(source_node_positions,node_motion,visible_node_mask)        
        edge_indices, down_sample_indices, up_sample_indices = self.get_graph_data()

        with torch.no_grad():    
            outputs = self.model(node_pos, curr_motion, historical_motion, edge_indices,down_sample_indices, up_sample_indices)
    
        node_motion,confidence = self.postprocess(outputs)    

        return node_motion, confidence



