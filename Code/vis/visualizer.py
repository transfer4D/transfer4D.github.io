# Visualize & log results using plotly or open3d
import os
import pynput # To pause/control and registration

import numpy as np
import logging
import colorsys
import matplotlib.pyplot as plt

class Visualizer: # Base Class which contains all the preprocessing to visualize results
    def __init__(self,opt):

        self.opt = opt

        self.log = logging.getLogger(__name__)


        if hasattr(opt,"exp",) and hasattr(opt,"ablation"):
            self.savepath = os.path.join(self.opt.datadir,"results",self.opt.exp,self.opt.ablation)
            os.makedirs(self.savepath,exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"images"),exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"video"),exist_ok=True)
            os.makedirs(os.path.join(self.savepath,"target_pcd"),exist_ok=True)
        self.colors = np.array([])

    def get_viridis_color(self,vals,min_val=0,max_val=1):
        """
            get viridis colormap
        """

        vals = vals.astype(np.float32)
        vals = (vals -min_val)/(max_val - min_val) # Normalize 

        colors = plt.get_cmap('viridis')(vals)

        return colors[:,:3]


    def get_color_from_labels(self,labels):
        num_labels = int(np.unique(labels).max())+1

        if num_labels > self.colors.shape[0]:

            colors = []
            for i in range(num_labels):
                if i < 10:
                    colors.append(plt.get_cmap("tab10")(i)[:3])
                elif i < 22:
                    colors.append(plt.get_cmap("Paired")(i-10)[:3])
                elif i < 30:
                    colors.append(plt.get_cmap("Accent")(i-22)[:3])
                else:
                    colors.append(np.random.random(3))

            self.colors = np.array(colors)
        return self.colors[labels,:]

    def get_color_from_matrix(self,weight_matrix):
        # Given a weight matrix showing importance of each weight calculate color as weighted sum 
        num_labels = weight_matrix.shape[1]

        if num_labels > self.colors.shape[0]:

            colors = []
            for i in range(num_labels):
                # if i < 10:
                #     colors.append(plt.get_cmap("tab10")(i)[:3])
                # elif i < 22:
                #     colors.append(plt.get_cmap("Paired")(i-10)[:3])
                # elif i < 30:
                #     colors.append(plt.get_cmap("Accent")(i-22)[:3])
                # else:
                colors.append(np.random.random(3))

            self.colors = np.array(colors)
        return weight_matrix@self.colors[:num_labels,:]
    
    def get_color(self,color):
        if color is None:
            return None
        elif len(np.squeeze(color).shape) == 1:         
            color = color.astype(np.uint64)
            # Get color from a label using matplotlib
            return self.get_color_from_labels(color)
        else: 
            assert color.shape[-1] == 3 or color.shape[-1] == 4, f"Expected information in RGB(Nx3) or RGBA(Nx4) found:{color.shape}" 

            # Normalize colors here for plotting
            if color.max() > 1: color = color.astype(np.float64) / 255
            return color


    # Make functions defined in sub-classes based on method used 
    def plot_graph(self,color,title="Embedded Graph",debug=False):
        """
            @parama:
                color: Color of nodes (could be a label, rgb color)
                debug: bool: Stop program to show the plot
        """
        raise NotImplementedError(f"Function:{self.plot_graph.__name__} not implemented for vis type: {self.opt.visualizer}")


    def init_plot(self,debug=False):    
        """
            Plot the initial TSDF and graph used for registration
        """
        raise NotImplementedError(f"Function:{self.init_plot.__name__} not implemented for vis type: {self.opt.visualizer}")
 
    def plot_alignment(self,source_frame_data,\
            target_frame_data,graph_data,skin_data,\
            model_data):
        """
            Plot Alignment similiar to neural tracking

        """
        raise NotImplementedError(f"Function:{self.plot_alignment.__name__} not implemented for vis type: {self.opt.visualizer}")

    def show(self,matches=None,debug=True):
        """
            For visualizing the tsdf integration: 
            1. Source RGBD + Graph(visible nodes)   2. Target RGBD as Point Cloud 
            3. Canonical Model + Graph              3. Deformed Model   
        """
        raise NotImplementedError(f"Function:{self.show.__name__} not implemented for vis type: {self.opt.visualizer}")

    # Image analysis
    def save_image(self):
        pass
    def create_video(self,video_path="polyscopy_plot.mp4",start_frame=0,framerate=5):
        

        image_path = os.path.join(self.savepath,"images",f"subplot_\%d.png")
        video_path = os.path.join(self.savepath,"video", video_path)


        self.log.info(f"Getting images from:{image_path}")
        self.log.info(f"Generating video...\nSaved at:{video_path}")

        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} {video_path}")

        # Create a pallete for gifs 
        palette_path = os.path.join(self.savepath,"images","palette.png")
        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")

        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")
        os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")
 


    # Main plotting functions 
    def plot_eroded_mesh(self,vertices,faces,non_eroded_indices):
        pass
