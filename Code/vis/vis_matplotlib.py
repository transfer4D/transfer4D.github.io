# Python Imports
import os
import numpy as np
import open3d as o3d

# import tkinter
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# Nueral Tracking Modules
from utils import image_proc

# Fusion Modules
from .visualizer import Visualizer

class VisualizeMatplotlib(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)
		self.fig = plt.figure(figsize=(16,4))
		
	# Todo needs to updated for arbitary subplots	
	def create_fig3D(self,azim=0,elev=-90,title="Matplotlib plot"):	
		self.ax = self.fig.add_subplot(111, projection='3d')
		self.title = title

		# Viewpoint 
		self.azim = azim
		self.elev = elev
		self.ax.view_init(azim=self.azim, elev=self.elev)
		
		# Mouse-click to update
		self.fig.canvas.mpl_connect('button_press_event', self.mouse_click)

	
	# Register mouse events
	def mouse_click(self,event):
		self.azim = self.ax.azim
		self.elev = self.ax.elev	

	def plot_tsdf_deformation(self):
		"""
			Plot tsdf deformation.
			Used for testing deformation 
		"""	

		assert hasattr(self,"fig"), "Matplotlib Figure not created. Run vis.create_fig() first"

		plt.cla()


		assert hasattr(self,"tsdf") and hasattr(self,"graph") and hasattr(self,"warpfield")	

		new_world_points,valid_points = self.warpfield.deform_tsdf()
		# Plot results

		self.ax.axis('off')
		self.ax.scatter(new_world_points[:, 0], new_world_points[:, 1], new_world_points[:, 2], marker='s', s=500,alpha=0.05)
		self.ax.scatter(self.warpfield.deformed_nodes[:, 0], self.warpfield.deformed_nodes[:, 1], self.warpfield.deformed_nodes[:, 2], s=100, c="green",alpha=1)
		self.ax.scatter(self.tsdf.world_pts[:, 0], self.tsdf.world_pts[:, 1], self.tsdf.world_pts[:, 2], marker='s', s=500, c="red",alpha=0.05)

		self.ax.set_title(self.title)

	def set_ax_lim(self,lim_min=-10,lim_max=10):
		
		if type(lim_min) == int or type(lim_min) == float: 
			lim_min = [lim_min]

		if type(lim_max) == int or type(lim_max) == float: 
			lim_max = [lim_max]

		assert (len(lim_min) == 1  and len(lim_max) == 1) or (len(lim_min) == 3  and len(lim_max) == 3), f"Axis limits must have length 1 or 3, found:{lim_min},{lim_max}"	

		if len(lim_min) == 1:
			lim_min = [lim_min[0]]*3

		if len(lim_max) == 1:
			lim_max = [lim_max[0]]*3


		assert hasattr(self,"fig"), "Matplotlib Figure not created. Run vis.create_fig() first"

		self.ax.set_xlim([lim_min[0], lim_max[0]])
		self.ax.set_ylim([lim_min[1], lim_max[1]])
		self.ax.set_zlim([lim_min[2], lim_max[2]])

	def show(self):
		plt.show()

	def draw(self,pause=0.1):	
		plt.draw()
		plt.pause(pause)	
	
	def plot_input_sequence(self,framerate = 15):
		
		files = sorted(os.listdir(os.path.join(self.opt.datadir,"depth")),key=lambda x: int(x.split('.')[0]))
		img_list = []
		for file in files:
			img_list.append(plt.imread(os.path.join(self.opt.datadir,"depth",file)))

		img_list = np.array(img_list)
		


		valid_depth = img_list>0

		img_list_max = np.max(1/img_list[valid_depth])
		img_list_min = np.min(1/img_list[valid_depth])


		img_list[valid_depth] = 1/img_list[valid_depth]/img_list_max

		for i,im in enumerate(img_list):
			H,W = im.shape
			save_im = self.get_viridis_color(im.reshape(-1)).reshape((H,W,3))
			save_im[im==0,:] = 1
			plt.imsave(os.path.join(self.savepath,"images",f"depth_{i}.png"),save_im)

		image_path = os.path.join(self.savepath,"images",f"depth_\%d.png")
		video_path = os.path.join(self.savepath,"video","depth.mp4")
		palette_path = os.path.join(self.savepath,"images","palette_depth.png")
		
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -vf palettegen {palette_path}")
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path.replace('mp4','gif')}")		
		os.system(f"ffmpeg -y -framerate {framerate} -i {image_path} -i {palette_path} -lavfi paletteuse {video_path}")		



	def plot_depth_images(self,depth_image_list,savename=None):
		plt.cla()	
		self.fig = plt.figure(figsize=(16,4))
		title_list = ["Warped Mask","Warped Depth", "Target Mask", "Target Depth", "Depth Error(MSE)"]

		for i,im in enumerate(depth_image_list):
			# print(100 + (i+1)*10 + len(depth_image_list))	
			ax = self.fig.add_subplot(100 + 10*len(depth_image_list) + i+1)

			im = im.detach().cpu().numpy()
			if im.dtype == np.float32 and im.max() > 1:
				im /= im.max()
			if im.shape[-1] > 3:
				im = im[...,0]
			# print(im.shape)
			ax.imshow(im)
			ax.set_title(title_list[i])
			ax.axis('off')
		plt.axis('off')
		plt.draw()

		if savename is not None:
			plt.savefig(os.path.join(self.savepath,"optimization_convergence_info",savename))

		# plt.pause(0.1)
		plt.close()

	def plot_tsdf_volume(self,tsdf_vol):

		tsdf_vol = (tsdf_vol + 1)/(1+1) # Normalize from -1 -> 1 to 0 -> 1

		W,H,D = tsdf_vol.shape

		print(tsdf_vol.shape)

		for i in range(max(W,H,D)):
			
			plt.cla()
			self.fig = plt.figure(figsize=(16,16))
			ax_x = self.fig.add_subplot(1,3,1) # X-axis
			ax_y = self.fig.add_subplot(1,3,2) # Y-axis
			ax_z = self.fig.add_subplot(1,3,3) # Z-axis

			ax_x.imshow(tsdf_vol[i%W].T,vmin=0, vmax=1)
			ax_x.set_title(f"Slice Y-Z plane:{i%W}") 
			ax_x.axis('off')

			ax_y.imshow(tsdf_vol[:,i%H].T,vmin=0, vmax=1) 
			ax_y.set_title(f"Slice X-Z plane/:{i%H}") 
			ax_y.axis('off')

			ax_z.imshow(tsdf_vol[:,:,i%D].T,vmin=0, vmax=1)
			ax_z.set_title(f"Slice X-Y plane:{i%D}")  
			ax_z.axis('off')

			print(i,tsdf_vol[i%W].shape,tsdf_vol[:,i%H].shape,tsdf_vol[:,:,i%D].shape)


			plt.axis('off')

			plt.draw()
			plt.savefig(os.path.join(self.savepath,"images",f"volume_rendering_{i}.png"))

			plt.close()

	def imshow(self,im):
		print("reached here?")
		if im.dtype == np.float32 and im.max() > 1:
			im /= im.max() 
		plt.imshow(im)
		plt.show()	

