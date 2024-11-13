def get_visualizer(opt):
	
	if not hasattr(opt,"vis"):
		opt.vis = opt.visualizer


	if opt.vis.lower() == "matplotlib": 
		from .vis_matplotlib import VisualizeMatplotlib
		return VisualizeMatplotlib(opt)

	elif opt.vis.lower() == "open3d": 
		from .vis_open3d import VisualizeOpen3D
		return VisualizeOpen3D(opt)

	elif opt.vis.lower() == "plotly": 
		from .vis_plotly import VisualizerPlotly
		return VisualizerPlotly(opt)

	elif opt.vis.lower() == "ipyvolume":
		from .vis_ipyvolume import VisualizerIpyVolume
		return VisualizerIpyVolume(opt)

	elif opt.vis.lower() == "polyscope":
		from .vis_polyscope import VisualizerPolyScope
		return VisualizerPolyScope(opt)
	
	else: 
		print("Reaching here")
		raise NotImplementedError("Current possible visualizers -> matplotlib,open3d,plotly. eg. --visualizers open3d")


