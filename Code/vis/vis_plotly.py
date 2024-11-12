import os
import json 
from .visualizer import Visualizer
import numpy as np
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objs as go

import pandas as pd


class VisualizerPlotly(Visualizer):
	def __init__(self,opt):
		super().__init__(opt)	

		self.nrrAblationFile = '<Code-Path>/evaluationScores/NRR-Ablations.csv'
		if os.path.isfile(self.nrrAblationFile):
			self.nrrAblation_pd = pd.read_csv(self.nrrAblationFile)

	def plot_motion_complete_tests_rec_error(self):
		datadir = self.opt.datadir
		rec_err = {}
		for i in range(2,6):
			filepath = os.path.join(datadir,"results",f"occ_fusion_test{i}.txt")
			with open(filepath) as f:
				data = [ float(x) for x in f.read().split() if len(x) > 0]
			rec_err[f"test_{i}"] = np.array(data)

		x = list(range(len(rec_err["test_2"])))

		fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Graph Nodes Reconstructon Error"])
		for i in range(2,6):

			if i == 2: 
				test_name = "Test:1 Ground Truth Source Nodes"
			elif i == 3:
				test_name = "Test:2 Predicted Source Nodes"
			elif i == 4: 
				test_name = "Test:3 Predicted Source Nodes + ARAP Loss"
			elif i == 5:
				test_name = "Test:4 Predicted Source Nodes + ARAP & Data Loss"
				
			fig.add_trace(go.Scatter(x = x,
									 y = rec_err[f"test_{i}"],
									 mode="lines + text",
									 name=test_name),row=1,col=1)
					 
		fig.update_layout(xaxis_title="Frame Index", yaxis_title="Rec Err.",font=dict(size=25))
		fig.show()

	def compare_convergance_info(self,plotting_variable_dict):
		sample_name = os.path.basename(self.opt.datadir)
		dir_path = os.path.join(self.opt.datadir,"results",self.opt.exp,'optimization_convergence_info')

		all_terms = ['target_id']
		for term_type in plotting_variable_dict:
			all_terms.extend(plotting_variable_dict[term_type])


		exp_data = {}
		for exp_name in os.listdir(dir_path):
			if not os.path.isdir(os.path.join(dir_path,exp_name)):
				self.log.warning(f"{os.path.join(dir_path,exp_name)} not a dir") 
				continue  
			else:
				self.log.warning(f"Loading exp:{exp_name}")
			# loss_terms = {'target_id':[],'lmdk':[],'arap':[],'silh':[],'depth':[]}
			exp_data[exp_name] = dict([(term_name,list()) for term_name in all_terms])
			files = sorted([ x for x in os.listdir(os.path.join(dir_path,exp_name)) if "optimization_convergence_info" in x],key=lambda x: int(x.split('_')[-1].split('.')[0]))

			for file in files: 
				target_id = int(file.split('_')[-1].split('.')[0])
				with open(os.path.join(dir_path,exp_name,file)) as f:
					convergance_info = json.load(f)

				for term in exp_data[exp_name]: 
					if term == 'target_id':
						exp_data[exp_name][term].append(target_id)
					else:     
						if term not in convergance_info:
							exp_data[exp_name][term].append(0)
						else:
							if len(convergance_info[term]) > 0:
								exp_data[exp_name][term].append(convergance_info[term][-1])
							else: 
								exp_data[exp_name][term].append(0)

		for term_type in plotting_variable_dict:
			fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Loss terms for {term_type} across timesteps for:{sample_name}"])
			
			for exp_name in exp_data:	
				for term in plotting_variable_dict[term_type]:					
					if term == 'target_id':
						continue
					fig.add_trace(go.Scatter(x = exp_data[exp_name]["target_id"],
											 y = exp_data[exp_name][term],
											 mode="lines + text",
											 name=f"Loss:{term} Exp:{exp_name}"),row=1,col=1)

					print(f"Exp:{exp_name} Term:{term} Score:{np.mean(exp_data[exp_name][term])}")



				score_filename = f'<Code-Path>/evaluationScores/{exp_name}.csv'

				if os.path.isfile(score_filename):
					self.pose_ev_pd = pd.read_csv(score_filename)
				else:
					self.pose_ev_pd = pd.DataFrame(columns=["Sample","silh",'depth','point_to_plane'])

				if sample_name not in self.pose_ev_pd.values:
					data = dict([ (term,np.mean(exp_data[exp_name][term])) for term in ['depth','silh','point_to_plane'] if term in exp_data[exp_name] ])
					data['Sample'] = sample_name
					self.pose_ev_pd = self.pose_ev_pd.append(data,ignore_index=True)
				self.pose_ev_pd.to_csv(score_filename,index=False)	
			

				


			fig.update_layout(xaxis_title="Target Frame Index", yaxis_title="Value.",font=dict(size=25))		
			# fig.show()
			fig.write_html(os.path.join(dir_path,f"compare_convergance_info_{term_type}.html"),auto_open=False)


	def plot_convergance_info(self):

		sample_name = os.path.basename(self.opt.datadir)
		dir_path = os.path.join(self.opt.datadir,"results","optimization_convergence_info")


		files = sorted([ x for x in os.listdir(dir_path) if "optimization_convergence_info" in x],key=lambda x: int(x.split('_')[-1].split('.')[0]))

		# loss_terms = {'target_id':[],'data':[],'total':[],'arap':[],'3D':[],'px':[],'py':[],'motion':[]}
		loss_terms = None

		for file in files: 
			target_id = int(file.split('_')[-1].split('.')[0])
			with open(os.path.join(dir_path,file)) as f:
				convergance_info = json.load(f)

			if loss_terms is None:	
				loss_terms = dict([(term,[]) for term in convergance_info ])	
				loss_terms['target_id'] = []


			for term in loss_terms: 
				if term == 'target_id':
					loss_terms[term].append(target_id)
				else:     
					if term not in convergance_info:
						loss_terms[term].append(0)
					else:
						if len(convergance_info[term]) > 0:
							loss_terms[term].append(convergance_info[term][-1])
						else: 
							loss_terms[term].append(0)
							
		fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Loss terms during Optimization for:{sample_name}"])

		for term in loss_terms:
			if term == "target_id": 
				continue

			fig.add_trace(go.Scatter(x = loss_terms["target_id"],
									 y = loss_terms[term],
									 mode="lines + text",
									 name=term),row=1,col=1)
					 
		fig.update_layout(xaxis_title="Target Frame Index", yaxis_title="Value.",font=dict(size=25))		
		fig.show()
		fig.write_html(os.path.join(dir_path,"convergance_info_plot.html"),auto_open=False)