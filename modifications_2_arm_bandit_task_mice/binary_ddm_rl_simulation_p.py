import os
import pdb
from simulation_functions_loki import Simulation, Simulation_artificial_data
import numpy as np
from time import time
import glob
import pandas as pd
from pickle_objects import save_object
import sys


data_type = sys.argv[1] # actual or artificial


home_path = os.path.expanduser('~')

exp_parameter_path = '../Data/processed_data/ideal_observer/'
artificial_data_path = "../Data/artificial_data_Krista/"
condition_key_path = os.path.join(home_path, 'Desktop/loki_1')
analysis_path = os.path.join(home_path, 'Desktop/loki_1/analysis/pseudobayes/')

#os.chdir(analysis_path)




# alpha: belief lr
# beta: cpp lr

mod_alpha = 1
mod_beta = 1

learning_rates = {'alpha': mod_alpha,'beta': mod_beta}




"""hypothesized update for bound + drift"""

# filename structure: 786_reward2_cond6530.csv

#condition_key_df = pd.read_csv(os.path.join(condition_key_path, 'reward_condition_key.csv')) # get reward codes
#condition_order_df = pd.read_csv(os.path.join(condition_key_path, 'reward_condition_order.csv')) # get condition sequence & skip subject that dropped the task
if data_type == "actual":
	subj_data_files = glob.glob(exp_parameter_path+'Day*.csv') # match pattern for exp. data file
elif data_type == "artificial":
	subj_data_files = glob.glob(artificial_data_path+'*reward*.csv') # match pattern for exp. data file


subj_data_files.sort()


#subject_ids = condition_order_df.coax_id.dropna().tolist()
#reward_codes = condition_key_df.code.tolist()
#lambda_vals = condition_key_df.volatility.tolist()



if data_type == "actual":

	def cpu_simulation(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring, subject_id, conflict_code, condition_code, day_code):
		sim = Simulation(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring, subject_id, conflict_code, condition_code, day_code)
		#pdb.set_trace()
		sim.calc_B_CPP()
		return sim
elif data_type == "artificial":
	def cpu_simulation(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring,reward_code,  condition_code, run_id):
		sim = Simulation_artificial_data(model, learning_rates, drift_start, bound_start, sp_start, tr_start,pathstring,reward_code,   condition_code, run_id)
		sim.adapt_ddm()
		return sim


av_model = 3
n_subjects = 4

drift_start = 0.001
bound_start = 0.3
sp_start = 0.5
tr_start = 0.2

if data_type == "actual":
	sim_data_path =  '../Data/processed_data/ideal_observer/simulated_data/'
elif data_type == "artificial":
	sim_data_path =  '../Data/processed_data/ideal_observer/simulated_data_artificial/'


if data_type == "actual":
	args = [(av_model, learning_rates, drift_start,
	bound_start, sp_start,
			tr_start, filen, filen.split('/')[-1].split('subject_')[1].split('_')[0], filen.split('/')[-1].split('conflict_')[1].split('_')[0], filen.split('/')[-1].split('condition')[1].split('.csv')[0], filen.split('/')[-1].split('Day_')[1].split('subject')[0] ) for filen in subj_data_files]
elif data_type == "artificial":
	args = [(av_model, learning_rates, drift_start,
	bound_start, sp_start,
			tr_start, filen,filen.split('/')[-1].split('_reward')[0], filen.split('/')[-1].split('cond')[1].split('_')[0], filen.split('/')[-1].split('run')[1].split('.csv')[0] ) for filen in subj_data_files]



sim_start_time = time()

#with Pool() as p:

if data_type == "actual":

	for i in np.arange(len(args)):
		av_model_sims = cpu_simulation(*args[i])
		save_object(av_model_sims, sim_data_path + 'sim_' + av_model_sims.subject_id + '_conflict_' + av_model_sims.conflict_id + '_condition_' + av_model_sims.condition_id +'_day_'+av_model_sims.day_id+  '.pkl')

elif data_type == "artificial":
	for i in np.arange(len(args)):
		av_model_sims = cpu_simulation(*args[i])
		save_object(av_model_sims, sim_data_path + 'sim' + '_condition_' + av_model_sims.condition_id +'_run_'+av_model_sims.run_id +"_reward_"+av_model_sims.reward_id+'.pkl')


sim_end_time = time()


sim_time = sim_end_time - sim_start_time



# save each simulation as a pickled object
#[save_object(sim, sim_data_path + 'sim' + sim.subject_id + '_reward' + sim.conflict_code + '_run' + sim.run_n + '.pkl') for sim in av_model_sims]
