#!/usr/bin/env python
# coding: utf-8



fig_target_dir = "/home/bahuguna/Work/CMU_project/Figures/ideal_observer/"


# In[189]:


import glob
import numpy as np
import pickle
import seaborn as sns
import pylab as pl
import pandas as pd
from pickle_objects import save_object, load_object
import sys

day_limit = int(sys.argv[1]) # Maybe displaying results on ;ater days show better results ?, -1, 0,2,3,5, 
# In[190]:
data_type = sys.argv[2]


if data_type == "actual":
	data_target_dir = "/home/bahuguna/Work/CMU_project/Data/processed_data/ideal_observer/simulated_data/"
	files = glob.glob(data_target_dir+"sim*.pkl")
elif data_type == "artificial":
	data_target_dir = "/home/bahuguna/Work/CMU_project/Data/processed_data/ideal_observer/simulated_data_artificial/"
	files = glob.glob(data_target_dir+"sim*.pkl")



# In[191]:


if data_type == "actual":
	B_CPP_df = pd.DataFrame(columns=["Day","Condition","Conflict","Subject","B-0","B-1","CPP","Trial","B_diff","change_point","Block"])
	for f in files:
		temp = pd.DataFrame(columns =["Day","Condition","Conflict","Subject","B-0","B-1","CPP","Trial","B_diff","change_point","Block"])
		sub = f.split('/')[-1].split('sim_')[1].split('_')[0]
		conf = f.split('/')[-1].split('conflict_')[1].split('_')[0]
		cond = f.split('/')[-1].split('condition_')[1].split('_')[0]
		day = f.split('/')[-1].split('day_')[1].split('.pkl')[0]
		
		sim = load_object(f)
		temp["B-0"] = sim.B[:,0]
		temp["B-1"] = sim.B[:,1]
		temp["CPP"] = sim.CPP
		temp["Day"] = day
		temp["Condition"] = cond
		temp["Conflict"] = conf
		temp["Subject"] = sub
		temp["Trial"] = np.arange(len(sim.B))
		temp["B_diff"] = sim.B_diff
		temp["signed_B_diff"] = sim.signed_B_diff
		temp["change_point"] = sim.expParam.cp
		temp["Block"] = sim.expParam.p_id_solution
		
		temp["ideal_B"] = sim.ideal_B		
		B_CPP_df = B_CPP_df.append(temp,ignore_index=True)

elif data_type == "artificial":
	B_CPP_df = pd.DataFrame(columns=["Reward","Run","Condition","B-0","B-1","CPP","Trial","B_diff","change_point"])
	for f in files:
		temp = pd.DataFrame(columns =["Reward","Run","Condition","B-0","B-1","CPP","Trial","B_diff","change_point"])
		rew = f.split('/')[-1].split('reward_')[1].split('.pkl')[0]
		run = f.split('/')[-1].split('run_')[1].split('_')[0]
		cond = f.split('/')[-1].split('condition_')[1].split('_')[0]
		
		sim = load_object(f)
		temp["B-0"] = sim.B[:,0]
		temp["B-1"] = sim.B[:,1]
		temp["CPP"] = sim.CPP
		temp["Reward"] = rew
		temp["Condition"] = cond
		temp["Run"] = run
		temp["Trial"] = np.arange(len(sim.B))
		temp["B_diff"] = sim.B_diff
		temp["signed_B_diff"] = sim.signed_B_diff
		temp["change_point"] = sim.expParam.cp

		temp["ideal_B"] = sim.ideal_B		
		B_CPP_df = B_CPP_df.append(temp,ignore_index=True)




    


if data_type == "actual":
	max_lag = 8 # -1 to 5
	cp_aligned_B = pd.DataFrame(columns=["Day","Condition","Conflict","Subject","B_diff","CPP","time_from_change_point","signed_B_diff","Block","ideal_B"])
	for grp in B_CPP_df.groupby(["Day","Condition","Conflict","Subject","Block"]):
		ind_cp = np.where(grp[1]["change_point"]==1.0)[0]
		#print(ind_cp)
		mask_cp = np.logical_and(ind_cp!=0,ind_cp!=len(grp[1])-1)
		ind_cp = ind_cp[mask_cp]
		#print(ind_cp)
		for ic in ind_cp:
			dat_slice = grp[1].iloc[ic-1:ic+max_lag]
			#print(dat_slice)
			for l in np.arange(max_lag):
				if l < len(dat_slice)-1:
					lag_dat = dat_slice.iloc[l]
					cp_aligned_B = cp_aligned_B.append({"Day":int(grp[0][0]),"Condition":grp[0][1],"Conflict":grp[0][2],"Subject":grp[0][3],"B_diff":lag_dat["B_diff"],"CPP":lag_dat["CPP"],"time_from_change_point":l-1,"signed_B_diff":lag_dat["signed_B_diff"],"Block":grp[0][4],"ideal_B":lag_dat["ideal_B"] },ignore_index=True)
				else:
					continue

	cp_aligned_B.to_csv(data_target_dir+"Change_point_aligned_B_CPP.csv")

elif data_type == "artificial":
	max_lag = 8 # -1 to 5
	cp_aligned_B = pd.DataFrame(columns=["Reward","Run","Condition","B_diff","CPP","time_from_change_point","signed_B_diff","ideal_B"])
	for grp in B_CPP_df.groupby(["Reward","Run","Condition"]):
		ind_cp = np.where(grp[1]["change_point"]==1.0)[0]
		#print(ind_cp)
		mask_cp = np.logical_and(ind_cp!=0,ind_cp!=len(grp[1])-1)
		ind_cp = ind_cp[mask_cp]
		#print(ind_cp)
		for ic in ind_cp:
			dat_slice = grp[1].iloc[ic-1:ic+max_lag]
			#print(dat_slice)
			for l in np.arange(max_lag):
				if l < len(dat_slice)-1:
					lag_dat = dat_slice.iloc[l]
					cp_aligned_B = cp_aligned_B.append({"Reward":grp[0][0],"Run":grp[0][1],"Condition":grp[0][2],"B_diff":lag_dat["B_diff"],"CPP":lag_dat["CPP"],"time_from_change_point":l-1,"signed_B_diff":lag_dat["signed_B_diff"],"ideal_B":lag_dat["ideal_B"] },ignore_index=True)
				else:
					continue

	cp_aligned_B.to_csv(data_target_dir+"Change_point_aligned_B_CPP.csv")



if data_type == "actual":
	cp_aligned_B["Condition + Subject"] = cp_aligned_B["Condition"] +" - "+cp_aligned_B["Subject"]
	cp_aligned_B["Condition + Block"] = cp_aligned_B["Condition"] +" - "+cp_aligned_B["Block"] 
	cp_aligned_B


	g1 = sns.catplot(x="time_from_change_point",y="ideal_B",data=cp_aligned_B.loc[cp_aligned_B["Day"]>day_limit],hue="Conflict",col="Condition + Block",kind='point',col_wrap=2)
	g1.fig.savefig(fig_target_dir+"Condition_Block_wise_B_diff_"+str(day_limit)+".png")


	g2 = sns.catplot(x="time_from_change_point",y="CPP",data=cp_aligned_B.loc[cp_aligned_B["Day"]>day_limit],hue="Conflict",col="Condition + Block",kind='point',col_wrap=2)
	g2.fig.savefig(fig_target_dir+"Condition_Block_wise_CPP_"+str(day_limit)+".png")


	g3 = sns.catplot(x="time_from_change_point",y="ideal_B",data=cp_aligned_B.loc[cp_aligned_B["Day"]>day_limit],hue="Conflict",col="Condition",kind='point')
	g3.fig.savefig(fig_target_dir+"Condition_B_diff_"+str(day_limit)+".png")

	g4 = sns.catplot(x="time_from_change_point",y="CPP",data=cp_aligned_B.loc[cp_aligned_B["Day"]>day_limit],hue="Conflict",col="Condition",kind='point')
	g4.fig.savefig(fig_target_dir+"Condition_CPP_"+str(day_limit)+".png")

elif data_type == "artificial":
	g3 = sns.catplot(x="time_from_change_point",y="ideal_B",data=cp_aligned_B,kind='point')
	g3.fig.savefig(fig_target_dir+"Condition_B_diff_artificial_data.png")

	g4 = sns.catplot(x="time_from_change_point",y="CPP",data=cp_aligned_B,kind='point')
	g4.fig.savefig(fig_target_dir+"Condition_CPP_artificial_data.png")




