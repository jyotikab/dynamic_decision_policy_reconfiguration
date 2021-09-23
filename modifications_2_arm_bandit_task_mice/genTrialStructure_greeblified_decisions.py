#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np 
import matplotlib.pyplot as plt 
import os 
import pandas as pd
from operator import sub, add
import glob
from random import shuffle
from itertools import product
from scipy import stats
import pdb


# In[66]:


#home_path = os.path.expanduser('~') #need this to get correct home dir. across operating systems


#exp_param_path = home_path + "/Documents/loki_1/experimental_parameters/reward_parameters_test/"
#img_path = home_path + "/Documents/loki_1/images/symm_greebles/"

#file_name = home_path+"/Documents/loki_1/experimental_parameters/training_greeble_images.csv"
#training_set_images = pd.read_csv(file_name)

#sample_itis = pd.read_csv(glob.glob(os.path.join(home_path,"Documents/loki_1/experimental_parameters/reward_parameters/*.csv")
#)[0]).iti
#os.chdir(exp_param_path)
#os.getcwd()
data_dir = "../Data/processed_data/"
fig_target_dir = "../Figures/"
data_target_dir = "../Data/processed_data/ideal_observer/"


final_df = pd.read_csv(data_dir+"final_df_all.csv")

opt_act = [  "Optimal" if x.split('-')[0]==y else "Sub-optimal"   for x,y in zip(final_df["Block"],final_df["Action_name"])    ]
final_df["Optimal_action"] = opt_act


# In[67]:


final_df


# In[68]:


def define_reward(prob, n_trials,sess_df): #, reward_mu=3, reward_std=1 # Reward is binary in this case

    trial_index = np.arange(n_trials)
    
    #define suboptimal choice reward probability
    subopt_p, opt_p = prob # Tuple
    
    #sample rewards
    #reward_values = np.random.normal(loc=reward_mu, scale=reward_std, size=n_trials)
    #reward_values = np.zeros((n_trials))
    reward_values = np.random.normal(loc=0.1, scale=0.01,size=n_trials)
    ind_rew = np.where(np.array(sess_df["Rewarded"])=="Rewarded")[0]
    #reward_values[ind_rew] = 1
    reward_values[ind_rew] = np.random.normal(loc=1, scale=0.01, size=len(ind_rew))

    # Optimal and sub-optimal choice depends on the block information. This information stored in Optimal_action
    
    
    #calcualte n_trials based on probabilities 
    n_opt_reward_trials = int(opt_p * n_trials)
    n_subopt_reward_trials = int(subopt_p * n_trials)

    #find indices for optimal and suboptimal choices 
    #opt_reward_idx = np.random.choice(trial_index, size=n_opt_reward_trials, replace=False)
    #subopt_reward_idx = np.setxor1d(trial_index, opt_reward_idx)
    opt_reward_idx = np.where(np.array(sess_df["Optimal_action"]=="Optimal"))[0]
    subopt_reward_idx = np.where(np.array(sess_df["Optimal_action"]=="Sub-optimal"))[0]
    
    #intialize reward vectors
    #reward_t1, reward_t2 = np.zeros((n_trials)),np.zeros((n_trials))
    reward_t1, reward_t2 = np.random.normal(loc=0.05, scale=0.01,size=n_trials),np.random.normal(loc=0.05, scale=0.01,size=n_trials)

    #assign rewards
    reward_t1[opt_reward_idx] = reward_values[opt_reward_idx] 
    reward_t2[subopt_reward_idx] = reward_values[subopt_reward_idx]
        
    return reward_t1, reward_t2 # reward_t1 == optimal, reward_t2 == suboptimal


# In[69]:


def define_changepoints(n_trials, reward_t1, reward_t2, sess_df ): #cp_lambda

    sess_df["Block_change"] = sess_df["Block"].shift(1, fill_value=sess_df["Block"].head(1)) != sess_df["Block"]
    
    cps = np.where(np.array(sess_df["Block_change"])==True)[0]
    n_cps = len(cps) #find approximate number of change points #int(n_trials/cp_lambda)
    
    cp_base = cps #np.cumsum(cps) #calculate cp indices
    
    cp_idx = np.insert(cp_base,0,0) #add 0
    cp_idx = np.append(cp_idx,n_trials-1) #add 0

    cp_idx = cp_idx[cp_idx < n_trials] 
    
    cp_indicator = np.zeros(n_trials)
    cp_indicator[cp_idx] = 1
    
    # passive record of volatility
    lam = (n_cps+2)/float(n_trials)
    
    return cp_idx, cp_indicator, sess_df, lam


# In[70]:


def define_epochs(n_trials, reward_t1, reward_t2, cp_idx, opt_p, sess_df):
    
    t1_epochs = []
    t2_epochs = []
    
    subopt_p = 1 - opt_p
    
    epoch_number = []
    epoch_trial = []
    epoch_length = []
    
    reward_p = []

    
    #p_id_solution = np.array(sess_df['Block']).tolist() #female greeble is always first 
    #p_id_solution = [ 0 if x == "Push" else 1 for x in sess_df["Action_name"]]
    p_id_solution = np.array(sess_df["Block"])
    
    #f_greeble = ord('f')
    #m_greeble = ord('m')
    h_push = ord('h')
    l_pull = ord('l')

    current_target = True
    for i in range(len(cp_idx)-1):
        if current_target: 
            t1_epochs.append(reward_t1[cp_idx[i]:cp_idx[i+1]])
            t2_epochs.append(reward_t2[cp_idx[i]:cp_idx[i+1]])
            reward_p.append(np.repeat(opt_p, cp_idx[i+1]-cp_idx[i]))
            #p_id_solution.append(np.repeat(f_greeble, cp_idx[i+1]-cp_idx[i])) 
        else: 
            t1_epochs.append(reward_t2[cp_idx[i]:cp_idx[i+1]])
            t2_epochs.append(reward_t1[cp_idx[i]:cp_idx[i+1]])
            reward_p.append(np.repeat(subopt_p, cp_idx[i+1]-cp_idx[i]))
            #p_id_solution.append(np.repeat(m_greeble, cp_idx[i+1]-cp_idx[i]))
        
        epoch_number.append(np.repeat(i, cp_idx[i+1]-cp_idx[i]))
        epoch_trial.append(np.arange(cp_idx[i+1]-cp_idx[i]))
        epoch_length.append(np.repeat(len(np.arange(cp_idx[i+1]-cp_idx[i])),repeats=len(np.arange(cp_idx[i+1]-cp_idx[i]))))

        if i == len(cp_idx)-2:
            if current_target:
                t1_epochs.append(reward_t1[-1])
                t2_epochs.append(reward_t2[-1])
                reward_p.append(opt_p)
                #p_id_solution.append(f_greeble)
            else:
                t1_epochs.append(reward_t2[-1])
                t2_epochs.append(reward_t1[-1])
                reward_p.append(opt_p)
                #p_id_solution.append(m_greeble)

            epoch_number.append(i)
            
    

        current_target = not(current_target)
    
    epoch_length[-1] = epoch_length[-1] + 1
    #flatten    
    epoch_number = np.hstack(epoch_number).astype('float')
    epoch_trial = np.hstack(epoch_trial).astype('float')
    epoch_length = np.hstack(epoch_length).astype('float')
    
    epoch_trial = np.append(epoch_trial, (epoch_trial[-1] + 1))
    epoch_length = np.append(epoch_length, epoch_length[-1])

    t1_epochs = np.hstack(t1_epochs)
    t2_epochs = np.hstack(t2_epochs)
    reward_p = np.hstack(reward_p).astype('float')
    reward_p[-1] = reward_p[-2]
    #p_id_solution = np.hstack(p_id_solution)

    return t1_epochs, t2_epochs, epoch_number, reward_p, p_id_solution, epoch_trial, epoch_length


# In[71]:


def define_observed_cps(n_trials, t1_epochs):
    
    observed_cp_indicator = np.hstack([False, np.diff(t1_epochs == 0)]).astype('float')
    
    return observed_cp_indicator


# In[72]:


def gen_itis(n_trials, iti_min=4, iti_max=16, rate_param=2.8): # change these parameters accordingly
        
    # Ask Julia about the ITI  - does not matter for the modeling
    lower, upper, scale = iti_min, iti_max, rate_param
    X = stats.truncexpon(b=(upper-lower)/scale, loc=lower, scale=scale)
    itis = X.rvs(n_trials)

#     itis = np.random.permutation(np.repeat(sample_itis, n_trials // len(sample_itis)))

    return itis


# In[73]:


# subject_ids = np.arange(initial_s_id, initial_s_id+len(subjects))
subject_ids = np.unique(final_df["Animal"])


#here you would specify minimum and maximum probabilities, along with the increment 
conflicts = dict({"No":(0.,1.) ,"Low":(0.1,0.9),"High":(0.25,0.75)})

#min_p = 0.65
#max_p = 0.95
#step_p = 0.10

#Here you would specify the # trials before a switch in the optimal target identity 
#min_e = 10
#max_e  = 40
#step_e = 10

#conflict_means = np.arange(min_p, max_p, step_p)
#vol_means = np.arange(min_e, max_e, step_e)

#print(conflict_means, vol_means)

#min_epoch_length = 4

#conf_vol_combinations = np.round(list(product(conflict_means, vol_means)),2)

#trials = np.arange(n_trials)


# In[74]:


print_concat_df = True

#for subject in subject_ids:
#    for session_n,session in enumerate(conf_vol_combinations): 

# Generate experimental parameters file per day, per animal, per conflict, per condition

for grp in final_df.groupby(["Animal","Day","Conflict","Condition"]):
    print(grp)
    sub,day,conf,cond = grp[0]
        
    print("Subject",sub)
    sess_df = grp[1].copy()
    subopt_p, opt_p = conflicts[grp[0][2]]
    
    #opt_p=session[0]
    #cp_lambda=session[1]
        
    #min_epoch_length=0
        
    #while min_epoch_length <=3: #need enough trials to analyze per epoch
    n_trials = len(grp[1])   
    trials = np.arange(0,n_trials)
    reward_t1, reward_t2 = define_reward(prob=conflicts[grp[0][2]], n_trials = n_trials,sess_df=sess_df)
    #print(reward_t1)
    #print(reward_t2)
    cp_idx,cp_indicator,sess_df,lam = define_changepoints(n_trials, reward_t1, reward_t2, sess_df)
    print(cp_idx)
    print(cp_indicator)
    
    t1_epochs, t2_epochs, epoch_number,reward_p, p_id_solution, epoch_trial, epoch_length = define_epochs(n_trials, reward_t1, reward_t2, cp_idx, opt_p,sess_df)
    
    print(t1_epochs)
    print(reward_p)
    print(epoch_number)
    observed_cp_indicator=define_observed_cps(n_trials, t1_epochs)
    
    itis = gen_itis(n_trials,4,16,lam)
            
    min_epoch_length = min(epoch_length)
    
    filename = "Day_" + str(day) + "subject" + "_" + str(sub) + '_' + "conflict"+"_"+conf+"_"+"condition"+cond
               
#         header = ('trial,r_t1,r_t2,cp,obs_cp,epoch_number,reward_p, m_image, f_image')
#         data = np.transpose(np.matrix((trials,t1_epochs, t2_epochs, cp_indicator, observed_cp_indicator, epoch_number, reward_p, training_set_images.m_image.values, training_set_images.f_image.values)))
#         np.savetxt(exp_param_path+filename, data, header=header, comments='',delimiter=',')

    header = ['r_t0','r_t1','cp','obs_cp','epoch_number','reward_p_t0', 
                   'iti',  'epoch_trial', 'epoch_length','day', 'subject','conflict','condition','p_id_solution',"Block_change","lambda","action_history","optimal"]

       
    # 0 = Push, 1 = Pull
    action_history = [ 0 if x == "Push" else 1 for x in sess_df["Action_name"]]
    optimal = [ 1 if x == "Optimal"  else 0 for x in sess_df["Optimal_action"] ] 

    
    print("t1_epochs",t1_epochs)
    print("t2_epochs",t2_epochs)
    print("cp_indicator",cp_indicator)
    print("observed_cp_indicator",observed_cp_indicator)
    print("epoch_number",epoch_number)
    print("reward_p",reward_p)
    print("itis",itis)
    print("p_id_solution",p_id_solution)
    print("epoch_trial",epoch_trial)
    print("epoch_length",epoch_length)

    
    append_data = [t1_epochs, t2_epochs, cp_indicator, observed_cp_indicator, epoch_number, reward_p, itis, epoch_trial, epoch_length, day,sub,conf,cond,p_id_solution,np.array(sess_df["Block_change"]),lam,action_history,optimal]
    data = pd.DataFrame(columns=header)
    for i,k in enumerate(header):
        data[k] = append_data[i]
    
    
    
    if print_concat_df: 
        data.to_csv(data_target_dir+filename + '.csv',index=False)
        
   
    #plt.hist(itis)






