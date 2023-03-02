from random import shuffle
import sys    
sys.path.append('data/')
sys.path.append('analysis/')
from drifting_gratings75 import DriftingGratings75
import os
import numpy as np
import copy 
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from spike_stim import SpikeStim
from spn.algorithms.Inference import log_likelihood


from neuro_spn import NeuroSPN

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger()

manifest_path = os.path.join("/home/koosha/Allen_data/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest = manifest_path)
sessions = cache.get_session_table()
session_ids = sessions.index.tolist()

result_path = "SPN_results/"
spn_path = "SPN_trees/"
all_areas = ['VISp', 'VISrl', 'VISl', 'VISal', 'VISam', 'VISpm']
min_range = 20
min_spike_percent = 1  ### percentage for inclusion of a neuron. e.g when one, we should see spikes in at least 1% of the data set  
chunk_size = 5
time_step = 20
per_bin = 1
start_time = 400
end_time = 800
shuffle_dur = 400 #end_time - start_time
high_contrast = 0 #### low = 0, high = 1
num_inits = 5
num_sh = 1

rdc_th = .3
min_data = 100
for shuffle in [False, True]:
    for si, session_id in enumerate(session_ids[27:]): ### since the first ones are all bob
        session = cache.get_session_data(session_id)
        if session.session_type != 'functional_connectivity':
            continue
        ds = DriftingGratings75(session_id)
        conds_all = ds.getConditions()
        conds_lh = ds.divideContrast(conds_all)
        conds_contrast = conds_lh [high_contrast] 
        for direction in range(4):
            conds_dir = ds.divideDirection(conds_contrast)  
            conds = conds_dir[direction]
            for area_name in all_areas:
                if area_name not in ds.session.units['ecephys_structure_acronym'].tolist():
                    continue
                print (start_time, end_time, si, session_id, area_name, direction, len(conds.keys()))
                file_name = ds.getStimName() + "_" + str(time_step) + int(shuffle) * "_shuffled" + "_" + str(ds.session_id) + "_" + area_name + "_" + str(min_range) + "_" + str(chunk_size) + "_" + str(start_time) + "_" + str(end_time)  + "_" + str(direction) + "_" + str(high_contrast) 
                spikes = ds.getSpikes(area_name, min_range)
                first_cond = list(spikes.keys())[0]
                num_n = len(list(spikes[first_cond].keys()))
                key_list = list(conds.keys())
                try:
                    file1 = result_path + file_name + "_1.npy"
                    file2 = result_path + file_name + "_l.npy"
                    print (np.load(file2) - np.load(file1))   
                    print ("already calculated")
                    continue
                except:
                    pass

               
                all_ll_sh = {}
                all_ll_deep = []
                all_spn_sh = {}
                all_spn_deep = []

                for i in range(1, 1 + num_sh):
                    all_ll_sh[i] = []

                    all_spn_sh[i] = []

                for chunk_i in range(0, len(key_list), int(chunk_size)):
                    conds_dict = {}
                    for j in range(chunk_i, int(chunk_size + chunk_i)):
                        if j > (len(key_list) - 1):     ###last chunk might be smaller due to this
                            continue
                        conds_dict[key_list[j]] = conds[key_list[j]] 
                    data = ds.getData(spikes, conds_dict, start_time, end_time, time_step)[:, :-2] #### since we do not need condition values 
                        
                    if shuffle:
                        data = SpikeStim.shuffleInTrial(data, trial_dur =  shuffle_dur, time_step = time_step, per_bin = per_bin)
                    
                    spike_rates = np.mean(data, axis = 0)
                    above_th_inds = np.where(spike_rates >= min_spike_percent / 100)[0]
                    data = data[:, above_th_inds]
                    if data.size < 1:
                        print ("No neurons left!")
                        continue
                    seed = session_id + (chunk_i * (direction + 1)) + 10 * ord(area_name[-1])
                    np.random.RandomState(seed).shuffle(data)

                    for num_c in range(1, 1 + num_sh):
                        #### fitting 
                        best_ll = -1000
                        best_spn = None
                        best_spn_pre = None
                        nspn = NeuroSPN(data)                
                        for _ in range(num_inits):
                            spn_i = nspn.shallowSPN(num_c)
                            spn_pre = copy.deepcopy(spn_i)
                            spn_i, ll = nspn.train(spn_i)
                            if ll > best_ll:
                                best_ll = ll
                                best_spn = copy.deepcopy(spn_i)
                                best_spn_pre = copy.deepcopy(spn_pre)
                        
                        print ("shallow: ", num_c, best_ll, end = " ")
 
                        
                    spn1_list = all_spn_sh[1] #pickle.load(open(spn_path + file_name + "_1.spn", "rb"))
                    nspn = NeuroSPN(data)          
                    min_data = 100
                    spn_l = nspn.learnSPN(min_data, rdc_th)      
                    while ((spn_l == None) and min_data < 500): 
                        min_data = int(min_data * 2)
                        print ("fail")
                        spn_l = nspn.learnSPN(min_data, rdc_th)
                    if (spn_l == None):
                        print ("FFFFail")
                        spn_l = spn1_list[int(chunk_i/chunk_size)]

                    ll = np.mean(log_likelihood(spn_l, data))
                    print ("learnSPN: ", ll) 

                    all_ll_deep.append(ll)
                    all_spn_deep.append(spn_l)
                    

                for num_c in range(1, 1 + num_sh):
                    np.save(result_path + file_name + "_" + str(num_c) + ".npy", np.array(all_ll_sh[num_c]))
                    file1 = open(spn_path + file_name + "_" + str(num_c) + ".spn", "wb")
                    pickle.dump(all_spn_sh[num_c], file1)
                    file1.close()

                np.save(result_path + file_name + "_l.npy", np.array(all_ll_deep))
                filec = open(spn_path + file_name +  "_l.spn", "wb")
                pickle.dump(all_spn_deep, filec)
                filec.close()

