from random import shuffle
import sys
sys.path.append('data/')
sys.path.append('analysis/')

from spike_stim import SpikeStim
from natural_movie import NaturalMovie 
import os
import numpy as np
import copy 
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from spn.algorithms.Inference import log_likelihood


from neuro_spn import NeuroSPN

import logging
logger = logging.getLogger()


import warnings
warnings.filterwarnings("ignore")

manifest_path = os.path.join("/home/koosha/Allen_data/", "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest = manifest_path)
sessions = cache.get_session_table()
session_ids = sessions.index.tolist()




#result_path = "result_files/"
result_path = "SPN_results/"
spn_path = "SPN_trees/"
all_areas = ['VISp', 'VISl', 'VISal', 'VISam', 'VISpm', 'VISrl']
min_spike_percent = 10 ** -40 ### percentage for inclusion of a neuron. e.g when one, we should see spikes in at least 1% of the data set  
time_step = 8
per_bin = 1
shuffle_dur = 340 #end_time - start_time
num_inits = 5
num_sh = 1
chunk_movies = 5
chunk_size = int(chunk_movies * (end_frame - start_frame))
shuffle = False
for start_frame in [240, 390, 490, 890]: 
    end_frame = start_frame + 10
    for si, session_id in enumerate(session_ids[27:]): ### since the first ones are all bob
        session = cache.get_session_data(session_id)
        if session.session_type != 'functional_connectivity':
            continue
        nm = NaturalMovie(session_id)
        conds_all = nm.getConditions()
        conds = nm.divideFrame(conds_all, start_frame)
        conds = conds[1]
        print ("START", len(conds.keys()))
        conds = nm.divideFrame(conds, end_frame)
        conds = conds[0]
        print ("END", len(conds.keys()))        
        
        key_list = list(conds.keys())
        for area_name in all_areas:
            if area_name not in nm.session.units['ecephys_structure_acronym'].tolist():
                continue
            print (si, session_id, area_name, len(conds.keys()))
            file_name = nm.getStimName() + "_" + str(time_step) + int(shuffle) * "_shuffled" + "_" + str(nm.session_id) \
                + "_" + area_name + "_" + str(chunk_movies) + "_" + str(start_frame) + "_" + str(end_frame)  
            spikes = nm.getSpikes(area_name)
            try:
                for i in range(num_sh):
                    file1 = result_path + file_name + "_" + str(i + 1) + ".npy" 
                    print (i + 1, np.load(file1))
                file2 = result_path + file_name + "_l.npy"
                print ("Already Computed: ", np.load(file2))
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
                if chunk_movies > 59:
                    conds_dict = conds
                else:
                    conds_dict = {}
                    for j in range(chunk_i, int(chunk_size + chunk_i)):
                        if j > (len(key_list) - 1):     ###last chunk might be smaller due to this
                            print ("ERRRROR: wrong movie frame length")
                            continue
                        conds_dict[key_list[j]] = conds[key_list[j]] 
                data = nm.getData(spikes, conds_dict, 0, 32, time_step)[:, :-1] #### since we do not need frame number 
                print (data.shape)    
                if shuffle:
                    data1 = SpikeStim.shuffleInTrial(data, trial_dur =  shuffle_dur, time_step = time_step, per_bin = per_bin)
                    for i in range(4):
                        data2 = SpikeStim.shuffleInTrial(data, trial_dur =  shuffle_dur, time_step = time_step, per_bin = per_bin)
                        data1 = np.vstack([data1, data2])
                    data = data1
                print (data.shape)
                spike_rates = np.mean(data, axis = 0)
                #print (spike_rates)
                above_th_inds = np.where(spike_rates >= min_spike_percent / 100)[0]
                data = data[:, above_th_inds]
                if data.size < 1:
                    print ("No neurons left!")
                    continue
                seed = session_id + 10 * ord(area_name[-1])
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
                    
 
                best_ll_cv = -1000
                spn1_list = all_spn_sh[1] #pickle.load(open(spn_path + file_name + "_1.spn", "rb"))
                nspn = NeuroSPN(data)                
                min_data = 100
                spn_l = nspn.learnSPN(min_data + int(shuffle) * min_data * 4, .1)
                while ((spn_l == None) and min_data < 300): 
                    min_data = int(min_data * 2)
                    print ("fail")
                    spn_l = nspn.learnSPN(min_data + int(shuffle) * min_data * 4, .1)
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
