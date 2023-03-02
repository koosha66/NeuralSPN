import pickle
from spike_stim import SpikeStim
import numpy as np 

class NaturalMovie(SpikeStim):
    def __init__(self, session_id, file_check = True, save = True, save_path = "data_files/"):
        super().__init__(session_id, file_check, save, save_path)

    @staticmethod
    def getStimName():
         return  'natural_movie_one_more_repeats'  # 'natural_movie_one_shuffled' #  #natural_movie_one'

    @staticmethod 
    def getNeededConditions():
         return ['frame']

    @staticmethod
    def getMovieLenSec():
         return  30


    def getConditions(self):
        file_chars = str(self.session_id) + "_" + self.getStimName() + ".pkl"
        if self.file_check:
            try:
                conds = pickle.load(open(self.save_path + "conditions_" + file_chars, "rb"))
                return conds
            except:
                pass      
            
        stim_table = self.session.stimulus_presentations
        stim_ids = stim_table.loc[(stim_table['stimulus_name'] == self.getStimName())].index.values
        frames = stim_table.loc[stim_ids].frame.values
        conds = {}
        for p_id, frame_num in zip(stim_ids, frames):
            conds[p_id] = [frame_num]
        if self.save:
            pickle.dump(conds, open(self.save_path + "conditions_" + file_chars, "wb"))
        return conds

    def getSpikes(self, area_name):
        file_chars = str(self.session_id) + "_" + area_name + "_" + self.getStimName()  + ".pkl"
        if self.file_check:
            try:
                spikes = pickle.load(open(self.save_path + "spikes_" + file_chars , "rb"))
                return spikes
            except:
                pass
        
        units = self.session.units
        units_area = units[units["ecephys_structure_acronym"] == area_name]
        unit_ids = units_area.index.values
        stim_table = self.session.stimulus_presentations
        
        spike_times_dict = self.session.spike_times
        
        start_movie_inds = stim_table[(stim_table['stimulus_name'] == self.getStimName()) & (stim_table['frame'] == 0)].index.values
        start_movie_times = stim_table[(stim_table['stimulus_name'] == self.getStimName()) & (stim_table['frame'] == 0)].start_time.tolist()
        
        spike_dict = {}
        
        conds = self.getConditions()
        for p_id in conds.keys():
            spike_dict[p_id] = {}
            for u_id in unit_ids:
                spike_dict[p_id][u_id] = []
        
        for start_i, start_m in zip(start_movie_inds, start_movie_times):
            for u_id in unit_ids:
                inds = np.where(spike_times_dict[u_id] > start_m)[0]
                spike_movie = spike_times_dict[u_id][inds]
                inds = np.where(spike_movie < start_m + self.getMovieLenSec())[0]
                spike_movie = spike_movie[inds] - start_m
                for time in spike_movie:
                    frame_number = int((time * 1000) / 33.34)
                    try:
                        spike_dict[start_i + frame_number][u_id].append((time * 1000) - frame_number * 33.34)
                    except:
                        pass
        
        if self.save:
            pickle.dump(spike_dict, open(self.save_path + "spikes_" + file_chars, "wb"))

        return spike_dict

    @staticmethod
    def divideFrame(conds, frame_th):
        conds_frame = {}
        for i in range(2):
            conds_frame[i] = {}
        for k, cond in conds.items():
            conds_frame[int(cond[0] > frame_th - 1)][k] = cond
        return conds_frame
    


  
