from drifting_gratings import DriftingGratings
import numpy as np
from spike_stim import SpikeStim


class DriftingGratings75 (DriftingGratings):
    def __init__(self, session_id, file_check = True, save = True, save_path = "data_files/"):
        super().__init__(session_id, file_check, save, save_path)
    
    @staticmethod
    def getStimName():
         return 'drifting_gratings_75_repeats'

    @staticmethod
    def getNeededConditions():
        return ['orientation', 'contrast']

    @staticmethod
    def getData(spikes, conds, start_time, end_time, time_step = 10, per_bin = 1, shuffle = False):
        data = SpikeStim.getData(spikes, conds, start_time, end_time, time_step, per_bin, shuffle)
        data[:, -2] = np.int_(data[:, -2] / 45)  
        data[:, -1] = np.int_(data[:, -1] > .5)  #### low contrast-> 0 , high_contrast -> 1
        return data 
    
    @staticmethod
    def divideContrast(conds):
        conds_contrast = {}
        for i in range(2):
            conds_contrast[i] = {}
        for k, cond in conds.items():
            conds_contrast[int(cond[1] > .5)][k] = cond
        return conds_contrast
    
    
    
    
    
