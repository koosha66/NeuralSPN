import numpy as np
from spike_stim import SpikeStim

MANIFEST_PATH = "/home/koosha/Allen_data/manifest.json"
SAVE_PATH = "data_files"

class DriftingGratings(SpikeStim):
    def __init__(self, session_id, file_check = True, save = True, save_path = SAVE_PATH, manifest_path = MANIFEST_PATH):
        super().__init__( session_id, file_check, save, save_path, manifest_path)

    @staticmethod
    def getStimName():
         return 'drifting_gratings'

    @staticmethod
    def getNeededConditions():
        return ['orientation', 'temproal frequency']

    @staticmethod
    def getData(spikes, conds, start_time, end_time, time_step = 10, per_bin = 1, shuffle = False):
        data = SpikeStim.getData(spikes, conds, start_time, end_time, time_step, per_bin, shuffle)
        data[:, -2] = np.int_(data[:, -2] / 45)  
        return data 
    
    @staticmethod
    def divideDirection(conds):
        conds_direction = {}
        for i in range(4):
            conds_direction[i] = {}
        for k, cond in conds.items():
            conds_direction[(cond[0] / 45) % 4][k] = cond
        return conds_direction
    
    @staticmethod
    def divideOrientation(conds):
        conds_direction = {}
        for i in range(8):
            conds_direction[i] = {}
        for k, cond in conds.items():
            conds_direction[int(cond[0] / 45)][k] = cond
        return conds_direction
    
    
    
    
