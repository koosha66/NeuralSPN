import numpy as np
import pickle
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import copy


MANIFEST_PATH = "/home/koosha/Allen_data/manifest.json"
SAVE_PATH = "data_files/"

class SpikeStim:
    def __init__(self, session_id, file_check = True, save = True, save_path = SAVE_PATH, manifest_path = MANIFEST_PATH):
        self.session_id = session_id
        cache = EcephysProjectCache.from_warehouse(manifest = manifest_path)
        self.session = cache.get_session_data(session_id)
        self.file_check = file_check
        self.save = save
        self.save_path = save_path

    @staticmethod
    def getStimName():
        raise NotImplementedError("Stimulus name function should be defined")

    @staticmethod 
    def getNeededConditions():
         raise NotImplementedError("Stimulus name function should be defined")    
        #if stim_name == "dot_motion":
        #    return ['Dir', 'coherence']

    def getSpikes(self, area_name, min_range = 0):
        file_chars = str(self.session_id) + "_" + area_name + "_" + self.getStimName() + "_" + str(min_range) + ".pkl"    
        if self.file_check:
            try:
                spikes = pickle.load(open(self.save_path + "spikes_" + file_chars , "rb"))
                return spikes
            except:
                pass
        units = self.session.units
        units_area = units[units["ecephys_structure_acronym"] == area_name]
        unit_ids = units_area.index.values
        stim_presentation_ids = self.session.stimulus_presentations.loc[
        (self.session.stimulus_presentations['stimulus_name'] == self.getStimName())
        ].index.values
        times = self.session.presentationwise_spike_times(
        stimulus_presentation_ids = stim_presentation_ids,
        unit_ids = unit_ids)  
        valid_ids = [] 
        spike_count = {}
        for u_id in unit_ids:
            spike_count[u_id] = []
            times_u_id = times[times['unit_id'] == u_id]
            cs = np.array(times_u_id.groupby(['stimulus_presentation_id']).unit_id.agg(['count'])['count'].tolist())
            l = cs.size
            sorted_cs = np.sort(cs)
            if sorted_cs[int(.975 *l)] - sorted_cs[int(.025*l)] > min_range:
                valid_ids.append(u_id)
        spike_dict = {}
        for p_id in stim_presentation_ids:
            spike_dict[p_id] = {}
            times_p_id = times[times['stimulus_presentation_id'] == p_id]
            for u_id in np.sort(valid_ids):
                spike_dict[p_id][u_id] = (times_p_id[times_p_id['unit_id'] == u_id]['time_since_stimulus_presentation_onset']*1000).tolist()
                
        if self.save:
            pickle.dump(spike_dict, open(self.save_path + "spikes_" + file_chars, "wb"))
        return spike_dict
        
    def getConditions(self):
        file_chars = str(self.session_id) + "_" + self.getStimName() + ".pkl"
        if self.file_check:
            try:
                conds = pickle.load(open(self.save_path + "conditions_" + file_chars, "rb"))
                return conds
            except:
                pass

        stim_presentation_ids = self.session.stimulus_presentations.loc[
        (self.session.stimulus_presentations['stimulus_name'] == self.getStimName())
        ].index.values
        stim_cond = self.session.stimulus_presentations[self.session.stimulus_presentations["stimulus_name"] == self.getStimName()]
        condition_names = self.getNeededConditions()
        conds = {}
        for p_id in stim_presentation_ids:
            all_cond = []
            for cond_name in condition_names:
                cond_val = stim_cond[cond_name][p_id]
                if cond_val != 'null': 
                    all_cond.append(cond_val)
            if len(all_cond) == len(condition_names): #### even one null means the data is not useful
                conds[p_id] = copy.copy(all_cond)  
        if self.save:
            pickle.dump(conds, open(self.save_path + "conditions_" + file_chars, "wb"))
        return conds


    def getVelocity(self): 
        file_chars = str(self.session_id) + "_" + self.getStimName()  + ".pkl"
        if self.file_check:
            try:
                velocity = pickle.load(open(self.save_path + "velocity_" + file_chars , "rb"))
                velocity_times = pickle.load(open(self.save_path + "velocity_times_" + file_chars , "rb"))
                return velocity, velocity_times
            except:
                pass

        stim_presentation_ids = self.session.stimulus_presentations.loc[
        (self.session.stimulus_presentations['stimulus_name'] == self.getStimName())
        ].index.values
        st = self.session.stimulus_presentations.loc[stim_presentation_ids].start_time.tolist()
        end = self.session.stimulus_presentations.loc[stim_presentation_ids].stop_time.tolist()

        velocity = {}
        velocity_times = {}
        for i, p_id in enumerate(stim_presentation_ids):
            inds = self.session.running_speed[(self.session.running_speed.start_time > st[i]) 
                                                & (self.session.running_speed.start_time < end[i])].index.values
            velocity[p_id] = np.array(self.session.running_speed.loc[inds].velocity.tolist())
            velocity_times[p_id] = np.array(((self.session.running_speed.loc[inds].start_time - st[i])*1000).tolist())
        if self.save:
            pickle.dump(velocity, open(self.save_path + "velocity_" + file_chars, "wb"))
            pickle.dump(velocity_times, open(self.save_path + "velocity_times_" + file_chars, "wb"))

        return velocity, velocity_times

    def getPupilDiameter(self): 
        file_chars = str(self.session_id) + "_" + self.getStimName()  + ".pkl"
        if self.file_check:
            try:
                pupil = pickle.load(open(self.save_path + "pupil_" + file_chars , "rb"))
                pupil_times = pickle.load(open(self.save_path + "pupil_times_" + file_chars , "rb"))
                return pupil, pupil_times
            except:
                pass

        #calculate min_d (.1%) of pupil:
        try:
            pupil_table = self.session.get_pupil_data()
            w = pupil_table.pupil_width.tolist()
        except:
            return None, None
        h = pupil_table.pupil_height.tolist()  
        w = np.array(pupil_table.pupil_width.tolist())
        h = np.array(pupil_table.pupil_height.tolist())
        d = np.max(np.vstack([w, h]), axis = 0)
        d = d[np.where(d != np.nan)[0]]
        min_d = np.sort(d)[int(d.size/10000)]


        stim_presentation_ids = self.session.stimulus_presentations.loc[
        (self.session.stimulus_presentations['stimulus_name'] == self.getStimName())
        ].index.values
        st = self.session.stimulus_presentations.loc[stim_presentation_ids].start_time.tolist()
        end = self.session.stimulus_presentations.loc[stim_presentation_ids].stop_time.tolist()

        pupil = {}
        pupil_times = {}
        for i, p_id in enumerate(stim_presentation_ids):
            inds = pupil_table[(pupil_table.index > st[i]) 
                                             & (pupil_table.index < end[i])].index.values

            w = np.array(pupil_table.loc[inds].pupil_width.tolist())
            h = np.array(pupil_table.loc[inds].pupil_height.tolist())
            d = np.max(np.vstack([w, h]), axis = 0)
            p_t = np.array(((pupil_table.loc[inds].index - st[i])*1000).tolist())
            p_t = p_t[np.where(d != np.nan)[0]] #bc of nans
            d = d[np.where(d != np.nan)[0]] #bc of nans
            pupil[p_id] = d / min_d
            pupil_times[p_id] = p_t
        
        if self.save:
            pickle.dump(pupil, open(self.save_path + "pupil_" + file_chars, "wb"))
            pickle.dump(pupil_times, open(self.save_path + "pupil_times_" + file_chars, "wb"))

        return pupil, pupil_times


    def divideVelocity(self, conds, th_l = 1, th_h = 3, start_time = 0, end_time = 2000):
        velocity, vt = self.getVelocity()
        start_index = np.where(vt > start_time)[0][0]
        end_index = np.where(vt < end_time)[0][-1]
        velocity = velocity[start_index : end_index]
        conds_vel = {}
        for i in range(2):
            conds_vel[i] = {}
        for k, cond in conds.items():
            vel = np.mean(velocity[k])
            if vel < th_l:
                conds_vel[0][k] = cond
            elif vel > th_h:
                conds_vel[1][k] = cond
        return conds_vel


    @staticmethod
    #### by specifying conds we get the data of subset of conditions
    def getData(spikes, conds, start_time = 0, end_time = 2000, time_step = 10, per_bin = 1, shuffle = False):
        start_index = int(start_time / (time_step * per_bin))
        end_index = int(end_time / (time_step * per_bin))
        first_p = list(conds.keys())[0]
        num_p = len(conds.keys())
        num_units = len(spikes[first_p].keys())
        num_f = len(conds[first_p])
        data = np.zeros([num_p * (end_index - start_index), per_bin * num_units + num_f])
        p = 0
        for p_key in conds.keys():
            u = 0
            p_units = spikes[p_key]
            for unit_times in p_units.values():
                for t in unit_times:
                    if t < start_time or t >= end_time:
                        continue
                    trial = int(t / (per_bin * time_step))
                    bin_num = int(int(t - start_time) % (time_step * per_bin) / time_step)
                    data[p * (end_index - start_index)
                            + trial - start_index, u * per_bin + bin_num] = 1
                u +=1
            for f in range(num_f):
                if conds[p_key][f] == 'null':
                    data[p * (end_index - start_index): (p+1)
                            * int(end_index - start_index), per_bin * num_units + f] = 1000  ### in case of corrupted data (in the new code this should not happen) TODO: test
                    continue
                data[p * (end_index - start_index): (p+1) * (end_index - start_index)
                        , per_bin * num_units + f] = conds[p_key][f]
            p +=1

        for f in range(num_f): #### TODO: after test of not happening this should be removed
            inds = np.where(data[:, -f] > 500)[0]
            data = np.delete(data, inds, axis = 0)

        if shuffle == False:
            return data
        num_trials = int(time_step * data.shape[0] / (end_time - start_time)) 
        for t in range(num_trials):
            i_t = range(int(t*(end_time - start_time)/time_step),int((t+1)*(end_time - start_time)/time_step))
            for n in range(data.shape[1]):
                d_t = np.array(data[i_t, n])
                np.random.RandomState(123).shuffle(d_t)
                data[i_t, n] = d_t
        return data 


    @staticmethod
    def shuffleInTrial(data, trial_dur = 400, time_step = 10, per_bin = 1):
        sh_data = np.copy(data)
        trial_size = int (trial_dur / (time_step * per_bin))
        if data.shape[0] %  trial_size > 0:
            print ("DATA not correct; should be dividible")
            return None
        for trial in range(0, sh_data.shape[0], trial_size):
            for c in range(data.shape[1]):
                np.random.shuffle(sh_data[trial:trial+trial_size, c])
        return sh_data



