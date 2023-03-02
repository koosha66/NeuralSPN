import numpy as np

#SPN part:
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.Base import Sum, Product, get_nodes_by_type
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.EM import EM_optimization
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering
from spn.algorithms.LearningWrappers import learn_parametric

from spn.structure.Base import Context



CLUSTERING_MODEL=  AgglomerativeClustering(n_clusters = 2, affinity= "precomputed", linkage= "average")
BATCH_SIZE = None 
TOTAL_ITERATIONS = 100
EPSILON = 10 ** -20



class NeuroSPN:
    ''' 
    The class for nSPN with different structures (shallow, deep, mixed) 
    Attributes:
    -----------
    data: Integer (binary) (numpy 2d array) 
        spiking data (spike/no spike in each time bin) dimensions: Time_bins * Neurons 
    
    num_neurons: Integer 
        number of neurons (data.shape[1]) / useful for readabilty of the code 
    
    I: numpy 2d array, dimensions: Neurons * Neurons
        Marginal pairwise mutual information matrix (base  = 2)  
    '''
    
    def __init__(self, data = np.array([])): # ,  I_th = 10 ** -5
        '''
        Constructor of nSPN
            build the marginal pairwise MI matrix based on the data

            Parameters: 
                data: Integer (binary) (numpy 2d array)  
                      dimensions: Time_bins * Neurons 
        '''
        self.data = data 
        if self.data.size == 0:
            self.num_neurons = 0
            return 
            
        self.num_neurons = self.data.shape[1]
        #calculate marginal pairwise MI 
        ps = np.vstack([np.mean(self.data, axis = 0), 1 - np.mean(self.data, axis = 0)])
        self.H = entropy(ps, base = 2, axis = 0)
        #self.I_th = I_th #### threshold of independence check on MI
        self.I = np.zeros([self.num_neurons,self.num_neurons])
        for i in range(self.num_neurons):
            for j in range(i + 1, self.num_neurons):
                pair_ps = np.histogramdd(self.data[:, [i,j]], bins = 2)[0].flatten()
                H_i_j = entropy(pair_ps, base = 2)
                self.I[i, j] = self.I[j, i] = self.H[i] + self.H[j] - H_i_j

    def train(self, spn, total_iters = TOTAL_ITERATIONS, num_iter_chunk = 1, ll_th = 5 * (10 ** -2), return_ll = True):
        if ll_th <= EPSILON:
            if not return_ll:
                EM_optimization (spn, self.data, iterations = total_iters)
                return spn
            num_iter_chunk = total_iters
        pre_ll = -1000
        for _ in range(0, total_iters, num_iter_chunk):
            EM_optimization (spn, self.data, iterations = num_iter_chunk)
            avg_ll = np.mean(log_likelihood(spn, self.data))
            if avg_ll - pre_ll  <  ll_th:
                #print ("EPOCH FINISHED", pre_ll, avg_ll, chunk_ind)
                break
            pre_ll = avg_ll
        return spn, pre_ll

    def shallowSPN(self, num_comps, scopes = np.array([])):
        if scopes.size < 1:
            scopes = np.arange(self.data.shape[1])
        comps = []
        for _ in range(num_comps):
            neurons = []
            for s in scopes:
                w_spike_init = np.random.random_sample()
                neurons.append(Sum(children = [Categorical(p = [1 - EPSILON, EPSILON], scope = int(s)), 
                Categorical(p = [EPSILON, 1 - EPSILON], scope = int(s))], weights = [1 - w_spike_init, w_spike_init]))
            comps.append(Product(children = neurons))
        spn = comps[0]
        if num_comps > 1:
            spn = Sum(weights = np.ones(num_comps) / num_comps, children = comps)
        assign_ids(spn)
        rebuild_scopes_bottom_up(spn)
        return spn 

    def learnSPN(self, min_data = 0, ind_th = .1, row_split = "kmeans"):
        if min_data == 0:
            min_data = int(self.data.shape[0] / 20) # default is 5% of the data
        ds_context = Context(parametric_types = self.num_neurons * [Categorical]).add_domains(self.data) 
        try:
            spn  = learn_parametric(self.data, ds_context, cols = "rdc", rows = row_split, min_instances_slice = min_data, threshold = ind_th)
        except:
            return None
        leaves = get_nodes_by_type(spn, ntype = Categorical)
        for l_node in leaves:
            if len(l_node.p) == 1:
                l_node.p = [1 - 10**-4, 10**-4]
            for i in range(2):
                if l_node.p[i] == 0:
                    l_node.p[i] = 10 ** -4
                    l_node.p[1 - i] = 1 - 10 ** -4
        return spn

