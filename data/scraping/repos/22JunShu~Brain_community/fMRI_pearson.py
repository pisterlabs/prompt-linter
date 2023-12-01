import igraph as ig
import leidenalg as la #https://leidenalg.readthedocs.io/en/latest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from igraph import Graph #https://python.igraph.org/en/stable/install.html
from modwt import modwt, modwtmra
from scipy.stats import t, pearsonr
from numpy.linalg import inv
import pickle
from scipy.signal import coherence
class fMRISeq():
    def __init__(self,data_seq,l_window = 20,step =None):
        self.time_seq = data_seq
        self.num_time = data_seq.shape[1]
        self.num_node = data_seq.shape[0]
        self.l_window = l_window
        self.corr_mats = None
        self.corr_mats_p = None
        self.rcor_mats = None
        self.MSC_mats_p = None
        self.temp_Q = None
        self.membership_community_array = None
        self.allegiance_matrix = None
        if step is None:
            self.step = l_window
        else:
            self.step = step
        self.n_window = int((self.num_time-self.l_window)/self.step)+1
        name91_str = loadmat(r"E:\NS\Primate fMRI\data\name91_str")['name91_str']
        map = {}
        name91_str = name91_str.reshape(-1)
        name91=  [name[0] for name in name91_str]
        name91[27] = "9-46v"
        name91[35] = "TEam-a"
        name91[81] = "TEam-p"
        name91[40] = "Core"
        name91[41] = "9-46d"
        name91[84] = "TH-TF"
        name91[49] = "PIR"
        name91[58] = "INS"
        name91[77] = "29-30"
        name91[25] = "ProSt"
        name91[63] = "Pi"
        self.node_names = name91
    def Modwt(self,method='db4',total_level = 5,level = 1):
        '''
        perform wavelet transform on the time series
        '''
        data_wt = modwt(self.time_seq,method,total_level)
        self.time_seq_wt = data_wt[level*self.num_node:(level+1)*self.num_node,:]
        return self.time_seq_wt
    def plot_Modwt(self,node_idx):
        '''
        plot the wavelet transform of a node
        '''
        plt.plot(self.time_seq_wt[node_idx,:])
        plt.show()
    def calc_pearson_mat(self,wavelet_transform = True,p_val = True):
        '''
        calculate the pearson correlation matrix using the pearsonr function
        '''
        l_window = self.l_window
        time_step = self.step
        n_window = self.n_window
        data_wt = self.time_seq
        if wavelet_transform:
            data_wt = self.Modwt()
        corr_mats = np.zeros((self.n_window,self.num_node,self.num_node))
        if p_val:
            corr_mats_p = np.zeros((self.n_window,self.num_node,self.num_node))
        for i in range(n_window):
            corr_mats[i,:,:] = np.array([[pearsonr(data_wt[j,i*time_step:i*time_step+l_window],data_wt[k,i*time_step:i*time_step+l_window])[0] for j in range(91)] for k in range(91)])
            if p_val:
                corr_mats_p[i,:,:] = np.array([[pearsonr(data_wt[j,i*time_step:i*time_step+l_window],data_wt[k,i*time_step:i*time_step+l_window])[1] for j in range(91)] for k in range(91)])
        self.corr_mats = corr_mats
        if p_val:
            self.corr_mats_p = corr_mats_p
            return corr_mats, corr_mats_p
        else:
            return corr_mats
    def cal_corcoeff_mat(self,wavelet_transform = False,p_val = False):
        '''
        calculate the correlation matrix using the corcoeff function
        '''
        l_window = self.l_window
        time_step = self.step
        n_window = self.n_window
        data_wt = self.time_seq
        if wavelet_transform:
            data_wt = self.Modwt()
        corr_mats = np.zeros((self.n_window,self.num_node,self.num_node))
        if p_val:
            corr_mats_p = np.zeros((self.n_window,self.num_node,self.num_node))
        for i in range(n_window):
            corr_mats[i,:,:] = np.corrcoef(data_wt[:,i*time_step:i*time_step+l_window])
            if p_val:
                corr_mats_p[i,:,:] = np.array([[pearsonr(data_wt[j,i*time_step:i*time_step+l_window],data_wt[k,i*time_step:i*time_step+l_window])[1] for j in range(91)] for k in range(91)])
        self.corr_mats = corr_mats
        if p_val:
            self.corr_mats_p = corr_mats_p
            return corr_mats, corr_mats_p
        else:
            return corr_mats
    def plot_corr_mat(self,window_idx,cmap='viridis'):
        '''
        plot the correlation matrix of a window
        '''
        plt.figure(figsize=(10,10))
        sns.heatmap(self.corr_mats[window_idx,:,:],cmap=cmap, center=0, vmin=-1, vmax=1, square=True)
        plt.show()
    def construct_pos_graph(self):
        """
        Construct a graph considering positive correlation as edges
        """
        corr_mats_pos = self.corr_mats_p.copy()
        if corr_mats_pos is None:
            corr_mats_pos = self.corr_mats.copy()
        corr_mats_pos[corr_mats_pos < 0 ] = 0
        graph_pos = Graph.Weighted_Adjacency(corr_mats_pos[0, :, :], mode='undirected', attr='weight',loops = False)
        graph_pos.vs['id'] = np.arange(91)
        graph_pos.vs['node_size'] = 91
        self.graph_pos = graph_pos
        return graph_pos
    def construct_neg_graph(self):
        """
        Construct a graph considering negative correlation as edges
        """
        corr_mats_neg = self.corr_mats_p.copy()
        if corr_mats_neg is None:
            corr_mats_neg = self.corr_mats.copy()
        corr_mats_neg[corr_mats_neg > 0 ] = 0
        graph_neg = Graph.Weighted_Adjacency(-corr_mats_neg[0, :, :], mode='undirected', attr='weight',loops = False)
        graph_neg.vs['id'] = np.arange(91)
        graph_neg.vs['node_size'] = 91
        self.graph_neg = graph_neg
        return graph_neg
    def partition_pos(self,optimize=False,resolution_parameter = 1,seed = None):
        """
        Partition the positive graph
        """
        graph_pos = self.construct_pos_graph()
        optimiser = la.Optimiser()
        partition_pos = la.RBConfigurationVertexPartition(graph_pos,
                                                weights='weight', resolution_parameter=resolution_parameter)
        if optimize:
            if seed is not None:
                optimiser.set_rng_seed(seed)
            diff_pos = optimiser.optimise_partition(partition_pos,n_iterations = -1)
        self.partition_pos_result = partition_pos
        return partition_pos
    def partition_neg(self,optimize=False,resolution_parameter = 1,seed=None):
        """
        Partition the negative graph
        """
        graph_neg = self.construct_neg_graph()
        optimiser = la.Optimiser()
        partition_neg = la.RBConfigurationVertexPartition(graph_neg,
                                                weights='weight', resolution_parameter=resolution_parameter)
        if optimize:
            if seed is not None:
                optimiser.set_rng_seed(seed)
            diff_neg = optimiser.optimise_partition(partition_neg,n_iterations = -1)
        self.partition_neg_result = partition_neg
        return partition_neg
    def partition_full(self,resolution_parameter = 1,seed = None):
        """
        Partition the full graph including neg and pos partitions

        returns the partition of the positive graph(negative is the same)
        """
        optimiser = la.Optimiser()
        if seed is not None:
            optimiser.set_rng_seed(seed)
        self.partition_pos(resolution_parameter=resolution_parameter)
        self.partition_neg(resolution_parameter=resolution_parameter)
        partition_pos = self.partition_pos_result
        partition_neg = self.partition_neg_result
        diff =  optimiser.optimise_partition_multiplex(
  [partition_pos, partition_neg],
  layer_weights=[1,-1])
        self.partition_pos_result = partition_pos
        self.partition_neg_result = partition_neg
        return partition_pos
    def MVDR(self,num1,num2,L = 4,time_range = None,k_list=range(1,5),step=1):
        if time_range is None:
            n = self.num_time
            time_range = [0,n]
        else:
            n = time_range[1] - time_range[0]
        seq = self.time_seq[:,time_range[0]:time_range[1]]
        s1 = np.zeros((L,n-L+1))
        s2 = np.zeros((L,n-L+1))    
        for i in range(n-L+1):
            # print(seq[num1,i*step:i*step+L],seq[num2,i*step:i*step+L])
            # print(seq)
            s1[:,i] = seq[num1,i*step:i*step+L]
            s2[:,i] = seq[num2,i*step:i*step+L]
        R1 = np.cov(s1)
        R2 = np.cov(s2)
        R12 = np.cov(s1,s2)[0:L,L:2*L]
        r_array = np.array([r_xy(i/4,R1,R2,R12,L=L) for i in k_list],dtype=float)
        r_mean = np.mean(r_array)
        return r_mean
    def calc_MSC(self,L = 4,k_list=range(1,5),save = False):
        time_step = self.step
        n_window = self.n_window
        rcor_mats = np.zeros((n_window,91,91))
        # set a progress bar
        from tqdm import tqdm
        for k in tqdm(range(n_window)):
            r_corr = np.zeros((91,91))
            for i in range(91):
                for j in range(i+1,91):
                    r_corr[i,j] = self.MVDR(i,j,L=L,k_list = k_list,time_range = [k*time_step,k*time_step+self.l_window])
                    r_corr[j,i] = r_corr[i,j]
            rcor_mats[k,:,:] = r_corr
        self.rcor_mats = rcor_mats
        # for k in range(n_window):
        #     r_corr = np.zeros((91,91))
        #     for i in range(91):
        #         for j in range(i+1,91):
        #             r_corr[i,j] = self.MVDR(i,j,L=L,k_list = range(1,5),time_range = [k*time_step,k*time_step+self.l_window]) # a strong correlation between the length of time range and the value of r
        #             r_corr[j,i] = r_corr[i,j]
        #     rcor_mats[k,:,:] = r_corr
        # self.rcor_mats = rcor_mats
        if save:
            self.save_results("results.pkl")
        return rcor_mats
    def welch_cor(self,num1,num2,nperseg = None,fs = 0.5,time_range = None):
        if time_range is None:
            n = self.num_time
            time_range = [0,n]
        else:
            n = time_range[1] - time_range[0]
        seq1 = self.time_seq[num1,time_range[0]:time_range[1]]
        seq2 = self.time_seq[num2,time_range[0]:time_range[1]]
        f, Cxy = coherence(seq1, seq2, fs = fs,nperseg = nperseg)
        # select the Cxy with f in [0.05,0.15]
        Cxy = Cxy[(f>0.05)&(f<0.15)]
        return np.mean(Cxy)
        
    def welch_MSC(self,nperseg = None,fs = 0.5,save = False):
        if nperseg is None:     
            nperseg = int(self.l_window/2)
        time_step = self.step
        n_window = self.n_window
        rcor_mats = np.zeros((n_window,91,91))
        for k in range(n_window):
            r_corr = np.zeros((91,91))
            for i in range(91):
                for j in range(i+1,91):
                    r_corr[i,j] = self.welch_cor(i,j,nperseg = nperseg,fs = fs,time_range=[k*time_step,k*time_step+self.l_window])
                    r_corr[j,i] = r_corr[i,j]
            rcor_mats[k,:,:] = r_corr
        self.rcor_mats = rcor_mats
        if save:
            self.save_results("results.pkl")
        return rcor_mats
    def pval_MSC(self,L = 4,k_list=range(1,5),p_thresh = 0.05):
        if self.rcor_mats is None:
            self.calc_MSC()
        n_window_r = self.rcor_mats.shape[0]
        rcor_mats = self.rcor_mats
        time_step = self.step
        pval_mats = np.zeros((n_window_r,91,91))
        for i in range(n_window_r):
            MSC_mat = rcor_mats[i,:,:]
            t_mat = MSC_mat * np.sqrt((time_step-L+1- 2)/(1 - MSC_mat**2))
            pval_mats[i,:,:] = t.sf(np.abs(t_mat), time_step-L+1-2) 
        MSC_mats_p = rcor_mats.copy()
        MSC_mats_p[pval_mats>p_thresh] = 0
        # show the MSC matrix of the first window
        # plt.figure(figsize=(10,10))
        # sns.heatmap(MSC_mats_p[0,:,:], cmap='viridis', center=0, square=True, cbar_kws={"shrink": .5})
        self.MSC_mats_p = MSC_mats_p
        return MSC_mats_p
    def cluster_MSC(self,pval = False,L = 4,k_list=range(1,5),resolution_parameter = 1,interslice_weight=1,seed = None,max_communities = 0,consider_empty_community = True):
        """
        perform clustering using MSC map

        return: membership_array_r (n_window,node_num)
        """
        if pval is True:
            if self.MSC_mats_p is None:
                self.pval_MSC()
            MSC_mats_p = self.MSC_mats_p
        else:
            if self.rcor_mats is None:
                self.calc_MSC()
            MSC_mats_p = self.rcor_mats
        n_window_r = self.n_window
        graph_list_r = []
        for i in range(n_window_r):
            graph_r = Graph.Weighted_Adjacency(MSC_mats_p[i, :, :], mode='undirected', attr='weight',loops = False)
            graph_r.vs['id'] = np.arange(91)
            graph_r.vs['node_size'] = 91
            graph_list_r.append(graph_r)

        optimiser = la.Optimiser()
        optimiser.consider_empty_community = consider_empty_community
        if seed is not None:
            optimiser.set_rng_seed(seed)
        optimiser.max_comm_size = max_communities
        layers_r,interSliceLayer_r,Gfull_r = la.time_slices_to_layers(graph_list_r,interslice_weight = interslice_weight)
        partitions_r = [la.RBConfigurationVertexPartition(H,
                                                weights='weight', resolution_parameter=resolution_parameter)
                    for H in layers_r]
        interslice_partition_r = la.CPMVertexPartition(interSliceLayer_r, resolution_parameter=0, node_sizes='node_size', weights='weight')
        diff_r = optimiser.optimise_partition_multiplex(partitions_r+[interslice_partition_r],layer_weights = [1]*n_window_r+[1],n_iterations = -1)
        membership_array_r = np.array(partitions_r[0].membership).reshape(-1,91)
        # membership_r = {(v['slice'], v['id']): m for v, m in zip(Gfull_r.vs, partitions_r[0].membership)}
        # membership_time_slices_r = []
        # for slice_idx, H in enumerate(graph_list_r):
        #     membership_slice_r = [membership_r[(slice_idx, v['id'])] for v in H.vs]
        #     membership_time_slices_r.append(list(membership_slice_r))
        self.membership_array_r = membership_array_r
        Q_list = []
        for item in partitions_r:
            Q_list.append(item.q)
        self.temp_Q = sum(Q_list)+interslice_partition_r.q
        return membership_array_r
    def plot_cluster(self,membership_array_r=None,cmap = 'viridis'):
        if membership_array_r is None:
            if self.membership_array_r is None:
                self.cluster_MSC()
            membership_array_r = self.membership_array_r
        plt.figure(figsize=(10,10))
        n_clusters =np.max(membership_array_r)+1
        color_list = sns.color_palette('hls', n_clusters)
        heatmap = sns.heatmap(membership_array_r, cmap=color_list, vmin=0, vmax=n_clusters-1, yticklabels=5)

        # Adjust the color bar ticks
        cbar = heatmap.collections[0].colorbar
        cbar.set_ticks(range(n_clusters))
        cbar.set_ticklabels(range(n_clusters))

        plt.ylabel('time point (%ds)'%(int(2*self.step)))
        plt.title('modularity change')
        plt.show()
        pass
    def node_switch(self):
        membership_array_r = self.membership_array_r
        # calculate the number of time that a node switch its community
        n_switch = np.zeros((91,1))
        for i in range(91):
            n_switch[i] = sum(np.diff(membership_array_r[:,i])!=0)
        self.n_switch = n_switch
        return n_switch
    def node_coverage(self):
        """
        calculate the coverage of each node, which is the number of communities that the node once belongs to
        """
        if self.membership_community_array is None:
            self.membership_community()
        membership_community = self.membership_community_array #(n_clusters,self.n_window,self.num_node)
        n_clusters = self.n_clusters
        membership_sumwindow = 1-np.abs(np.prod(membership_community-1,axis = 1)) #(n_clusters,self.num_node)
        n_coverage = np.sum(membership_sumwindow,axis = 0)/n_clusters
        self.n_coverage = n_coverage
        return n_coverage
    def edge_switch(self):
        membership_array_r = self.membership_array_r
        # calculate the number of time that a edge switch its community
        n_switch = np.zeros((91,91))
        for i in range(91):
            for j in range(i+1,91):
                n_switch[i,j] = sum(np.diff(membership_array_r[:,i]!=membership_array_r[:,j])!=0)
                n_switch[j,i] = n_switch[i,j]
        self.n_switch = n_switch
        return n_switch
    def save_results(self,filename='results.pkl'):
        # save the network to a file
        with open(filename,'wb') as f:
            pickle.dump(self,f)
    def load_results(self,filename):
        # load the network from a file
        with open(filename,'rb') as f:
            network = pickle.load(f)
        return network
    def visualize_single(self,membership):
        """
        convert to the form {"V1":1,"F1":2}
        """
        membership_dict = {}
        for i in range(91):
            membership_dict[self.node_names[i]] = membership[i]
        return membership_dict
    def visualize_all(self):
        """
        return all membership dict in a trial

        the dicts are put in a list
        """
        membership_array = self.membership_array_r
        membership_dict_list = []
        for i in range(membership_array.shape[0]):
            membership = membership_array[i,:]
            membership_dict = self.visualize_single(membership)
            membership_dict_list.append(membership_dict)
        return membership_dict_list
    def print_membership_dicts(self,membership_dict_list):
        """
        membership_dict_list should be a list
        """
        import json
        for membership_dict in membership_dict_list:
            print(json.dumps(membership_dict))
        pass
    def membership_community(self,membership_array_r=None):
        """
        turn the membership array to a binary array (n_clusters,self.n_window,self.num_node)
        """
        if self.membership_array_r is None:
            self.cluster_MSC(L = 4,k_list=range(1,5))
        membership_array_r = self.membership_array_r
        # find the total number of community
        n_clusters = np.max(membership_array_r)+1
        self.n_clusters = n_clusters

        # find the member of each community at each time point; 0: not a member; 1: a member
        membership_community = np.zeros((n_clusters,self.n_window,self.num_node))
        for i in range(n_clusters):
            membership_community[i,:,:] = (membership_array_r == i)
        self.membership_community_array = membership_community
        return membership_community
    def stable_community(self,L = 4,k_list=range(1,5)):
        '''
        find the stable community structure according to the membership array
        '''
        if self.membership_community_array is None:
            self.membership_community()
        membership_community = self.membership_community_array
        n_clusters = np.max(self.membership_array_r)+1
        self.n_clusters = n_clusters

        # calculate the number of same nodes in each community at adjacent time points (performing AND operation)
        n_same = np.zeros((n_clusters,self.n_window-1))
        for i in range(n_clusters):
            for j in range(self.n_window-1):
                n_same[i,j] = np.sum(membership_community[i,j,:]*membership_community[i,j+1,:])

        # calculate the total number of nodes in each community at adjacent time points (performing OR operation)
        n_total = np.zeros((n_clusters,self.n_window-1))
        for i in range(n_clusters):
            for j in range(self.n_window-1):
                n_total[i,j] = np.sum(membership_community[i,j,:]+membership_community[i,j+1,:]) - n_same[i,j] + 1e-8
        
        # calculate the ratio of same/total
        ratio = n_same/n_total # size: (n_clusters,n_window-1)
        ratio_sum = np.sum(ratio,axis = 1) # size: (n_clusters,1)
        
        # calculate whether the communities appear in each time point, 0: not appear; 1: appear
        community_appear = np.zeros((n_clusters,self.n_window))
        for i in range(n_clusters):
            for j in range(self.n_window):
                community_appear[i,j] = [0,1][np.sum(membership_community[i,j,:])>0]
        
        # calculate the number of time that a community appear consecutively, which is the number of time that (1,1) appears in the community_appear matrix 
        n_consecutive = np.sum((1-np.diff(community_appear))*community_appear[:,1:],axis = 1) # size: (n_clusters,1)
        
        # calculate the stability of each community
        stability = ratio_sum/n_consecutive # size: (n_clusters,1)

        self.stability = stability
        return stability
    def node_entropy(self,member_array_r = None):
        '''
        calculate the entropy of each node
        '''
        from scipy.stats import entropy
        if member_array_r is None:
            member_array_r = self.membership_array_r
            if member_array_r is None:
                self.cluster_MSC()
                member_array_r = self.membership_array_r
        n_window = self.n_window
        entropy_list = []
        for i in range(91):
            # count the number of times that each community appears
            count = np.zeros((self.n_clusters,1))
            for j in range(self.n_clusters):
                count[j] = sum(member_array_r[:,i] == j)
            # calculate the entropy
            entropy_list.append(entropy(count))
        self.entropy = entropy_list
        return entropy_list
    def node_entropy_plot(self,entropy_list = None):
        '''
        plot the entropy of each node
        '''
        if entropy_list is None:
            if self.entropy is None:
                self.node_entropy()
            entropy_list = self.entropy
        plt.figure(figsize=(20,5))
        sns.barplot(x = np.arange(91),y = entropy_list)
        plt.show()
    def central_nodes(self,top_n = 5):
        '''
        find the central nodes of each community
        '''
        if self.membership_community is None:
            self.stable_community()
        membership_community = self.membership_community
        # find the top n nodes with the highest frequency of each community
        central_nodes = np.zeros((self.n_clusters,top_n))
        freq_mat = np.sum(membership_community,axis = 1)
        for i in range(self.n_clusters):
            # find the frequency of each node in the community
            freq = freq_mat[i,:]
            # find the top 5 nodes with the highest frequency
            central_nodes[i,:] = np.argsort(freq)[-top_n:]
        self.centralNodes = central_nodes
        return central_nodes
    def mutual_info(self,membership_array_r = None,show_mat = False):
        '''
        calculate the mutual information between the members
        '''
        if membership_array_r is None:
            if self.membership_array_r is None:
                self.cluster_MSC()
            membership_array_r= self.membership_array_r
        from sklearn.metrics import mutual_info_score
        mutual_info_mat = np.zeros((self.num_node,self.num_node))
        for i in range(self.num_node):
            for j in range(i,self.num_node):
                mutual_info_mat[i,j] = mutual_info_score(membership_array_r[:,i],membership_array_r[:,j])
                mutual_info_mat[j,i] = mutual_info_mat[i,j]
        self.mutual_info_mat = mutual_info_mat
        if show_mat:
            sns.heatmap(mutual_info_mat,cmap = 'viridis')
            plt.show()
        return mutual_info_mat

    def modular_allegiance_matrix(self,membership_community= None):
        """
        for each time window, construct the modular allegiance matrix T whose binary elements Tij indicate if two nodes i and j have been assigned to the same module or not

        Input argument: 
            membership_community 
                binary matrix with shape: (n_clusters,self.n_window,self.num_node) 

        Return:
            T: modular allegiance matrix with shape (n_window,num_node,num_node)
            self.allegiance_matrix = T
        """
        if membership_community is None:
            if self.membership_community_array is None:
                self.membership_community()
            membership_community_array = self.membership_community_array.copy()
        else:
            membership_community_array = membership_community.copy()
        n_clusters = self.n_clusters
        n_window = self.n_window
        num_node = self.num_node
        assert membership_community_array.shape == (n_clusters,n_window,num_node)
        # change the array to (n_window,n_clusters,num_node) and (n_window, num_node,n_clusters)
        membership_community_array_1 = np.transpose(membership_community_array,(1,0,2))
        membership_community_array_2 = np.transpose(membership_community_array,(1,2,0))
        # calculate the modular allegiance matrix
        T = np.matmul(membership_community_array_2,membership_community_array_1)     # binary matrix of size(n_window,num_node,num_noed)
        assert T.shape == (n_window,num_node,num_node)
        self.allegiance_matrix = T
        return T
    # calculate the strength of connection of each node; return a matrix of size 91*num_communities
    def calc_strength(cor_mat,membership):
        """
        Calculate the strengths(total edge weights) of node i's edges to all communities

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1)

        Return: strength_mat(n_nodes,num_communities)
        """
        cor_mat = np.multiply(cor_mat,1-np.eye(cor_mat.shape[0]))
        strength_mat = np.zeros((cor_mat.shape[0],max(membership)+1))
        membership = np.array(membership)
        for i in range(max(membership)+1):
            strength_mat[:,i] = np.mean(cor_mat[:,membership==i],axis=1)
        return strength_mat
    def participation_coef(self,cor_mat,membership):
        """
        Calculate the participation coefficient of each node

        $P_i = 1- \sum_{k=1}^{num_communities}(\frac{S_i{C_k}}{S_{i}})^2$

        $ S_i{C_k} = \sum_{j\in C_k}A_{ij}$

        $S_i = \sum_{j}A_{ij}$

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1)

        Return: p_coef(n_nodes,1)
        """
        strength_mat = self.calc_strength(cor_mat,membership)
        s_tot = np.sum(strength_mat,axis=1) + 1e-7
        s_ratio = strength_mat/s_tot[:,np.newaxis]
        p_coef = 1-np.sum(np.power(s_ratio,2),axis=1)
        return p_coef
    def intra_community_strength(self,cor_mat,membership):
        """
        Calculate the intra-community strength of each node

        $z_i = \frac{S_i{C_i}-\langle S_{C_i}\rangle}{\sigma_{C_i}}$

        $\langle S_{C_i}\rangle = \frac{1}{N_{C_i}}\sum_{j\in C_i}S_{jC_i}$

        $\sigma_{C_i}}$ is the standard deviation of $S_{jC_i}$

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1)
        Return: z_intra(n_nodes,1)
        """
        strength_mat = self.calc_strength(cor_mat,membership)
        membership = np.array(membership)
        mean_commu_strength = np.zeros((strength_mat.shape[1],1))
        std_commu_strength = np.zeros((strength_mat.shape[1],1))    
        z_intra = np.zeros((strength_mat.shape[0],1))
        for i in range(strength_mat.shape[1]):
            mean_commu_strength[i] = np.mean(strength_mat[membership==i,i])
            std_commu_strength[i] = np.std(strength_mat[membership==i,i]) + 1e-7
        for i in range(strength_mat.shape[0]):
            membership_i = membership[i]
            z_intra[i] = (strength_mat[i,membership_i] - mean_commu_strength[membership_i])/std_commu_strength[membership_i]
        return z_intra
    def between_module_connectivity(cor_mat,membership,mod1,mod2):
        """
        Calculate the between-module connectivity of two modules 

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1), mod1(int), mod2(int)

        Return: I_ij(float)

        $I_{ij} = \frac{\sum_{i\in C_i}\sum_{j\in C_j}A_{ij}}{N_iN_j}$
        
        """
        cor_mat = np.multiply(cor_mat,1-np.eye(cor_mat.shape[0]))
        membership = np.array(membership)
        I_12 = np.sum(cor_mat[membership==mod1,:][:,membership==mod2])/np.sum(membership==mod1)/np.sum(membership==mod2)
        return I_12
    def relative_interaction_strength(self,cor_mat,membership,mod1,mod2):
        """
        Calculate the relative interaction strength between two modules

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1), mod1(int), mod2(int)

        Return: R_12(float)

        $R_{ij} = \frac{I_{ij}}{\sqrt{I_{ii}I_{jj}}}$
        """
        I_12 = self.between_module_connectivity(cor_mat,membership,mod1,mod2) + 1e-7
        I_11 = self.between_module_connectivity(cor_mat,membership,mod1,mod1) + 1e-7
        I_22 = self.between_module_connectivity(cor_mat,membership,mod2,mod2) + 1e-7
        R_12 = I_12/np.sqrt(I_11*I_22)
        return R_12
    def relative_interaction_strength_mat(self,cor_mat,membership):
        """
        Calculate the relative interaction strength between all pairs of modules

        Input: cor_mat(n_nodes,n_nodes), membership(n_nodes,1)

        Return: R_mat(n_clusters,n_clusters)

        """
        n_clusters = max(membership)+1
        R_mat = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(i,n_clusters):
                R_mat[i,j] = self.relative_interaction_strength(cor_mat,membership,i,j)
                R_mat[j,i] = R_mat[i,j]
        return R_mat

def f_k(k,L=4):
    return np.array([np.exp(1j*2*np.pi*k*i/L) for i in range(L)]).reshape(-1,1)/np.sqrt(L)
def r_xy(k,R1,R2,R12,L=4):
    f_k_vec = f_k(k,L)
    R1_inv = inv(R1)
    R2_inv = inv(R2)
    # the conjugate transpose of f_k
    f_k_conj = np.conj(f_k_vec).T
    # calculate S(k) = fkH*R1−1* R12*R2-1*fk/([fkH R−1 x1x1 fk ][fkH R−1 x2x2 fk ])
    r_k = (np.linalg.norm(f_k_conj @ R1_inv @ R12 @ R2_inv @ f_k_vec))**2/ np.linalg.norm(f_k_conj @ R1_inv @ f_k_vec)/np.linalg.norm(f_k_conj @ R2_inv @ f_k_vec)
    return r_k


        
