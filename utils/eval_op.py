'''
get_srr_precision_recall
get_srr_precision_recall_fast
get_recall_precision_at_k
'''
import numpy as np
import tensorflow as tf

# hash keycomes in different way
class HashTable2:
    def __init__(self, hash_idx_set, labelh, nbucket, nlabel):
        '''
        Args:
            hash_idx_set - Numpy 2d array [nhash, nk]
                    Every value in [0, nbucket)
            labelh - Numpy 1D array [nhash]
                the label of each hash key [0, nlabel)
            nbucket- the number of buckets
            nlabel - the number of labels
        '''
        self.hash_idx_set=hash_idx_set
        self.nhash, _ = self.hash_idx_set.shape
        self.labelh = labelh
        self.nbucket = nbucket
        self.nlabel = nlabel

        assert self.nhash==len(self.labelh), "The number of label doesn't match with the number of hash key"

        self.table = list() # hash table
        self.hash_distribution = np.zeros([self.nbucket, self.nlabel]) 
        for b_idx in range(self.nbucket):
            self.table.append(list())

        for h_idx in range(self.nhash):
            for bucket_idx in self.hash_idx_set[h_idx]:
                self.hash_distribution[bucket_idx][self.labelh[h_idx]]+=1
                self.table[bucket_idx].append(h_idx)

    def get_nmi(self):
        '''
        Return:
            nmi - int
        '''
        ncluster, nlabel = self.hash_distribution.shape
        cluster_array = np.sum(self.hash_distribution, axis=1)
        label_array = np.sum(self.hash_distribution, axis=0)

        ndata = np.sum(self.hash_distribution)

        cluster_prob = cluster_array/ndata
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

        label_prob = label_array/ndata
        label_entropy = 0
        for l_idx in range(nlabel):
            if label_prob[l_idx]!=0:
                label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

        mutual_information = 0
        for c_idx in range(ncluster):
            for l_idx in range(nlabel):
                if self.hash_distribution[c_idx][l_idx]!=0:
                    mutual_information += self.hash_distribution[c_idx][l_idx]/ndata*np.log2(ndata*self.hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
        norm_term = (cluster_entropy+label_entropy)/2
        return mutual_information/norm_term

    def get_suf(self):
        '''
        Return:
            suf - float
        '''
        srr = 0
        for h_idx in range(self.nhash):
            retrieve_set = set()
            for bucket_idx in self.hash_idx_set[h_idx]:
                retrieve_set |= set(self.table[bucket_idx])
            retrieve_set -= set([h_idx]) # exclude myself
            retrieve_set = np.array(list(retrieve_set))
            srr += len(retrieve_set)/(self.nhash-1)
        srr/=self.nhash
        suf=1.0/srr
        return suf

# HashTable insert index on the key
class HashTable:
    def __init__(self, hash_key, labelh, nlabel):
        '''
        Args:
            hash_key - Numpy 2d array [nhash, nbucket]
                should be binary 0 or 1
                Add index in the following hash key buckets
            labelh - Numpy 1D array [nhash]
                the label of each hash key
            nlabel - the number of labels
        '''
        self.hash_key = hash_key
        self.nhash, self.nbucket = self.hash_key.shape
        self.labelh = labelh
        self.nlabel = nlabel

        assert self.nhash==len(self.labelh), "Wrong label"

        self.table = list()
        self.hash_count = np.zeros(nlabel)
        for hash_idx in range(self.nhash): 
            self.hash_count[self.labelh[hash_idx]]+=1

        self.hash_distribution = np.zeros([self.nbucket, self.nlabel]) 
        for hash_idx in range(self.nhash):
            for b_idx in range(self.nbucket):
                if self.hash_key[hash_idx][b_idx]==1:
                    self.hash_distribution[b_idx][self.labelh[hash_idx]]+=1


        for b_idx in range(self.nbucket):
            self.table.append(list())

        for hash_idx in range(self.nhash):
            for b_idx in range(self.nbucket):
                if self.hash_key[hash_idx][b_idx]==1:
                    self.table[b_idx].append(hash_idx)

    def get_retrieve_set(self, query_key):
        '''
        Args:
            query_key - Numpy 1D array
        '''
        assert query_key.ndim==1, "Query key dimension should be 1"
        retrieve_set = set()
        for idx in range(len(query_key)):
            if query_key[idx]==1:
                retrieve_set |= set(self.table[idx])
        return list(retrieve_set)

    def get_nmi(self):
        '''
        Return:
            nmi - int
        '''
        ncluster, nlabel = self.hash_distribution.shape
        cluster_array = np.sum(self.hash_distribution, axis=1)
        label_array = np.sum(self.hash_distribution, axis=0)

        ndata = np.sum(self.hash_distribution)

        cluster_prob = cluster_array/ndata
        cluster_entropy = 0
        for c_idx in range(ncluster):
            if cluster_prob[c_idx]!=0:
                cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

        label_prob = label_array/ndata
        label_entropy = 0
        for l_idx in range(nlabel):
            if label_prob[l_idx]!=0:
                label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

        mutual_information = 0
        for c_idx in range(ncluster):
            for l_idx in range(nlabel):
                if self.hash_distribution[c_idx][l_idx]!=0:
                    mutual_information += self.hash_distribution[c_idx][l_idx]/ndata*np.log2(ndata*self.hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
        norm_term = (cluster_entropy+label_entropy)/2
        return mutual_information/norm_term

    def get_srr_recall_precision_at_k_hash(self, dist_matrix, query_key, labelq, base_activate_k, k_set, issame):
        '''
        Args:
            dist_matrix - Numpy 2d array [nquery, nhash]
                Distance from query from each data
            query_key - Numpy 2d array [nquery, nbucket] 
            labelq - Numpy 1D array [nquery]
            base_activate_k - int
            k_set - list (nk)
            issame - bool
                True => if query and dist is same 
                False => otherwise
        Return:
            srr - Numpy 2D array [nk, nquery] 
            recall_value_set - list [nk]
            precision_value_set - list [nk]
        '''
        nquery = len(query_key)
        assert query_key.shape==(nquery, self.nbucket), "Wrong query key shape"
        assert dist_matrix.shape == (nquery, self.nhash), "Wrong dist matrix"
        assert issame==(nquery== self.nhash), "Wrong issame parameter"

        nk, k_max = len(k_set), max(k_set) 
        recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
        srr = np.zeros([nk, nquery])

        for q_idx in range(nquery):
            for k_idx in range(nk):
                nretrieve = k_set[k_idx]
                query_sort = np.argsort(-query_key[q_idx])
                retrieve_set = set()

                exclude_set = set([q_idx]) if issame else set() # exclude my self if issame 
                for idx in range(base_activate_k):
                    retrieve_set |= set(self.table[query_sort[idx]])
                retrieve_set -= exclude_set

                idx = base_activate_k
                while len(retrieve_set)<nretrieve:
                    retrieve_set |= set(self.table[query_sort[idx]])
                    retrieve_set -= exclude_set
                    idx+=1 
                retrieve_set = np.array(list(retrieve_set))
                srr[k_idx][q_idx] = len(retrieve_set)
                retrieve_set = retrieve_set[np.argsort(dist_matrix[q_idx][retrieve_set])][:nretrieve]
                flag = 0
                for r_idx in retrieve_set:
                    if labelq[q_idx]==self.labelh[r_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag==0:
                            recall_correct_set[k_idx]+=1
                            flag=1
        if issame:
            srr/=(self.nhash-1)
        else:
            srr/=(self.nhash)

        recall_value_set = list()
        precision_value_set = list()
        for k_idx in range(nk):
            recall_value_set.append(recall_correct_set[k_idx]/nquery)
            precision_value_set.append(precision_correct_set[k_idx]/nquery/k_set[k_idx])

        return srr, recall_value_set, precision_value_set

def get_recall_precision_at_k(dist_matrix, labelq, labelh, k_set, issame=False):
    '''Get recall value with
    search space and query is different
    Dependency: numpy as np
    Args:
        dist_matrix - 2D numpy array [nquery, nhash]
        labelq - 1D numpy array [nquery]
        labelh - 1D numpy array [nhash]
        k_set - list of int which is the k value for recall [nk]
        issame - bool determine wheter two matrix and same
    Return:
        recall_value_set - list of float with length k_set
        precision_value_set - list of float with length k_set
    '''
    nquery, nhash = len(labelq), len(labelh)
    nk, k_max  = len(k_set), max(k_set) 

    assert dist_matrix.shape == (nquery, nhash), "Wrong dist_matrix shape dist_matrix shape : {}, and labelq({}), labelh({})".format(dist_matrix.shape, label.shape, labelh.shape)
    assert issame==(nquery==nhash), "label should be same"

    recall_correct_set, precision_correct_set = np.zeros(nk), np.zeros(nk)
    for idx1 in range(nquery):
        count = 0 # prevent useless counting
        idx_close_from_i = np.argsort(dist_matrix[idx1])
        flag_set = np.zeros(nk) # for recall double counting for recall
        for idx2 in idx_close_from_i:
            if issame and idx2==idx1:
                continue #if data1, and data2 is same, exclude same idx
            count+=1
            if labelq[idx1]==labelh[idx2]:
                for k_idx in range(nk):
                    if count<=k_set[k_idx]:
                        precision_correct_set[k_idx]+=1
                        if flag_set[k_idx]==0:
                            recall_correct_set[k_idx]+=1
                            flag_set[k_idx]=1
            if count>=k_max:
                break

    recall_value_set = list()
    precision_value_set = list()
    for k_idx in range(nk):
        k = k_set[k_idx]
        recall_value_set.append(recall_correct_set[k_idx]/nquery)
        precision_value_set.append(precision_correct_set[k_idx]/nquery/k)
    return recall_value_set, precision_value_set 

def get_nmi_suf_quick(index_array, label_array, ncluster, nlabel):
    '''
    Args:
        index_array - [ndata, k]
                value should be [0, ncluster)
        label_array - [ndata]
                value should be [0, nlabel)
        ncluster -  int
        nlabel - int
    Return:
        nmi - float
        suf - float
    '''
    if index_array.ndim==1:
        index_array = np.expand_dims(index_array, axis=-1)
    ndata, k_value = index_array.shape
    hash_distribution = np.zeros([ncluster, nlabel])
    
    for idx1 in range(ndata):
        tmp_l = label_array[idx1]
        for idx2 in range(k_value):
            hash_distribution[index_array[idx1][idx2]][tmp_l]+=1

    cluster_array = np.sum(hash_distribution, axis=1) # [ncluster]
    label_array = np.sum(hash_distribution, axis=0) # [nlabel]

    total_size = ndata*k_value

    cluster_prob = cluster_array/total_size
    cluster_entropy = 0
    for c_idx in range(ncluster):
        if cluster_prob[c_idx]!=0:
            cluster_entropy -= cluster_prob[c_idx]*np.log2(cluster_prob[c_idx])

    label_prob = label_array/total_size
    label_entropy = 0
    for l_idx in range(nlabel):
        if label_prob[l_idx]!=0:
            label_entropy -= label_prob[l_idx]*np.log2(label_prob[l_idx])

    mutual_information = 0
    for c_idx in range(ncluster):
        for l_idx in range(nlabel):
            if hash_distribution[c_idx][l_idx]!=0:
                mutual_information += hash_distribution[c_idx][l_idx]/total_size*np.log2(total_size*hash_distribution[c_idx][l_idx]/cluster_array[c_idx]/label_array[l_idx])
    norm_term = (cluster_entropy+label_entropy)/2
    nmi = mutual_information/norm_term

    suf = np.sum(cluster_array)/np.sum(np.square(cluster_array))
    suf *= ndata
    suf /= k_value
    return nmi, suf
