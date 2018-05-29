import numpy as np
import random

class BasicDatamanager(object):
    def __init__(self, image, label, nclass):
        self.image = image
        self.label = label
        self.nclass = nclass
        self.ndata = len(self.label)

        self.fullidx = np.arange(self.ndata)
        self.start = 0
        self.end = 0

    def print_shape(self):
        print("Image shape : {}({})".format(self.image.shape, self.image.dtype))
        print("Label shape : {}({})".format(self.label.shape, self.label.dtype))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata):
            counter[int(self.label)]+=1
        return counter

    def next_batch(self, batch_size):
        '''
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            image
            label
        '''
        if self.start == 0 and self.end ==0:
            np.random.shuffle(self.fullidx) # shuffle first
        
        if self.end + batch_size > self.ndata:
            self.start = self.end
            self.end = (self.end + batch_size)%self.ndata
            self.subidx = np.append(self.fullidx[self.start:self.ndata], self.fullidx[0:self.end])
            self.start = 0
            self.end = 0
        else:
            self.start = self.end
            self.end += batch_size
            self.subidx = self.fullidx[self.start:self.end]

        return self.image[self.subidx], self.label[self.subidx].astype('int32')

class ContrastDatamanager(object):
    def __init__(self, image, label, nclass):
        '''
        Args:
            image -
            label - 
            nclass - number of total classes
        '''
        self.image = image
        self.label = label
        self.nclass = nclass

        self.ndata = len(self.label)

        self.pos_label_idx_set = [[idx for idx in range(self.ndata) if self.label[idx]==class_idx] for class_idx in range(self.nclass)]
        self.neg_label_idx_set = [[idx for idx in range(self.ndata) if self.label[idx]!=class_idx] for class_idx in range(self.nclass)]

        self.fullidx = np.arange(self.ndata)
        self.start = 0
        self.end = 0

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata):
            counter[int(self.label)]+=1
        return counter

    def change_nsclass(self, value):
        self.nsclass = value

    def next_batch(self, batch_size):
        '''
        Make batch data which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            image - batch_size image
            image_pair - batch_size image
            label - batch_size binary label 
                    pos to be 1
                    neg to be 0
        '''
        if self.start==0 and self.end==0:
            np.random.shuffle(self.fullidx) # shuffle first
       
        npositive = batch_size//2
        nnegative = batch_size-npositive
        self.binary_label = np.append(np.ones(npositive), np.zeros(nnegative))
        
        if self.end + batch_size > self.ndata:
            self.start = self.end
            self.end = (self.end + batch_size)%self.ndata
            self.subidx = np.append(self.fullidx[self.start:self.ndata], self.fullidx[0:self.end])
            self.start = 0
            self.end = 0
        else:
            self.start = self.end
            self.end += batch_size
            self.subidx = self.fullidx[self.start:self.end]

        self.subidx_pair = list()
        for i in range(batch_size):
            anc_label = self.label[self.subidx[i]]
            if i < npositive:
                while True:
                    pos_sample = random.sample(self.pos_label_idx_set[anc_label], 1)[0]
                    if pos_sample!=self.subidx[i]:
                        break
                self.subidx_pair.append(pos_sample)
            else:
                self.subidx_pair.append(random.sample(self.neg_label_idx_set[anc_label], 1)[0])

        self.subidx_pair = np.array(self.subidx_pair)
        return self.image[self.subidx], self.image[self.subidx_pair], self.binary_label

class TripletDatamanager(object):
    def __init__(self, image, label, nclass, nsclass=2):
        '''
        Args:
            image -
            label - 
            nclass - number of total classes
            nsclass - When we select the batch, the number of classes it contain
        '''
        self.image = image
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass 

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        
        self.label_idx_set = list()
        for cls_idx in range(self.nclass):
            self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata):
            self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass):
            self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])
        
        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(v) for v in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[cls_idx], dtype=np.int32) for cls_idx in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata):
            counter[int(self.label)]+=1
        return counter

    def change_nsclass(self, value):
        self.nsclass = value

    def next_batch(self, batch_size):
        '''
        Make batch data which containes (batch_size/nsclass) data in nsclass

        Args:
            batch_size - int
                return batch size numbers of pairs
        Return:
            image - batch_size image
            label - batch_size label
        '''
        assert batch_size%self.nsclass == 0, "Batchsize(%d) should be divided by nsclass(%d)"%(batch_size, self.nsclass)

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first
        
        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class

                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.subidx = np.concatenate(self.subidx, axis=0)

        return self.image[self.subidx], self.label[self.subidx].astype('int32')

class NpairDatamanager(object):
    def __init__(self, image, label, nclass, nsclass):
        self.image = image
        self.label = label
        self.nclass = nclass
        self.nsclass = nsclass

        self.ndata = len(self.label)

        # list with [self.nclass] each element = idx set of which label is cls_idx
        # initialize
        self.label_idx_set = list()
        for cls_idx in range(self.nclass):
            self.label_idx_set.append(list())
        # append
        for d_idx in range(self.ndata):
            self.label_idx_set[self.label[d_idx]].append(d_idx)
        # to numpy
        for cls_idx in range(self.nclass):
            self.label_idx_set[cls_idx] = np.array(self.label_idx_set[cls_idx])

        self.valid_class_set = [cls_idx for cls_idx in range(self.nclass) if len(self.label_idx_set[cls_idx])>1]
        self.ndata_idx = np.array([len(vlist) for vlist in self.label_idx_set])
        self.fullidx = [np.arange(self.ndata_idx[index], dtype=np.int32) for index in range(self.nclass)]
        self.start = np.zeros(self.nclass, dtype=np.int32)
        self.end = np.zeros(self.nclass, dtype=np.int32)

    def print_shape(self):
        print("Image shape : {}".format(self.image.shape))
        print("Label shape : {}".format(self.label.shape))

    def count_label(self):
        counter = np.zeros(self.nclass)
        for i in range(self.ndata): counter[int(self.label)]+=1
        return counter

    def next_batch(self, batch_size):
        '''
        Args:
            batch_size - int
                return batch size numbers of pairs
        Return :
            anc_img
            pos_img
            anc_label - label of anc img
            pos_label - label of pos img
                anc_label and pos_label is idential just for checking
        '''
        assert batch_size%(2*self.nsclass) == 0, "Batchsize(%d) should be multiple of (2*nsclass)(=%d)"%(batch_size, 2*self.nsclass) 

        batch_per_class = batch_size//self.nsclass

        for index in self.valid_class_set: 
            if self.start[index] == 0 and self.end[index] ==0:
                np.random.shuffle(self.fullidx[index]) # shuffle first

        sclass = np.array(random.sample(self.valid_class_set, self.nsclass))
         
        self.subidx = list()
        for cls_idx in sclass:
            if self.end[cls_idx] + batch_per_class > self.ndata_idx[cls_idx]:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] = (self.end[cls_idx] + batch_per_class)%self.ndata_idx[cls_idx]
                self.subidx.append(self.label_idx_set[cls_idx][
                                        np.append(
                                            self.fullidx[cls_idx][self.start[cls_idx]:self.ndata_idx[cls_idx]],\
                                            self.fullidx[cls_idx][0:self.end[cls_idx]])])
                self.start[cls_idx] = 0
                self.end[cls_idx] = 0
            else:
                self.start[cls_idx] = self.end[cls_idx]
                self.end[cls_idx] += batch_per_class
                self.subidx.append(self.label_idx_set[cls_idx][self.fullidx[cls_idx][self.start[cls_idx]:self.end[cls_idx]]])

                if self.end[cls_idx]==self.ndata_idx[cls_idx]:
                    self.start[cls_idx]=0
                    self.end[cls_idx]=0

        self.anc_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==0] for v in self.subidx], axis=0)
        self.pos_subidx = np.concatenate([[v[idx] for idx in range(len(v)) if idx%2==1] for v in self.subidx], axis=0)

        assert len(self.anc_subidx)==len(self.pos_subidx), "Both anc and pos have same length"

        return self.image[self.anc_subidx],\
               self.image[self.pos_subidx],\
               self.label[self.anc_subidx].astype('int32'),\
               self.label[self.pos_subidx].astype('int32')

DATAMANAGER_DICT = {
    'basic' : BasicDatamanager,
    'contrast' : ContrastDatamanager,
    'triplet' : TripletDatamanager,
    'npair' : NpairDatamanager
    }
