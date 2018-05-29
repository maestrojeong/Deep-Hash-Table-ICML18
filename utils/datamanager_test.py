import sys
sys.path.append('../configs')
sys.path.append('../utils')
sys.path.append('../tfops')

# ../utils
from reader import read_npy
# ../config
from path import CIFARPROCESSED
from info import CIFARNCLASS

def test1():
    val_embed = read_npy(CIFARPROCESSED+'val_image.npy')    
    val_label = read_npy(CIFARPROCESSED+'val_label.npy')    

    cifar = TripletDatamanager(val_embed, val_label, CIFARNCLASS, nsclass=10) 
    count = np.zeros(cifar.nclass)
    nbatch = cifar.ndata//50+1
    for i in range(nbatch):
        _, label = cifar.next_batch(50)
        for index in range(len(label)):
            count[label[index]]+=1
    print(count)

def test2():
    val_embed = read_npy(CIFARPROCESSED+'val_image.npy')    
    val_label = read_npy(CIFARPROCESSED+'val_label.npy')    

    cifar = NpairDatamanager(val_embed, val_label, CIFARNCLASS, nsclass=4)
    _, _, anc_l, pos_l = cifar.next_batch(32)
    print(anc_l)
    print(pos_l)

if __name__=='__main__':
    test1()
    test2()
