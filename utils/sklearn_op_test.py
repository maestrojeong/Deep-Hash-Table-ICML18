import sys
sys.path.append('../utils/')
#../utils/
from gpu_op import selectGpuById

# ./
from sklearn_op import KMeansClustering

import tensorflow as tf
import numpy as np

def test1(gpu_id=0):
    '''
    Results :
	X : [[1 2]
	 [1 4]
	 [1 0]
	 [4 2]
	 [4 4]
	 [4 0]]
	centers : [[ 1.  2.]
	 [ 4.  1.]
	 [ 4.  4.]]
	hash : 
	[[ -0. -10. -13.]
	 [ -4. -18.  -9.]
	 [ -4. -10. -25.]
	 [ -9.  -1.  -4.]
	 [-13.  -9.  -0.]
	 [-13.  -1. -16.]]
    '''
    selectGpuById(gpu_id)
    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    kmc = KMeansClustering(X, 3)
    sess = tf.Session()
    
    print("X : {}".format(X))
    print("centers : {}".format(kmc.centers))
    print("hash : \n{}".format(kmc.k_hash(X, sess)))

if __name__=='__main__':
    test1()

