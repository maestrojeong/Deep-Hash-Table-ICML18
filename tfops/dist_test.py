import sys
sys.path.append('../utils/')
#../utils/
from gpu_op import selectGpuById

# ./
from dist import pairwise_distance_euclid, pairwise_distance_euclid_v2

import tensorflow as tf

def test2(gpu_id=0):
    '''
    Results:
	feature1 :
	[[ 1.  2.  3.]
	 [-4. -2.  0.]
	 [ 5. -1.  1.]
	 [ 3. -1.  1.]] 
	feature2 :
	[[ 1.  2.  3.]
	 [-4. -2.  0.]]
	[[  0.  50.]
	 [ 50.   0.]
	 [ 29.  83.]
	 [ 17.  51.]]
	[[  0.  50.  29.  17.]
	 [ 50.   0.  83.  51.]
	 [ 29.  83.   0.   4.]
	 [ 17.  51.   4.   0.]]
    '''
    selectGpuById(gpu_id)
    sess=tf.Session()
    feature1 = tf.constant([[1,2,3], [-4,-2,0], [5,-1,1], [3,-1,1]], dtype=tf.float32) 
    feature2 = tf.constant([[1,2,3], [-4,-2,0]], dtype=tf.float32) 
    print("feature1 :\n{} ".format(sess.run(feature1)))
    print("feature2 :\n{}".format(sess.run(feature2)))
    print(sess.run(pairwise_distance_euclid_v2(feature1, feature2)))
    print(sess.run(pairwise_distance_euclid(feature1, squared=True)))

if __name__ == '__main__':
    test2()
