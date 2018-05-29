import sys
sys.path.append('../utils/')

# ../utils/
from gpu_op import selectGpuById

from hash_dist import pairwise_distance_w_obj1

def test1():
    selectGpuById(0)
    sess = tf.Session()
    feature = tf.constant([[1,2,3], [-1,-1,0], [1,0,2], [1,2,1]], dtype=tf.float32) 
    objective = tf.constant([[0,0,1], [0,0,1], [0,0,1], [0,1,0]], dtype=tf.float32) 
    print("feature : ", sess.run(feature))
    print("objective : ", sess.run(objective))
    print("pairwisedist : \n", sess.run(pairwise_distance_w_obj1(feature, objective)))

if __name__ == '__main__':
    test1()
