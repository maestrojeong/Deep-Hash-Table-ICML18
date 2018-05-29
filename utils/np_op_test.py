from np_op import activate_k_1D, activate_k_2D, pairwise_dist, hamming_dist

import numpy as np

def test1():
    # test for activate_k_1D
    arr = np.array([3,6,5,4])
    print("arr :", arr)
    print("activated arr :", activate_k_1D(arr, 2))

def test2():
    # test for activate_k_2D
    arr = np.array([[3,6,5,4],[2,1,3,4]])
    print("arr :", arr)
    print("activated arr :", activate_k_2D(arr, 2))
    
    # test for activate_k_2D_fast
    with tf.Session() as sess:
        print("activated arr :", activate_k_2D(arr, 2, sess))
        print("arr : ", arr)

def test3(): 
    # test for hamming_dist, interdist
    components = np.array([[1,0,1],[0,1,1],[1,1,1]])
    print("components : ",components)
    print(pairwise_dist(components, hamming_dist, identity=3))


if __name__ == '__main__':
    test1()
    test2()
    test3()

