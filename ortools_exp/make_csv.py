import sys
sys.path.append('../utils')

# ../utils
from ortools_op import SolveMaxMatching
from writer import write_npy
from csv_op import CsvWriter3

import numpy as np
import time
import os

n_c_set = [32, 64, 128, 256, 512]
d_set = [32, 64, 128, 256, 512]
k = 4
lamb = 1.0
n_iter = 20

cw = CsvWriter3()

for idx1 in range(len(n_c_set)):
    for idx2 in range(len(d_set)):
        print("idx : {}, {}".format(idx1, idx2))
        mcf = SolveMaxMatching(nworkers=n_c_set[idx1], ntasks=d_set[idx2], k=k, pairwise_lamb=lamb)
        time_record = 0 
        for _ in range(n_iter):
            unary = np.random.random([n_c_set[idx1], d_set[idx2]])
            start_time = time.time()
            results = mcf.solve(unary)
            end_time = time.time()
            time_record+=end_time-start_time
        time_record/=n_iter

        cw.add_content(n_c_set[idx1], d_set[idx2], time_record)

cw.write('./time_record.csv', n_c_set, d_set)
