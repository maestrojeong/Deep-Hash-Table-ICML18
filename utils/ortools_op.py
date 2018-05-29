from ortools.graph import pywrapgraph

import numpy as np
import copy

class SolveMaxMatching:
    def __init__(self, nworkers, ntasks, k, value=10000, pairwise_lamb=0.1):
        '''
        This can be used when nworkers*k > ntasks
        Args:
            nworkers - int
            ntasks - int
            k - int
            value - int 
                should be large defaults to be 10000

            pairwise_lamb - int

        '''
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value
        self.k = k

        self.source = 0
        self.sink = self.nworkers+self.ntasks+1

        self.pairwise_cost = int(pairwise_lamb*value)

        self.supplies = [self.nworkers*self.k]+(self.ntasks+self.nworkers)*[0]+[-self.nworkers*self.k]
        self.start_nodes = list()
        self.end_nodes = list() 
        self.capacities = list()
        self.common_costs = list()

        for work_idx in range(self.nworkers):
            self.start_nodes.append(self.source)
            self.end_nodes.append(work_idx+1)
            self.capacities.append(self.k)
            self.common_costs.append(0)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(self.nworkers+1+task_idx)
                self.end_nodes.append(self.sink)
                self.capacities.append(1)
                self.common_costs.append(work_idx*self.pairwise_cost)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(work_idx+1)
                self.end_nodes.append(self.nworkers+1+task_idx)
                self.capacities.append(1)

        self.nnodes = len(self.start_nodes)

    def solve(self, array):
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)

        self.array = self.value*array
        self.array = -self.array # potential to cost
        self.array = self.array.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(self.array[work_idx][task_idx])

        costs = np.array(costs)
        costs = (costs.tolist())

        assert len(costs)==self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for idx in range(self.nnodes):
             min_cost_flow.AddArcWithCapacityAndUnitCost(self.start_nodes[idx], self.end_nodes[idx], self.capacities[idx], costs[idx])
        for idx in range(self.ntasks+self.nworkers+2):
            min_cost_flow.SetNodeSupply(idx, self.supplies[idx])

        min_cost_flow.Solve()
        results = list()
        for arc in range(min_cost_flow.NumArcs()):
            if min_cost_flow.Tail(arc)!=self.source and min_cost_flow.Head(arc)!=self.sink:
                if min_cost_flow.Flow(arc)>0:
                    results.append([min_cost_flow.Tail(arc)-1, min_cost_flow.Head(arc)-self.nworkers-1])

        return results

