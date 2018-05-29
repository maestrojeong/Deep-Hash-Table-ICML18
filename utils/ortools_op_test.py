from ortools_op import SolveMaxMatching
import numpy as np

def results2objective(results, nworkers, ntasks):
    objective = np.zeros([nworkers, ntasks]) 
    for i,j in results: objective[i][j]=1
    return objective

def test1():
    '''
    Results :
	unary potential :
	[[ 0.89292312  0.64831936  0.56726688  0.13208358  0.05779465  0.48978515]
	 [ 0.83382476  0.08014071  0.61772549  0.95149459  0.04179085  0.92253984]
	 [ 0.76766159  0.6634347   0.91049119  0.6748744   0.17438728  0.51890275]
	 [ 0.90997762  0.18447894  0.81440657  0.09081913  0.46642204  0.47917976]
	 [ 0.72631254  0.94356716  0.05386514  0.57434492  0.69070927  0.39979905]]
	objective :
	[[ 1.  1.  0.  0.  0.  0.]
	 [ 0.  0.  0.  1.  0.  1.]
	 [ 0.  0.  1.  1.  0.  0.]
	 [ 1.  0.  1.  0.  0.  0.]
	 [ 0.  1.  0.  0.  1.  0.]]
    '''
    pairwise_lamb = 0.1
    nworkers = 5
    ntasks = 6
    k = 2
    mcf  = SolveMaxMatching(nworkers=nworkers, ntasks=ntasks, k=k, pairwise_lamb=pairwise_lamb)   
    unary_potential = np.random.random([nworkers, ntasks])
    mcf_results = mcf.solve(unary_potential)
    objective = results2objective(results=mcf_results, nworkers=nworkers, ntasks=ntasks)

    print("unary potential :\n{}".format(unary_potential))
    print("objective :\n{}".format(objective))

if __name__ =='__main__':
    test1()

