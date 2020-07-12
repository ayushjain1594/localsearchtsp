from tsp import TSP
from pandas import DataFrame, set_option
import time

def createRandomCompleteGraph(n, symmetric=False):
	"""
	create a complete symmetric graph with 
	random edge weight
	"""
	from random import randint
	v = [i for i in range(1, n+1)]

	e = []
	M = [[0 for v1 in v] for v2 in v]
	for ind1, i in enumerate(v):
		ind2 = ind1+1
		for j in v[ind1+1:]:
			if i != j:
				w = randint(1, 100)
				e.append((i,j,w))
				e.append((j,i,w))
				M[ind1][ind2] = w
				if symmetric:
					M[ind2][ind1] = w
				else:
					w = randint(1, 100)
					M[ind2][ind1] = w
				ind2 += 1

	import numpy as np
	M = np.array(M)
	#print(M, "\n")
	return v, e, M


def callAndTime(input_):
	"""
	Wrapper for calling and timing a function
	args:
		input: List 
			func at zeroth index and
			rest are arguments
	"""
	func= input_[0]
	args = input_[1:]
	
	start = time.time()
	ret = func(*args)
	timetaken = time.time() - start
	return ret, timetaken


def test2(n=50):
	"""
	Given number of nodes -
	Test greedy tour using default starting node, 
	two optimal tour using greedy tour
	three optimal tour using greedy tour 
	three optimal tour using two optimal tour
	"""
	v, e, M = createRandomCompleteGraph(n)
	tsp = TSP(v, e)

	print("Greedy tour")
	greedytour, greedytourlen = tsp.greedyTour()
	print(greedytour, greedytourlen)

	print("\n2OPT")
	twoopttour, twooptlen = tsp.twoOPT(greedytour)
	print(twoopttour, twooptlen)

	print("\n3OPT Using greedytour")
	(threeopttour, threeoptlen), time = callAndTime((tsp.threeOPT, greedytour))
	print(threeopttour, threeoptlen, time)

	print("\n3OPT Using 2OPT tour")
	(threeopttour, threeoptlen), time = callAndTime((tsp.threeOPT, twoopttour))
	print(threeopttour, threeoptlen, time)


def testResultDataFrame(rows, columns=None):
	"""
	Generic function to create pandas dataframe
	on results of different test runs 
	with different starting node
	"""
	if columns == None:
		columns_ = ["StartNode", "GreedyTour", 
		"TwoOPT", "ThreeOPT_Greedy", 
		"Time1", "ThreeOPT_TwoOPT", "Time2"]
	else:
		columns_ = columns

	
	df = DataFrame(rows, columns=columns_)
	set_option('display.max_columns', None)
	print(df)

	df.to_csv('sampleresult.csv', index=False)

	print(f"GREEDY SOL AVG LENGTH = {df.GreedyTour.mean()}")
	print(f"2 OPT SOL AVG LENGTH = {df.TwoOPT.mean()}")
	print(f"3 OPT FROM GREEDY SOL AVG LENGTH = {df['ThreeOPT_Greedy'].mean()}  time = {df.Time1.mean()}")
	print(f"3 OPT FROM 2 OPT SOL AVG LENGTH = {df['ThreeOPT_TwoOPT'].mean()} time = {df.Time2.mean()}")


def test3(n=30, tsp_object=None):
	"""
	Test sequentially over all possible starting
	node
	"""
	if isinstance(tsp_object, TSP):
		tsp = tsp_object
	else:
		v, e, M = createRandomCompleteGraph(n)
		tsp = TSP(v, e)

	rows = []
	for v_ in tsp.nodes:
		greedytour, greedytourlen = tsp.greedyTour(v_)
		twoopttour, twooptlen = tsp.twoOPT(greedytour)
		(threeopttour1, threeoptlen1), time1 = callAndTime((tsp.threeOPT, greedytour))
		(threeopttour2, threeoptlen2), time2 = callAndTime((tsp.threeOPT, twoopttour))
		rows.append(
			[v_, greedytourlen,
			twooptlen,
			threeoptlen1, time1,
			threeoptlen2, time2]
		)
	testResultDataFrame(rows)


def test4(n=30, tsp_object=None):
	"""
	Test concurrently over all possible 
	starting node using multiprocessing module
	"""
	if isinstance(tsp_object, TSP):
		tsp = tsp_object
	else:
		v, e, M = createRandomCompleteGraph(n)
		tsp = TSP(v, e)
	rows = []

	from multiprocessing import Pool
	from itertools import product

	
	p = Pool(8)
	greedysol = p.map(tsp.greedyTour, tsp.nodes)
	twooptsol = p.map(tsp.twoOPT, [sol[0] for sol in greedysol])
	threeoptsol1 = p.map(callAndTime, [(tsp.threeOPT, sol[0]) for sol in greedysol])
	threeoptsol2 = p.map(callAndTime, [(tsp.threeOPT, sol[0]) for sol in twooptsol])

	for ind in range(n):
		rows.append(
			[tsp.nodes[ind],
			greedysol[ind][1], 
			twooptsol[ind][1],
			threeoptsol1[ind][0][1], threeoptsol1[ind][1],
			threeoptsol2[ind][0][1], threeoptsol2[ind][1]]
		)
	testResultDataFrame(rows)


def test6(n=50):
	v, e, M = createRandomCompleteGraph(n)
	tsp = TSP(v, e)
	#ret, time = callAndTime((test3, n, tsp))
	#print(ret, time)

	ret, time = callAndTime((test4, n, tsp))
	print(ret, time)


if __name__ == '__main__':
	test6()