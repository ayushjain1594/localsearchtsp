from random import choice

class TSP:
	"""
	Class to initiate and solve instances
	of Traveling Salesman Problem
	"""

	def __init__(self, vertices, edges=[]):
		self.nodes = vertices
		self.n = len(vertices)

		self.adjacency = {v: [] for v in self.nodes}
		self.edges = {}

		if len(edges) > 0:
			for edge in edges:
				self.addEdge(*edge)


	def addEdge(self, u, v, w):
		for node in [u, v]:
			if self.adjacency.get(node) == None:
				raise KeyError(
					f"Node {node} does not belong to graph"
				)
		if isinstance(w, int) or isinstance(w, float):
			self.adjacency[u].append((v, w))
			self.edges[(u, v)] = w

		else:
			raise TypeError(
				f"Weight {w} must be either type int or float"
			)


	def sortAdjacency(self):
		self.adjacency = {
			v: sorted(self.adjacency[v], key=lambda e: e[1])
			for v in self.nodes
		}


	def greedyTour(self, startnode=None, randomized=False):
		nodevisited = {v: False for v in self.nodes}
		tourlength = 0
		tour = []

		self.sortAdjacency()

		try:
			if startnode:
				currentnode = startnode
			else:
				currentnode = self.nodes[0]
				startnode = currentnode
			nodevisited[startnode] = True
			tour.append(startnode)
			

			while(len(tour) < self.n):
				flag = False
				adjacentnodes = self.adjacency.get(currentnode)

				if len(adjacentnodes) == 0:
					print("Disconnected Graph")
					return tour, tourlength

				if randomized:
					nextthree = []
					count = 0
					for v, w in self.adjacency.get(currentnode):
						if not nodevisited[v]:
							nextthree.append((v, w))
							count += 1
						if count == 2:
							break

					if len(nextthree) > 0:
						v, w = nextthree[choice(range(len(nextthree)))]
						tour.append(v)
						nodevisited[v] = True
						tourlength += w
						currentnode = v
						flag = True


				else:
					for v, w in self.adjacency.get(currentnode):
						if not nodevisited[v]:
							tour.append(v)
							nodevisited[v] = True
							tourlength += w
							currentnode = v
							flag = True
							break

				if flag == False:
					print("Disconnected graph")
					return tour, tourlength


			tour.append(startnode)
			for v, w in self.adjacency.get(currentnode):
				if v == startnode:
					tourlength += w

		except IndexError as e:
			print(e)
		except KeyError as e:
			print(e)

		return tour, tourlength


	@staticmethod
	def swapEdges(tour, i, j):
		vi, vj = tour[i+1], tour[j+1]
		tour[j+1], tour[i+1] = vi, vj
		return tour


	def calculateTourLength(self, tour):
		tourlen = 0
		for i in range(len(tour)-1):
			tourlen += self.edges[(tour[i], tour[i+1])]
		return tourlen


	def twoOPT(self, tour):
		tourlen = self.calculateTourLength(tour)
		n = len(tour)
		improved = True
		while improved:
			improved = False
			for i in range(n-4):
				for j in range(i+2, n-2):
					neighboringtour = TSP.swapEdges(tour.copy(), i, j)
					neighbortourlen = self.calculateTourLength(neighboringtour)

					if neighbortourlen < tourlen:
						print(neighbortourlen - tourlen)
						tour = neighboringtour
						tourlen = neighbortourlen
						improved = True

		return tour, tourlen

	def threeOPT(self):
		pass

def createGraph(n):
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
				M[ind2][ind1] = w
				ind2 += 1

	import numpy as np
	M = np.array(M)
	#print(M)
	return v, e, M

def test2(n=30):
	v, e, M = createGraph(n)
	tsp = TSP(v, e)

	greedytour, greedytourlen = tsp.greedyTour()
	twoopttour, twooptlen = tsp.twoOPT(greedytour)

	print(greedytour, greedytourlen)
	print(twoopttour, twooptlen)

def test(n=150):
	
	from multiprocessing import Pool
	import concurrent.futures
	from pytsp import k_opt_tsp, nearest_neighbor_tsp
	import time

	
	v, e, M = createGraph(n)

	#e = [(i,j,randint(1, 10)) for i in v for j in v if i!=j and i!=5]
	tsp = TSP(v, e)
	start = time.time()

	p = Pool(n)
	greedysols = p.map(tsp.greedyTour, v)
	twooptsols = p.map(tsp.twoOPT, [sol[0] for sol in greedysols])

	delta = time.time() - start
	print(delta)

	for ind in range(n):
		startnode = v[ind]
		print(f"Start node {startnode}, " + \
			f"greedy = {greedysols[ind][1]} 2opt = {twooptsols[ind][1]}")
	
	start = time.time()
	greedysols = []
	twooptsols = []
	for startnode in v:
		sol = tsp.greedyTour(startnode)
		greedysols.append(sol)
		twooptsols.append(tsp.twoOPT(sol[0]))

	delta = time.time() - start
	print(delta)

	for ind in range(n):
		startnode = v[ind]
		print(f"Start node {startnode}, " + \
			f"greedy = {greedysols[ind][1]} 2opt = {twooptsols[ind][1]}")
	
	
	'''
	print(mingreedy, mintwoopt)

	
	print(tour, tourlen)

	
	print(newtour, newtourlen)

	newtour, newtourlen = tsp.twoOPT(newtour, newtourlen)
	print(newtour, newtourlen)

	actualnewtourlen = 0
	for i in range(len(newtour)-1):
		actualnewtourlen += tsp.edges[(newtour[i], newtour[i+1])]

	print(actualnewtourlen)

	
	start = time.time()

	sols = []
	for startnode in v:
		sols.append(tsp.greedyTour(startnode, False))
	for sol in sols:
		print(sol[1])
	
	print(time.time() - start)

	print("\n")

	start = time.time()
	
	p = Pool(n)
	
	sols = p.map(tsp.greedyTour, v)
	
	with concurrent.futures.ProcessPoolExecutor() as executor:
		sols = [executor.submit(tsp.greedyTour, v1, False) for v1 in v]
		for f in concurrent.futures.as_completed(sols):
			print(f.result()[1])

	#for sol in sols:
	#	print(sol[1])
	print(time.time()-start)

	start = time.time()
	nntsp_tour = nearest_neighbor_tsp.nearest_neighbor_tsp(M) + [0]
	print(time.time()-start)

	start = time.time()
	opt_tour = k_opt_tsp.tsp_3_opt(M) + [0]
	print(time.time()-start)

	tourlen_nntsp = 0
	for ind in range(len(nntsp_tour)-1):
		tourlen_nntsp += M[nntsp_tour[ind], nntsp_tour[ind+1]]

	tourlen_3opt = 0
	for ind in range(len(opt_tour)-1):
		tourlen_3opt += M[opt_tour[ind], opt_tour[ind+1]]
	print("\n")
	#print([v[ind] for ind in nntsp_tour], tourlen_nntsp)
	#print([v[ind] for ind in opt_tour], tourlen_3opt)
	print(tourlen_3opt)
	'''

if __name__ == '__main__':
	test2()




