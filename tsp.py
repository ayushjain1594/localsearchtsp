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
	def swapEdgesTwoOPT(tour, i, j):

		newtour = tour[:i+1]
		newtour.extend(reversed(tour[i+1:j+1]))
		newtour.extend(tour[j+1:])

		return newtour

	@staticmethod
	def swapEdgesThreeOPT(tour, i, j, k, case):

		if case == 1:
			newtour = TSP.swapEdgesTwoOPT(tour.copy(), i, k)

		elif case == 2:
			newtour = TSP.swapEdgesTwoOPT(tour.copy(), i, j)

		elif case == 3:
			newtour = TSP.swapEdgesTwoOPT(tour.copy(), j, k)

		elif case == 4:
			newtour = tour[:i+1]
			newtour.extend(tour[j+1:k+1])
			newtour.extend(reversed(tour[i+1:j+1]))
			newtour.extend(tour[k+1:])

		elif case == 5:
			newtour = tour[:i+1]
			newtour.extend(reversed(tour[j+1:k+1]))
			newtour.extend(tour[i+1:j+1])
			newtour.extend(tour[k+1:])

		elif case == 6:
			newtour = tour[:i+1]
			newtour.extend(reversed(tour[i+1:j+1]))
			newtour.extend(reversed(tour[j+1:k+1]))
			newtour.extend(tour[k+1:])

		elif case == 7:
			newtour = tour[:i+1]
			newtour.extend(tour[j+1:k+1])
			newtour.extend(tour[i+1:j+1])
			newtour.extend(tour[k+1:])

		return newtour


	def calculateTourLength(self, tour):
		tourlen = 0
		for i in range(len(tour)-1):
			tourlen += self.edges[(tour[i], tour[i+1])]
		return tourlen


	def twoOPT(self, tour):
		n = len(tour)
		if n <= 2:
			return tour, 0

		tourlen = self.calculateTourLength(tour)
		
		improved = True

		while improved:
			improved = False

			for i in range(n):
				for j in range(i+2, n-1):

					a = self.edges[(tour[i],tour[i+1])]
					b = self.edges[(tour[j], tour[j+1])]
					c = self.edges[(tour[i], tour[j])]
					d = self.edges[(tour[i+1], tour[j+1])]
					delta = - a - b +  c + d
					if delta < 0:
						#print(delta, i, j)
						tour = TSP.swapEdgesTwoOPT(tour.copy(), i, j)
						tourlen += delta
						improved = True

		return tour, tourlen


	def threeOPT(self, tour):
		n = len(tour)
		if n <= 2:
			return [], 0

		tourlen = self.calculateTourLength(tour)
		improved = True
		while improved:
			improved = False
			for i in range(n):
				for j in range(i+2, n-1):
					for k in range(j+2, n-2+(i>0)):
						#print(i, j, k)
						a, b = tour[i], tour[i+1]
						c, d = tour[j], tour[j+1]
						e, f = tour[k], tour[k+1]

						deltacase = {
							1: self.edges[a,e] + self.edges[b,f] \
								- self.edges[a,b] - self.edges[e,f],

							2: self.edges[a,c] + self.edges[b,d] \
								- self.edges[a,b] - self.edges[c,d],

							3: self.edges[c,e] + self.edges[d,f] \
								- self.edges[c,d] - self.edges[e,f],

							4: self.edges[a,d] + self.edges[e,c] + self.edges[b,f]\
								- self.edges[a,b] - self.edges[c,d] - self.edges[e,f],

							5: self.edges[a,e] + self.edges[d,b] + self.edges[c,f]\
								- self.edges[a,b] - self.edges[c,d] - self.edges[e,f],

							6: self.edges[a,c] + self.edges[b,e] + self.edges[d,f]\
								- self.edges[a,b] - self.edges[c,d] - self.edges[e,f],

							7: self.edges[a,d] + self.edges[e,b] + self.edges[c,f]\
								- self.edges[a,b] - self.edges[c,d] - self.edges[e,f],
						}

						bestcase = min(deltacase, key=deltacase.get)

						if deltacase[bestcase] < 0:
							#print(deltacase[bestcase], i, j, k, bestcase)
							tour = TSP.swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
							#print(self.calculateTourLength(tour), tourlen + deltacase[bestcase])
							tourlen += deltacase[bestcase]
							improved = True

		return tour, tourlen


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
	#print(M, "\n")
	return v, e, M


def callAndTime(func, args):
	import time
	start = time.time()
	ret = func(args)
	timetaken = time.time() - start
	return ret, timetaken


def test2(n=200):
	v, e, M = createGraph(n)
	tsp = TSP(v, e)

	greedytour, greedytourlen = tsp.greedyTour()
	print(greedytourlen)

	print("\n2OPT")
	twoopttour, twooptlen1 = tsp.twoOPT(greedytour)
	print(twooptlen1)

	print("\n3OPT Using greedy")
	(threeopttour, threeoptlen), time = callAndTime(tsp.threeOPT, greedytour)
	print(threeoptlen, time)

	print("\n3OPT Using 2OPT")
	(threeopttour, threeoptlen), time = callAndTime(tsp.threeOPT, twoopttour)
	print(threeoptlen, time)


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




