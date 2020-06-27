from random import choice

class TSP:
	"""
	Class to initiate and solve instances
	of Traveling Salesman Problem using
	Greedy, 2OPT and 3OPT.
	"""

	def __init__(self, vertices, edges=[]):
		"""
		Initialize TSP Object
		args:
			vertices: List of nodes in graph
			edges: (Optional) List of tuples where 
				each tuple is format (u,v,w)
		"""
		self.nodes = vertices
		self.n = len(vertices)

		self.adjacency = {v: [] for v in self.nodes}
		self.edges = {}

		if len(edges) > 0:
			for edge in edges:
				self.addEdge(*edge)


	def addEdge(self, u, v, w):
		"""
		Method to add directed edge from 
		vertex u to v with numeric weight w.
		"""
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
		"""
		Method to sort outgoing edges out of each vertex
		based on edge weight
		"""
		self.adjacency = {
			v: sorted(self.adjacency[v], key=lambda e: e[1])
			for v in self.nodes
		}


	def greedyTour(self, startnode=None, randomized=False):
		"""
		Method to create a greedy tour on object's 
		graph with optional randomization on the choice
		of next edge to be added.
		"""
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
		"""
		Method to swap two edges and replace with 
		their cross.
		"""
		newtour = tour[:i+1]
		newtour.extend(reversed(tour[i+1:j+1]))
		newtour.extend(tour[j+1:])

		return newtour

	@staticmethod
	def swapEdgesThreeOPT(tour, i, j, k, case):
		"""
		Method to swap edges from 3OPT
		"""
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
		"""
		Method to return length of a tour
		"""
		tourlen = 0
		for i in range(len(tour)-1):
			try:
				tourlen += self.edges[(tour[i], tour[i+1])]
			except KeyError:
				print(f"({tour[i]}, {tour[i+1]}) edge is not part of graph")
		return tourlen


	def twoOPT(self, tour):
		"""
		Method to create new tour using 2OPT
		"""
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
		"""
		Method to create new tour using 3OPT
		"""
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


def createRandomCompleteGraph(n):
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


def test2(n=30):
	v, e, M = createRandomCompleteGraph(n)
	tsp = TSP(v, e)

	print("Greedy tour")
	greedytour, greedytourlen = tsp.greedyTour()
	print(greedytour, greedytourlen)

	print("\n2OPT")
	twoopttour, twooptlen1 = tsp.twoOPT(greedytour)
	print(twoopttour, twooptlen1)

	print("\n3OPT Using greedytour")
	(threeopttour, threeoptlen), time = callAndTime(tsp.threeOPT, greedytour)
	print(threeopttour, threeoptlen, time)

	print("\n3OPT Using 2OPT tour")
	(threeopttour, threeoptlen), time = callAndTime(tsp.threeOPT, twoopttour)
	print(threeopttour, threeoptlen, time)


if __name__ == '__main__':
	test2()




