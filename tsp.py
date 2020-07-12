from random import choice

from graph import Graph

class TSP(Graph):
	"""
	Class to initiate and solve instances
	of Traveling Salesman Problem using
	Greedy, 2OPT and 3OPT.
	"""
	def __init__(self, vertices, edges=[]):
		"""
		Initialize graph object 
		args:
			vertices: List of nodes in graph
			edges: (Optional) List of tuples where 
				each tuple is format (u,v,w)
		"""
		super().__init__(vertices, edges)
		

	def sortAdjacency(self):
		"""
		Method to sort inplace outgoing edges out of each vertex
		based on edge weight
		"""

		# second argument of tuple 'e' is weight
		self.adjacency = {
			v: sorted(self.adjacency[v], key=lambda e: e[1])
			for v in self.nodes
		}


	def greedyTour(self, startnode=None, randomized=False):
		"""
		Method to create a greedy tour on object's 
		graph with optional randomization on the choice
		of next edge to be added.
		args:
			startnode(optional): specify a starting node 
				for greedy algorithm. Keyerror is raised
				if not part of graph nodes.
			randomized(optional): boolean
				If true, algorithm will randomly (uniformly)
				choose one of next three nodes with lowest 
				added cost.
		return:
			tour: List of nodes in the tour including
				start node added at the end
			tourlen: int/float based on edge weights
				This is the length of tour.
		"""

		# tracker for nodes that have been visited
		nodevisited = {v: False for v in self.nodes}

		# initializing empty tour
		tourlength = 0
		tour = []

		# sort adjacency lists of outgoing edges for each vertex
		self.sortAdjacency()

		try:
			# if specified, else pick first node in the graph
			if startnode:
				currentnode = startnode
			else:
				currentnode = self.nodes[0]
				startnode = currentnode

			nodevisited[startnode] = True
			tour.append(startnode)
			

			while(len(tour) < self.n):

				# track if next node was found and added to tour
				flag = False

				# get list of tuples (v, weight) of adjacent nodes
				adjacentnodes = self.adjacency.get(currentnode)

				if len(adjacentnodes) == 0:
					# there are no outgoing edges from current node.
					# the graph is disconnnect
					print("Disconnected Graph")
					return tour, tourlength

				if randomized:
					nextthree = []
					count = 0
					# get next (up to) three nodes that have not been visited
					# from sorted adjacency list
					for v, w in self.adjacency.get(currentnode):
						if not nodevisited[v]:
							nextthree.append((v, w))
							count += 1
						if count == 3:
							break

					if len(nextthree) > 0:
						# uniformly choose one
						v, w = nextthree[choice(range(len(nextthree)))]
						tour.append(v)
						nodevisited[v] = True
						tourlength += w
						currentnode = v
						flag = True


				else:
					# if not randomized
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

			# add starting node at the end of tour
			tour.append(startnode)

			# add weight of last edges
			flag = False
			for v, w in self.adjacency.get(currentnode):
				if v == startnode:
					tourlength += w
					flag = True
			if flag == False:
				print(f"Missing edge ({currentnode}, {startnode})")
				print("Tour may not be feasible")

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
		Method to return length of a tour given
		all tour edges are part of graph
		args:
			tour: List of nodes of graph
		return:
			tourlen: int/float 
				Length of tour. If any edges is 
				missing, returns zero.
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
		args:
			tour: List of nodes forming a cycle
		return:
			tour: List of nodes forming a cycle
				Two optimal tour
			tourlen: int/float
				Length of two optimal tour
		"""
		n = len(tour)
		if n <= 2:
			# no cycle possible
			return tour, 0

		# length of provided tour
		tourlen = self.calculateTourLength(tour)
		
		# tracking improvemnt in tour
		improved = True

		while improved:
			improved = False

			for i in range(n):
				for j in range(i+2, n-1):

					a = self.edges[(tour[i],tour[i+1])]
					b = self.edges[(tour[j], tour[j+1])]
					c = self.edges[(tour[i], tour[j])]
					d = self.edges[(tour[i+1], tour[j+1])]

					# benefit from swapping i,i+1 and j,j+1
					# with i,j and i+1,j+1
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
		args:
			tour: List of nodes forming a cycle
		return:
			tour: List of nodes forming a cycle
				Three optimal tour
			tourlen: int/float
				Length of three optimal tour
		"""
		n = len(tour)
		if n <= 2:
			# no cycle possible
			return [], 0

		# length of provided tour
		tourlen = self.calculateTourLength(tour)

		# tracking improvemnt in tour
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

						# possible cases of removing three edges 
						# and adding three
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

						# get the case with most benefit
						bestcase = min(deltacase, key=deltacase.get)

						if deltacase[bestcase] < 0:
							#print(deltacase[bestcase], i, j, k, bestcase)
							tour = TSP.swapEdgesThreeOPT(tour.copy(), i, j, k, case=bestcase)
							#print(self.calculateTourLength(tour), tourlen + deltacase[bestcase])
							tourlen += deltacase[bestcase]
							improved = True

		return tour, tourlen



if __name__ == '__main__':
	pass





