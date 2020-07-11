class Graph:
	"""
	Class to create a directed graph.
	"""

	def __init__(self, vertices, edges=[]):
		"""
		Initialize Graph object with vertices 
		and (optional) directed edges
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