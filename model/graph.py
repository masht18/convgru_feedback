import pprint
from collections import defaultdict

class Node:
    def __init__(self):
        self.val = 0 #TODO: what format should val take?
        self.in_nodes = [] #nodes passing values into current node #contains Node objects
        self.out_nodes = [] #nodes being passed with values from current node #contains Node objects
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank = [] #default value

class Graph(object):
    """ A brain architecture graph object, directed by default. """
    def __init__(self, connections, conn_strength,input_node, output_node, directed=True):
        self.conn = connections #adjacency matrix
        self.directed = directed #flag for whether the connections are directed 
        self.num_node = len(self.conn)
        self.max_rank = -1
        self.num_edge = 0 #assuming directed edge. Will need to change if graph is undirected
        self.nodes = self.generate_node_list(conn_strength)
        self.nodes[output_node].dist = 0
        self.max_length = self.find_longest_path_length()
        self.rank_node(self.nodes[input_node],self.nodes[output_node], 0, 0)



    def generate_node_list(self,conn_strength):
        self.nodes = []
        # initialize node list
        for node in range(self.num_node):
            n = Node()
            self.nodes.append(n)

        for in_node in range(self.num_node):
            for out_node in range(self.num_node):
                if (self.conn[in_node][out_node] != 0 ):
                    self.num_edge = self. num_edge + 1 # i dont want to use numpy so this is how im calculating the number of connections
                    self.nodes[in_node].out_nodes.append(self.nodes[out_node])
                    self.nodes[out_node].in_nodes.append(self.nodes[in_node])
                    #TODO: add conn strength
    
    def rank_node(self, current_node, output_node, rank_val, num_pass):
        ''' ranks each node in the graph'''
        current_node.rank.append[rank_val]
        rank_val = rank_val + 1
        for node in self.nodes[current_node].out_nodes:
            num_pass = num_pass + 1
            if (rank_val == self.max_length and num_pass == self.num_edge):
                self.max_rank = current_node.rank.max()
                return
            else:
                self.rank_node(node, output_node, rank_val, num_pass)