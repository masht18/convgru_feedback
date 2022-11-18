import pprint
from collections import defaultdict
from topdown_gru import ConvGRUTopDownCell
import torch
from torch import nn

class Node:
    def __init__(self, input_size = None, input_dim = None, hidden_dim = None, kernal_size = None):
        self.in_nodes = [] #nodes passing values into current node #contains Node index (int)
        self.out_nodes = [] #nodes being passed with values from current node #contains Node index (int)
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank_list = [] #default value


        #cell params
        self.input_height, self.input_width = input_size #Height and width of input tensor as (height, width).
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernal_size = kernal_size 


class Graph(object):
    """ A brain architecture graph object, directed by default. """
    def __init__(self, connections, conn_strength, input_node, output_node, output_size, directed=True, reps = 2，topdown = True, dtype = torch.FloatTensor, bias = False):
        self.conn = connections #adjacency matrix
        self.conn_strength = conn_strength
        self.directed = directed #flag for whether the connections are directed 
        self.num_node = len(self.conn)
        self.max_rank = -1
        self.num_edge = 0 #assuming directed edge. Will need to change if graph is undirected
        self.topdown = topdown
        self.nodes = self.generate_node_list(conn_strength)
        self.dtype = dtype 
        self.bias = bias
        self.output_size = output_size
        self.rep = reps
        #self.nodes[output_node].dist = 0
        #self.max_length = self.find_longest_path_length()
        self.rank_node(self.nodes[input_node],self.nodes[output_node], 0, 0)



    def generate_node_list(self,conn_strength):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            node = Node()
            nodes.append(node)

        for in_node in range(self.num_node):
            for out_node in range(self.num_node):
                if (self.conn[in_node][out_node] != 0 ):
                    self.num_edge = self.num_edge + 1 # i dont want to use numpy so this is how im calculating the number of connections
                    nodes[in_node].out_nodes.append(out_node)
                    nodes[out_node].in_nodes.append(in_node) 
                    #TODO: add conn strength
        return nodes
    
    def rank_node(self, current_node, output_node, rank_val, num_pass):
        ''' ranks each node in the graph'''
        current_node.rank_list.append[rank_val]
        rank_val = rank_val + 1
        for node in self.nodes[current_node].out_nodes:
            num_pass = num_pass + 1
            if (rank_val == self.max_length and num_pass == self.num_edge):
                self.max_rank = current_node.rank.max()
                return
            else:
                self.rank_node(node, output_node, rank_val, num_pass)

    def build_architecture(self):
        architecture = Architecture(self)

    def find_feedforward_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].in_nodes:
            if ((t-1) in n.rank_list):
                in_nodes.append(n)
        return in_nodes

    def find_feedback_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].out_nodes:
            if ((t+1) in n.rank_list):
                in_nodes.append(n)
        return in_nodes

class Architecture(object):
    def __init__(self, graph):
        self.graph = graph
        cell_list = []
        for node in range(0, graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=(self.height, self.width),
                                         input_dim=self.graph.nodes[node].hidden_dim, #TODO: discuss with Mashbayar
                                         hidden_dim = self.graph.nodes[node].hidden_dim,
                                         topdown_dim= self.graph.nodes[node].hidden_dim, #TODO: discuss with Mashbayar
                                         kernel_size=self.graph.nodes[node].kernel_size,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        self.fc1 = nn.Linear(self.graph.height*self.graph.width*self.graph.nodes[graph.output_node].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, self.graph.output_size)

        self.input_conv = nn.Conv2d(in_channels= self.graph.nodes[self.graph.input_node].input_dim,
                              out_channels= self.graph.nodes[self.graph.input_node].hidden_dim,
                              kernel_size=3,
                              padding=1)

    def forward(self, input_tensor, topdown, topdown_input):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
        :param topdown: size (b,hidden,h,w) 
        :return: label 
        """

        seq_len = 1 #for easy testing purposes
        #seq_len = input_tensor.size(1)
        hidden_state_prev = self._init_hidden(batch_size=input_tensor.size(0))
        hidden_state_cur = self._init_hidden(batch_size=input_tensor.size(0))
        # Images in sequence goes through all layers one by one
        # This is a reverse of the original convgru code, where the entire sequence went through one layer before advancing to next
        for rep in range(self.reps):
            for t in range(max(seq_len, self.graph.max_rank)): #TODO: discuss with Mashbayar
                current_input = self.input_conv(input_tensor[:, t, :, :, :])
                #current_input = self.input_conv(input_tensor[:, :, :, :])#[32,1,28,28]

                for node in range(self.num_layers):
                    # Current state
                    h = hidden_state_prev[node] 

                    input_cells = self.graph.find_feedforward_cells(node, t)
                    # Bottom-up signal is either the state from the bottom layer or the input if it's the bottom-most layer
                    if node == self.graph.input_node:
                        bottomup =  current_input
                    else: #cur node not input_node
                        is_first = True
                        bottomup = None
                        for i in (input_cells):
                            if is_first:
                                bottomup = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                            else:
                                bottomup = bottomup + self.graph.conn_strength[i][node]*hidden_state_prev[i]
                            if_First = False


                    #now we handle topdown signal
                    if self.graph.topdown == False:
                    #Disable topdown if using the no-topdown version of model
                        topdown_sig = None
                    #topdown signal present
                    elif bottomup == None: #if there is no bottomup input, then ignore topdown feedback
                        topdown_sig = None 
                    else:  #current node receiving some kind of bottomup and therefore topdown info 
                        if node == self.graph.output_node:
                            topdown_sig = topdown_input #TODO: discuss with Mashbayar
                        else: #not output node
                            modulate_cells = self.graph.find_feedback_cells(node, t)
                            is_first = True
                            topdown_sig = None
                            for i in (modulate_cells):
                                if is_first:
                                    topdown_sig = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                                else:
                                    topdown_sig = topdown_sig + self.graph.conn_strength[i][node]*hidden_state_prev[i]
                                if_First = False

                    #now we are done with gathering bottomup and topdown inputs for current cell
                    
                    #TODO: some form of reshaping topdown sig 
                    #Update hidden state of this layer by feeding bottom-up, top-down and current cell state into gru cell
                    h = self.cell_list[node](input_tensor=bottomup, # (b,t,c,h,w)
                                                  h_cur=h, topdown=topdown_sig)
                    hidden_state_cur[node] = h
                #we are done with iterating through all cells at the current timestep
                hidden_state_prev = hidden_state_cur

        if self.return_bottom_layer == True:
            #TODO: adjust dimention & process 
            pred = hidden_state_cur[0]
        else:
            #TODO: adjust dimention & process 

            pred = hidden_state_cur[-1]
        
        #TODO: process pred through relu
        #         
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states