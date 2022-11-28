from model.topdown_gru import ConvGRUTopDownCell
import torch
from torch import nn
import torch.nn.functional as F

class Node:
    def __init__(self, index, input_size = (28, 28), input_dim = 3, hidden_dim = 10, kernal_size = (3,3)):
        self.index = index
        self.in_nodes_index = [] #nodes passing values into current node #contains Node index (int)
        self.out_nodes_index = [] #nodes being passed with values from current node #contains Node index (int)
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank_list = [] #default value


        #cell params
        (self.input_height, self.input_width) = input_size #Height and width of input tensor as (height, width).
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernal_size 

    def __eq__(self, other): 
        return self.index == other.index


class Graph(object):
    """ A brain architecture graph object, directed by default. """
    def __init__(self, connections, conn_strength, input_node, output_node_index, output_size = 10, directed=True, dtype = torch.FloatTensor, topdown = True, bias = False, reps = 2):
        self.input_node_index = input_node
        self.output_node_index = output_node_index
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
        self.reps = reps
        #self.nodes[output_node].dist = 0
        self.max_length = self.find_longest_path_length()
        self.rank_node(self.nodes[input_node],self.nodes[output_node_index], 0, 0)
        # for node in range (self.num_node):
        #     print(self.nodes[node].rank_list)



    def generate_node_list(self,conn_strength):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            node = Node(n)
            nodes.append(node)

        for in_node in range(self.num_node):
            for out_node in range(self.num_node):
                if (self.conn[in_node][out_node] != 0 ):
                    self.num_edge = self.num_edge + 1 # i dont want to use numpy so this is how im calculating the number of connections
                    nodes[in_node].out_nodes_index.append(out_node)
                    nodes[out_node].in_nodes_index.append(in_node) 
                    #TODO: add conn strength
        return nodes
    
    def rank_node(self, current_node, output_node_index, rank_val, num_pass):
        ''' ranks each node in the graph'''
        current_node.rank_list.append(rank_val)
        rank_val = rank_val + 1
        for node in current_node.out_nodes_index:
            num_pass = num_pass + 1
            if (current_node == output_node_index and num_pass == self.num_edge):
                self.max_rank = current_node.rank.max()
                return
            else:
                self.rank_node(self.nodes[node], output_node_index, rank_val, num_pass)

    def build_architecture(self):
        architecture = Architecture(self)
        return architecture

    def find_feedforward_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].in_nodes_index:
            if ((t-1) in self.nodes[n].rank_list):
                in_nodes.append(n)
        return in_nodes

    def find_feedback_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].out_nodes_index:
            if ((t+1) in self.nodes[n].rank_list):
                in_nodes.append(n)
        return in_nodes

    def find_longest_path_length(self): #TODO: 
        return 3

class Architecture(nn.Module):
    def __init__(self, graph):
        super(Architecture, self).__init__()

        self.graph = graph
        cell_list = []
        for node in range(0, graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=(28,28), #TODO:
                                         input_dim=self.graph.nodes[node].hidden_dim, #TODO: discuss with Mashbayar
                                         hidden_dim = self.graph.nodes[node].hidden_dim,
                                         topdown_dim= self.graph.nodes[node].hidden_dim, #TODO: discuss with Mashbayar
                                         kernel_size=self.graph.nodes[node].kernel_size,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        #this is some disgusting line of code
        self.fc1 = nn.Linear(self.graph.nodes[self.graph.output_node_index].input_height * self.graph.nodes[self.graph.output_node_index].input_width * self.graph.nodes[self.graph.output_node_index].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, self.graph.output_size)

        self.input_conv = nn.Conv2d(in_channels= self.graph.nodes[self.graph.input_node_index].input_dim,
                              out_channels= self.graph.nodes[self.graph.input_node_index].hidden_dim,
                              kernel_size=3,
                              padding=1)


    def forward(self, input_tensor, topdown_input, topdown = True):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
        :param topdown: size (b,hidden,h,w) 
        :return: label 
        """

        seq_len = 1 #for easy testing purposes
        #seq_len = input_tensor.size(1)
        hidden_state_prev = self._init_hidden(batch_size=input_tensor.size(0))
        hidden_state_cur = self._init_hidden(batch_size=input_tensor.size(0))
        input_tensor = input_tensor.permute(0, 2, 1, 3, 4) #why  is the input_seqs shaped like this?
        # Images in sequence goes through all layers one by one
        # This is a reverse of the original convgru code, where the entire sequence went through one layer before advancing to next
        for rep in range(self.graph.reps):
            for t in range(max(seq_len, self.graph.max_rank)): 
                #before permuet: b c t h w
                
                current_input = self.input_conv(input_tensor[:, t, :, :, :])
                #current_input = self.input_conv(input_tensor[:, :, :, :])#[32,1,28,28]

                for node in range(self.graph.num_node):
                    print('node:',node)
                    # Current state
                    node_hidden_state = hidden_state_prev[node] 

                    input_cells = self.graph.find_feedforward_cells(node, t)
                    # Bottom-up signal is either the state from the bottom layer or the input if it's the bottom-most layer
                    if node == self.graph.input_node_index:
                        bottomup =  current_input
                    else: #cur node not input_node
                        is_first = True
                        #bottomup = None
                        for i in (input_cells):
                            if is_first:
                                bottomup = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                            else:
                                bottomup = torch.cat([bottomup, self.graph.conn_strength[i][node]*hidden_state_prev[i]], dim=1)
                            is_first = False


                    #now we handle topdown signal
                    if topdown == False:#Disable topdown if using the no-topdown version of model
                        topdown_sig = None

                    #topdown signal present
                    elif bottomup == None: #if there is no bottomup input, then ignore topdown feedback
                        topdown_sig = None 
                    else:  #current node receiving some kind of bottomup and therefore topdown info 
                        if node == self.graph.output_node_index: #TODO: output_node not necessary the node receiving topdown_signal
                            topdown_sig = topdown_input #TODO: discuss with Mashbayar
                        else: #not output node
                            modulate_cells = self.graph.find_feedback_cells(node, t)
                            is_first = True #boolean to judge if is first pass
                            #topdown_sig = None
                            for i in (modulate_cells):
                                if is_first:
                                    topdown_sig = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                                else:
                                    topdown_sig = torch.cat([topdown_sig, self.graph.conn_strength[i][node]*hidden_state_prev[i]], dim=1) #TODO: torch cat into channel dimension 
                                is_first = False
                        
                        print('t:',t)
                        print('bot:',bottomup.shape,)
                        print('topdown_sig:',topdown_sig.shape)
                        print('node_hidden_state',node_hidden_state.shape)
                        m = nn.Linear(topdown_sig.shape[1]*topdown_sig.shape[2]*topdown_sig.shape[3], (bottomup.shape[1]+node_hidden_state.shape[1])*bottomup.shape[2]*bottomup.shape[3])
                        (_,a,b,c) = topdown_sig.shape
                        topdown_sig = m(torch.flatten(topdown_sig,start_dim=1))
                        topdown_sig = torch.reshape(topdown_sig, (topdown_sig.shape[0],bottomup.shape[1]+node_hidden_state.shape[1], bottomup.shape[2],bottomup.shape[3]))

                    #now we are done with gathering bottomup and topdown inputs for current cell
                    
                    #TODO: some form of reshaping topdown sig 
                    #Update hidden state of this layer by feeding bottom-up, top-down and current cell state into gru cell
                    
                    h = self.cell_list[node](input_tensor=bottomup,
                                                  h_cur=node_hidden_state, topdown=topdown_sig)
                    hidden_state_cur[node] = h
                #we are done with iterating through all cells at the current timestep
                hidden_state_prev = hidden_state_cur

        pred = self.fc1(F.relu(torch.flatten(hidden_state_cur[self.graph.output_node_index], start_dim=1)))
        pred = self.fc2(F.relu(pred))
          
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.graph.num_node):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states