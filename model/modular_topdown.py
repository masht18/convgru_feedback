import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from model.topdown_gru import ConvGRUTopDownCell


class ConvGRUExplicitTopDown(nn.Module): #output of my module
    def __init__(self, block_list, output_size, connection_strengths, topdown_input, reps = 1, dtype = torch.FloatTensor, 
                topdown_type = 'image', topdown = True, batch_first=True, bias=True, return_bottom_layer=False):
        """
        Full model with modularized top-down architecture

        block_list: (block_object)
            a list of length n, containing n cells of various type (e.g. ConvGRUTopDownCell)
            each cell has input_size, input_dim, hidden_dim, kernal_size as parameters. 
            what is expected PER CELL:
                :param input_size: (int, int)
                    Height and width of input tensor as (height, width).
                :param input_dim: int
                    Number of channels of input tensor.
                :param hidden_dim: int
                    Number of channels of hidden state. Can be custom for each layer or the same
                :param kernel_size: (int, int) 
                    Size of the convolutional kernel. Can be custom for each layer or the same     
        output_size: int
            Number of output classes
        connection_strengths: len(block_list) x len(block_list)
            strength of connections between layers
        topdown_input: preprocessed topdown signal of appropriate size
        reps: int
            How many times the model sees the sequence before output
        dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        topdown_type: str
            'image' or 'text' topdown signal.
        topdown: bool
            whether or not to include topdown signal
        batch_first: bool
            if the first position of array is batch or not
        bias: bool
            Whether or not to add the bias.
        return_bottom_layer: bool
            Whether to use the bottom layer's hidden state for output, if not model uses top layer's 
        """
        super(ConvGRUExplicitTopDown, self).__init__()

        self.num_layers = len(block_list)

        #self.height, self.width = input_size
        #self.input_dim = input_dim

        #self.hidden_dim = hidden_dim
        #self.kernel_size = kernel_size
        #self.connection_strengths = connection_strengths
        self.dtype = dtype
        self.batch_first = batch_first
        self.bias = bias
        self.topdown = topdown
        self.topdown_type = topdown_type
        self.return_bottom_layer = return_bottom_layer
        self.reps = reps
        
        self.orig_topdown_input = topdown_input #saves the original topdown input
        # Mostly here to increase the number of channels from in_channel --> hidden dim. This layer doesn't receive top down feedback.
        # Sort of a hack to make sure input dimensions to the bottom layer matches the top layer in the recurrence
        self.input_conv = nn.Conv2d(in_channels=block_list[0].input_dim,
                              out_channels=block_list[0].hidden_dim,
                              kernel_size=3, #why three in orig code?
                              padding=1) #why 1? another design choice?
        
        # List of linear layers between areas (to broadcast hidden state of upper areas to appropriate shape)
        feedback_linear_list = []
        for i in range(0, self.num_layers):
            if i!=0:
                block_list[i].input_dim = block_list[i-1].hidden_dim 
            if i!= (self.num_layer-1):
                block_list[i].topdown_dim = block_list[i+1].hidden_dim
            feedback_linear_list.append(nn.Linear(block_list[i].hidden_dim, 2*block_list[i].hidden_dim))
        self.block_list = nn.ModuleList(block_list)
        self.feedback_linear_list = nn.ModuleList(feedback_linear_list)
        
        # TODO: here we should figure out the feedforward/backward ranking of the cells using connection_strengths
        # discuss with Mashbayar
        #output fc
        self.fc1 = nn.Linear(block_list[-1].height*block_list[-1].width*block_list[-1].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, input_tensor):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
        :return: label 
        """
        
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        seq_len = 1 #for easy testing purposes
        #seq_len = input_tensor.size(1)

        # Images in sequence goes through all layers one by one
        # This is a reverse of the original convgru code, where the entire sequence went through one layer before advancing to next
        for rep in range(self.reps):
            for t in range(seq_len):
                current_input = self.input_conv(input_tensor[:, t, :, :, :])
                #current_input = self.input_conv(input_tensor[:, :, :, :])#[32,1,28,28]

                for layer_idx in range(self.num_layers):
                    # Current state
                    h = hidden_state[layer_idx] #if use a non-linear connection, have to change this

                    # Bottom-up signal is either the state from the bottom layer or the input if it's the bottom-most layer
                    bottomup = hidden_state[layer_idx-1] if layer_idx != 0 else current_input #if use a non-linear connection, have to change this
                    #bottomup = self._connection_decay(hidden_state[layer_idx-1], self.connection_strengths[layer_idx-1]) if layer_idx != 0 else current_input

                    if self.topdown == False:
                    #Disable topdown if using the no-topdown version of model
                        topdown_sig = None   
                    elif layer_idx+1 != self.num_layers:
                    #If non-final layer, use the next state's hidden as topdown input
                        m = nn.Linear(self.block_list[layer_idx].height* self.block_list[layer_idx].width*self.block_list[layer_idx].hidden_dim[layer_idx], self.block_list[layer_idx+1]*self.block_list[layer_idx+1].width*self.block_list[layer_idx+1].hidden_dim)
                        topdown_sig = m(torch.flatten(hidden_state[layer_idx],start_dim=1))
                    else:
                    #If final layer and topdown is not turned off, use the given topdown
                        topdown_sig = self.orig_topdown_input#stores the given topdown input to the current layer
                        
                        m = nn.Linear(self.block_list[layer_idx].height*self.block_list[layer_idx].width*self.block_list[layer_idx].hidden_dim[layer_idx], self.block_list[layer_idx].height*self.block_list[layer_idx].width*self.block_list[layer_idx].hidden_dim[layer_idx]) #changed dim here
                        topdown_sig = m(torch.flatten(hidden_state[layer_idx],start_dim=1))

                    topdown_sig = torch.reshape(topdown_sig, (topdown_sig.shape[0],self.block_list[layer_idx].hidden_dim,self.block_list[layer_idx].height,self.block_list[layer_idx].width))
                    # Update hidden state of this layer by feeding bottom-up, top-down and current cell state into gru cell
                    h = self.block_list[layer_idx](input_tensor=bottomup, # (b,t,c,h,w)
                                                  h_cur=h, topdown=topdown_sig)
                    hidden_state[layer_idx] = h

        if self.return_bottom_layer == True:
            pred = self.fc1(F.relu(torch.flatten(hidden_state[0], start_dim=1)))
        else:
            pred = self.fc1(F.relu(torch.flatten(hidden_state[-1], start_dim=1)))
        
        pred = self.fc2(F.relu(pred))
        
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.block_list[i].init_hidden(batch_size)) #requires each cell to have a init_hidden function
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    
    @staticmethod
    def _connection_decay(signal, loss_percent):
        '''
        Masks out given percentage of signal. Used to model low strength connections. SUBJECT TO CHANGE
        '''
        mask = torch.cuda.FloatTensor(signal.shape).uniform_() > loss_percent
        return signal*mask
