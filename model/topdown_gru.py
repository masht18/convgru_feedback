import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConvGRUTopDownCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, topdown_dim, kernel_size, bias, dtype):
        """
        Single ConvGRU block with topdown
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param topdown_dim: int
            Number of channels of topdown signal
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUTopDownCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.topdown_dim = topdown_dim
        self.bias = bias
        self.dtype = dtype

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,  # for update_gate,reset_gate + 2*topdown
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)
        
        if topdown_dim == 0:
            topdown_channel_dim = 1 
        else: 
            topdown_channel_dim = topdown_dim
        
        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur, topdown):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: topdown: (b, c_topdown, h, w),
            topdown signal, either a direct clue or hidden of top layer
        """
            
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)
     
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
           
        if topdown == None:
            topdown = torch.zeros_like(combined_conv) #TODO: ask

        a = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        b = (F.relu(topdown) + 1)
        combined =  a*b
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next


class ConvGRUExplicitTopDown(nn.Module):
    def __init__(self, input_size, output_size, input_dim, hidden_dim, kernel_size, connection_strengths, num_layers,
                 reps = 1, dtype = torch.cuda.FloatTensor, topdown_type = 'image', topdown = True, batch_first=True, 
                 bias=True, return_bottom_layer=False):
        """
        Full model with multiple blocks of ConvGRU
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param output_size: int
            Number of output classes
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int or list of ints with len=num_layers
            Number of channels of hidden state. Can be custom for each layer or the same
        :param kernel_size: (int, int) or list of (int, int) with len=num_layers
            Size of the convolutional kernel. Can be custom for each layer or the same
        :param connection_strengths: list of floats between 0-1. Must have len=num_layers-1
            strength of connections between layers
        :param num_layers: int
            Number of ConvGRU layers
        :param reps: int
            How many times the model sees the sequence before output
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param topdown_type: str
            'image' or 'text' topdown signal. Determines 
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_bottom_layer
            Whether to use the bottom layer's hidden state for output, if not model
            uses top layer's 
        """
        super(ConvGRUExplicitTopDown, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.connection_strengths = connection_strengths
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.topdown = topdown
        self.topdown_type = topdown_type
        self.return_bottom_layer = return_bottom_layer
        self.reps = reps
        
        # Mostly here to increase the number of channels from in_channel --> hidden dim. This layer doesn't receive top down feedback.
        # Sort of a hack to make sure input dimensions to the bottom layer matches the top layer in the recurrence
        self.input_conv = nn.Conv2d(in_channels=input_dim,
                              out_channels=hidden_dim[0],
                              kernel_size=3,
                              padding=1)
        
        # List of ConvGRU blocks
        cell_list = [] 
        
        # List of linear layers between areas (to broadcast hidden state of upper areas to appropriate shape)
        feedback_linear_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = hidden_dim[0] if i == 0 else hidden_dim[i - 1]
            next_dim = 0 if i == self.num_layers-1 else hidden_dim[i + 1]
            cell_list.append(ConvGRUTopDownCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         topdown_dim=next_dim,
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))
            if i != 0:
                feedback_linear_list.append(nn.Linear(hidden_dim[i], 2*hidden_dim[i-1]))
                #
        self.cell_list = nn.ModuleList(cell_list)
        self.feedback_linear_list = nn.ModuleList(feedback_linear_list)
        
        # The Topdown signal may consist of different modalities. 
        # This top layer processes the signal depending on its modality 
        # this section is left unused.
        if topdown_type == 'image':
            self.topdown_gru = ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=input_dim,
                                         hidden_dim=self.hidden_dim[-1]*2,
                                         kernel_size=self.kernel_size[-1],
                                         bias=self.bias,
                                         dtype=self.dtype)
        elif topdown_type == 'text':
            self.topdown_gru = nn.GRUCell(input_size=27,
                                         hidden_size=self.hidden_dim[-1]*2*self.height*self.width)
        elif topdown_type == 'audio':
            self.topdown_gru = nn.Sequential(ConvGRUCell(input_size=(64, 64),
                                         input_dim=1,
                                         hidden_dim=4,
                                         kernel_size=self.kernel_size[-1],
                                         bias=self.bias,
                                         dtype=self.dtype), nn.Flatten(), nn.ReLU(), nn.Linear(64*64*4, self.height*self.width*self.hidden_dim[-1]*2))
        else:
            raise ValueError('Topdown modality not implemented')
        
        #output fc
        self.fc1 = nn.Linear(self.height*self.width*hidden_dim[-1], 100) 
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, input_tensor, topdown):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
        :param topdown: size (b,hidden,h,w) 
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
                    h = hidden_state[layer_idx] 

                    # Bottom-up signal is either the state from the bottom layer or the input if it's the bottom-most layer
                    bottomup = hidden_state[layer_idx-1] if layer_idx != 0 else current_input
                    #bottomup = self._connection_decay(hidden_state[layer_idx-1], self.connection_strengths[layer_idx-1]) if layer_idx != 0 else current_input

                    if self.topdown == False:
                    #Disable topdown if using the no-topdown version of model
                        topdown_sig = None   
                    elif layer_idx+1 != self.num_layers:
                    #If non-final layer, use the next state's hidden as topdown input
                        m = nn.Linear(self.height*self.width*self.hidden_dim[layer_idx],self.height*self.width*self.hidden_dim[layer_idx+1])
                        topdown_sig = m(torch.flatten(hidden_state[layer_idx],start_dim=1))
                        #topdown_sig = self.feedback_linear_list[layer_idx](flattened)
                        #topdown_sig = self._connection_decay(topdown_sig, self.connection_strengths[layer_idx])
                    else:
                    #If final layer and topdown is not turned off, use the given topdown
                        topdown_sig = self.topdown_gru(topdown)
                        
                        m = nn.Linear(self.height*self.width*self.hidden_dim[layer_idx],self.height*self.width*self.hidden_dim[layer_idx]) #changed dim here
                        topdown_sig = m(torch.flatten(hidden_state[layer_idx],start_dim=1))

                    topdown_sig = torch.reshape(topdown_sig, (topdown_sig.shape[0],self.hidden_dim[layer_idx],self.height,self.width))
                    # Update hidden state of this layer by feeding bottom-up, top-down and current cell state into gru cell
                    h = self.cell_list[layer_idx](input_tensor=bottomup, # (b,t,c,h,w)
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
            init_states.append(self.cell_list[i].init_hidden(batch_size))
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
