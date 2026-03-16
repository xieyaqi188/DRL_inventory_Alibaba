### Custom neural network architectures

import torch
from torch import nn
from ds.configs import MAX_REPLENISH


class NeuralNetworkCreator:
    """
    Create DS model by given data; called in main.py
    Uses lazy initialization - model weights are initialized during first forward pass
    and inherit the random seed from global state set by fix_seed()
    """

    def __init__(self):
        self.architectures = {
            'ds_none': DSNone,
            'ds_none_coeff': DSNoneCoeff,
            'ds_base': DSBase,  # input includes inventory; i.e., base_stock with inv features
            'ds_base_coeff': DSBaseCoeff,
        }
        self.args = None

    def create_neural_network(self, args, device='cpu'):
        self.args = args
        ds_name = args.action_mode
        model = self.architectures[ds_name](args, device=device)
        return model.to(device)


class MyNeuralNetwork(nn.Module):

    def __init__(self, args, device='cpu'):
        super(MyNeuralNetwork, self).__init__()
        self.device = device

        self.trainable = True
        self.activation_funcs = {
            'relu': nn.ReLU(),
            'elu': nn.ELU(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
        }

        # Store args for lazy initialization
        self.args = args
        self.net = None
        self.layers = None
        self._initialized = False

    def initialize_network_if_needed(self, input_tensor):
        """Initialize network on first forward pass with actual input size"""
        if not self._initialized:
            input_size = input_tensor.shape[-1]  # Get the last dimension as input size
            self.net, self.layers = self.create_ds_net(self.args, input_size)
            
            # Move network to the same device as the input tensor
            self.net = self.net.to(input_tensor.device)
            
            # Apply initial bias after network creation
            # initial_action_bias is a ratio (0.0-1.0) of max_replenish
            if self.args.initial_action_bias is not None:
                if self.args.output_layer_activation is not None:
                    position = -2
                else:
                    position = -1

                actual_bias_value = self.args.initial_action_bias * MAX_REPLENISH
                self.layers[position].bias.data.fill_(actual_bias_value)
            
            self._initialized = True

    def forward(self, current_state):
        raise NotImplementedError

    def create_ds_net(self, args, input_size):
        neurons_per_hidden_layer = args.neurons_per_hidden_layer
        inner_layer_activation = args.inner_layer_activation
        output_layer_activation = args.output_layer_activation
        output_size = args.output_size

        layers = []
        prev_size = input_size
        
        for output_neurons in neurons_per_hidden_layer:
            layers.append(nn.Linear(prev_size, output_neurons))
            layers.append(self.activation_funcs[inner_layer_activation])
            prev_size = output_neurons

        layers.append(nn.Linear(prev_size, output_size))

        if output_layer_activation is not None:
            layers.append(self.activation_funcs[output_layer_activation])

        return nn.Sequential(*layers), layers

    def flatten_then_concatenate_states(self, state_list, dim=1):
        """
        flatten features and concatenate them into a tensor of shape (sample_num, 1)
        """
        return torch.cat([
            tensor.flatten(start_dim=dim) for tensor in state_list
        ], dim=dim)


class DSNone(MyNeuralNetwork):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)

    def forward(self, current_state):
        # input_state = self.flatten_then_concatenate_states(
        #     [current_state['features'], current_state['inventory'] / MAX_REPLENISH], dim=1)

        input_state = self.flatten_then_concatenate_states(
            [current_state['features_org'][:, 0:1], current_state['inventory']], dim=1)
        # print('input', input_state.shape, input_state)

        self.initialize_network_if_needed(input_state)
        output = self.net(input_state)

        return torch.clip(output, min=0, max=MAX_REPLENISH)


class DSNoneCoeff(MyNeuralNetwork):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)

    def forward(self, current_state):
        # 'features' have been normalized in data
        input_state = self.flatten_then_concatenate_states(
            [current_state['features'], current_state['inventory'] / MAX_REPLENISH], dim=1)

        self.initialize_network_if_needed(input_state)
        output = self.net(input_state)

        reg_input = current_state['reg_input']
        output_dim = output.shape[1] # = 4
        # coeffs in [0, 1]
        weights = torch.clip(output[:, :output_dim - 1], min=0, max=1)  # shape: (batch_size, 4)
        bias = output[:, output_dim - 1]  # shape: (batch_size,)
        final_output = (weights * reg_input).sum(dim=1) * 28 + bias  # shape: (batch_size,)

        return torch.clip(final_output.unsqueeze(1), min=0, max=MAX_REPLENISH)


class DSBase(MyNeuralNetwork):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)

    def forward(self, current_state):

        input_state = self.flatten_then_concatenate_states(
            [current_state['features_org'][:, 0:1], current_state['inventory']], dim=1)

        self.initialize_network_if_needed(input_state)
        output = self.net(input_state)
        inv_pos = current_state['inventory'].sum(dim=1)

        return torch.clip(output - inv_pos.unsqueeze(1), min=0, max=MAX_REPLENISH)


class DSBaseCoeff(MyNeuralNetwork):

    def __init__(self, args, device='cpu'):
        super().__init__(args, device)

    def forward(self, current_state):
        # 'features' have been normalized in data
        # normalize current_state['inventory'], min=0, max=max_replenish
        input_state = self.flatten_then_concatenate_states(
            [current_state['features'], current_state['inventory'] / MAX_REPLENISH], dim=1)

        self.initialize_network_if_needed(input_state)
        output = self.net(input_state)

        reg_input = current_state['reg_input']  # shape: (batch_size, 4)
        output_dim = output.shape[1]
        # Coeffs in [0, 1]
        weights = torch.clip(output[:, :output_dim-1], min=0, max=1)  # shape: (batch_size, 4)
        bias = output[:, output_dim-1]  # shape: (batch_size,)

        final_output = (weights * reg_input).sum(dim=1) * 28 + bias  # shape: (batch_size,)

        inv_pos = current_state['inventory'].sum(dim=1)

        return torch.clip(final_output.unsqueeze(1) - inv_pos.unsqueeze(1), min=0, max=MAX_REPLENISH)