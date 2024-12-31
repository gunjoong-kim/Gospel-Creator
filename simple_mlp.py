import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FusedLinear(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(FusedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        # 가중치 초기화 (Xavier Uniform)
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        # 선형 변환과 활성화 함수를 결합하여 연산
        return self.activation(F.linear(input.float(), self.weight.float(), self.bias.float()))

class SimpleMLP(nn.Module):
    def __init__(self, n_input_dims, n_output_dims, n_hidden_layers, n_neurons,
                 activation='ReLU', output_activation='None', w0=1.0):
        super(SimpleMLP, self).__init__()
        
        # 활성화 함수 매핑
        activation_funcs = {
            'ReLU': nn.ReLU(inplace=False),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(inplace=False),
            'ELU': nn.ELU(inplace=False),
            'Sine': torch.sin,
            'None': nn.Identity(),
        }
        hidden_activation = activation_funcs.get(activation, nn.ReLU(inplace=False))
        output_activation = activation_funcs.get(output_activation, nn.Identity())
        
        layers = []
        input_dim = n_input_dims
        
        # 히든 레이어 생성 (FusedLinear 사용)
        for layer_idx in range(n_hidden_layers):
            fused_layer = FusedLinear(input_dim, n_neurons, hidden_activation)
            layers.append(fused_layer)
            input_dim = n_neurons
        
        # 출력 레이어 생성
        fused_layer = FusedLinear(input_dim, n_output_dims, output_activation)
        layers.append(fused_layer)
        
        # Sequential로 레이어 묶기
        self.network = nn.Sequential(*layers)
        
        # 가중치 초기화
        self.initialize_weights(activation, w0)
        
    def initialize_weights(self, activation, w0):
        for idx, module in enumerate(self.network):
            if isinstance(module, FusedLinear):
                if activation == 'Sine':
                    self.siren_initialization(module, idx == 0, w0)
                else:
                    # Xavier Uniform Initialization
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def siren_initialization(self, layer, is_first_layer, w0):
        with torch.no_grad():
            num_input = layer.weight.size(1)
            if is_first_layer:
                # initialize_siren_uniform_first
                bound = 1 / num_input
            else:
                # initialize_siren_uniform
                bound = np.sqrt(6 / num_input) / w0
            nn.init.uniform_(layer.weight, -bound, bound)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -bound, bound)
        
    def forward(self, x):
        return self.network(x)
    
    def params(self):
        return torch.cat([p.view(-1) for p in self.parameters()], 0)
