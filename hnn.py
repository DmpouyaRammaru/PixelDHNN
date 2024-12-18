# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
import numpy as np

from nn_models import MLP
from utils import rk4
from nn_models import MLPAutoencoder

class HNN(torch.nn.Module):
    '''Learn arbitrary vector fields that are sums of conservative and solenoidal fields'''
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                    baseline=False, assume_canonical_coords=True):
        super(HNN, self).__init__()
        self.baseline = baseline
        self.differentiable_model = differentiable_model
        self.assume_canonical_coords = assume_canonical_coords
        self.M = self.permutation_tensor(input_dim) # Levi-Civita permutation tensor
        self.field_type = field_type

    def forward(self, x):
        # traditional forward pass
        if self.baseline:
            return self.differentiable_model(x)

        y = self.differentiable_model(x)
        return y

    def rk4_time_derivative(self, x, dt):
        return rk4(fun=self.time_derivative, y0=x, t=0, dt=dt)

    def time_derivative(self, x, t=None, separate_fields=False):
        '''NEURAL ODE-STLE VECTOR FIELD'''
        if self.baseline:
            return self.differentiable_model(x)

        '''NEURAL HAMILTONIAN-STLE VECTOR FIELD'''
        # F2を作って，分けてみる？
        F1 = self.forward(x)

        conservative_field = torch.zeros_like(x) # start out with both components set to 0
        solenoidal_field = torch.zeros_like(x)

        if self.field_type != 'solenoidal':
            conservative_field = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for conservative field

        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F1.sum(), x, create_graph=True)[0] # gradients for solenoidal field
            dHdq, dHdp = torch.split(dF2, dF2.shape[-1]//2, dim=1)
            q_dot_hat, p_dot_hat = dHdp, -dHdq
            solenoidal_field = torch.cat([q_dot_hat, p_dot_hat], axis=-1)

        if separate_fields:
            return [conservative_field, solenoidal_field]

        return conservative_field + solenoidal_field

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M


class PixelHNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, autoencoder, nonlinearity='tanh', baseline=False):
        super(PixelHNN, self).__init__()
        self.autoencoder = autoencoder
        self.baseline = baseline

        output_dim = 1
        # hamiltonian neural network
        nn_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity)
        # dissipative neural network
        nn_model_diss = MLP(input_dim, hidden_dim, output_dim, nonlinearity)

        self.hnn = HNN(input_dim, differentiable_model=nn_model, field_type='solenoidal', baseline=baseline)
        self.diss = HNN(input_dim, differentiable_model=nn_model_diss, field_type='conservative', baseline=baseline)

    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def time_derivative(self, z, separate_fields=False):
        return self.hnn.time_derivative(z, separate_fields)
    
    def time_derivative_diss(self, z, separate_fields=False):
        return self.diss.time_derivative(z, separate_fields)

    def forward(self, x):
        z = self.encode(x)
        z_next = z + self.time_derivative(z) + self.time_derivative_diss(z)
        return self.decode(z_next)


class DHNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DHNN, self).__init__()  # Inherit the methods of the Module constructor
        self.mlp_h = MLP(input_dim, hidden_dim, output_dim = 1, nonlinearity='tanh')  # Instantiate an MLP for learning the conservative component
        self.mlp_d = MLP(input_dim, hidden_dim, output_dim = 1, nonlinearity='tanh')  # Instantiate an MLP for learning the dissipative component
        self.autoencoder = MLPAutoencoder(2*28**2, 200, 2, nonlinearity='relu')
    
    def encode(self, x):
        return self.autoencoder.encode(x)

    def decode(self, z):
        return self.autoencoder.decode(z)

    def forward(self, x, as_separate=False): 
        inputs = x
        D = self.mlp_d(inputs)
        H = self.mlp_h(inputs)

        irr_component = torch.autograd.grad(D.sum(), x, create_graph=True)[0]  # Take their gradients
        rot_component = torch.autograd.grad(H.sum(), x, create_graph=True)[0]

        # For H, we need the symplectic gradient, and therefore
        # we split our tensor into 2 and swap the chunks.
        dHdq, dHdp = torch.split(rot_component, rot_component.shape[-1]//2, dim=1)
        q_dot_hat, p_dot_hat = dHdp, -dHdq
        rot_component = torch.cat([q_dot_hat, p_dot_hat], axis=-1)
        if as_separate:
            return irr_component, rot_component  # Return the two fields seperately, or return the composite field. 

        return rot_component + irr_component # return decomposition if as_separate else sum of fields
