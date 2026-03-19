import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# class BayesLinear(torch.nn.Module):
#     r"""
#     Applies Bayesian Linear
#     Arguments:
#         prior_mu (Float): mean of prior normal distribution.
#         prior_sigma (Float): sigma of prior normal distribution.

#     .. note:: other arguments are following linear of pytorch 1.2.0.
#     https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
#     """
#     __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

#     def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
#         super(BayesLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.prior_mu = prior_mu
#         self.prior_sigma = prior_sigma
#         self.prior_log_sigma = math.log(prior_sigma)

#         self.weight_mu = torch.nn.Parameter(torch.Tensor(out_features, in_features))
#         self.weight_log_sigma = torch.nn.Parameter(torch.Tensor(out_features, in_features))

#         self.bias_mu = torch.nn.Parameter(torch.Tensor(out_features))
#         self.bias_log_sigma = torch.nn.Parameter(torch.Tensor(out_features))

#         self.bias = bias

#         self.reset_parameters()

#     def reset_parameters(self):
#         # Initialization method of Adv-BNN
#         stdv = 1. / math.sqrt(self.weight_mu.size(1))
#         self.weight_mu.data.uniform_(-stdv, stdv)
#         self.weight_log_sigma.data.fill_(self.prior_log_sigma)
#         # if self.bias:
#         self.bias_mu.data.uniform_(-stdv, stdv)
#         self.bias_log_sigma.data.fill_(self.prior_log_sigma)

#     def forward(self, input, weight_eps, bias_eps):
#         if weight_eps is None:
#             weight_eps = torch.randn(size=(self.out_features,self.in_features))
#         if bias_eps is None:
#             bias_eps = torch.randn(size=(1,self.out_features))
#         weight = self.weight_mu + torch.exp(self.weight_log_sigma) * weight_eps.to(self.weight_mu.device)
#         bias = self.bias_mu + torch.exp(self.bias_log_sigma) * bias_eps.to(self.bias_mu.device)
#         return torch.nn.functional.linear(input, weight, bias)

#     def loss(self):
#         return 0.
    
#     def extra_repr(self):
#         r"""
#         Overriden.
#         """
#         return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'. \
#             format(self.prior_mu,
#                    self.prior_sigma,
#                    self.in_features,
#                    self.out_features,
#                    self.bias is not None)

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        
        self.mu_weight = nn.Parameter(torch.Tensor(out_features, in_features).normal_(0, 0.1))
        self.sigma_weight = nn.Parameter(torch.Tensor(out_features, in_features).fill_(0.1))
        
        self.mu_bias = nn.Parameter(torch.Tensor(out_features).normal_(0, 0.1))
        self.sigma_bias = nn.Parameter(torch.Tensor(out_features).fill_(0.1))
        
        self.epsilon_weight = None
        self.epsilon_bias = None
        
    def forward(self, x, eps_weight, eps_bias):
        if eps_weight is None:
            eps_weight = torch.normal(torch.zeros_like(self.mu_weight))
        if eps_bias is None:
            eps_bias = torch.normal(torch.zeros_like(self.mu_bias))
        
        weight = self.mu_weight + self.sigma_weight * eps_weight.to(self.mu_weight.device)
        bias = self.mu_bias + self.sigma_bias * eps_bias.to(self.mu_bias.device)
        return nn.functional.linear(x, weight, bias)
    
    def loss(self):
        return self._kl_divergence()
    
    def _kl_divergence(self):
        kl_weight = 0.5 * torch.sum(
            1 + 2 * torch.log(self.sigma_weight) - torch.square(self.mu_weight) - torch.square(self.sigma_weight)
        ) / (self.prior_sigma ** 2)
        
        kl_bias = 0.5 * torch.sum(
            1 + 2 * torch.log(self.sigma_bias) - torch.square(self.mu_bias) - torch.square(self.sigma_bias)
        ) / (self.prior_sigma ** 2)
        
        return kl_weight + kl_bias