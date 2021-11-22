import math
import copy
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 


"""implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for 
inference, prior, and generating models."""

# prior is the previous encoding distribution

class VRNN(nn.Module):
	def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
		super(VRNN, self).__init__()

		self.x_dim = x_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU())

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim,z_dim),
			nn.Softplus())

		#prior is either a standard normal (first time-step) or previous encoding distribution
		
		
		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU())
		self.dec_std = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Softplus())
		#self.dec_mean = nn.Linear(h_dim, x_dim)
		self.dec_mean = nn.Sequential(
			nn.Linear(h_dim, x_dim),
			nn.Sigmoid())

		#recurrence
		#self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)


	def forward(self, x):

		all_enc_mean, all_enc_std = [], []
		all_dec_mean, all_dec_std = [], []
		kld_loss = 0
		nll_loss = 0

		enc_mean_t,enc_std_t = None, None
		prev_enc_mean_t,prev_enc_std_t = torch.zeros(x.shape[1],self.z_dim),torch.ones(x.shape[1],self.z_dim)
		for t in range(x.size(0)):
			phi_x_t = self.phi_x(x[t])

			if enc_mean_t != None:
				prev_enc_mean_t,prev_enc_std_t = torch.clone(enc_mean_t),torch.clone(enc_std_t)

			#encoder
			enc_t = self.enc(phi_x_t)
			enc_mean_t = self.enc_mean(enc_t)
			enc_std_t = self.enc_std(enc_t)

			# TODO: change so that it is the distribution from the previous timestep
			# if first time step: prior is a standard normal
			if t == 0:
				#prior_mean_t,prior_std_t = torch.zeros(x.shape[1],self.z_dim).cuda(3),torch.ones(x.shape[1],self.z_dim).cuda(3)
				prior_mean_t,prior_std_t = torch.zeros(x.shape[1],self.z_dim),torch.ones(x.shape[1],self.z_dim)
			# otherwise: prior is the previous encoding distribution
			else:
				prior_mean_t,prior_std_t = prev_enc_mean_t,prev_enc_std_t

			#sampling and reparameterization
			z_t = self._reparameterized_sample(enc_mean_t, enc_std_t, prev_enc_mean_t, prev_enc_std_t)
			phi_z_t = self.phi_z(z_t)

			#decoder
			dec_t = self.dec(phi_z_t)
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)

			#computing losses
			kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
			#kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, 0,1)
			#nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
			nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

			all_enc_std.append(enc_std_t)
			all_enc_mean.append(enc_mean_t)
			all_dec_mean.append(dec_mean_t)
			all_dec_std.append(dec_std_t)

		return kld_loss, nll_loss, \
			(all_enc_mean, all_enc_std), \
			(all_dec_mean, all_dec_std)


	def sample(self, seq_len):

		sample = torch.zeros(seq_len, self.x_dim)
		#print('x dim',self.x_dim,self.h_dim,self.z_dim,seq_len)
		#prev_dec_mean_t,prev_dec_std_t=None,None
		dec_mean_t,dec_std_t=None,None

		h = Variable(torch.zeros(self.n_layers, 1, self.h_dim))#.cuda(3)
		for t in range(seq_len):

			#prior
			if dec_mean_t == None:
				#prior_mean_t,prior_std_t = torch.zeros(1,self.z_dim).cuda(3),torch.ones(1,self.z_dim).cuda(3)
				prior_mean_t,prior_std_t = torch.zeros(1,self.z_dim),torch.ones(1,self.z_dim)
			else:
				prior_mean_t, prior_std_t = dec_mean_t, dec_std_t
			z_t = self._reparameterized_sample(prior_mean_t,prior_std_t,prior_mean_t,prior_std_t)
			#sampling and reparameterization
			phi_z_t = self.phi_z(z_t)
			
			#decoder
			dec_t = self.dec(phi_z_t)
			dec_mean_t = self.dec_mean(dec_t)
			dec_std_t = self.dec_std(dec_t)
			
			phi_x_t = self.phi_x(dec_mean_t)

			#sample[t] = dec_mean_t.data
			sample[t] = self._reparameterized_sample(dec_mean_t,dec_std_t,dec_mean_t,dec_std_t)
	
		return sample


	def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)


	def _init_weights(self, stdv):
		pass


	def _reparameterized_sample(self, mean, std, prev_mean, prev_std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()#.cuda(3)
		eps = Variable(eps)
		#return eps.mul(std).add_(mean)
		return eps.mul(prev_std).add_(prev_mean)


	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""
		eps = 1e-5
		kld_element =  (2 * torch.log(std_2+eps) - 2 * torch.log(std_1+eps) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)


	def _nll_bernoulli(self, theta, x):
		eps = 1e-5
		return - torch.sum(x*torch.log(theta+eps) + (1-x)*torch.log(1-theta+eps))


	def _nll_gauss(self, mean, std, x):
		pass