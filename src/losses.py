import torch
import torch.func
import numpy as np

# Score loss as derived from Fisher divergence by Hyvärinen 2005. 
# Score is grad ln p, where p is the (model) data distribution.
# def loss_fn_score(model, input):
#     score = torch.vmap(model)(input).squeeze()
#     jacobian = torch.vmap(torch.func.jacfwd(model))(input).squeeze()
#     loss = torch.einsum('bjj', jacobian) + torch.einsum('bi, bi -> b', score, score)/2
#     return loss

# Yang Song denoising score.
def loss_fn(
	model, 
	x, 
	marginal_prob_std, 
	device, 
	eps=1e-4
):
	"""The loss function for training score-based generative models.

	Args:
	model: A PyTorch model instance that represents a 
	time-dependent score-based model.
	x: A mini-batch of training data.    
	marginal_prob_std: A function that gives the standard deviation of 
	the perturbation kernel.
	eps: A tolerance value for numerical stability.
	"""
	
	random_t = torch.clamp(
		(1. - torch.cos(torch.rand(x.shape[0], device=device) * np.pi / 2.))**2, 
		min=eps,
		max=1.,
	)
	# error = torch.gt(torch.zeros(x.shape[0], device=device), random_t)
	# if torch.any(error):
	#     print(random_t)
	z = torch.randn_like(x)
	std = marginal_prob_std(random_t, device=device)
	perturbed_x = x + z * std[:, None]
	score = model(perturbed_x, random_t)
	# loss_avg = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1)) #DDPM

	tmploss = torch.sum((score * std[:, None] + z)**2, dim=1)
	ones = torch.ones(x.shape[0], device=device)
	sml = torch.gt(0.1*ones, random_t).int()
	sml = sml/torch.sum(sml)
	med = torch.gt(0.3*ones, random_t).int()
	med = med/torch.sum(med)
	big = torch.gt(0.5*ones, random_t).int()
	big = big/torch.sum(big)
	comp = torch.stack((sml, med, big), dim=1) # Not controlled dtype.
	specloss_avg = torch.einsum('bi, b -> i', comp, tmploss)
	loss_avg = torch.mean(tmploss)
	return loss_avg, specloss_avg