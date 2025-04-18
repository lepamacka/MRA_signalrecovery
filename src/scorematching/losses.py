import torch
import torch.nn as nn
import torch.func as tfunc

# Score loss as derived from Fisher divergence by Hyvärinen 2005. Not sliced.
# Score is grad ln p, where p is the (model) data distribution.
def loss_fn_score(model, input):
    score = model(input)
    jacobian = torch.vmap(tfunc.jacfwd(model))(input.unsqueeze(1)).squeeze()
    loss = torch.einsum('bjj -> b', jacobian) + torch.einsum('bi, bi -> b', score, score)/2
    return loss