import numpy as np
from utils.utils import *

def get_returns(rewards, masks, gamma):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    returns = (returns - returns.mean()) / returns.std()
    return returns

def get_loss(actor, returns, states, actions):
    mu, std = actor(torch.Tensor(states))
    log_policy = log_prob_density(torch.Tensor(actions), mu, std)
    returns = returns.unsqueeze(1)

    loss = log_policy * returns
    loss = loss.mean()
    return loss

# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, states, b, nsteps, residual_tol=1e-10):
    x = torch.zeros(b.size())
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for i in range(nsteps): # nsteps=10
        f_Ax = hessian_vector_product(actor, states, p, cg_damping=1e-1)
        alpha = rdotr / torch.dot(p, f_Ax)
        x += alpha * p
        r -= alpha * f_Ax
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        
        rdotr = new_rdotr
        if rdotr < residual_tol: # residual_tol = 0.0000000001
            break
    return x

def train_model(actor, memory, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    # ----------------------------
    # step 1: get returns
    returns = get_returns(rewards, masks, args.gamma)

    # ----------------------------
    # step 2: get gradient of loss and hessian of kl
    loss = get_loss(actor, returns, states, actions)
    loss_grad = torch.autograd.grad(loss, actor.parameters())
    loss_grad = flat_grad(loss_grad)

    step_dir = conjugate_gradient(actor, states, loss_grad.data, nsteps=10)

    # ----------------------------
    # step 3: get step direction and step size and update actor
    params = flat_params(actor)
    new_params = params + 0.5 * step_dir
    update_model(actor, new_params)