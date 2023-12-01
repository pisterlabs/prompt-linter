import torch
from utils import *
from minresqlp import MinresQLP

def train_critic(critic, states, returns, critic_optim, critic_iter, batch_size, device):
    criterion = torch.nn.MSELoss(reduction='sum').to(device)
    for i in range(critic_iter):
        values = critic(states).squeeze()
        loss = 0.5*criterion(values, returns)/batch_size
        critic_optim.zero_grad()
        loss.backward()
        critic_optim.step()
    return loss
        
def get_loss(critic, states, returns, log_probs, batch_size):
    critic.eval()
    values = critic(states).squeeze().detach()
    loss = -log_probs * ( returns - values )
    loss = loss.sum()/batch_size
    return loss,values.sum()/batch_size

def fisher_vector_product(actor, kl_grad, p, eps=0.1):
    kl_grad_p = (kl_grad * p).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(),create_graph=True)
    kl_hessian_p = flat_hessian(kl_hessian_p)
    return kl_hessian_p + eps * p


# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, policies, b, nsteps, residual_tol=1e-10, device='cpu',eps=0.1):
    new = policies.clone()
    old = new.detach()
    kl = kl_divergence(new,old).mean()
#     kl = kl_divergence(new,old).sum()/policies.shape[0]
    kl_grad = flat_grad(torch.autograd.grad(kl, actor.parameters(), create_graph=True))
        
    x = torch.zeros(b.size(), device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for i in range(nsteps):

        _Avp = fisher_vector_product(actor, kl_grad, p, eps)
        alpha = rdotr / torch.dot(p, _Avp)
        x += alpha * p
        r -= alpha * _Avp
        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr
        p = r + betta * p
        rdotr = new_rdotr
#         if (i+1)%10==0:
#             print(f'step {i+1}, rdotr: {rdotr}')
        if rdotr < residual_tol:
            break
#     print(f'nsteps used: {i+1}')
    return x,kl_grad,rdotr

def fvp(actor, kl_grad, p, eps, device='cpu'):
    '''
    p: np or jnp object
    output: object on torch device
    '''
    assert(len(p.squeeze().shape)==1)
    try:
        p_device = torch.FloatTensor(np.array(p)).to(device)
    except:
        p_device = p
    kl_grad_p = (kl_grad @ p_device).squeeze()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, actor.parameters(),create_graph=True)
    kl_hessian_p = flat_hessian(kl_hessian_p)
    if len(p.shape)>1:
        kl_hessian_p = kl_hessian_p.reshape(len(kl_hessian_p),-1)
    return kl_hessian_p + eps * p_device

def minres_qlp(actor, policies, van_grad, nsteps, residual_tol=1e-10, device='cpu',eps=0.):  
    kl = kl_divergence(policies.clone(),policies.clone().detach()).mean()
    kl_grad = flat_grad(torch.autograd.grad(kl, actor.parameters(), create_graph=True))
    
    afun = lambda x: fvp(actor,kl_grad,x,eps,device).cpu().numpy()
    nat_grad = MinresQLP(afun,van_grad.cpu().numpy(),residual_tol,nsteps)

    
#     fng_true = fvp(actor,kl_grad,nat_grad.squeeze(),0,device)
#     err = (van_grad-fng_true).norm()/van_grad.norm()
    return torch.FloatTensor(nat_grad[0]).squeeze().to(device),kl_grad,nat_grad[5].item()


def train_model(actor, critic, memory, a_lr, actor_optim, critic_optim, critic_iter=5,order=2,lam_ent = 0.,optimizer='adam',device='cpu', cg_tol=1e-10, eff_lr=False, nsteps=100, method='minresqlp',shift=0):
    
    batch_size = memory[2].shape[0]
    policies, critic_states, returns, log_probs = [m.reshape(-1,*(m.size()[2:])).to(device) for m in memory] # flat out the batch and time dimensions

    # ----------------------------
    # step 1: train critic several steps with respect to returns
    critic_loss = train_critic(critic, critic_states, returns, critic_optim, critic_iter, batch_size, device)

    # ----------------------------
    # step 2: get gradient of loss and hessian of kl
    loss,baseline = get_loss(critic, critic_states, returns, log_probs, batch_size)
    actor_optim.zero_grad()
    loss.backward(create_graph=True)
    vanilla_grad = []
    for p in actor.parameters():
        vanilla_grad.append(p.grad)
    vanilla_grad = flat_grad(vanilla_grad).data
    
    if order == 1:
        step_dir = vanilla_grad
        a_lr_eff = a_lr
    elif order == 2:
        if method == 'cg':
            natural_grad,_,_ = conjugate_gradient(actor, policies, vanilla_grad, nsteps=nsteps, residual_tol=cg_tol, device=device,eps=shift)
        elif method == 'minresqlp':
            natural_grad,_,_ = minres_qlp(actor, policies, vanilla_grad, nsteps=nsteps, residual_tol=cg_tol, device=device,eps=shift)
            
        if eff_lr:
            norm = 1 / torch.sqrt(torch.abs(torch.dot(vanilla_grad.T, natural_grad) + 1e-16))
            step_dir = natural_grad*norm
            a_lr_eff = a_lr*norm.cpu().item()
        else:
            step_dir = natural_grad
            a_lr_eff = a_lr

    # ----------------------------
    # step 3: get step direction and step size and update actor
    params = flat_params(actor)
    if optimizer=='gd':
        new_params = params - a_lr*step_dir
        update_model(actor, new_params)
    else: # allow adam or sgd
        update_grad(actor, step_dir)
        actor_optim.step()
        
    # step 5: analysis data
    entropy = get_entropies(policies).mean().cpu().detach().item()
    pmin,pmax,pmean = meas_prob(policies)
    
    return [loss.detach().cpu().item(), critic_loss.detach().cpu().item(), a_lr_eff, entropy, pmin, pmax, pmean, baseline.cpu().item()]

def train_model_pg(actor, memory, a_lr, actor_optim, order=2, optimizer='adam', device='cpu', cg_tol=1e-10, eff_lr=False, nsteps=100, method='minresqlp', shift=0):
    
    batch_size = memory[2].shape[0]
    policies, states, returns, log_probs = [m.reshape(-1,*(m.size()[2:])).to(device) for m in memory]

    # ----------------------------
    # step 1: get gradient of loss and hessian of kl
    loss = -log_probs * returns
    loss = loss.sum()/batch_size
    
    try: actor_optim.zero_grad()
    except: pass
    loss.backward(create_graph=True)
    vanilla_grad = []
    for p in actor.parameters():
        vanilla_grad.append(p.grad)
    vanilla_grad = flat_grad(vanilla_grad).data
    
#     return vanilla_grad
    
    if order == 1:
        step_dir = vanilla_grad
        a_lr_eff = a_lr
    elif order == 2:
        if method == 'cg':
            natural_grad,kl_grad,residual = conjugate_gradient(actor, policies, vanilla_grad, nsteps=nsteps, residual_tol=cg_tol, device=device,eps=shift)
        elif method == 'minresqlp':
            natural_grad,kl_grad,residual = minres_qlp(actor, policies, vanilla_grad, nsteps=nsteps, residual_tol=cg_tol, device=device,eps=shift)
    
        fisher_nat_grad = fvp(actor,kl_grad,natural_grad,0,device)
        fisher_van_grad = fvp(actor,kl_grad,vanilla_grad,0,device)
        diff = vanilla_grad-fisher_nat_grad
        fisher_diff = fvp(actor,kl_grad,diff,0,device)
        
        van_grad_err = (diff.norm()/vanilla_grad.norm()).item()
        van_grad_err1 = (diff.norm()/natural_grad.norm()).item()
        van_grad_err_fisher = np.sqrt((diff.dot(fisher_diff)/vanilla_grad.dot(fisher_van_grad)).item())
        van_grad_err_fisher1 = np.sqrt((diff.dot(fisher_diff)/natural_grad.dot(fisher_nat_grad)).item())
    
        if eff_lr:
            norm = 1 / torch.sqrt(torch.abs(torch.dot(vanilla_grad.T, natural_grad) + 1e-16))
            step_dir = natural_grad*norm
            a_lr_eff = a_lr*norm.cpu().item()
        else:
            step_dir = natural_grad
            a_lr_eff = a_lr
#         print(f'Learning rate: {a_lr} {a_lr_eff}')
#     return vanilla_grad,step_dir

    # ----------------------------
    # step 2: get step direction and step size and update actor
    params = flat_params(actor)
    if optimizer=='gd':
        new_params = params - a_lr*step_dir
        update_model(actor, new_params)
    elif optimizer=='adam':
        update_grad(actor, step_dir)
        actor_optim.step()
    elif optimizer=='adam-manual':
        new_params = params - actor_optim.step(natural_grad)
        update_model(actor, new_params)      
        
    # step 5: analysis data
    entropy = get_entropies(policies).mean().cpu().detach().item()
    pmin,pmax,pmean = meas_prob(policies)
    
    return [loss.detach().cpu().item(), a_lr_eff, entropy, pmin, pmax, pmean, van_grad_err, van_grad_err1, van_grad_err_fisher, van_grad_err_fisher1, residual]





'''def train_model(actor, critic, memory, a_lr, actor_optim, critic_optim, critic_iter=5,order=2,lam_ent = 0.,optimizer='adam',device='cpu', cg_tol=1e-10, eff_lr=False):
    lengths = [len(ret) for ret in memory[2]]
    batch_size = len(lengths)
    try:
        policies, critic_states, returns, log_probs = [torch.cat(m) for m in memory]
    except:
        policies, critic_states, returns, log_probs = [m.reshape(-1,*(m.size()[2:])).to(device) for m in memory]


    # ----------------------------
    # step 1: train critic several steps with respect to returns
    critic_loss = train_critic(critic, critic_states, returns, critic_optim, critic_iter, batch_size, device)

    # ----------------------------
    # step 2: get gradient of loss and hessian of kl
    loss = get_loss(critic, critic_states, returns, log_probs, batch_size)
    actor_optim.zero_grad()
    loss.backward(create_graph=True)
    vanilla_grad = []
    for p in actor.parameters():
        vanilla_grad.append(p.grad)
#     vanilla_grad = torch.autograd.grad(loss, actor.parameters(), create_graph=True)
    vanilla_grad = flat_grad(vanilla_grad).data
    if order == 1:
        step_dir = vanilla_grad
    elif order == 2:
        natural_grad = conjugate_gradient(actor, policies, vanilla_grad, nsteps=10, device=device,residual_tol=cg_tol)
        if eff_lr:
            norm = 1 / torch.sqrt(torch.abs(torch.dot(vanilla_grad.T, natural_grad) + 1e-16))
            step_dir = natural_grad*norm
            a_lr_eff = a_lr*norm.cpu().item()
        else:
            step_dir = natural_grad
            a_lr_eff = a_lr
#         print(f'Learning rate: {a_lr} {eff_a_lr}')

    # step 3: add entropy
    entropy = (0.5*log_probs**2).sum()/batch_size
    entropy_grad = torch.autograd.grad(entropy, actor.parameters(), create_graph=True)
    entropy_grad = flat_grad(entropy_grad).data
    step_dir -= lam_ent*entropy_grad

    # ----------------------------
    # step 4: get step direction and step size and update actor
    params = flat_params(actor)
    if optimizer=='gd':
        new_params = params - a_lr*step_dir
        update_model(actor, new_params)
    elif optimizer=='adam':
        update_grad(actor, step_dir)
        actor_optim.step()
        
    # step 5: analysis data
    entropy = get_entropies(policies).mean().cpu().detach().item()
    pmin,pmax,pmean = meas_prob(policies)
    
    return [loss.detach().cpu().item(), critic_loss.detach().cpu().item(), a_lr_eff, entropy, pmin, pmax, pmean]
'''

