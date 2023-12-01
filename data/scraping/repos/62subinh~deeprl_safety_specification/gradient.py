import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten


def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten


def kl_divergence(new_actor, old_actor, obs, deterministic=True):
    if deterministic:
        mu = new_actor(obs)

        mu_old = old_actor(obs)
        mu_old = mu_old.detach()

        kl = (mu_old - mu).pow(2) / 2.0
        return kl.sum(1, keepdim=True)
    else:
        mu, std = new_actor(obs)
        mu_old, std_old = old_actor(obs)
        mu_old = mu_old.detach()
        std_old = std_old.detach()

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu_old, std_old / pi_new -> mu, std
        # be careful of calculating KL-divergence. It is not symmetric metric.
        kl = torch.log(std / std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)


def hessian_vector_product(new_actor, old_actor, obs, p, cg_damping=1e-1):
    p.detach()
    kl = kl_divergence(new_actor=new_actor, old_actor=old_actor, obs=obs)
    kl = kl.mean()

    kl_grad = torch.autograd.grad(kl, new_actor.parameters(), create_graph=True)
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p).sum()
    kl_hessian = torch.autograd.grad(kl_grad_p, new_actor.parameters())
    kl_hessian = flat_hessian(kl_hessian)

    return kl_hessian + p * cg_damping


# from openai baseline code
# https://github.com/openai/baselines/blob/master/baselines/common/cg.py
def conjugate_gradient(actor, target_actor, obs, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros(b.size(), device=device)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for i in range(nsteps):
        Ap = hessian_vector_product(actor, target_actor, obs, p, cg_damping=1e-1)
        alpha = rdotr / torch.dot(p, Ap)

        x += alpha * p
        r -= alpha * Ap

        new_rdotr = torch.dot(r, r)
        betta = new_rdotr / rdotr

        p = r + betta * p
        rdotr = new_rdotr

        if rdotr < residual_tol:
            break
    return x