from typing import Tuple
from coherence.custom_types import Batch

import jax
import jax.numpy as jnp

import haiku as hk
import optax

from tqdm import tqdm

from coherence.data import get_data, decimate, normalize, sanitize, get_data_by_class
from coherence.train_with_state import network_and_loss, do_training, update_params, net_accuracy
from coherence.models.cnn import cifar_vgg_11_fn

from coherence.pruning.runner import masked_update, masked_update_with_state, imp
from coherence.pruning.pruning import extract_masked_params, global_threshold_prune, Rule, create_plan, init_mask

from coherence.coherence import ptwise, ptwise_with_state, get_coherence, subnetwork_coherence

from coherence.utils import ravel_pytree

from functools import partial

import matplotlib.pyplot as plt

# load mnist data
train, train_eval, test_eval = get_data("cifar10",batch_size=100,format_fun=sanitize)

dsets_by_class = get_data_by_class("cifar10",batch_size=100,format_fun=sanitize)

# cnn, loss, params
net, xent_loss = network_and_loss(cifar_vgg_11_fn)

key = jax.random.PRNGKey(42)
params, state = net.init(key, next(train)["image"])

# optimization of network
opt = optax.adam(1e-3)
opt_state = opt.init(params)
accuracy_fn = net_accuracy(net)

def calc_coherence(loss_fn):
    
    ptwise_fn = ptwise_with_state(loss_fn)

    # @jax.jit
    def helper(
      params: hk.Params,
      state: hk.State,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:

        outputs = []

        for ind,ds in enumerate(dsets_by_class):

            epoch_grad = None
            coh = None

            hs = []
            masks = []

            for _ in tqdm(range(60)):

                # Should add has_aux input to make_pointwise
                pt_grads, _ = ptwise_fn(params, state, next(ds))

                # c = get_coherence(batch_grads)
                grad = jax.tree_map(lambda g: jnp.sum(g,axis=0), pt_grads)
                coh = jax.tree_map(lambda g: jnp.sum(jnp.concatenate([jnp.abs(g[i,:]) for i in range(g.shape[0])],axis=0),axis=0))

                if epoch_grad is None:
                    epoch_grad = grad
                    epoch_coh = coh
                else:
                    epoch_grad = jax.tree_map(lambda a,b: a+b, grad, epoch_grad)
                    epoch_coh = jax.tree_map(lambda a,b: a+b, grad, epoch_coh)
                
                
                c = jax.tree_map(lambda a,b: jnp.abs(a)/b)

                # Create histogram from non-excluded params
                rules = [Rule('batch_norm',1)]
                plan = create_plan(c,rules,default_value=0.9)
                old_mask = init_mask(c,plan)

                c_allowed = jax.tree_map(extract_masked_params,c,old_mask)
                c_flat = ravel_pytree(c_allowed)
                h = jnp.histogram(c_flat, bins=100)
                hs.append(h)

                mask = global_threshold_prune(c,old_mask,plan)
                masks.append(mask)

            outputs.append((masks,hs))

        return outputs

    return helper
    
    
f = calc_coherence(xent_loss)

outputs = f(params,state,None)

