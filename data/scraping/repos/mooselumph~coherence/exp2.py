from typing import Tuple
from coherence.custom_types import Batch

import jax
import jax.numpy as jnp

import haiku as hk
import optax

from coherence.data import get_data, decimate, normalize, sanitize, get_data_by_class
from coherence.train_with_state import network_and_loss, do_training, update_params, net_accuracy
from coherence.models.cnn import cifar_vgg_11_fn

from coherence.pruning.runner import masked_update, masked_update_with_state, imp
from coherence.pruning.pruning import global_threshold_prune, Rule, create_plan, init_mask

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

cs_in = [[] for _ in range(10)]
cs_out = [[] for _ in range(10)]
hs = [[] for _ in range(10)]

def calc_coherence(loss_fn,mask):

    # @jax.jit
    def helper(
      params: hk.Params,
      state: hk.State,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:

        ptwise_fn = ptwise_with_state(loss_fn)
        
        for ind,ds in enumerate(dsets_by_class):

            pt_grads, _ = ptwise_fn(params, state, next(ds))
            c = get_coherence(pt_grads)

            c_in, c_out= subnetwork_coherence(c, mask)
            cs_in[ind].append(c_in)
            cs_out[ind].append(c_out)

            c_flat = ravel_pytree(c)
            h = jnp.histogram(c_flat, bins=100)
            hs[ind].append(h)

    return helper
    
def train_fn_mask(mask, key):

    # params = net.init(key, next(train)["image"])
    update_fn = masked_update_with_state(opt,xent_loss,mask)

    # train
    final_params = do_training(update_fn, accuracy_fn, params, state, opt_state, train, train_eval, test_eval,epochs=31)

    return final_params

def train_fn_trace(mask):

    update_fn = update_params(opt,xent_loss)
    aux_fn = calc_coherence(xent_loss,mask)
    final_params = do_training(
                    update_fn, accuracy_fn, 
                    params, state, opt_state, 
                    train, train_eval, test_eval, 
                    epochs=101, 
                    aux_fn=aux_fn,
                    aux_epoch=10,
                    )

    return final_params

rules = [Rule('/w',1),]
plan = create_plan(params,rules=rules,default_value=0.95)
mask = init_mask(params,plan)

masks, branches = imp(key,train_fn_mask,partial(global_threshold_prune,plan=plan,fraction=0.95),params,mask,num_reps=1)

train_fn_trace(masks[-1])

print("cs_in: ", cs_in)
print("cs_out: ", cs_out)