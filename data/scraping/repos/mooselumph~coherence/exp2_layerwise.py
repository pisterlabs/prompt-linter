from typing import Tuple
from coherence.custom_types import Batch

import jax
import jax.numpy as jnp

import haiku as hk
import optax

from coherence.data import get_data, decimate, normalize
from coherence.train import network_and_loss, do_training, update_params, net_accuracy
from coherence.models.mlp import lenet_fn

from coherence.pruning.runner import masked_update, imp
from coherence.pruning.pruning import Rule, create_plan, layerwise_threshold_prune, init_mask

from coherence.coherence import ptwise, get_coherence, subnetwork_coherence

from functools import partial

import matplotlib.pyplot as plt

# load mnist data
train, train_eval, test_eval = get_data("mnist",batch_size=100,format_fun=normalize)

# cnn, loss, params
net, xent_loss = network_and_loss(lenet_fn)

key = jax.random.PRNGKey(42)
params = net.init(key, next(train)["image"])

# optimization of network
opt = optax.adam(1e-3)
opt_state = opt.init(params)
accuracy_fn = net_accuracy(net)

cs_in = []
cs_out = []

def calc_coherence(loss_fn,mask):

    # @jax.jit
    def helper(
      params: hk.Params,
      batch: Batch,
    ) -> Tuple[hk.Params, optax.OptState]:

        ptwise_fn = ptwise(loss_fn)
        pt_grads = ptwise_fn(params, batch)

        c = get_coherence(pt_grads)

        c_in, c_out= subnetwork_coherence(c, mask)
        cs_in.append(c_in)
        cs_out.append(c_out)

    return helper
    
def train_fn_mask(mask, key):

    # params = net.init(key, next(train)["image"])
    update_fn = masked_update(opt,xent_loss,mask)

    # train
    final_params = do_training(update_fn, accuracy_fn, params, opt_state, train, train_eval, test_eval, epochs=1001)

    return final_params

def train_fn_trace(mask):

    update_fn = update_params(opt,xent_loss)
    aux_fn = calc_coherence(xent_loss,mask)
    final_params = do_training(
                    update_fn, accuracy_fn, 
                    params, opt_state, 
                    train, train_eval, test_eval, 
                    epochs=1001, 
                    aux_fn=aux_fn,
                    aux_epoch=100,
                    )

    return final_params


rules = [Rule('linear_2',lambda v: 1 - 2*(1 - v)),Rule('/b',1)]
plan = create_plan(params,rules=rules,default_value=0.95)

print(plan)

mask = init_mask(params,plan)

masks, branches = imp(key,train_fn_mask,partial(layerwise_threshold_prune,plan=plan),params,mask,num_reps=1)

train_fn_trace(masks[-1])

print("cs_in: ", cs_in)
print("cs_out: ", cs_out)