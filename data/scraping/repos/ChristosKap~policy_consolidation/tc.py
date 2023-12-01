# Code for Policy Consolidation model which is adapted
# from PPO2 implementation in OpenAI baselines

import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from baselines import logger
from collections import deque
from baselines.common import explained_variance
import itertools
import pickle
from policies import MlpPolicy


class Model(object):
    def __init__(self, *, policy, ob_space, ac_space, nbatch_act, nbatch_train, nsteps, ent_coef, vf_coef,
                 max_grad_norm, cascade_depth, flow_factor, mesh_factor, var_init='random',
                 imp_sampling='normal', imp_clips=[5,-5], dynamic_neglogpacs=False,full_kl=False,
                 kl_beta=1.0, targs=None,value_cascade=False,separate_value=False, prox_value_fac=False,
                 reverse_kl=False,cross_kls=['new','new'],multiagent=False,model_scope=""):
        
        sess = tf.get_default_session()

        # Assuming observation space for each agent is the same,
        # set ob_space to that of first agent
        if multiagent:
            ob_space=ob_space.spaces[0]
            ac_space=ac_space.spaces[0]
            
        # Create first policy in cascade (or only policy for baselines)
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1,
                           reuse=False,scope=model_scope+"model")
        train_model = policy(sess, ob_space, ac_space, nbatch_train,
                             nsteps, reuse=True, scope=model_scope+"model")
        
        if reverse_kl:
            print("reverse kl")

        if separate_value: # If using separate policy and value networks
            act_value_model = policy(sess, ob_space, ac_space, nbatch_act,
                                     1, reuse=False,scope=model_scope+"value_model",
                                     obs=act_model.X)
            train_value_model = policy(sess, ob_space, ac_space, nbatch_train,
                                       nsteps, reuse=True,scope=model_scope+"value_model",
                                       obs=train_model.X)
        else:
            act_value_model = act_model
            train_value_model = train_model
            
        policies = [train_model]
        act_policies = [act_model]
        train_value_models = [train_value_model]
        act_value_models = [act_value_model]

        # Create hidden policies
        for k in range(1,cascade_depth):
            scope = model_scope+"hidden"+str(k)+"pol"
            print(scope)
            hidden_act_policy = policy(sess, ob_space, ac_space, nbatch_act,
                                       1, reuse=False,scope=scope,obs=act_model.X)
            hidden_policy = policy(sess, ob_space, ac_space, nbatch_train,
                                   nsteps, reuse=True,scope=scope,obs=train_model.X)
            act_policies.append(hidden_act_policy)
            policies.append(hidden_policy)
            if separate_value:
                hidden_act_value_model = policy(sess, ob_space, ac_space, nbatch_act,
                                                1, reuse=False,
                                                scope=model_scope+"hidden"+str(k)+"value",
                                                obs=act_model.X)
                hidden_value_model = policy(sess, ob_space, ac_space, nbatch_train,
                                            nsteps, reuse=True,
                                            scope=model_scope+"hidden"+str(k)+"value",
                                            obs=train_model.X)
            else:
                hidden_act_value_model = hidden_act_policy
                hidden_value_model = hidden_policy
                
            act_value_models.append(hidden_act_value_model)
            train_value_models.append(hidden_value_model)
            
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        # negative log action probabilities for clipped PPO
        OLDNEGLOGPACS = [tf.placeholder(tf.float32, [None])
                         for i in range(cascade_depth)]
        # summed neglogpacs over partial trajectories, used for importance sampling ratios
        CUM_OLDNEGLOGPACS = [tf.placeholder(tf.float32, [None])
                             for i in range(cascade_depth)]
            
        OLDVPREDS = [tf.placeholder(tf.float32, [None])
                     for i in range(cascade_depth)]
        LRS = [tf.placeholder(tf.float32, [])
               for i in range(cascade_depth)]
        
        neglogpacs = [pol.pd.neglogp(A) for pol in policies] 
        
        # Importance factors NOT used for results in paper
        IMP_FACTORS = []
        if imp_sampling == 'normal':
            for k in range(cascade_depth):
                imp_factor = tf.exp(CUM_OLDNEGLOGPACS[0]-CUM_OLDNEGLOGPACS[k])
                if full_kl:
                    IMP_FACTORS.append(tf.concat([[1.],imp_factor[:-1]],0))
                else:
                    IMP_FACTORS.append(imp_factor)
        elif imp_sampling == 'clipped':
            print('imp factors clipped')
            assert len(imp_clips)==2
            for k in range(cascade_depth):
                imp_factor = tf.exp(tf.clip_by_value(
                        CUM_OLDNEGLOGPACS[0]-CUM_OLDNEGLOGPACS[k],
                        imp_clips[0],imp_clips[1]))
                if full_kl:
                    IMP_FACTORS.append(tf.concat([[1.],imp_factor[1:]],0))
                else:
                    IMP_FACTORS.append(imp_factor)
        elif imp_sampling =='none':
            print('No importance sampling')
            IMP_FACTORS = [tf.constant(1.0) for i in range(cascade_depth)]
        
        entropy = tf.reduce_mean(train_model.pd.entropy()) 
        vpred = train_value_model.vf
        
        # For storing probability distributions for policies from previous time step
        OLD_PDPARAMS = [tf.placeholder(tf.float32,
                                       [*policies[i].pdparam.get_shape()])
                        for i in range(cascade_depth)]
        OLD_PDS = [policies[i].pdtype.pdfromflat(OLD_PDPARAMS[i])
                   for i in range(cascade_depth)]

        ratios = [tf.exp(old - curr)
                  for (old,curr) in zip(OLDNEGLOGPACS,neglogpacs)]
        CLIPRANGES = [tf.placeholder(tf.float32, [])
                      for i in range(cascade_depth)]
        clipfrac = tf.reduce_mean(tf.to_float(
                tf.greater(tf.abs(ratios[0] - 1.0), CLIPRANGES[0])))
        
        if full_kl: # for PC, fixed KL and adaptive KL models
            KL_BETAS = [tf.placeholder(tf.float32,[])
                        for i in range(cascade_depth)]
            kl_losses = []
            cross_kl_losses = []
            vf_loss = tf.reduce_mean(tf.square(vpred - R))
            # Factor for limit size of value step, not used in paper
            if prox_value_fac: 
                assert isinstance(prox_value_fac, (int, float))
                vf_loss += prox_value_fac * tf.reduce_mean(
                    tf.square(vpred - OLDVPREDS[0]))

            # create loss terms keeping value functions close to their
            # previous values and close to neighbouring values (not used in paper)
            if value_cascade:
                assert isinstance(prox_value_fac, (int, float))
                cross_value_losses = [vf_coef * tf.reduce_mean(
                        tf.square(vpred - OLDVPREDS[1]))]
                for k in range(1,cascade_depth):
                    cross_value_loss = mesh_factor*tf.reduce_mean(
                        tf.square(OLDVPREDS[k-1]-train_value_models[k].vf))
                    if k < (cascade_depth-1):
                        cross_value_loss += tf.reduce_mean(
                            tf.square(train_value_models[k].vf - OLDVPREDS[k+1]))
                    cross_value_losses.append(cross_value_loss)
                    vf_loss += prox_value_fac * np.power(mesh_factor,k) * tf.reduce_mean(
                        tf.square(train_value_models[k].vf-OLDVPREDS[k]))
                vf_loss += tf.reduce_sum(cross_value_losses)
                    
            pg_loss = tf.reduce_mean(-ADV * ratios[0])
            if reverse_kl: # directionality of individual PPO-like KL terms
                kl_losses.append(tf.reduce_mean(OLD_PDS[0].kl(policies[0].pd)))
            else:
                kl_losses.append(tf.reduce_mean(policies[0].pd.kl(OLD_PDS[0])))
            
            if cascade_depth > 1:
                cross_kl_losses.append(
                    tf.reduce_mean(flow_factor * policies[0].pd.kl(OLD_PDS[1])))

            # create extra loss terms for PC model
            for k in range(1,cascade_depth):
                if reverse_kl:
                    kl_losses.append(tf.reduce_mean(OLD_PDS[k].kl(policies[k].pd)))
                else:
                    kl_losses.append(tf.reduce_mean(policies[k].pd.kl(OLD_PDS[k])))
                # set up directions of kl terms between adjacent policies
                if cross_kls[0]=='old':
                    cross_kl_losses.append(
                        tf.reduce_mean(mesh_factor*OLD_PDS[k-1].kl(policies[k].pd)))
                elif cross_kls[0]=='new':
                    cross_kl_losses.append(
                        tf.reduce_mean(mesh_factor*policies[k].pd.kl(OLD_PDS[k-1])))
                if k < (cascade_depth-1):
                    if cross_kls[1]=='old':
                        cross_kl_losses.append(
                            tf.reduce_mean(OLD_PDS[k+1].kl(policies[k].pd)))
                    elif cross_kls[1]=='new':
                        cross_kl_losses.append(
                            tf.reduce_mean(policies[k].pd.kl(OLD_PDS[k+1])))

            # add PPO-like KL terms for each policy
            kl_loss = tf.constant(0.0)
            for i in range(cascade_depth):
                kl_loss += KL_BETAS[i]*tf.reduce_mean(kl_losses[i])
            cross_kl_loss = tf.reduce_sum(cross_kl_losses)
            loss = pg_loss + kl_loss + cross_kl_loss + vf_loss + entropy * ent_coef

            if targs:
                targ_greater = tf.greater(kl_loss,targs)
                targ_less = tf.less(kl_loss,targs)
            
        else: # Clipped PPO / Clipped cascade - clipped cascade not used in the paper
            vpredclipped = OLDVPREDS[0] + tf.clip_by_value(
                train_value_model.vf - OLDVPREDS[0], - CLIPRANGES[0], CLIPRANGES[0])
            vf_losses1 = tf.square(vpred - R)
            vf_losses2 = tf.square(vpredclipped - R)
            vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
            pg_losses = -ADV * ratios[0]
            pg_losses2 = -ADV * tf.clip_by_value(
                ratios[0], 1.0 - CLIPRANGES[0], 1.0 + CLIPRANGES[0])
            if cascade_depth < 2:
                kl_losses = kl_losses2 = tf.constant(0.0)
            else: 
                # add kl loss with second network in cascade
                if dynamic_neglogpacs:
                    kl_losses = -neglogpacs[0] + neglogpacs[1]
                    kl_losses2 = tf.clip_by_value(
                        -neglogpacs[0],
                         tf.log(1.0-CLIPRANGES[0])-OLDNEGLOGPACS[0],
                         tf.log(1.0+CLIPRANGES[0])-OLDNEGLOGPACS[0])+neglogpacs[1]
                else:
                    kl_losses = -neglogpacs[0] + OLDNEGLOGPACS[1]
                    kl_losses2 = tf.clip_by_value(
                        -neglogpacs[0],
                         tf.log(1.0-CLIPRANGES[0])-OLDNEGLOGPACS[0],
                         tf.log(1.0+CLIPRANGES[0])-OLDNEGLOGPACS[0])+OLDNEGLOGPACS[1]
            pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
            kl_loss = flow_factor*tf.reduce_mean(tf.maximum(kl_losses, kl_losses2))
            loss = pg_loss + kl_loss - entropy * ent_coef + vf_loss * vf_coef    
            
            # Construction of additional loss terms for clipped version of PC model (not used for paper)
            if cascade_depth > 1:
                if dynamic_neglogpacs:
                    for k in range(1,cascade_depth):
                        hidden_pg_losses = mesh_factor*IMP_FACTORS[k-1]*(
                            neglogpacs[k]-neglogpacs[k-1])
                        hidden_pg_losses2 = mesh_factor*IMP_FACTORS[k-1]*(
                            -neglogpacs[k-1]-tf.clip_by_value(
                                -neglogpacs[k],
                                 tf.log(1.0-CLIPRANGES[k])-OLDNEGLOGPACS[k],
                                 tf.log(1.0+CLIPRANGES[k])-OLDNEGLOGPACS[k]))
                        if k < (cascade_depth-1):
                            hidden_pg_losses2 += IMP_FACTORS[k]*(tf.clip_by_value(
                                    -neglogpacs[k],
                                     tf.log(1.0-CLIPRANGES[k])-OLDNEGLOGPACS[k],
                                     tf.log(1.0+CLIPRANGES[k])-OLDNEGLOGPACS[k])+neglogpacs[k+1])
                        hidden_pg_loss = tf.reduce_mean(
                            tf.maximum(hidden_pg_losses, hidden_pg_losses2))
                        loss += hidden_pg_loss 
                else:    
                    for k in range(1,cascade_depth):
                        # We optimize each hidden policy to be close to the policy 
                        # of the adjacent hidden policies at the previous time step.
                        # It is twice as important to be close to 'shallower' policy,
                        # reflecting wider 'tube width' between shallower policies
                        hidden_pg_losses = mesh_factor*IMP_FACTORS[k-1]*(neglogpacs[k]-OLDNEGLOGPACS[k-1])
                        hidden_pg_losses2 = mesh_factor*IMP_FACTORS[k-1]*(
                            -OLDNEGLOGPACS[k-1]-tf.clip_by_value(
                                -neglogpacs[k],
                                 tf.log(1.0-CLIPRANGES[k])-OLDNEGLOGPACS[k],
                                 tf.log(1.0+CLIPRANGES[k])-OLDNEGLOGPACS[k]))
                        hidden_pg_loss = tf.reduce_mean(tf.maximum(hidden_pg_losses, hidden_pg_losses2))
                        if k < (cascade_depth-1):
                            # Here we assume that importance factors are fixed during one epoch
                            hidden_pg_losses = IMP_FACTORS[k]*(OLDNEGLOGPACS[k+1]-neglogpacs[k]) 
                            hidden_pg_losses2 = IMP_FACTORS[k]*(
                                tf.clip_by_value(
                                    -neglogpacs[k],
                                     tf.log(1.0-CLIPRANGES[k])-OLDNEGLOGPACS[k],
                                     tf.log(1.0+CLIPRANGES[k])-OLDNEGLOGPACS[k])+OLDNEGLOGPACS[k+1])
                            hidden_pg_loss += tf.reduce_mean(tf.maximum(hidden_pg_losses, hidden_pg_losses2))
                        loss += hidden_pg_loss 
                
        # ops for recording importance factors
        imp_fac_mean = tf.reduce_mean(IMP_FACTORS)        
        imp_fac_mean_last = tf.reduce_mean(IMP_FACTORS[-1])

        approxkl = .5 * tf.reduce_mean(tf.square(neglogpacs[0] - OLDNEGLOGPACS[0]))
        if cascade_depth > 1:
            approxkl12 = .5 * tf.reduce_mean(tf.square(neglogpacs[0] - neglogpacs[1]))
        pg_kl_loss_ratio = pg_loss / (kl_loss + 1e-8)
        
        # Optimizer set up
        training_ops = []
        init_ops = [] # for initializing hidden vars
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope+'model')
        for k in range(cascade_depth):
            curr_scope = model_scope+'hidden'+str(k)+"pol" if k>0 else model_scope+'model'
            curr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=curr_scope)
            curr_grads = tf.gradients(loss, curr_vars)
            if max_grad_norm is not None:
                curr_grads, _curr_grad_norm = tf.clip_by_global_norm(curr_grads, max_grad_norm)
            curr_grads = list(zip(curr_grads,curr_vars))
            curr_trainer = tf.train.AdamOptimizer(learning_rate = LRS[k])
            if var_init == "equal": # Initialise all hidden policy networks to same as visible policy
                for i in range(len(curr_vars)):
                    init_ops.append(tf.assign(curr_vars[i], model_vars[i].initialized_value()))
            training_ops.append(curr_trainer.apply_gradients(curr_grads))

        if separate_value: # not used in paper
            if value_cascade:
                for k in range(len(train_value_models)):
                    curr_scope = model_scope+'hidden'+str(k)+"value" if k>0 else model_scope+'value_model'
                    curr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=curr_scope)
                    curr_grads = tf.gradients(loss, curr_vars)
                    if max_grad_norm is not None:
                        curr_grads, _curr_grad_norm = tf.clip_by_global_norm(curr_grads, max_grad_norm)
                    curr_grads = list(zip(curr_grads,curr_vars))
                    curr_trainer = tf.train.AdamOptimizer(learning_rate = LRS[k])
                    training_ops.append(curr_trainer.apply_gradients(curr_grads))
            else:
                curr_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope+'value_model')
                curr_grads = tf.gradients(loss, curr_vars)
                if max_grad_norm is not None:
                    curr_grads, _curr_grad_norm = tf.clip_by_global_norm(curr_grads, max_grad_norm)
                curr_grads = list(zip(curr_grads,curr_vars))
                curr_trainer = tf.train.AdamOptimizer(learning_rate = LRS[0])
                training_ops.append(curr_trainer.apply_gradients(curr_grads))
                        
        self.policies = policies
        self.act_policies = act_policies
        self.initial_states = []
        for p in self.policies:
            self.initial_states.append(p.initial_state)
        _train = tf.group(*training_ops)            
        _init_op = tf.group(*init_ops)

        # Operations for running
        
        act_ops = [act_policies[i].a for i in range(cascade_depth)]
        val_ops = [act_value_models[i].v for i in range(cascade_depth)]
        neglogpac_ops = [act_policies[i].neglogp for i in range(cascade_depth)]
        pdparam_ops = [act_policies[i].pdparam for i in range(cascade_depth)] 
        run_ops = list(itertools.chain(act_ops, val_ops, neglogpac_ops, pdparam_ops))
        if self.initial_states[0] is not None:
            state_ops = [act_policies[i].state for i in range(cascade_depth)]
            run_ops = run_ops.extend(state_ops)
        
        def run(obs, states, dones):
            td_map = {}
            td_map[act_model.X]=obs
            # Augment td_map for recurrent models
            if states[0] is not None:
                for i in range(len(self.policies)):
                    td_map[act_policies[i].S]=states[i]
            if act_policies[0].M:
                for i in range(len(self.policies)):
                    td_map[act_policies[i].M]=dones
                    
            results = sess.run(run_ops, feed_dict=td_map)
            
            if states[0] is None:
                results += self.initial_states

            # actions, values, neglogpacs, pdparams, states    
            return tuple(results[i:i+cascade_depth]
                         for i in range(0,len(results), cascade_depth))
 
        def train(lrs, clipranges, kl_betas, obs, returns,
                  masks, actions, values, neglogpacs,
                  cumneglogpacs, pdparams, states=[None]):

            advs = returns - values[:,0]
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map ={}
            td_map.update({train_model.X:obs,
                           A:np.take(actions,0,axis=1),
                           ADV:advs,
                           R:returns})
            for k in range(cascade_depth):
                td_map.update({LRS[k]:lrs[k],
                               CLIPRANGES[k]:clipranges[k],
                               OLDNEGLOGPACS[k]:neglogpacs[:,k],
                               OLDVPREDS[k]:values[:,k],
                               CUM_OLDNEGLOGPACS[k]:cumneglogpacs[:,k],
                               OLD_PDPARAMS[k]:pdparams[:,k]})
                if kl_betas is not None:
                    td_map[KL_BETAS[k]] = kl_betas[k]
            if states[0] is not None:
                for k in range(cascade_depth):
                    td_map[policies[k].S] = states[k]
                #td_map[train_model.S] = states
                #td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac,
                 imp_fac_mean, imp_fac_mean_last, pg_kl_loss_ratio,
                 kl_losses, _train],
                td_map
            )[:-1]

        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy',
                           'approxkl', 'clipfrac', 'imp factor mean',
                           'imp factor mean last','PG/KL loss ratio',
                           'KL losses']

        def save(save_path):
            ps = sess.run(tf.trainable_variables())
            joblib.dump(ps, save_path)

        def load(load_path):
            # If you want to load weights,
            # also save/load observation scaling inside VecNormalize
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(tf.trainable_variables(), loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        # Only loads first policy beaker
        def load_with_scope(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            print(len(loaded_params),
                  len(tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, scope=model_scope+'model')))
            for p, loaded_p in zip(tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES,scope=model_scope+'model'),
                                   loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            
        self.run = run
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_value_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        self.load_with_scope = load_with_scope
        self.cascade_depth = cascade_depth
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
        if var_init is not 'random':
            sess.run(_init_op)
            print("copied vars")
            
    def get_depth(self):
        return self.cascade_depth
    
# For running single agent experiments and collecting data
class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.obs = np.zeros((nenv,)+env.observation_space.shape,
                            dtype=model.train_model.X.dtype.name)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_states
        self.dones = [False for _ in range(nenv)]
        
    def run(self): # not used in paper, left over from OpenAI code
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(
                self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions)
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        print(mb_neglogpacs[0])
        print(mb_neglogpacs.shape)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
    # obs, returns, masks, actions, values, neglogpacs, states = runner.run()
    
    def run_cascade(self): # function for running single agent experiments
        depth = self.model.get_depth()
        mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_pdparams = [],[],[],[],[],[],[]
        mb_states = self.states 
        epinfos = []
        print(self.obs.shape)
        for _ in range(self.nsteps):
            actions, values, neglogpacs, pdparams, self.states = self.model.run(
                self.obs, self.states, self.dones)
            mb_obs.append(self.obs.copy())
            mb_dones.append(self.dones)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_pdparams.append(pdparams)
            self.obs[:], rewards, self.dones, infos = self.env.step(actions[0])
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions).swapaxes(0,1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0,1)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).swapaxes(0,1)
        mb_pdparams = np.asarray(mb_pdparams, dtype=np.float32).swapaxes(0,1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states[0], self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[0][t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[0][t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values[0]

        # calculate cumulative neglogps for calculating importance sampling ratios
        mb_cumneglogpacs = np.cumsum(mb_neglogpacs,axis=1)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones)),
                 *map(sf012, (mb_actions, mb_values, mb_neglogpacs, mb_cumneglogpacs, mb_pdparams)),
                 mb_states, epinfos) 

# Runner for selfplay
class SelfPlayRunner(object):

    def __init__(self, *, env, models, nsteps, gamma, lam, dense_decay = False):
        self.env = env
        self.models = models
        self.nplayers = len(models)
        nenv = env.num_envs
        self.obs = np.zeros((nenv,) + (len(env.observation_space.spaces),)+env.observation_space.spaces[0].shape,
                            dtype=models[0].train_model.X.dtype.name)
        print(self.obs.shape)
        self.obs[:] = env.reset()
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = [model.initial_states for model in models]
        self.dones = np.array([[False for _ in range(self.nplayers)] for _ in range(nenv)])
        self.dense_decay = dense_decay

        if self.dense_decay:
            self.dense_frac = 1.0-self.dense_decay
        
        print(self.obs[:,0,:].shape)

    def run(self):
        depth = self.models[0].get_depth()
        mb_obs, mb_rewards, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_pdparams = [],[],[],[],[],[],[]
        mb_states = self.states[0] 
        epinfos = []
        for _ in range(self.nsteps):
            all_actions = []
            # only record data from first player
            actions, values, neglogpacs, pdparams, self.states[0] = self.models[0].run(
                self.obs[:,0,:], self.states[0], self.dones[:,0])
            mb_obs.append(self.obs[:,0,:].copy())
            mb_dones.append(self.dones[:,0])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_pdparams.append(pdparams)
            all_actions.append(actions[0])
            # get actions and states of remaining players
            for i in range(1,self.nplayers):
                actions,_,_,_,self.states[i] = self.models[i].run(
                    self.obs[:,i,:], self.states[i], self.dones[:,i])
                all_actions.append(actions[0])
            all_actions = np.stack(all_actions, axis=1)
            self.obs[:], rewards, self.dones, infos = self.env.step(all_actions)
            rewards = np.array(rewards)
            rewards = rewards.swapaxes(0,1)[0]
            infos = np.array(infos).swapaxes(0,1)[0]
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            # Decay dense reward 
            if self.dense_decay:
                rewards = np.array([self.dense_frac*info['shaping_reward']+(1-self.dense_frac)*info['main_reward'] for info in infos])
            mb_rewards.append(rewards)
        if self.dense_decay:
            self.dense_frac = max(0, self.dense_frac - self.dense_decay)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions).swapaxes(0,1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0,1)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32).swapaxes(0,1)
        mb_pdparams = np.asarray(mb_pdparams, dtype=np.float32).swapaxes(0,1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.models[0].value(self.obs[:,0,:], self.states[0], self.dones[:,0])
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones[:,0]
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[:,0][t+1]
                nextvalues = mb_values[0][t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[0][t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values[0]

        # calculate cumulative neglogps for calculating importance sampling ratios
        mb_cumneglogpacs = np.cumsum(mb_neglogpacs,axis=1)
        return (*map(sf01, (mb_obs, mb_returns, mb_dones)),
                 *map(sf012, (mb_actions, mb_values, mb_neglogpacs, mb_cumneglogpacs, mb_pdparams)),
                 mb_states, epinfos)
    
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def sf012(arr):

    s= arr.shape
    return arr.swapaxes(0,2).reshape(s[1]*s[2],s[0],*s[3:])

def constfn(val):
    def f(_):
        return val
    return f

def constfn_arr(val,length):
    def f(_):
        return [val for i in range(length)]
    return f

def decayfn_arr(start, decay, length):
    def f(_):
        return [start*decay**i for i in range(length)]
    return f

def adapt_betas(betas, kl_losses, targs, leeway, mult_factor):
    for i in range(len(betas)):
        if kl_losses[i]<(targs[i]/leeway):
            betas[i] /= mult_factor
        elif kl_losses[i]>(leeway*targs[i]):
            betas[i] *= mult_factor
    return betas

# Learn function for single agent experiments
def learn(*, policy, env, nsteps, total_timesteps, ent_coef,
             lr, lr_decay, var_init, vf_coef=0.5,
             max_grad_norm=0.5, gamma=0.99, lam=0.95,
             log_interval=10, nminibatches=4, noptepochs=4,
             cliprange=0.2, imp_sampling='normal',
             imp_clips=[5,-5], dynamic_neglogpacs=False,
             full_kl=False, separate_value = False,
             value_cascade=False, prox_value_fac = False,
             save_interval=0, cascade_depth=1,
             flow_factor = 1.0, mesh_factor = 2.0,
             kl_type='fixed', reverse_kl=False,
             kl_beta=1.0, cross_kls=['new','new'],
             adaptive_targ=None, targ_leeway=1.5,
             beta_mult_fac=2.0,load_path=False,
             prev_model=None,epoch_num=0):

    if isinstance(lr, float):
        if lr_decay:
            lrs = decayfn_arr(lr,1.0/mesh_factor,cascade_depth) # learning rates exponentially smaller for deeper policies in cascade
        else:
            lrs = constfn_arr(lr,cascade_depth)
    else: assert callable(lr)
    if isinstance(cliprange, float): clipranges = decayfn_arr(cliprange,1.0/mesh_factor,cascade_depth)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    targs=None
    if full_kl:
        kl_betas = decayfn_arr(kl_beta, mesh_factor, cascade_depth)(0)
        if kl_type=='adaptive':
            targs = decayfn_arr(adaptive_targ, 1.0/mesh_factor, cascade_depth)(0)
    else:
        kl_betas=None
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    print('nbatch: '+str(nbatch))
    nbatch_train = nbatch // nminibatches

    # create model
    if prev_model is None:
        make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                                    nbatch_act=nenvs, nbatch_train=nbatch_train,
                                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                    max_grad_norm=max_grad_norm,cascade_depth=cascade_depth,
                                    flow_factor=flow_factor, mesh_factor=mesh_factor,
                                    var_init=var_init, imp_sampling=imp_sampling,
                                    imp_clips=imp_clips, dynamic_neglogpacs=dynamic_neglogpacs,
                                    full_kl=full_kl, cross_kls=cross_kls,
                                    separate_value=separate_value, value_cascade=value_cascade,
                                    prox_value_fac=prox_value_fac,targs=targs, reverse_kl=reverse_kl)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        model = make_model()
    else:
        model=prev_model
    if load_path:
        model.load(load_path=load_path)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    # Policy update iteration loop
    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lrs(frac)
        cliprangenow = clipranges(frac)
        if full_kl:
            kl_betasnow = kl_betas
        else:
            kl_betasnow = None
        obs, returns, masks, actions, values, neglogpacs, cumneglogpacs, pdparams, states, epinfos = runner.run_cascade() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        kl_lossvals = []
        if states[0] is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds]
                              for arr in (obs, returns, masks, actions,
                                          values, neglogpacs, cumneglogpacs, pdparams))
                    mblossval = model.train(lrnow, cliprangenow, kl_betasnow, *slices)
                    mblossvals.append(mblossval[:-1])
                    kl_lossvals.append(mblossval[-1])
        else: # recurrent version, never used - leftover from OpenAI code
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds]
                              for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        kl_lossvals = np.mean(kl_lossvals,axis=0)
        if kl_type=='adaptive' and full_kl: # adapt beta for adaptive KL baseline
            kl_betas = adapt_betas(kl_betas, kl_lossvals, targs, targ_leeway, beta_mult_fac)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values[:,0], returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            if kl_type=='adaptive' and full_kl:
                logger.logkv('KL beta 1',kl_betas[0])
                logger.logkv('KL loss 1', kl_lossvals[0])
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, "epoch"+str(epoch_num)+"_"+'%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            if env.ob_rms:
                pickle.dump([env.ob_rms, env.ret_rms, env.ret], open(savepath+"_vecnorm.pkl","wb"))
    env.close()
    return savepath, model

# Learn function for selfplay setting
def learn_multi(*, policy, env, nsteps, total_timesteps, ent_coef,
                   lr, lr_decay, var_init, vf_coef=0.5,  max_grad_norm=0.5,
                   gamma=0.99, lam=0.95, log_interval=10, nminibatches=4,
                   noptepochs=4, cliprange=0.2, imp_sampling='normal',
                   imp_clips=[5,-5], dynamic_neglogpacs=False,
                   full_kl=False, separate_value = False, value_cascade=False,
                   prox_value_fac = False, save_interval=0, cascade_depth=1,
                   flow_factor = 1.0, mesh_factor = 2.0, kl_type='fixed',
                   reverse_kl=False, kl_beta=1.0, cross_kls=['new','new'],
                   adaptive_targ=None, targ_leeway=1.5, beta_mult_fac=2.0,
                   load_path=False,prev_model=None, test_history=False,
                   num_test_eps=30,test_env=None,test_model_dir=False,
                   dense_decay = False):

    if isinstance(lr, float):
        if lr_decay:
            lrs = decayfn_arr(lr,1.0/mesh_factor,cascade_depth)
        else:
            lrs = constfn_arr(lr,cascade_depth)
    else: assert callable(lr)
    if isinstance(cliprange, float): clipranges = decayfn_arr(
        cliprange,1.0/mesh_factor,cascade_depth)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    targs=None
    if full_kl:
        kl_betas = decayfn_arr(kl_beta, mesh_factor, cascade_depth)(0)
        if kl_type=='adaptive':
            targs = decayfn_arr(adaptive_targ, 1.0/mesh_factor, cascade_depth)(0)
    else:
        kl_betas=None
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    print('nbatch: '+str(nbatch))
    nbatch_train = nbatch // nminibatches

    if prev_model is None:
        make_model = lambda : Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                                    nbatch_act=nenvs, nbatch_train=nbatch_train,
                                    nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                                    max_grad_norm=max_grad_norm,cascade_depth=cascade_depth,
                                    flow_factor=flow_factor, mesh_factor=mesh_factor,
                                    var_init=var_init, imp_sampling=imp_sampling,
                                    imp_clips=imp_clips, dynamic_neglogpacs=dynamic_neglogpacs,
                                    full_kl=full_kl, cross_kls=cross_kls,
                                    separate_value=separate_value, value_cascade=value_cascade,
                                    prox_value_fac=prox_value_fac,targs=targs, reverse_kl=reverse_kl,
                                    multiagent=True,model_scope="learn_")
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        model = make_model()
    else:
        model=prev_model
    if load_path:
        model.load(load_path=load_path)

    # If planning to test model on its historic self, create another duplicate model to load prev weights into
    if test_history:
        make_test_model = lambda m_scope : Model(policy=policy, ob_space=ob_space,
                                                 ac_space=ac_space, nbatch_act=nenvs,
                                                 nbatch_train=nbatch_train,
                                                 nsteps=nsteps, ent_coef=ent_coef,
                                                 vf_coef=vf_coef,max_grad_norm=max_grad_norm,
                                                 cascade_depth=cascade_depth,
                                                 flow_factor=flow_factor, mesh_factor=mesh_factor,
                                                 var_init=var_init, imp_sampling=imp_sampling,
                                                 imp_clips=imp_clips, dynamic_neglogpacs=dynamic_neglogpacs,
                                                 full_kl=full_kl, cross_kls=cross_kls,
                                                 separate_value=separate_value, value_cascade=value_cascade,
                                                 prox_value_fac=prox_value_fac,targs=targs,
                                                 reverse_kl=reverse_kl, multiagent=True,model_scope=m_scope)
        test_model = make_test_model("test_")
        hist_model = make_test_model("hist_")

    if test_model_dir:
        pass
    else:
        runner = SelfPlayRunner(env=env, models=[model,model], nsteps=nsteps, gamma=gamma, lam=lam, dense_decay=dense_decay)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lrs(frac)
        cliprangenow = clipranges(frac)
        if full_kl:
            kl_betasnow = kl_betas
        else:
            kl_betasnow = None
        obs, returns, masks, actions, values, neglogpacs, cumneglogpacs, pdparams, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        kl_lossvals = []
        if states[0] is None or states[0][0] is None: # nonrecurrent version
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds]
                              for arr in (obs, returns, masks, actions, values,
                                          neglogpacs, cumneglogpacs, pdparams))
                    mblossval = model.train(lrnow, cliprangenow, kl_betasnow, *slices)
                    mblossvals.append(mblossval[:-1])
                    kl_lossvals.append(mblossval[-1])
        else: # recurrent version - havent used for multi learning
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            envsperbatch = nbatch_train // nsteps
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks,
                                                          actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        kl_lossvals = np.mean(kl_lossvals,axis=0)
        if kl_type=='adaptive' and full_kl:
            kl_betas = adapt_betas(kl_betas, kl_lossvals, targs, targ_leeway, beta_mult_fac)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
            if test_history:
                scores = [False]*test_history
            else:
                scores = [False]
            ev = explained_variance(values[:,0], returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
            logger.logkv("explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('epdenserewmean', safemean([epinfo['dr'] for epinfo in epinfobuf]))
            logger.logkv('epcentrerewmean', safemean([epinfo['cr'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            logger.logkv('time_elapsed', tnow - tfirststart)
            if kl_type=='adaptive' and full_kl:
                logger.logkv('KL beta 1',kl_betas[0])
                logger.logkv('KL loss 1', kl_lossvals[0])
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
            if test_history:
                test_model.load(load_path=savepath)
                for j in range(min(test_history,update//save_interval)):
                    hist_path = osp.join(checkdir, '%.5i'%(update-j*save_interval))
                    hist_model.load(load_path=hist_path)
                    scores[j] = test_play(
                        env=test_env, models=[test_model,hist_model], max_eps = num_test_eps)[0]
                #runner.obs[:] = env.reset()
        if update % log_interval == 0 or update == 1:
            for i in range(len(scores)):
                logger.logkv("test_scores_"+str(i), scores[i])
            logger.dumpkvs()
    env.close()
    test_env.close()
    return savepath, model

# Function for playing two agents against each other
def test_play(*, env, models, max_eps):

    nplayers = len(models)
    nenv = env.num_envs
    obs = np.zeros((nenv,) + (len(env.observation_space.spaces),)+env.observation_space.spaces[0].shape,
                   dtype=models[0].train_model.X.dtype.name)
    obs[:] = env.reset()
    states = [model.initial_states for model in models]
    dones = np.array([[False for _ in range(nplayers)] for _ in range(nenv)])
    epinfos = []
    num_eps = 0
    scores = np.zeros(nplayers,dtype=np.float32)
    wins = np.zeros(nplayers,dtype=np.float32)
    while num_eps < max_eps:
        all_actions = []
        for i in range(nplayers):
            actions,_,_,_,states[i] = models[i].run(
                obs[:,i,:], states[i], dones[:,i])
            all_actions.append(actions[0])
        all_actions = np.stack(all_actions, axis=1)
        obs[:], rewards, dones, infos = env.step(all_actions)
        #rewards = np.array(rewards)
        #rewards = rewards.swapaxes(0,1)[0]
        infos = np.array(infos)
        for info in infos:
            maybeepinfo = info[0].get('episode')
            if maybeepinfo:
                num_eps += 1
                winner = False
                for i in range(len(info)):
                    if 'winner' in info[i]:
                        scores[i] += 1.0
                        wins[i] += 1
                        winner=True
                if not winner:
                    scores += 0.5
    print("scores: ", scores)
    print("num_eps: ", num_eps)
    print("wins: ", wins)
    return scores / num_eps, wins

# Function for loading self-play model and playing it against its past selves
def historic_test_play(*, env, model_dir, test_dir, max_eps,
                          just_latest=False, checkpoint_time=False,
                          checkpoint_match=False):

    model = load_model_from_dir(env=env, model_dir=model_dir, name='now_')
    test_model = load_model_from_dir(env=env, model_dir=test_dir, name='past_')
    
    test_params = pickle.load(open(test_dir+"/params.pkl", "rb"))
    model_params = pickle.load(open(model_dir+"/params.pkl", "rb"))
    
    if 'save_interval' in model_params:
        model_save_interval = model_params['save_interval']
    else:
        model_save_interval = 25
    
    if 'save_interval' in test_params:
        test_save_interval = test_params['save_interval']
    else:
        test_save_interval = 25
        
    test_time_step = test_save_interval*test_params['ncpu']*test_params['traj_length']
    model_time_step = model_save_interval*model_params['ncpu']*model_params['traj_length']

    len_test_hist = test_params['num_timesteps'] // test_time_step
    len_model_hist = model_params['num_timesteps'] // model_time_step
    test_checkdir = osp.join(test_dir, 'checkpoints')
    model_checkdir = osp.join(model_dir,'checkpoints')

    scores = []
    wins = []
    
    # Find checkpoint corresponding to specified checkpoint_time
    if checkpoint_time:
        checkpoint = myround(checkpoint_time//(
                model_params['ncpu']*model_params['traj_length']),
                             model_save_interval)
        model_path = osp.join(model_checkdir, '%.5i'%checkpoint)
    else:
        # otherwise find latest checkpoint
        for i in range(1,len_model_hist+1):
            temp_path = osp.join(model_checkdir, '%.5i'%(i*model_save_interval))
            if osp.isfile(temp_path):
                model_path = temp_path
            else:
                break
    # Load weights from checkpoint
    print(model_path)
    model.load_with_scope(load_path=model_path)

    # To test on just the latest version of opponent..
    if just_latest:
        for i in range(1,len_test_hist+1):
            temp_path = osp.join(test_checkdir, '%.5i'%(i*test_save_interval))
            if osp.isfile(temp_path):
                hist_path = temp_path
            else:
                break
        print(hist_path)
        test_model.load_with_scope(load_path=hist_path)
        return test_play(env=env, models=[model,test_model], max_eps = max_eps)[0]
    # .. or on all historic versions 
    for i in range(1,len_test_hist+1):
        hist_path = osp.join(test_checkdir, '%.5i'%(i*test_save_interval))
        if not osp.isfile(hist_path):
            break
        test_model.load_with_scope(load_path=hist_path)
        print(hist_path)
        if checkpoint_match: # Both models trained up until same checkpoint
            model_path = osp.join(model_checkdir, '%.5i'%(i*model_save_interval))
            model.load_with_scope(load_path=model_path)
        mean_scores, num_wins = test_play(
            env=env, models=[model,test_model], max_eps = max_eps)
        scores.append(mean_scores[0])
        wins.append(num_wins)
        print(i, scores[-1])

    timesteps = range(test_time_step,test_params['num_timesteps'],test_time_step)
    
    return timesteps, scores, wins
        
def load_model_from_dir(*,env, model_dir, name, multiagent=True):

    params = pickle.load(open(model_dir+"/params.pkl", "rb"))

    nenvs = env.num_envs
    nsteps = params['traj_length']
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch_act = nenvs
    nbatch_train = nenvs*nsteps

    if 'cross_kls' not in params:
        params['cross_kls'] = ['new','new']

    if 'reverse_kl' not in params:
        params['reverse_kl'] = False
        
    policy = MlpPolicy
    
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space,
                  nbatch_act=nenvs, nbatch_train=nbatch_train,
                  nsteps=nsteps, ent_coef=params['ent_coef'],
                  vf_coef=params['vf_coef'],max_grad_norm=0.5,
                  cascade_depth=params['cascade_depth'],
                  flow_factor=params['flow_factor'],
                  mesh_factor=params['mesh_factor'],
                  var_init=params['var_init'],
                  imp_sampling=params['imp_sampling'],
                  imp_clips=params['imp_clips'],
                  dynamic_neglogpacs=params['dynamic_neglogpacs'],
                  full_kl=params['full_kl'], cross_kls=params['cross_kls'],
                  separate_value=params['separate_value'],
                  value_cascade=params['value_cascade'],
                  prox_value_fac=params['prox_value_fac'],targs=None,
                  reverse_kl=params['reverse_kl'],
                  multiagent=multiagent, model_scope=name)

    return model

# Test agent on single player game
def single_env_test_play(*, env, model, max_eps, depth):

    nenv = env.num_envs
    obs = np.zeros((nenv,) + env.observation_space.shape,
                   dtype=model.train_model.X.dtype.name)
    obs[:] = env.reset()
    states = model.initial_states
    dones = [False for _ in range(nenv)]
    ep_rewards = []
    ep_lens = []
    num_eps = 0
    
    while num_eps < max_eps:
        actions,_,_,_,states = model.run(obs, states, dones)
        obs[:], rewards, dones, infos = env.step(actions[depth])
        
        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                num_eps += 1
                ep_rewards.append(maybeepinfo['r'])
                ep_lens.append(maybeepinfo['l'])
                
    return safemean(ep_rewards), safemean(ep_lens),num_eps

# Test all cascade policies on a single player game for a particular model
def single_env_cascade_test_play(*, env, model, max_eps, depth):

    cascade_rewards = []
    cascade_num_eps = []
    cascade_ep_lens = []
    
    for i in range(depth):
        rewards, ep_lens, num_eps = single_env_test_play(
            env=env, model=model, max_eps = max_eps, depth=i)
        cascade_rewards.append(rewards)
        cascade_num_eps.append(num_eps)
        cascade_ep_lens.append(ep_lens)
        
    print(cascade_rewards)
    return cascade_rewards, cascade_ep_lens,cascade_num_eps

# Test all cascade policies on a single player game on historical checkpoints of model
def single_env_cascade_historic_test_play(*, env, model,model_dir, epoch, max_eps):
    
    model_params = pickle.load(open(model_dir+"/params.pkl", "rb"))

    if 'save_interval' in model_params:
        model_save_interval = model_params['save_interval']
    else:
        model_save_interval = 25

    model_time_step = model_save_interval*model_params['ncpu']*model_params['traj_length']
    len_model_hist = model_params['num_timesteps'] // model_time_step
    model_checkdir = osp.join(model_dir,'checkpoints')
    
    hist_rewards = []
    hist_num_eps = []
    hist_ep_lens = []
    
    for i in range(1,len_model_hist+1):
        temp_path = osp.join(model_checkdir,
                             'epoch'+str(epoch)+'_'+'%.5i'%(i*model_save_interval))
        if osp.isfile(temp_path):
            hist_path = temp_path
        else:
            break
        print(hist_path)
        model.load(load_path=hist_path)
        # load normalisation factor for state variables
        vecnorm = pickle.load(open(hist_path+"_vecnorm.pkl","rb"))
        env.ob_rms = vecnorm[0]
        env.ret_rms = vecnorm[1]
        env.ret = vecnorm[2]
        
        rewards, ep_lens, num_eps = single_env_cascade_test_play(
            env=env, model=model, max_eps=max_eps, depth=model_params['cascade_depth'])
        hist_rewards.append(rewards)
        hist_num_eps.append(num_eps)
        hist_ep_lens.append(ep_lens)
        
    return hist_rewards, hist_ep_lens, hist_num_eps
    
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def myround(x, base=5):
    return int(base * round(float(x)/base))
