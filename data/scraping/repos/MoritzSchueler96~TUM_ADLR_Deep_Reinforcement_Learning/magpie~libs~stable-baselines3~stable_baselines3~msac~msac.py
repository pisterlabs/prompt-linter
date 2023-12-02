from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.meta_off_policy_algorithm import MetaOffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.msac.policies import SACPolicy


class mSAC(MetaOffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 0,
        
        n_traintasks: int = 0,
        n_evaltasks: int = 0,
        n_epochtasks: int = 0,
        
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(mSAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            n_traintasks,
            n_evaltasks,
            n_epochtasks,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        self.indices = None
        self.context = None
        
        self.ent_coef_losses, self.ent_coefs = [], []
        self.actor_losses, self.critic_losses = [], []
        self.kl_losses = []
        self.l_z_means, self.l_z_vars = [], []
        


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(mSAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
    def sample_context(self, indices, buff = None):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
#        print('indices in contextsampling', indices)

        
        final = th.zeros(len(indices),100,self.obs_dim+self.act_dim+1)
        
      
   #     print(indices)
    #    print(self.encodermap['end'])
     #   print(self.replaymap['end'])
        if len(indices) >1:
            
            for i,idx in enumerate(indices):
 
                sample = self.RBList_encoder[idx].sample(batch_size=100) 

                final[i]=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)
        else:
            if buff is not None:
                sample = buff[indices[0]].sample(batch_size=100) 

                final=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)           
            
            else:
                sample = self.RBList_encoder[indices[0]].sample(batch_size=100) 

                final=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)
            final = final.view(1, 100, self.obs_dim+self.act_dim+1)
        return final

    ##### Training #####
    def _do_training(self, indices):
        num_updates = 1

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.actor.clear_z(num_tasks=len(indices))
        context = context_batch
        
        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            #[:, i * mb_size: i * mb_size + mb_size, :]
            
            self._take_step(indices, context)

            # stop backprop
            self.actor.detach_z() 

    def _take_step(self, indices, context) -> None:

        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.actor.context_optimizer, self.critic.optimizer] 
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)
        
        self.actor.infer_posterior(context)   
        self.actor.sample_z()
        task_z = self.actor.z 
        num_tasks = len(indices)
        
        batch_size =256   
            
        
        obs = th.zeros(16,batch_size,self.obs_dim)
        next_observations = th.zeros(16,batch_size,self.obs_dim)
        actions = th.zeros(16,batch_size,self.act_dim)
        rewards = th.zeros(16,batch_size,1)
        dones = th.zeros(16,batch_size,1)

        for i,idx in enumerate(indices):
        
            sample = self.RBList_replay[idx].sample(batch_size=batch_size)

            obs[i]=sample.observations
            next_observations[i]=sample.next_observations
            actions[i]=sample.actions
            rewards[i]=sample.rewards
            dones[i]=sample.dones

        if self.use_sde:
            self.actor.reset_noise()        
    
       

        
        t, b, _ = obs.size()
        
      #  next_observations = next_observations.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        dones = dones.view(t * b, -1)
        actions = actions.view(t * b, -1)


        new_actions, mean_actions, log_std, log_prob, expected_log_prob, std,mean_action_log_prob, pre_tanh_value, task_z  = self.actor(obs, reparameterize=True ,return_log_prob=True)

        local_means = self.actor.z_means.detach().numpy()
        local_vars = self.actor.z_vars.detach().numpy()
        self.l_z_means.append(local_means)
        self.l_z_vars.append(local_vars) 
        
         # KL constraint on z if probabilistic
        self.actor.context_optimizer.zero_grad()
        kl_div = self.actor.compute_kl_div()
        kl_loss = 0.1 * kl_div
        kl_loss.backward(retain_graph=True)
        self.kl_losses.append(kl_loss.detach().numpy())   

        

        # run policy, get log probs and new actions
        
        log_prob = log_prob.reshape(-1, 1)
#            print(actions_pi, actions_pi.shape)

        ent_coef_loss = None
        if self.ent_coef_optimizer is not None:
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            ent_coef = th.exp(self.log_ent_coef.detach())
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_losses.append(ent_coef_loss.item())
        else:
            ent_coef = self.ent_coef_tensor

        self.ent_coefs.append(ent_coef.item())

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if ent_coef_loss is not None:
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            
        with th.no_grad():
            # Select action according to policy
            
            next_actions, _, _, next_log_prob, _, _,_, _, _  = self.actor(next_observations, reparameterize=True ,return_log_prob=True)
            #task_z = [z.repeat(b, 1) for z in task_z]
            #task_z = th.cat(task_z, dim=0)
            # Compute the target Q value: min over all critics targets
            next_actions_and_z = th.cat([next_actions, task_z.detach()], dim=1)
            next_observations = next_observations.view(t * b, -1)
            targets = th.cat(self.critic_target(next_observations, next_actions_and_z), dim=1)
            target_q, _ = th.min(targets, dim=1, keepdim=True)
            
            target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
            
            q_backup = (rewards * 5) + (1 - dones) * self.gamma * target_q

        # Q and V networks
        # encoder will only get gradients from Q nets
        # Get current Q estimates for each critic network
        # using action from the replay buffer
        
        actions_and_z = th.cat([actions, task_z], dim=1)
        obs = obs.view(t * b, -1)
        current_q_estimates = self.critic(obs, actions_and_z)
        

        # Compute critic loss
        critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
        #critic_loss = th.mean([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
        #critic_loss= (th.mean((current_q_estimates[0] - q_backup) ** 2) + th.mean((current_q_estimates[1] - q_backup) ** 2)) / 2
        self.critic_losses.append(critic_loss.item())
        self.critic.optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        # Optimize the critic
        self.critic.optimizer.step()
        self.actor.context_optimizer.step()
        
        


#        meany = np.mean([np.mean(i.bias.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
 #       maxy = np.max([np.max(i.bias.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
  #      miny = np.min([np.min(i.bias.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
   #     mediany = np.median([np.median(i.bias.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
    #    print('mean: ', meany, 'max:', maxy, 'min:', miny, 'median:', mediany)

#        meany = np.mean([np.mean(i.weight.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
 #       maxy = np.max([np.max(i.weight.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
  #      miny = np.min([np.min(i.weight.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
   #     mediany = np.median([np.median(i.weight.data.detach().numpy()) for i in self.actor.context_encoder[::2]])
    #    print('mean: ', meany, 'max:', maxy, 'min:', miny, 'median:', mediany)

        
        
        
        #Compute actor loss
        # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        own_actions_and_z = th.cat([new_actions, task_z.detach()], dim=1)
        q_values_pi = th.cat(self.critic.forward(obs, own_actions_and_z), dim=1)
        min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

        #actor_loss = th.mean(log_prob - q_values_pi[1])
        actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
        #actor_loss = ((ent_coef * log_prob - min_qf_pi)**2).mean()
        #actor_loss = F.mse_loss(ent_coef * log_prob, min_qf_pi)
        self.actor_losses.append(actor_loss.item())

        # Optimize the actor
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()
    
        # Update target networks
        polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += 1

        logger.record(key = "train/n_updates", value=self._n_updates, exclude="tensorboard")
        #logger.record(key = "train/ent_coef", value=self.ent_coefs)
        logger.record(key = "train/actor_loss", value=actor_loss.item())
        logger.record(key = "train/critic_loss", value = critic_loss.item())
        logger.record(key = "train/KL_loss", value= kl_loss.detach().numpy().item())
        logger.record(key = "train/avg. z", value = np.mean(local_means))
        logger.record(key = "train/avg. z var", value = np.mean(local_vars))
        if len(self.ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", self.ent_coef_losses)
        
        #self._dump_logs()
        logger.dump(step=self._n_updates)


        print('KL_DIV:', kl_div)
        print('KL_LOSS:', kl_loss)
        print('Critic_LOSS:',critic_loss)
        print('Actor_LOSS:',actor_loss)


    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "mSAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> MetaOffPolicyAlgorithm:

        return super(mSAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(mSAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
