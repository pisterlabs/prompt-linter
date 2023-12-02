import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
#from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

############################################
from src.exploration_policy.policies_MOD_DISC import ActorCriticPolicy
from src.exploration_policy.buffers_MOD_DISC import DictRolloutBuffer
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import obs_as_tensor
import gym
import cv2

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
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
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        aux_pred_dim=2,
        proprio_dim=11,
        cut_out_aux_head=2,
        deact_aux=False
    ):

        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
            batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

        #NEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEW

        buffer_cls = DictRolloutBuffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            use_aux=True,
            aux_dim = aux_pred_dim
        )
        #self.policy_class is one of the policies in plocies.py for PPO -> ActorCriticPolicy
        self.policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
            use_aux=True,
            aux_pred_dim=aux_pred_dim,
            proprio_dim=proprio_dim,
            cut_out_aux_head=cut_out_aux_head,
            deact_aux=deact_aux
        )

        self.policy = self.policy.to(self.device)
        self.criteria1 = th.nn.CrossEntropyLoss()
        #loss components
        #self.loss1 = (th.ones((self.batch_size,1))*np.pi).to(self.device)
        #self.loss2 = (th.ones((self.batch_size,1))*(2*np.pi)).to(self.device)
        #self.loss1 = (th.ones((self.batch_size,1))*1).to(self.device)
        #self.loss2 = (th.ones((self.batch_size,1))*2).to(self.device)

    def _setup_model(self) -> None:
        super(PPO, self)._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def my_loss(self,output, target):#,sample_removal_mask):
        #print("here it goes")
        #print(output.size(),target.size())
        #diff = (output - target)+self.loss1
        #print(diff.size())
        #diff = diff % self.loss2
        #diff = diff - self.loss1
        #loss = th.sum(diff)
        #output lies in range [-1,1] due to Tanh()
        #scale back to [-pi,pi]
        #THE PREV ONE !!!!
        #output = th.multiply(output,np.pi)
        #print(f"out 1 {output.shape}")
        output = th.atan2(output[:,0],output[:,1]).unsqueeze(1)
        #output = (1-th.cos(output-target))
        #output[sample_removal_mask] = 0.0
        #print(f"out2 {output.shape}")
        #print(f"out3 {target.shape}")
        #print(output[0:3])
        return (1-th.cos(output-target)).sum()
        #output[sample_removal_mask] = 0.0
        #print(f"out2 {output.shape}")
        #print(f"out3 {target.shape}")
        #print(output[0:3])

        #return output.sum()
        #print(loss)
        #return loss
    def my_loss2(self,output, target,sample_removal_mask):
        loss = th.abs(output - target)
        loss[sample_removal_mask] = 0.0
        #print("LOSS:",loss.sum())
        return loss.mean()


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        #added by Fabi
        
        aux_losses_1 = []
        aux_losses_2 = []
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy, aux_angle = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)


                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
               
                   
                
                #sample_removal_mask = rollout_data.aux_dist_gt == 20.0
                #print("REMOVAL:",sample_removal_mask,sample_removal_mask.size(),sample_removal_mask.sum())
                #aux_loss_1 = self.my_loss(aux_angle, rollout_data.aux_angle_gt)#recently modified 05.05.22
                #F.mse_loss(aux, rollout_data.true_aux)#self.my_loss(aux, rollout_data.true_aux)##F.l1_loss(aux, rollout_data.true_aux)##th.nn.SmoothL1Loss(aux, rollout_data.true_aux)##
                #print(aux_angle.size()," others: ",rollout_data.aux_angle_gt.long().size())
                aux_loss_1 = self.criteria1(aux_angle, rollout_data.aux_angle_gt.long().squeeze(1))
                #aux_loss_2 = F.l1_loss(aux_dist, rollout_data.aux_dist_gt)#self.my_loss2(aux_dist, rollout_data.aux_dist_gt,sample_removal_mask)#
                
                #aux_losses_2.append(aux_loss_2.item())

                aux_losses_1.append(aux_loss_1.item())

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
    
                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                #print("loss1 :{} and loss2 {}".format(aux_loss_1,aux_loss_2))
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + 0.1*aux_loss_1# + aux_loss_2
                
                #print("different losses: pol: {}, ent: {}, val: {}, aux: {}".format(policy_loss,entropy_loss,value_loss,aux_loss))
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        #Fabi
        
        self.logger.record("train/aux_loss_angle", np.mean(aux_losses_1))
        #self.logger.record("train/aux_loss_dist", np.mean(aux_losses_2))

        self.logger.record("train/aux_ep_SR", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.aux_episode_succ])))
        
        #self.logger.record("train/aux_ep_Coll", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.aux_episode_coll_rate])))

        self.logger.record("train/success_rate", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.succ_rate])))
        #self.logger.record("train/collision_rate", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.collision_rate])))
        #for i in self.env.scene_succ.keys():
        #    self.logger.record((f"train/{i}"), np.mean(np.array(self.env.scene_succ[i])))
        """
        #MAPS
        self.logger.record("train/Ben1SR", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.ben1_sr])))
        self.logger.record("train/Ben2SR", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.ben2_sr])))
        self.logger.record("train/BeechSR", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.rs_sr])))
        self.logger.record("train/RsSR", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.beech])))
        
        #numb_objects
        self.logger.record("train/3-objects", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.numb_obj_3])))
        self.logger.record("train/4-objects", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.numb_obj_4])))
        self.logger.record("train/5-objects", np.mean(np.array([np.mean(np.array(sub_ele)) for sub_ele in self.env.numb_obj_5])))
        """
        #self.logger.record("train/success_rate", np.mean(np.array(self.env.envii.episode_stats['succ_rate'])))
        #self.logger.record("train/collision_rate", np.mean(np.array(self.env.envii.episode_stats['collision'])))


        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super(PPO, self).learn(
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


    
    #Overwritten Method since we use it here for auxilliary Task
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                #added by Fabi
                
                actions, values, log_probs, aux_angle  = self.policy.forward(obs_tensor)
                
            actions = actions.cpu().numpy()

            #added by Fabi
            
            aux_angle = aux_angle.cpu().numpy()
            
            

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            #Added by Fabi .step(clipped_actions) and no dict
            
            aux_actions = [{"action":clipped_actions[i],"aux_angle":aux_angle[i]} for i in range(self.env.numb_envs)]
            
            #for i in range(self.env.numb_envs):
            #    aux_actions.append({"action":clipped_actions[i],"aux_angle":aux_angle[i],"aux_dist":aux_dist[i]})
            
            new_obs, rewards, dones, infos = env.step(aux_actions)
            
            
            #Reason why stacked obs not working ? infos are getting stacked as well in vec_frame_stack and I might need to use [-1] instead of [0]
            aux_angle_gt = []#
            #aux_dist_gt = []
            #print(f"ppo: True aux:{infos},new_obs : \n{new_obs['task_obs']} shape: {new_obs['task_obs'].shape}, -- {new_obs['image'].shape}")

            for i in range(self.env.numb_envs):
                aux_angle_gt.append([infos[i]['true_aux']])#[:2])
                #aux_dist_gt.append([infos[i]['true_aux'][0]])
                #print(f"ppo True aux: {infos[i]['true_aux'][1]}, observation: {new_obs['task_obs'][i]}")
            aux_angle_gt = np.array(aux_angle_gt)
            #aux_dist_gt = np.array(aux_dist_gt)
            #print("ANGLE :",aux_angle.shape ," and GT :",aux_angle_gt.shape, " dist: ",aux_dist.shape," GT: ",aux_dist_gt.shape)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            #added aux here Fabi
            """
            print("returned HL cumulative_rewards:",rewards)
            for key, value in new_obs.items():

                if key == "image":
                    for i in range(2):
                        cv2.imshow("coarse new_obs{}".format(i),value[i].transpose(1,2,0).astype(np.uint8))
                        cv2.waitKey(1)

                elif key == "image_global":
                    for i in range(2):
                        cv2.imshow("global new_obs{}".format(i),value[i].transpose(1,2,0).astype(np.uint8))
                        cv2.waitKey(1) 

            
            input()  
            """   
            rollout_buffer.add2(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,aux_angle,aux_angle_gt)
            

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            #added _ here Fabi
            
            _, values, _,_ = self.policy.forward(obs_tensor)
            
            


        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True
