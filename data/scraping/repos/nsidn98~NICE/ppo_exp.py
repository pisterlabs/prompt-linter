"""
    Proximal Policy Optimisation(PPO)
    Taken from OpenAI SpinningUp repo
    Have made modiffications to it so that 
    we do not have to install spinningup
    as well as make it well suited to our needs.
"""
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from tqdm import tqdm

from RL.models.model import MLPActorCritic
from RL.utils.mpi_utils import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from RL.utils.mpi_utils import mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from RL.utils.utils import print_box, combined_shape, count_vars, discount_cumsum


class PPOBuffer:
    """
        A buffer for storing trajectories experienced by a PPO agent interacting
        with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
        for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim:int, act_dim:int, size:int, 
                gamma:float=0.99, lam:float=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        # NOTE act_dim = () hence hardcoding 87 here
        self.act_mask_buf = np.zeros(combined_shape(size, 87), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, act_mask=None):
        """
            Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.act_mask_buf[self.ptr] = act_mask
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
            Call this at the end of a trajectory, or when one gets cut off
            by an epoch ending. This looks back in the buffer to where the
            trajectory started, and uses rewards and value estimates from
            the whole trajectory to compute advantage estimates with GAE-Lambda,
            as well as compute the rewards-to-go for each state, to use as
            the targets for the value function.

            The "last_val" argument should be 0 if the trajectory ended
            because the agent reached a terminal state (died), and otherwise
            should be V(s_T), the value function estimated for the last state.
            This allows us to bootstrap the reward-to-go calculation to account
            for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
            Call this at the end of an epoch to get all of the data from
            the buffer, with advantages appropriately normalized (shifted to have
            mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, act_mask=self.act_mask_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def save_checkpoint(path, model, args):
    """
        To save the checkpoint/model weights
        Parameters:
        –––––––––––
        path: str
            Path to save the weights
        model: nn.Module
            the model we want to save
        args: argparse.Namespace
            Arguments used for the experiment
    """
    checkpoint = {}
    checkpoint['args'] = args       # save as argparse.Namespace
    checkpoint['args_dict'] = vars(args) # save as a Dict
    checkpoint['actor_state_dict'] = model.state_dict()
    torch.save(checkpoint, path)

def ppo(env, test_env, actor_critic=MLPActorCritic, ac_kwargs=dict(), seed:int=0, 
        steps_per_epoch:int=4000, epochs:int=50, gamma:float=0.99, 
        clip_ratio:float=0.2, pi_lr:float=3e-4, vf_lr:float=1e-3, 
        train_pi_iters:int=80, train_v_iters:int=80, lam:float=0.97, 
        max_ep_len:float=1000, target_kl:float=0.01, save_freq:int=10, 
        num_test_rollouts:int=10, eval_freq:int=10000, logger=None, 
        log_losses:bool=False, log_overqual:bool=False, log_req:bool=False, 
        mask_actions:bool=False):
    """
        Proximal Policy Optimization (by clipping), 

        with early stopping based on approximate KL

        Parameters:
        -----------
        env : An instantiation of the environment.
            The environment must satisfy the OpenAI Gym API.
        
        test_env: An instantiation of the environment to test policy after certain number of steps.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        num_test_rollouts (int): For how many episodes do we evalaute the policy

        eval_freq (int): After how many steps do we evaluate the policy

        logger (WandbLogger):
            A wandb logger wrapper
        
        log_losses (bool):
            Whether we want to log PPO losses or not
        
        log_overqual (bool):
            Whether we want to log overqualification differences or not

        log_req (bool):
            Whether we want to log training requirement completions or not

        mask_actions (bool):
            Whether we want to mask invalid actions and choose only from valid pilots
    """

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
    if proc_id() == 0:
        print_box('Actor Critic Network Architecture', 80)
        print_box(ac, 80)
        print_box('Number of parameters in the network: \t pi: %d, \t v: %d'%var_counts, 80)
        print_box(f'Number of processes: {num_procs()}')

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old, act_mask = data['obs'], data['act'], data['adv'], data['logp'], data['act_mask']

        # Policy loss
        if mask_actions:
            pi, logp = ac.pi(obs, act, act_mask)
        else:
            pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    def update(epoch:int):
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        if logger and (proc_id() == 0) and log_losses:
            # print(f"Logging:{epoch}")
            logger.writer.add_scalar('LossPi', pi_l_old, epoch)
            logger.writer.add_scalar('LossV', v_l_old, epoch)
            logger.writer.add_scalar('KL', kl, epoch)
            logger.writer.add_scalar('Entropy', ent, epoch)
            logger.writer.add_scalar('ClipFraction', cf, epoch)
            logger.writer.add_scalar('DeltaLossPi', loss_pi.item() - pi_l_old, epoch)
            logger.writer.add_scalar('DeltaLossV', loss_v.item() - v_l_old, epoch)

    def eval_agent(num_test_rollouts:int=num_test_rollouts):
        """
            Evaluate the agent for 'num_test_rollouts' episodes
            using a deterministic policy and return the following:
            • Total episodic rewards averaged over 'num_test_rollouts' episodes
            • Fraction of episode completed averaged over 'num_test_rollouts' episodes
            • Total overqualification of pilots averaged over 'num_test_rollouts' episodes
            • Mean of Standard Deviation of overqualification
            • Overqualification per pilot-event assignment averaged over 
                'num_test_rollouts' episodes
            • Total requirements completed averaged over 'num_test_rollouts' episodes
        """
        rollout_rewards = []
        ep_frac_lengths = []
        ep_lengths = []
        mu_overqual = []
        total_overqual = []
        std_overqual = []
        for test_rollouts in range(num_test_rollouts):
            o = test_env.reset()
            valid_actions = test_env.getValidPilotsVec()
            rollout_reward = 0; ep_len = 0
            while True:
                if mask_actions:
                    a, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), 
                                        deterministic=True, 
                                        act_mask=torch.as_tensor(valid_actions, 
                                        dtype=torch.float32))    
                else:
                    a, _, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), 
                                        deterministic=True)
                next_o, r, d, _ = test_env.step(a)
                valid_actions = test_env.getValidPilotsVec()
                rollout_reward += r; ep_len += 1
                o = next_o
                timeout = ep_len == max_ep_len
                terminal = d or timeout
                if terminal:
                    rollout_rewards.append(rollout_reward)
                    ep_frac_lengths.append(ep_len / test_env.episode_horizon)
                    ep_lengths.append(ep_len)
                    # if the agent fails to even place one pilot, append zeros
                    if len(test_env.episode_overquals):
                        mu_overqual.append(np.mean(test_env.episode_overquals))
                        std_overqual.append(np.std(test_env.episode_overquals))
                    else:
                        mu_overqual.append(0)
                        std_overqual.append(0)
                    total_overqual.append(test_env.episode_qual_diff)
                    break
        avg_overqual_per_event = np.array(total_overqual) / np.array(ep_lengths)
        return (np.mean(rollout_rewards), np.mean(ep_frac_lengths), 
                np.mean(total_overqual), np.mean(mu_overqual), 
                np.mean(std_overqual))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    valid_actions = env.getValidPilotsVec()
    ep_num, step_num = 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for epoch in tqdm(range(epochs)):
        for t in range(local_steps_per_epoch):
            if mask_actions:
                a, v, logp, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), 
                                        act_mask=torch.as_tensor(valid_actions, 
                                        dtype=torch.float32))
            else:
                a, v, logp, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            if mask_actions:
                buf.store(o, a, r, v, logp, act_mask=valid_actions)
            else:
                buf.store(o, a, r, v, logp)
            # logger.store(VVals=v)
            if proc_id() == 0:
                for i in range(args.cpu):
                    # making a for loop here across all process just for increasing 
                    # the 'step_num' by considering all the parallel processes
                    if (step_num % eval_freq == 0) and logger:
                        # print('Evaluating agent')
                        eval_reward, eval_frac, eval_overqual, \
                            eval_avg_overqual, eval_std_overqual = eval_agent()
                        logger.writer.add_scalar('EvalReward', eval_reward, step_num)
                        logger.writer.add_scalar('EvalFrac', eval_frac, step_num)
                        if log_overqual:
                            logger.writer.add_scalar('EvalOverqual',
                                                    eval_overqual, step_num)
                            logger.writer.add_scalar('EvalAvgOverqual',
                                                    eval_avg_overqual, step_num)
                            logger.writer.add_scalar('EvalStdOverQual', 
                                                    eval_std_overqual, step_num)
                    step_num += 1
            
            # Update obs (critical!)
            o = next_o
            valid_actions = env.getValidPilotsVec()

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    if mask_actions:
                        _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32), 
                                            act_mask=torch.as_tensor(valid_actions, 
                                                                dtype=torch.float32))
                    else:
                        _, v, _, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    if logger and (proc_id() == 0):
                        logger.writer.add_scalar('EpisodeReward', ep_ret, ep_num)
                        logger.writer.add_scalar('FractionEpisodeCompeted', 
                                                ep_len/env.episode_horizon, ep_num)
                    ep_num += 1
                o, ep_ret, ep_len = env.reset(), 0, 0
                valid_actions = env.getValidPilotsVec()

        # Save model
        if ((epoch % save_freq == 0) or (epoch == epochs - 1)) and logger and (proc_id() == 0):
            save_checkpoint(logger.weight_save_path, ac, args)

        # Perform PPO update!
        update(epoch)
        epoch_eval_rew, epoch_eval_frac, epoch_eval_overqual, \
            epoch_eval_avg_overqual, epoch_eval_std_overqual = eval_agent(num_test_rollouts=1)
        if proc_id() == 0:
            print()
            print_box(f'Epoch: {epoch+1}, EvalFrac: {epoch_eval_frac:.3f}')

if __name__ == '__main__':

    from RL.utils.utils import print_args, check_steps_per_MPI_task
    from RL.configs.config import args
    from RL.utils.logger import WandbLogger
    from pilotRLEnv.env import PilotRLEnv

    args = check_steps_per_MPI_task(args, num_procs())
    # setup the Wandb Logger
    logger = None
    if not args.dryrun:
        logger = WandbLogger(experiment_name='ppo', save_folder='ppo', 
                            project='NICE', 
                            entity='puckboard', args=args)
    if proc_id() == 0:
        print_args(args, 80)

    seed = args.seed + 100 * proc_id()
    env = PilotRLEnv(args, seed=seed)
    test_env = PilotRLEnv(args, seed=seed+1)

    ac_kwargs=dict(hidden_sizes=[args.hid]*args.l)

    ppo(env=env, test_env=test_env, actor_critic=MLPActorCritic, 
        ac_kwargs=ac_kwargs, seed=args.seed, steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs, gamma=args.gamma, clip_ratio=args.clip_ratio, 
        pi_lr=args.pi_lr, vf_lr=args.vf_lr, train_pi_iters=args.train_pi_iters, 
        train_v_iters=args.train_v_iters, lam=args.lam, max_ep_len=args.max_ep_len,
        target_kl=args.target_kl, save_freq=args.save_freq, 
        num_test_rollouts=args.num_test_rollouts, eval_freq=args.eval_freq,
        logger=logger, log_losses=args.log_losses, mask_actions=args.mask_actions)