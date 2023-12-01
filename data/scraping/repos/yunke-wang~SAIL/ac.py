from my_utils import *
from core_nn.nn_ac import *

""" Basic Actor-Critic with GAE. This is not A2C, since the networks are updated in an epoch style like PPO """
class AC():
    def __init__(self, state_dim, action_dim, args, a_bound=1, encode_dim=0): 
        self.state_dim = state_dim + encode_dim
        self.action_dim = action_dim
        self.a_bound = a_bound
        self.gamma = args.gamma

        self.tau_soft = args.tau_soft
        self.gae_tau = args.gae_tau
        self.gae_l2_reg = args.gae_l2_reg 
        self.mini_batch_size = args.mini_batch_size
        self.train_epoch = 10
        
        self.entropy_coef = torch.Tensor(1).fill_(args.entropy_coef).to(device) # alpha version??
        
        self.initilize_nets(args)

    def initilize_nets(self, args):                
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=0).to(device)
        self.value_net = Value(self.state_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate_pv)
        
    def update_policy(self, states, actions, next_states, rewards, masks):
        with torch.no_grad():
            values = self.value_net.get_value(states).data

        """ get GAE from trajectories """
        advantages, returns = self.estimate_advantages(rewards, masks, values)   # advantage is GAE and returns is the TD(lambda) return.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6).detach().data 

        train = data_utils.TensorDataset(states, actions, returns, advantages)
        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size, shuffle=True)

        for _ in range(self.train_epoch):
            for batch_idx, (states_batch, actions_batch, returns_batch, advantages_batch) in enumerate(train_loader):
                log_probs = self.policy_net.get_log_prob(states_batch, actions_batch)
                cur_entropy = self.policy_net.compute_entropy().mean()

                policy_loss = -(advantages_batch * log_probs).mean() - self.entropy_coef * cur_entropy

                value_loss = (self.value_net.get_value(states_batch) - returns_batch).pow(2).mean()
                if self.gae_l2_reg > 0:
                    for params in self.value_net.parameters():
                        value_loss += self.gae_l2_reg * params.norm(2)

                self.optimizer_policy.zero_grad() 
                policy_loss.backward() 
                self.optimizer_policy.step()

                self.optimizer_value.zero_grad() 
                value_loss.backward() 
                self.optimizer_value.step()

    def sample_action(self, x):
        return self.policy_net.sample_action(x)

    def greedy_action(self, x):
        return self.policy_net.greedy_action(x)

    def policy_to_device(self, device):
        self.policy_net = self.policy_net.to(device) 

    def save_model(self, path):
        torch.save( self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau_soft) + param.data * self.tau_soft)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def estimate_advantages(self, rewards, masks, values):
        """ Estimate genralized advantage estimation (GAE) from the TD error. Also compute return along trajectories."""
        rewards = rewards.to(device_cpu)
        masks = masks.to(device_cpu)
        values = values.to(device_cpu)

        tensor_type = type(rewards)
        deltas = tensor_type(rewards.size(0), 1)
        advantages = tensor_type(rewards.size(0), 1)

        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values[i]
            advantages[i] = deltas[i] + self.gamma * self.gae_tau * prev_advantage * masks[i]
            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        """ 
            Use TD(lambda) returns as the target V-function, as done in baslines. 
            This is more stable (low variance) compared to Monte-Carlo return (below).
        """
        returns = advantages + values   #TD(lambda) returns
    
        advantages = advantages.to(device)
        returns = returns.to(device)

        return advantages, returns

""" TRPO. Adapted from OpenAI's baselines and https://github.com/Khrylx/PyTorch-RL """
class TRPO(AC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, encode_dim=0):  
        super().__init__(state_dim, action_dim, args, a_bound, encode_dim)
        self.trpo_max_kl = args.trpo_max_kl
        self.trpo_damping = args.trpo_damping 
        self.update_type = "on_policy"

    def initilize_nets(self, args):                
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=0).to(device)
        self.value_net = Value(self.state_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=args.learning_rate_pv)

    def update_policy(self, states, actions, next_states, rewards, masks):
        """ get advantage estimation from trajectories """
        values = self.value_net.get_value(states).data.detach() 
        advantages, returns = self.estimate_advantages(rewards, masks, values)   # advantage is GAE and returns is the TD(lambda) return.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        """update v-value function """
        # OpenAI's baselines use train_epoch=3 with batch size 128
        train = data_utils.TensorDataset(states, returns)
        train_loader = data_utils.DataLoader(train, batch_size=128, shuffle=True)

        for _ in range(0, 3):
            for batch_idx, (states_batch, returns_batch) in enumerate(train_loader):
                values_pred = self.value_net.get_value(states_batch)
                value_loss = (values_pred - returns_batch).pow(2).mean()

                if self.gae_l2_reg > 0:
                    for params in self.value_net.parameters():
                        value_loss += self.gae_l2_reg * params.norm(2)

                self.value_net.zero_grad()
                value_loss.backward()
                self.optimizer_value.step()  
                
        """ for policy update"""
        with torch.no_grad():
            fixed_log_probs = self.policy_net.get_log_prob(states, actions).detach().data
        self.policy_net.zero_grad()

        """define the loss function for TRPO. Flip sign since the code below is for minimization. """
        def get_loss(volatile=False):
            log_probs = self.policy_net.get_log_prob(states, actions)
            loss = (advantages * torch.exp(log_probs - fixed_log_probs)).mean() + self.entropy_coef * self.policy_net.compute_entropy()
            return -loss 

        """use fisher information matrix for Hessian*vector"""
        def Fvp_fim(v):
            M, mu, info = self.policy_net.get_fim(states)
            mu = mu.view(-1)
            filter_input_ids = set([info['std_id']])

            t = ones(mu.size()).requires_grad_()
            mu_t = (mu * t).sum()
            Jt = compute_flat_grad(mu_t, self.policy_net.parameters(), filter_input_ids=filter_input_ids, create_graph=True)
            Jtv = (Jt * v).sum()
            Jv = torch.autograd.grad(Jtv, t, retain_graph=True)[0]
            MJv = M * Jv.data
            mu_MJv = (MJv * mu).sum()
            JTMJv = compute_flat_grad(mu_MJv, self.policy_net.parameters(), filter_input_ids=filter_input_ids, retain_graph=True).data
            JTMJv /= states.shape[0]
            std_index = info['std_index']
            JTMJv[std_index: std_index + M.shape[0]] += 2 * v[std_index: std_index + M.shape[0]]
            return JTMJv + v * self.trpo_damping

        loss = get_loss()
        grads = torch.autograd.grad(loss, self.policy_net.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).data
        stepdir = self.conjugate_gradients(Fvp_fim, -loss_grad, 10)

        shs = 0.5 * (stepdir.dot(Fvp_fim(stepdir)))
        lm = math.sqrt(self.trpo_max_kl / shs)
        fullstep = stepdir * lm
        expected_improve = -loss_grad.dot(fullstep)

        prev_params = get_flat_params_from(self.policy_net)
        success, new_params = self.line_search(self.policy_net, get_loss, prev_params, fullstep, expected_improve)
        set_flat_params_to(self.policy_net, new_params)

        return success

    def conjugate_gradients(self, Avp_f, b, nsteps, rdotr_tol=1e-10):
        x = zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            Avp = Avp_f(p)
            alpha = rdotr / torch.dot(p, Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < rdotr_tol:
                break
        return x

    def line_search(self, model, f, x, fullstep, expected_improve_full, max_backtracks=10, accept_ratio=0.1):
        fval = f(True).data

        for stepfrac in [.5**x for x in range(max_backtracks)]:
            x_new = x + stepfrac * fullstep
            set_flat_params_to(model, x_new)
            fval_new = f(True).data
            actual_improve = fval - fval_new
            expected_improve = expected_improve_full * stepfrac
            ratio = actual_improve / expected_improve

            if ratio > accept_ratio:
                return True, x_new
        return False, x

""" PPO. Adapted from OpenAI's spinningup """
class PPO(AC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, is_discrete=False, encode_dim=0): 
        self.is_discrete = is_discrete
        self.separate_net = args.ppo_separate_net
        self.cnn = args.cnn 
        super().__init__(state_dim, action_dim, args, a_bound, encode_dim)
        self.ppo_clip = args.ppo_clip 
        self.ppo_early = args.ppo_early 
        self.ppo_gradient_clip = 0
        self.full_batch = True 
        self.update_type = "on_policy"

    def initilize_nets(self, args):                
        if self.separate_net:
            if self.is_discrete:
                self.policy_net = Policy_Categorical(self.state_dim, self.a_bound, hidden_size=args.hidden_size, activation=args.activation).to(device)
            else:
                self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=0).to(device)

            self.value_net = Value(self.state_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

            self.optimizer_policy = torch.optim.Adam( self.policy_net.parameters(), lr=args.learning_rate_pv)
            self.optimizer_value = torch.optim.Adam( self.value_net.parameters(), lr=args.learning_rate_pv)
        else:
            if self.is_discrete:
                from core_nn.nn_policy import Policy_Categorical_V
                self.policy_net = Policy_Categorical_V(self.state_dim, self.a_bound, hidden_size=args.hidden_size, activation=args.activation).to(device)
            else:
                from core_nn.nn_policy import Policy_Gaussian_V
                self.policy_net = Policy_Gaussian_V(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=0, log_std=args.log_std, a_bound=self.a_bound, squash_action=0).to(device)

            self.value_net = self.policy_net 
            self.optimizer_pv = torch.optim.Adam( self.policy_net.parameters(), lr=args.learning_rate_pv)

    def update_policy(self, states, actions, next_states, rewards, masks):
        with torch.no_grad():
            values = self.value_net.get_value(states).data
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """ get GAE from trajectories """
        advantages, returns = self.estimate_advantages(rewards, masks, values)   # advantage is GAE and returns is the TD(lambda) return.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        fixed_log_probs = fixed_log_probs.detach().data

        target_kl = 0.01
        
        train = data_utils.TensorDataset(states, actions, returns, advantages, fixed_log_probs)
        if self.full_batch:
            self.train_epoch = 80
            train_loader = data_utils.DataLoader(train, batch_size=states.size(0), shuffle=True)
        else:
            train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size, shuffle=True)


        for _ in range(self.train_epoch):
            for batch_idx, (states_batch, actions_batch, returns_batch, advantages_batch, fixed_log_probs_batch) in enumerate(train_loader):
                log_probs = self.policy_net.get_log_prob(states_batch, actions_batch)
                cur_entropy = self.policy_net.compute_entropy().mean()

                ratio = torch.exp(log_probs - fixed_log_probs_batch)                     
                min_adv = torch.where(advantages_batch>0, (1+self.ppo_clip)*advantages_batch, (1-self.ppo_clip)*advantages_batch)
                policy_loss = -torch.min(ratio * advantages_batch, min_adv).mean() - self.entropy_coef * cur_entropy

                value_loss = (self.value_net.get_value(states_batch) - returns_batch).pow(2).mean()
                if self.gae_l2_reg > 0:
                    for params in self.value_net.parameters():
                        value_loss += self.gae_l2_reg * params.norm(2)

                if self.separate_net:
                    self.optimizer_policy.zero_grad() 
                    policy_loss.backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_policy.step()

                    self.optimizer_value.zero_grad() 
                    value_loss.backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_value.step()

                else:
                    self.optimizer_pv.zero_grad() 
                    (policy_loss + value_loss).backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_pv.step()
                    
                ## early stopping based on KL 
                if self.ppo_early:
                    kl = (fixed_log_probs_batch - log_probs).mean()
                    if kl > 1.5 * target_kl:
                        # print('Early stopping at step %d due to reaching max kl.' % i)
                        if self.full_batch:
                            return 
                        else:
                            break

""" Soft actor-critic. Adapted from pranz24/pytorch-soft-actor-critic and https://github.com/sfujim/TD3"""
class SAC(AC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, encode_dim=0):  
        super().__init__(state_dim, action_dim, args, a_bound, encode_dim)    
        self.update_type = "off_policy" 
        self.symmetric = args.symmetric   

    def initilize_nets(self, args):       
        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=1, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)    
        self.policy_net_target = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=1, log_std=args.log_std, a_bound=self.a_bound, squash_action=1).to(device)

        self.hard_update(self.policy_net_target, self.policy_net)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv)

        self.value_net = Value_2(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_target = Value_2(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_value = torch.optim.Adam( self.value_net.parameters(), lr=args.learning_rate_pv)
        self.hard_update(self.value_net_target, self.value_net)

        self.target_entropy = -torch.Tensor(1).fill_(args.target_entropy_scale * self.action_dim).to(device)
        self.log_entropy_coef = torch.zeros(1, requires_grad=True, device=device)
        self.entropy_coef = torch.exp(self.log_entropy_coef).to(device)

        self.optimizer_entropy = torch.optim.Adam([self.log_entropy_coef], lr=args.learning_rate_pv)   

    def update_policy(self, states, actions, next_states, rewards, masks):    

        current_q1, current_q2 = self.value_net(states, actions)  

        sample_actions_next, log_probs_next, _, _ = self.policy_net_target.sample_full(next_states, symmetric=self.symmetric)
        """ Anti-thetic sampling """
        if self.symmetric:  #double sample size
            next_states = torch.cat((next_states, next_states), 0)   
            rewards = torch.cat((rewards, rewards), 0)   
            masks = torch.cat((masks, masks), 0)   
            current_q1 = torch.cat((current_q1, current_q1), 0)   
            current_q2 = torch.cat((current_q2, current_q2), 0)  

        """ Update value """  
        target_q1, target_q2 = self.value_net_target(next_states, sample_actions_next)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (masks * self.gamma * (target_q - self.entropy_coef * log_probs_next)).detach()
        value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
        self.soft_update(self.value_net_target, self.value_net)
        
        """ Update policy """  
        sample_actions, log_probs, action_mean, action_log_std = self.policy_net.sample_full(states, symmetric=self.symmetric)

        if self.symmetric:  #double sample size
            states = torch.cat((states, states), 0) 
        q_new, _ = self.value_net(states, sample_actions) 

        policy_loss = -( q_new - self.entropy_coef.detach() * log_probs).mean()  # we backprop throught both q_new and log_probs.
        policy_reg = (action_mean ** 2).mean() + (action_log_std ** 2).mean() ## as done by the original code (/sac/distributions/normal.py).
        policy_loss += 0.001 * policy_reg

        self.optimizer_policy.zero_grad()
        policy_loss.backward()
        self.optimizer_policy.step()
        self.soft_update(self.policy_net_target, self.policy_net)

        """ Update entropy coef """
        entropy_loss = -(self.log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
        self.optimizer_entropy.zero_grad() 
        entropy_loss.backward() 
        self.optimizer_entropy.step()
        self.entropy_coef = torch.exp(self.log_entropy_coef)

""" TD3. From https://github.com/sfujim/TD3 """
class TD3(AC): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, encode_dim=0): 
        super().__init__(state_dim, action_dim, args, a_bound, encode_dim)
        self.update_type = "off_policy"

        self.policy_freq = 2
        self.policy_noise = args.explore_std
        self.noise_clip = 0.5
        self.p_counter = 0

    def initilize_nets(self, args):    

        self.policy_net = Policy_Gaussian(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, param_std=-1, log_std=np.log(args.explore_std), a_bound=self.a_bound, squash_action=-1).to(device)
        
        self.value_net_1 = Value(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_2 = Value(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_target_1 = Value(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.value_net_target_2 = Value(self.state_dim + self.action_dim, hidden_size=args.hidden_size, activation=args.activation).to(device)

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=args.learning_rate_pv)
        self.optimizer_value = torch.optim.Adam( list(self.value_net_1.parameters()) + list(self.value_net_2.parameters()), lr=args.learning_rate_pv)
        self.hard_update(self.value_net_target_1, self.value_net_1)
        self.hard_update(self.value_net_target_2, self.value_net_2)

    def update_policy(self, states, actions, next_states, rewards, masks):

        noise = torch.FloatTensor(actions.size()).data.normal_(0, self.policy_noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        sample_actions_next = (self.policy_net(next_states)[0] + noise).clamp(min=-self.a_bound, max=self.a_bound)
    
        sa = torch.cat((states, actions), 1)
        current_q1, current_q2 = self.value_net_1(sa), self.value_net_2(sa)

        sa_next = torch.cat((next_states, sample_actions_next), 1)
        target_q1, target_q2 = self.value_net_target_1(sa_next), self.value_net_target_2(sa_next)
        target_q = torch.min(target_q1, target_q2)
        target_q = rewards + (masks * self.gamma * target_q).detach()

        value_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
    
        self.soft_update(self.value_net_target_1, self.value_net_1) 
        self.soft_update(self.value_net_target_2, self.value_net_2) 

        # Delayed policy updates
        self.p_counter += 1
        if self.p_counter % self.policy_freq == 0:
            policy_loss = -self.value_net_1(torch.cat((states, self.policy_net(states)[0]), 1)).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

            """ Update target nets"""
            self.p_counter = 0
        
