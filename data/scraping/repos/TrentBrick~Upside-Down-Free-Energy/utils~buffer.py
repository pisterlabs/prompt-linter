# pylint: disable=no-member
import numpy as np 
import torch 
import random 
import bisect
def combined_shape(length, shape=None):
    # taken from openAI spinning up. 
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class SortedBuffer:
    """
    Buffer that efficiently remains sorted.  
    """
    def __init__(self, obs_dim, act_dim, size,
                use_td_lambda_buf=False ):
        self.obs_buf = None
        self.obs2_buf= None
        self.act_buf= None
        self.discounted_rew_to_go_buf= None
        self.cum_rew= None
        self.horizon= None
        self.rollout_length = None
        self.final_obs = None
        self.buffer_dict = dict(obs=self.obs_buf, obs2=self.obs2_buf, 
                                act=self.act_buf, 
                                discounted_rew_to_go=self.discounted_rew_to_go_buf,
                                cum_rew=self.cum_rew, horizon=self.horizon, 
                                rollout_length=self.rollout_length, 
                                final_obs=self.final_obs)
        
        self.use_td_lambda_buf = use_td_lambda_buf
        if self.use_td_lambda_buf:
            self.rollout_end_ind_buf = None
            self.raw_rew_buf = None
            
            self.buffer_dict['rollout_end_ind'] = self.rollout_end_ind_buf
            self.buffer_dict['raw_rew'] = self.raw_rew_buf
        
        self.size, self.max_size = 0, size
        self.total_num_steps_added = 0

    def add_rollouts(self, list_of_rollout_dicts):
        # sorted in ascending order. largest values at the back. 
        for rollout in list_of_rollout_dicts:
            # dont bother adding rollouts that are worse than the worst in the buffer. 
            # but still count them to the overall number of rollouts seen. 
            len_rollout = len(rollout['terminal'])
            self.total_num_steps_added += len_rollout
            if self.size == self.max_size and rollout['cum_rew'][0] <= self.buffer_dict['cum_rew'][-1]:
                continue 
            
            self.size = min(self.size+len_rollout, self.max_size)
            
            if self.buffer_dict['obs'] is not None:
                # find where everything from this rollout should be inserted into 
                # each of the numpy buffers. Uses the cumulative/terminal rewards
                # minus so that highest values are at the front. 
                sort_ind = np.searchsorted(-self.buffer_dict['cum_rew'], -rollout['cum_rew'][0]  )
                end_ind = len_rollout+sort_ind 
            else: 
                end_ind = len_rollout 
            
            if self.use_td_lambda_buf:
                # will be appended and treated like everything else. 
                end_ind = np.repeat(end_ind, len_rollout)
                rollout['rollout_end_ind'] = end_ind

            for key in self.buffer_dict.keys(): 
                # NOTE: assumes that buffer and rollout use the same keys!
                # needed at init!
                if self.buffer_dict[key] is None:
                    self.buffer_dict[key] = rollout[key]
                else: 
                    self.buffer_dict[key] = np.insert(self.buffer_dict[key], sort_ind, rollout[key], axis=0)

                if key == 'rollout_end_ind': 
                    self.buffer_dict[key][end_ind[0]:] = self.buffer_dict[key][end_ind[0]:]+len_rollout

                if self.size >= self.max_size:
                    # buffer is full. Need to trim!
                    # this will have a bias in that it will favour 
                    # the longer horizons at the end of the training data
                    # but it shouldnt make a major diff. 
                    self.buffer_dict[key] = self.buffer_dict[key][:self.max_size]

    def retrieve_path(self, start_index):
        end_index = self.buffer_dict['rollout_end_ind'][start_index]
        if end_index<= start_index:
            print("for sorted buffer shouldnt have looping here!")
            # we have looping 
            obs = np.concatenate( [self.buffer_dict['obs'][start_index:], self.buffer_dict['obs'][:end_index]], axis=0)
            rew = np.concatenate( [self.buffer_dict['raw_rew'][start_index:], self.buffer_dict['raw_rew'][:end_index]], axis=0)
        else: 
            obs = self.buffer_dict['obs'][start_index:end_index]
            rew = self.buffer_dict['raw_rew'][start_index:end_index]

        return torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(rew, dtype=torch.float32)


    def get_desires(self, last_few = 75):
        """
        This function calculates the new desired reward and new desired horizon based on the replay buffer.
        New desired horizon is calculted by the mean length of the best last X episodes. 
        New desired reward is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
        where X is the hyperparameter last_few.
        """
        # it is the first occurence of each cumulative ind. 
        unique_cum_rews, unique_cum_inds = np.unique(-self.buffer_dict['cum_rew'], return_index=True)
        #unique returns a sorted dictionary need to reverse it. 
        unique_cum_rews, unique_cum_inds = -unique_cum_rews[:last_few], unique_cum_inds[:last_few]
        #The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
        new_desired_horizon = round( self.buffer_dict['rollout_length'][unique_cum_inds].mean()  )

        # from these returns calc the mean and std
        mean_returns = np.mean(unique_cum_rews)
        std_returns = np.std(unique_cum_rews)

        new_desired_state = self.buffer_dict['final_obs'][unique_cum_inds].mean(axis=0) 

        return mean_returns, std_returns, new_desired_horizon, new_desired_state

    def __getitem__(self, idx):
        # turn this into a random value!
        #rand_ind = np.random.randint(0,self.size) # up to current max size. 
        return self.sample_batch(idxs=idx)

    def __len__(self):
        return self.size #self.num_batches_per_epoch

    def sample_batch(self, idxs=None, batch_size=256):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        return {key:torch.as_tensor(arr[idxs],dtype=torch.float32) for key, arr in self.buffer_dict.items()}


class RingBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    # Taken from OpenAI spinning up. 
    """
    def __init__(self, obs_dim, act_dim, size, use_td_lambda_buf=False):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        if act_dim==1:
            self.act_buf = np.zeros(size, dtype=np.float32)
        else: 
            self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.discounted_rew_to_go_buf = np.zeros(size, dtype=np.float32)
        self.cum_rew = np.zeros(size, dtype=np.float32)
        self.final_obs = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.horizon_buf = np.zeros(size, dtype=np.float32)
        #self.terminal_buf = np.zeros(size, dtype=np.int8)

        self.buf_list = [self.obs_buf, 
                        self.obs2_buf, 
                        self.discounted_rew_to_go_buf,
                        self.act_buf, 
                        self.cum_rew, 
                        self.horizon_buf,
                        self.final_obs
                        ]

        self.value_names = ['obs', 
                            'obs2', 
                            'discounted_rew_to_go', 
                            'act', 
                            'cum_rew', 
                            'horizon',
                            'final_obs'
                            ]

        self.use_td_lambda_buf = use_td_lambda_buf
        if self.use_td_lambda_buf:
            self.rollout_end_ind_buf = np.zeros(size, dtype=np.int32)
            self.raw_rew_buf = np.zeros(size, dtype=np.float32)
            
            self.buf_list.append(self.raw_rew_buf)
            self.value_names.append('raw_rew')

        self.ptr, self.size, self.max_size = 0, 0, size
        self.total_num_steps_added = 0

    def retrieve_path(self, start_index):
        end_index = self.rollout_end_ind_buf[start_index]
        if end_index<= start_index:
            # we have looping
            obs = np.concatenate( [self.obs_buf[start_index:], self.obs_buf[:end_index]], axis=0)
            rew = np.concatenate( [self.raw_rew_buf[start_index:], self.raw_rew_buf[:end_index]], axis=0)
        else: 
            obs = self.obs_buf[start_index:end_index]
            rew = self.raw_rew_buf[start_index:end_index]

        return torch.as_tensor(obs, dtype=torch.float32), torch.as_tensor(rew, dtype=torch.float32)

    def add_to_buffer(self,np_buf, iters_adding, data ):
        if (self.ptr+iters_adding)>self.max_size:
            amount_pre_loop = self.max_size-self.ptr
            amount_post_loop = iters_adding-amount_pre_loop
            np_buf[self.ptr:] = data[:amount_pre_loop]
            np_buf[:amount_post_loop] = data[amount_pre_loop:]
        else: 
            np_buf[self.ptr:self.ptr+iters_adding] = data

    def add_rollouts(self, list_of_rollout_dicts):
        for rollout in list_of_rollout_dicts:
            iters_adding = len(rollout['terminal'])
            self.total_num_steps_added += iters_adding

            for np_buf, key in zip(self.buf_list,
                                self.value_names ):
                self.add_to_buffer(np_buf, iters_adding, rollout[key])
                
            if self.use_td_lambda_buf:
                end_ind = int((self.ptr+iters_adding) % self.max_size)
                assert end_ind >=0
                end_ind = np.repeat(end_ind, iters_adding)
                self.add_to_buffer(self.rollout_end_ind_buf, iters_adding, end_ind)

            self.ptr = (self.ptr+iters_adding) % self.max_size
            self.size = min(self.size+iters_adding, self.max_size)

    def __getitem__(self, idx):
        # turn this into a random value!
        #rand_ind = np.random.randint(0,self.size) # up to current max size. 
        return self.sample_batch(idxs=idx)

    def __len__(self):
        return self.size #self.num_batches_per_epoch

    def sample_batch(self, idxs=None, batch_size=32):
        if idxs is None:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     discounted_rew_to_go=self.discounted_rew_to_go_buf[idxs],
                     #terminal=self.terminal_buf[idxs],
                     horizon=self.horizon_buf[idxs],
                     cum_rew=self.cum_rew[idxs],
                     final_obs=self.final_obs[idxs]
                     )
        if self.use_td_lambda_buf:
            batch['start_index'] = idxs
            batch['rollout_end_ind'] = self.rollout_end_ind_buf[idxs]
            batch['raw_rew'] = self.raw_rew_buf[idxs]
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


######## OLD buffers no longer in use!


class ReplayBuffer:
    def __init__(self, max_size, seed, batch_size, num_grad_steps):
        self.max_size = max_size
        self.buffer = []
        self.batch_size = batch_size
        self.num_grad_steps = num_grad_steps
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def add_sample(self, states, actions, rewards, length):
        #assert length < 300, "episode is too long!"
        episode = {"states": states, "actions":actions, 
            "rewards": rewards, "summed_rewards":sum(rewards), 
            "length":length}
        self.buffer.append(episode)
        
    def sort(self):
        #sort buffer
        self.buffer = sorted(self.buffer, key = lambda i: i["summed_rewards"],reverse=True)
        # keep the max buffer size
        self.buffer = self.buffer[:self.max_size]
    
    def get_episodes(self, batch_size):
        idxs = np.random.randint(0, len(self.buffer), batch_size)
        batch = [self.buffer[idx] for idx in idxs]
        # returns a list of dictionaries that contain full rollouts.  
        return batch

    def sample_batch(self, batch_size):

        episodes = self.get_episodes(batch_size)
        
        batch_states = []
        batch_rewards = []
        batch_horizons = []
        batch_actions = []
        
        for episode in episodes:
            # this is from a named tuple. 
            # samples one thing from each episode?????? this is weird. 
            T = episode['length'] # one over what it needs to be for [s:e] indexing in sum.
            t1 = np.random.randint(0, T-1) # -1 so cant sample last time index
            # for sparse and everything really?
            t2 = T
            #t2 = np.random.randint(t1+1, T+1)
            dr = sum(episode['rewards'][t1:t2])
            dh = t2 - t1
            
            st1 = episode['states'][t1]
            at1 = episode['actions'][t1]
            
            batch_states.append(st1)
            batch_actions.append(at1)
            batch_rewards.append(dr)
            batch_horizons.append(dh)
        
        batch_states = torch.FloatTensor(batch_states)
        batch_rewards = torch.FloatTensor(batch_rewards)
        batch_horizons = torch.FloatTensor(batch_horizons)
        batch_actions = torch.LongTensor(batch_actions)

        return dict(obs=batch_states, rew=batch_rewards, horizon=batch_horizons, act=batch_actions)

    def get_nbest(self, n):
        return self.buffer[:n]

    def get_desires(self, last_few = 75):
        """
        This function calculates the new desired reward and new desired horizon based on the replay buffer.
        New desired horizon is calculted by the mean length of the best last X episodes. 
        New desired reward is sampled from a uniform distribution given the mean and the std calculated from the last best X performances.
        where X is the hyperparameter last_few.
        """
        
        top_X = self.get_nbest(last_few)
        #The exploratory desired horizon dh0 is set to the mean of the lengths of the selected episodes
        #print([len(i["states"]) for i in top_X])
        #print([i["length"] for i in top_X])
        new_desired_horizon = round(np.mean([i["length"] for i in top_X]))
        # save all top_X cumulative returns in a list 
        returns = [i["summed_rewards"] for i in top_X]
        # from these returns calc the mean and std
        mean_returns = np.mean(returns)
        std_returns = np.std(returns)
        # sample desired reward from a uniform distribution given the mean and the std
        #new_desired_reward = np.random.uniform(mean_returns, mean_returns+std_returns)

        return mean_returns, std_returns, new_desired_horizon
    
    def __getitem__(self, idx):
        return self.sample_batch(self.batch_size)

    def __len__(self):
        return self.num_grad_steps

class TrainBufferDataset:
    def __init__(self, init_train_data, max_buffer_size, 
        key_to_check_lengths='terminal'):

        self.key_to_check_lengths = key_to_check_lengths
        self.max_buffer_size = max_buffer_size
        # init the buffer.
        self.buffer = init_train_data
        self.buffer_index = len(init_train_data[self.key_to_check_lengths])

    def add(self, train_data):
        curr_buffer_size = len(self.buffer[self.key_to_check_lengths])
        length_data_to_add = len(train_data[self.key_to_check_lengths])
        # dict agnostic length checker::: len(self.buffer[list(self.buffer.keys())[0]])
        if curr_buffer_size < self.max_buffer_size:
            print('growing buffer')
            for k in self.buffer.keys():
                self.buffer[k] += train_data[k]
            print('new buffer size', len(self.buffer[self.key_to_check_lengths]))
            self.buffer_index += length_data_to_add
            #if now exceeded buffer size: 
            if self.buffer_index>self.max_buffer_size:
                self.max_buffer_size=self.buffer_index
                self.buffer_index = 0
        else: 
            # buffer is now full. Rewrite to the correct index.
            if self.buffer_index > self.max_buffer_size-length_data_to_add:
                print('looping!')
                # going to go over so needs to loop around. 
                amount_pre_loop = self.max_buffer_size-self.buffer_index
                amount_post_loop = length_data_to_add-amount_pre_loop

                for k in self.buffer.keys():
                    self.buffer[k][self.buffer_index:] = train_data[k][:amount_pre_loop]

                for k in self.buffer.keys():
                    self.buffer[k][:amount_post_loop] = train_data[k][amount_pre_loop:]
                self.buffer_index = amount_post_loop
            else: 
                print('clean add')
                for k in self.buffer.keys():
                    self.buffer[k][self.buffer_index:self.buffer_index+length_data_to_add] = train_data[k]
                # update the index. 
                self.buffer_index += length_data_to_add
                self.buffer_index = self.buffer_index % self.max_buffer_size

    def __len__(self):
        return len(self.buffer[self.key_to_check_lengths])


# Datasets: 
class ForwardPlanner_GeneratedDataset(torch.utils.data.Dataset):
    """ This dataset is inspired by those from dataset/loaders.py but it 
    doesn't need to apply any transformations to the data or load in any 
    files.

    :args:
        - transform: any tranforms desired. Currently these are done by each rollout
        and sent back to avoid performing redundant transforms.
        - data: a dictionary containing a list of Pytorch Tensors. 
                Each element of the list corresponds to a separate full rollout.
                Each full rollout has its first dimension corresponding to final_obs. 
        - seq_len: desired length of rollout sequences. Anything shorter must have 
        already been dropped. (currently done in 'combine_worker_rollouts()')

    :returns:
        - a subset of length 'seq_len' from one of the rollouts with all of its relevant features.
    """
    def __init__(self, transform, data, seq_len): 
        self._transform = transform
        self.data = data
        self._cum_size = [0]
        self._buffer_index = 0
        self._seq_len = seq_len

        # set the cum size tracker by iterating through the data:
        for d in self.data['terminal']:
            self._cum_size += [self._cum_size[-1] +
                                   (len(d)-self._seq_len)]

    def __getitem__(self, i): # kind of like the modulo operator but within rollouts of batch size. 
        # binary search through cum_size
        rollout_index = bisect(self._cum_size, i) - 1 # because it finds the index to the right of the element. 
        # within a specific rollout. will linger on one rollout for a while iff random sampling not used. 
        seq_index = i - self._cum_size[rollout_index] # references the previous file length. so normalizes to within this file's length. 
        obs_data = self.data['obs'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        if self._transform:
            obs_data = self._transform(obs_data.astype(np.float32))
        action = self.data['actions'][rollout_index][seq_index:seq_index + self._seq_len + 1]
        reward, terminal = [self.data[key][rollout_index][seq_index:
                                      seq_index + self._seq_len + 1]
                            for key in ('rewards', 'terminal')]
        return obs_data, action, reward.unsqueeze(1), terminal
        
    def __len__(self):
        return self._cum_size[-1]