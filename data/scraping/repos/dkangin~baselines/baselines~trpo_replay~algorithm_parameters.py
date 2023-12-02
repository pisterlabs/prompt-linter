# The code is based on the ACKTR implementation from OpenAI baselines
# This code implements method by Kangin & Pugeault "On-Policy Trust Region Policy Optimisation with Replay Buffers"

class algorithm_parameters:
    def __init__(self):
        self.gamma=0.99
        self.lam=0.97
        self.timesteps_per_batch=1000
        self.desired_kl=0.1
        self.num_timesteps=1000000
        self.animate = False
        self.advantage_replay_buffer_size = 5
        self.advantage_sample_size  = 5
        self.policy_replay_buffer_size = 5
