import torch
import math

class OrnsteinUhlenbeck():
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    # (actually based on OpenAI baselines)
    # (actually copied directly from OpenAI baselines just switching numpy to pytorch XD)
    def __call__(self):
        x = self.prev + self.theta * (self.mu - self.prev) * self.dt + self.sigma * math.sqrt(self.dt) * torch.randn(self.mu.shape)
        self.prev = x
        return x

    def reset(self):
        self.prev = torch.zeros(self.mu.shape)
