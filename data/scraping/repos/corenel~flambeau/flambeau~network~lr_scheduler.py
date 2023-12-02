import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        """
        Constant learning rate scheduler

        :param optimizer:
        :type optimizer:
        :param last_epoch:
        :type last_epoch:
        """
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


class PolynomialLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 max_iter,
                 decay_iter=1,
                 gamma=0.9,
                 last_epoch=-1):
        """
        Polynomial learning rate scheduler

        :param optimizer: given optimizer
        :type optimizer: torch.optim.Optimizer
        :param max_iter: maximum value of iteration
        :type max_iter: int
        :param decay_iter: interval of decaying learning rate
        :type decay_iter: int
        :param gamma: decay factor
        :type gamma: float
        :param last_epoch: last epoch
        :type last_epoch: int
        """
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch % self.decay_iter or self.last_epoch % self.max_iter:
            return [base_lr for base_lr in self.base_lrs]
        else:
            factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 model_depth,
                 warmup_steps,
                 factor=1.0,
                 min_lr=None,
                 last_epoch=-1):
        """
        Noam Learning rate scheduler

        This corresponds to increasing the learning rate linearly for the
        first ``warmup_steps`` training steps, and decreasing it thereafter
        proportionally to the inverse square root of the step number,
        scaled by the inverse square root of the dimensionality of the model.
        Time will tell if this is just madness or it's actually important.

        :param optimizer: given optimizer
        :type optimizer: torch.optim.Optimizer
        :param model_depth: depth which dominates the number of parameters in your model
        :type model_depth: int
        :param warmup_steps: number of steps to linearly increase the learning rate
        :type warmup_steps: int
        :param factor: overall scale factor for the learning rate decay
        :type factor: float
        :param min_lr: minimum value of learning rate
        :type min_lr: float
        :param last_epoch: last epoch
        :type last_epoch: int
        """
        super().__init__(optimizer, last_epoch=last_epoch)
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.model_size = model_depth
        self.min_lr = min_lr

    def get_lr(self):
        step = max(self.last_epoch, 1)
        scale = self.factor * (self.model_size ** (-0.5) *
                               min(step ** (-0.5), step * self.warmup_steps ** (-1.5)))
        if self.min_lr is not None:
            scale = max(self.min_lr, scale)

        return [scale for _ in range(len(self.base_lrs))]


class LinearLR(_LRScheduler):
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 min_lr=1e-6,
                 last_epoch=-1):
        """
        Linearly annealed learning rate scheduler


        :param optimizer: given optimizer
        :type optimizer: torch.optim.Optimizer
        :param warmup_steps: number of steps to linearly increase the learning rate
        :type warmup_steps: int
        :param min_lr: minimum value of learning rate
        :type min_lr: float
        :param last_epoch: last epoch
        :type last_epoch: int
        """
        super().__init__(optimizer, last_epoch)
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

    def get_lr(self):
        return [max(self.min_lr + (base_lr - self.min_lr) * (1.0 - self.last_epoch / self.warmup_steps),
                    self.min_lr)
                for base_lr in self.base_lrs]


def constant(base_lr, global_step):
    """
    Return constant learning rate

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :return: scheduled learning rate
    :rtype: float
    """
    return base_lr


def noam_linear(base_lr,
                global_step,
                warmup_steps=0,
                anneal_steps=1000,
                min_lr=1e-5):
    """
    Noam learning rate decay (from section 5.3 of Attention is all you need)

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param warmup_steps: number of steps for warming up
    :type warmup_steps: int
    :param anneal_steps: number of steps for linear annealing
    :type anneal_steps: int
    :param min_lr: minimum learning rate
    :type min_lr: float
    :return: scheduled learning rate
    :rtype: float
    """
    step_num = max(0, global_step) + 1.
    if step_num < warmup_steps:
        lr = base_lr * step_num / float(warmup_steps)
    else:
        lr = base_lr * (
                anneal_steps / float(step_num + anneal_steps - warmup_steps)) ** 0.5
        # lr = min_lr + (base_lr - min_lr) * (1.0 - global_step / warmup_steps)
        lr = max(lr, min_lr)
    return lr


def noam_decay(base_lr, global_step, warmup_steps=4000, min_lr=1e-4):
    """
    Noam learning rate decay (from section 5.3 of Attention is all you need)

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param warmup_steps: number of steps for warming up
    :type warmup_steps: int
    :param min_lr: minimum learning rate
    :type min_lr: float
    :return: scheduled learning rate
    :rtype: float
    """
    step_num = global_step + 1.
    lr = base_lr * warmup_steps ** 0.5 * np.minimum(
        step_num ** -0.5,
        step_num * float(warmup_steps) ** -1.5)

    if global_step >= warmup_steps:
        lr = max(min_lr, lr)
    return lr


def linear_warmup(base_lr, curr_epoch, warmup_epochs=2, decay_epochs=5, total_epochs=10):
    if curr_epoch <= warmup_epochs:
        return base_lr * min(1., curr_epoch / warmup_epochs)
    elif curr_epoch >= decay_epochs:
        return base_lr * max(0., 1 - (curr_epoch - decay_epochs) / (total_epochs - decay_epochs))


def openai(base_lr, num_processed_images, num_epochs, num_warmup_epochs):
    """
    Learning rate scheduling strategy from openai/glow

    :param base_lr: base learning rate
    :type base_lr: float
    :param num_processed_images: number of processed images
    :type num_processed_images: int
    :param num_epochs: number of total epochs to train
    :type num_epochs: int
    :param num_warmup_epochs: number of epochs to warm up
    :type num_warmup_epochs: int
    :return: scheduled learning rate
    :rtype: float
    """
    lr = base_lr * min(1., num_processed_images / (num_epochs * num_warmup_epochs))
    return lr


def linear_anneal(base_lr, global_step, warmup_steps, min_lr):
    """
    Linearly annealed learning rate from 0 in the first warming up epochs.

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param warmup_steps: number of steps for warming up
    :type warmup_steps: int
    :param min_lr: minimum learning rate
    :type min_lr: float
    :return: scheduled learning rate
    :rtype: float
    """
    lr = max(min_lr + (base_lr - min_lr) * (1.0 - global_step / warmup_steps),
             min_lr)
    return lr


def step_anneal(base_lr,
                global_step,
                anneal_rate=0.98,
                anneal_interval=30000,
                min_lr=None):
    """
    Annealing learning rate by steps

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param anneal_rate: rate of annealing
    :type anneal_rate: float
    :param anneal_interval: interval steps of annealing
    :type anneal_interval: int
    :param min_lr: minimum value of learning rate
    :type min_lr: float
    :return: scheduled learning rate
    :rtype: float
    """

    lr = base_lr * anneal_rate ** (global_step // anneal_interval)
    if min_lr is not None:
        lr = max(min_lr, lr)
    return lr


def cyclic_cosine_anneal(base_lr, global_step, t, m):
    """
    Cyclic cosine annealing (from section 3 of SNAPSHOT ENSEMBLES: TRAIN 1, GET M FOR FREE)

    :param base_lr: base learning rate
    :type base_lr: float
    :param global_step: global training steps
    :type global_step: int
    :param t: total number of epochs
    :type t: int
    :param m: number of ensembles we want
    :type m: int
    :return: scheduled learning rate
    :rtype: float
    """
    lr = (base_lr / 2.) * (np.cos(np.pi * (
            (global_step - 1) % (t // m)) / (t // m)) + 1.)
    return lr


lr_scheduler_dict = {
    'constant': lambda **kwargs: constant(**kwargs),
    'noam': lambda **kwargs: noam_decay(**kwargs),
    'noam_linear': lambda **kwargs: noam_linear(**kwargs),
    'linear_warmup': lambda **kwargs: linear_warmup(**kwargs),
    'openai': lambda **kwargs: openai(**kwargs),
    'linear': lambda **kwargs: linear_anneal(**kwargs),
    'step': lambda **kwargs: step_anneal(**kwargs),
    'cyclic_cosine': lambda **kwargs: cyclic_cosine_anneal(**kwargs),
}

lr_scheduler_neo_dict = {
    'constant': ConstantLR,
    'poly': PolynomialLR,
    'noam': NoamLR,
    'step': StepLR,
    'multi_step': MultiStepLR,
    'cosine': CosineAnnealingLR,
    'linear': LinearLR,
    'exp': ExponentialLR,
}
