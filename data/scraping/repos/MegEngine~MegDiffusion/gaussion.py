"""Gaussion Diffusion.

Modified from OpenAI improved/guided diffusion codebase:
https://github.com/openai/guided-diffusion/blob/master/guided_diffusion/gaussian_diffusion.py

OpenAI's code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
"""

import numpy as np

import megengine as mge
import megengine.functional as F

from tqdm import tqdm

from ..loss import normal_kl, discretized_gaussian_log_likelihood
from .schedule import linear_schedule
from ..utils import batch_broadcast, mean_flat

class GaussionDiffusion:

    def __init__(
        self,
        *,
        timesteps = None,
        betas = None,
        model = None, 
        model_mean_type = "EPSILON",
        model_var_type = "FIXED_SMALL",
        loss_type = "SIMPLE",
        rescale_timesteps = False,
    ) -> None:

        assert model_mean_type in ["PREVIOUS_X", "START_X", "EPSILON"]
        assert model_var_type in ["FIXED_SMALL", "FIXED_LARGE", "LEARNED", "LEARNED_RANGE"]
        assert loss_type in ["SIMPLE", "VLB", "HYBRID"]

        # define beta schedule
        self.betas = linear_schedule(timesteps=1000) if betas is None else betas
        self._pre_calculate(self.betas)

        self.timesteps = len(self.betas) if timesteps is None else timesteps
        self.model = model
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

    def _pre_calculate(self, betas):
        """Pre-calculate constant values frequently used in formulas appears in paper.
        Calculated values will be copied to GPU (if it's default device) in advance.
        It can prevent lots of copy operations in subsequent processes.
        
        Args:
            betas: a 1-D np.array including scheduled beta values.
        """

        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # define alphas and alphas_cumprod
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)  # 1 > alphas_cumprod > 0
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])  # alphas_cumprod_{t-1}
        alphas_cumprod_next = np.append(alphas_cumprod[1:], 0.)   # alphas_cumprod_{t+1}
        sqrt_recip_alphas = np.sqrt(1. / alphas)

        # calculations for diffusion q(x_t | x_0), see :meth:`q_sample`
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        one_minus_alphas_cumprod = 1. - alphas_cumprod
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)

        # calculations for predicting x_0 with given x_t and model predicted noise
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        log_posterior_variance = np.log(np.append(posterior_variance[1], posterior_variance[1:]))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        # calculations for posterior q(x_{0} | x_t, x_{t-1})
        frac_coef1_coef2 = (posterior_mean_coef1 /  # to avoid dividing zero
            np.append(posterior_mean_coef2[1], posterior_mean_coef2[1:]))

        def host2device(data):
            return mge.Tensor(data, dtype="float32")

        # copy and store these values on GPU device (if exists) in advance
        self.betas = host2device(betas)
        self.alphas = host2device(alphas)
        self.alphas_cumprod = host2device(alphas_cumprod)
        self.alphas_cumprod_prev = host2device(alphas_cumprod_prev)
        self.alphas_cumprod_next = host2device(alphas_cumprod_next)
        self.sqrt_recip_alphas = host2device(sqrt_recip_alphas)
        self.sqrt_alphas_cumprod = host2device(sqrt_alphas_cumprod)
        self.one_minus_alphas_cumprod = host2device(one_minus_alphas_cumprod)
        self.log_one_minus_alphas_cumprod = host2device(log_one_minus_alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = host2device(sqrt_recip_alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = host2device(sqrt_recipm1_alphas_cumprod)
        self.posterior_variance = host2device(posterior_variance)
        self.log_posterior_variance = host2device(log_posterior_variance)
        self.posterior_mean_coef1 = host2device(posterior_mean_coef1)
        self.posterior_mean_coef2 = host2device(posterior_mean_coef2)
        self.frac_coef1_coef2 = host2device(frac_coef1_coef2)

    def q_mean_variance(self, x_start, t):
        """Get the distribution q(x_t | x_0).
        
        Args:
            x_start: the [N x C x ...] tensor of noiseless inputs.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.

        Return:
             A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        shape = x_start.shape
        mean = batch_broadcast(self.sqrt_alphas_cumprod[t], shape) * x_start
        variance = batch_broadcast(self.one_minus_alphas_cumprod[t], shape)
        log_variance = batch_broadcast(self.log_one_minus_alphas_cumprod[t], shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0) using reparameterization trick.
        
        Args:
            x_start: the initial data batch.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            noise: if specified, the split-out normal noise.

        Return:
            A noisy version of ``x_start``, i.e ``x_t``.
        """
        shape = x_start.shape
        noise = mge.random.normal(0, 1, shape) if noise is None else noise

        mean, _, log_var = self.q_mean_variance(x_start, t)
        return mean + F.exp(0.5 * log_var) * noise

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)

        Args:
            x_start: the (predicted) initial data batch.
            x_t: the noisy data batch.
            t: the number of diffusion steps (minus 1). Here, 0 means one step.
            
        Return:
             A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        shape = x_start.shape
        posterior_mean = (batch_broadcast(self.posterior_mean_coef1[t], shape) * x_start
            + batch_broadcast(self.posterior_mean_coef2[t], shape) * x_t)
        posterior_variance = batch_broadcast(self.posterior_variance[t], shape)
        log_posterior_variance = batch_broadcast(self.log_posterior_variance[t], shape)
        return posterior_mean, posterior_variance, log_posterior_variance

    def p_mean_variance(self, x_t, t, clip_denoised=True):
        """Apply the model to get p(x_{t-1} | x_t), as well as a prediction of the initial x.

        Args:
            x_t: the [N x C x ...] tensor at time t.
            t: a 1-D Tensor of timesteps.
            clip_denoised: if True, clip the denoised signal into [-1, 1].

        Return:
            A tuple (mean, variance, log_variance, model_ouput), all of x_start's shape.
            Note ``model_ouput`` has been processed according to learning variance or not.
        """
        shape = x_t.shape

        model_output = self.model(
            x_t, 
            t * 1000.0 / self.timesteps if self.rescale_timesteps else t
        )

        # Handle with model_output according to the variance type (fixed or learned)
        # Then get model log variance and variance values
        if self.model_var_type == "FIXED_SMALL":
            model_log_var = batch_broadcast(self.log_posterior_variance[t], shape)
        elif self.model_var_type == "FIXED_LARGE":
            model_log_var = batch_broadcast(
                F.concat((self.log_posterior_variance[1], F.log(self.betas[1:])), axis=1),
                shape,  # set the initial (log-)variance to get a better decoder log likelihood.
            )
        else:  # model's output contains learned variance value (the 2nd half part on channels)
            model_output, model_var_values = F.split(model_output, 2, axis=1)
            if self.model_var_type == "LEARNED":  # learned log variance directly
                model_log_var = model_var_values
            elif self.model_var_type == "LEARNED_RANGE":  # IDDPM Eq. (15)
                min_log = batch_broadcast(self.log_posterior_variance[t], shape)
                max_log = batch_broadcast(F.log(self.betas[t]), shape)
                # The model_var_values is [-1, 1] and should convert to [0, 1] as coff.
                frac = (model_var_values + 1) / 2
                model_log_var = frac * max_log + (1 - frac) * min_log
        model_variance = F.exp(model_log_var)

        # Handle with model_output to get ``predict_x_start`` commonly then get model_mean
        if self.model_mean_type == "PREVIOUS_X":  # model_ouput is x_{t-1}
            predict_x_start = (  # formula x_0 = (x_{t-1} - coef2 * x_t) / coef1, not mentioned in papaer
                batch_broadcast(1.0 / self.posterior_mean_coef1[t], shape) * model_output -
                batch_broadcast(self.frac_coef1_coef2[t], shape) * x_t
            )
        elif self.model_mean_type == "EPSILON":  # model_output is the noise between x_{0} and x_{t}
            predict_x_start = (
                batch_broadcast(self.sqrt_recip_alphas_cumprod[t], shape) * x_t -
                batch_broadcast(self.sqrt_recipm1_alphas_cumprod[t], shape) * model_output
            )
        else:  # model_output is x_0 directly
            predict_x_start = model_output

        # All the image values are scaled to [-1, 1], so clip them here if needed
        if clip_denoised:
            predict_x_start = F.clip(predict_x_start, -1., 1.)

        # get predicted x_{t-1} from predicted x_{0} and input x_{t}
        model_mean = (
            batch_broadcast(self.posterior_mean_coef1[t], shape) * predict_x_start
            + batch_broadcast(self.posterior_mean_coef2[t], shape) * x_t
        )

        # model_output will be used in other place, so return it here
        return model_mean, model_variance, model_log_var, model_output


    def p_sample(self, x_t, t, clip_denoised=True):
        """Sample from p_{theta} (x_{t-1} | x_t) using reparameterization trick.
        
        Args:
            x: the current tensor at x_{t-1}.
            t: the value of t, starting at 0 for the first diffusion step.
            clip_denoised: if True, clip the x_start prediction to [-1, 1].

        Return:
            a random sample from the model, i.e x_{t-1}
        """
        shape = x_t.shape

        # if t == 0, the sample do not need to be denoised, so add a mask here
        nozero_mask = batch_broadcast(t != 0, shape)
        noise = nozero_mask * mge.random.normal(0, 1, shape)

        model_mean, _, model_log_var, _ = self.p_mean_variance(x_t, t, clip_denoised)
        
        return model_mean + F.exp(0.5 * model_log_var) * noise

    def p_sample_loop(self, shape):
        x = mge.random.normal(0, 1, shape)
        for i in tqdm(reversed(range(0, self.timesteps)), 
            desc="Generating image from noise", total=self.timesteps):
            x = self.p_sample(x, F.full((shape[0],), i))
        return x
        
    def training_loss(self, x_start, t=None, noise=None):

        shape = x_start.shape

        if t is None:
            t = mge.Tensor(np.random.randint(0, self.timesteps, len(x_start)))
        noise = mge.random.normal(0, 1, shape) if noise is None else noise

        x_t = self.q_sample(x_start, t, noise)

        true_mean, _, true_log_var = self.q_posterior_mean_variance(x_start, x_t, t)
        pred_mean, _, pred_log_var, model_output = self.p_mean_variance(x_t, t)  # model forward here

        def _vlb_loss(rescale=False):
            """calculate VLB bound bits per dimensions"""
            
            # L_{t-1} := D_{KL} ( q(x_{t-1} | x_t, x_0) || p (x_{t-1} | x_t))
            kl = normal_kl(true_mean, true_log_var, pred_mean, pred_log_var)
            kl = mean_flat(kl) / F.log(2.)  # get bit per dimension loss

            # L_{0} := -log p(x_0 | x_1)
            # To evaluate L0 for images, we assume that each color component is divided into 256 bins,
            # and we compute the probability of pθ (x0 |x1) landing in the correct bin
            # (which is tractable using the CDF of the Gaussian distribution).
            l0_nll = -discretized_gaussian_log_likelihood(
                x_start, means=pred_mean, log_scales=0.5 * pred_log_var
            )
            l0_nll = mean_flat(kl) / F.log(2.)

            # L_{t} is not need to be trained so ignore here

            loss = F.where((t == 0), l0_nll, kl)

            if rescale:
                loss = loss * self.timesteps
            return loss

        def _simple_loss():
            
            loss = mean_flat(({
                "PREVIOUS_X": true_mean,
                "START_X": x_start,
                "EPSILON": noise,
            }[self.model_mean_type] - model_output) ** 2) # MSE

            return loss

        def _hybrid_loss(lamb=0.001):
            """
            See IDDPM Eq. (16) and default config ``rescale_learned_sigmas=True`` in original code.
            Divide by 1000 for equivalence with initial implementation.
            Without a factor of 1/1000, the VB term hurts the MSE term.
            """
            return lamb * _vlb_loss() + _simple_loss()

        return { 
            "VLB": _vlb_loss,
            "SIMPLE": _simple_loss,
            "HYBRID": _hybrid_loss,
        }[self.loss_type]()