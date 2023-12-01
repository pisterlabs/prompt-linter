"""Reconstruction guidance"""
from collections.abc import Callable
import torch as th
from torch import nn
from src.diffusion.base import DiffusionSampler, extract
import pytorch_lightning as pl
from src.guidance.base import Guidance


class ReconstructionSampler:
    """Sampling from reconstruction guided DDPM"""

    def __init__(self, diff_model: nn.Module, diff_proc: DiffusionSampler, guidance: Guidance, verbose=False):
        self.diff_model = diff_model
        self.diff_model.eval()
        self.diff_proc = diff_proc
        self.guidance = guidance
        self.verbose = verbose

    @th.no_grad()
    def sample(self, num_samples: int, classes: th.Tensor, device: th.device, shape: tuple):
        """Sampling from the backward process
        Sample points from the data distribution

        Args:
            model (model to predict noise)
            num_samples (number of samples)
            classes (num_samples, ): classes to condition on (on for each sample)
            device (the device the model is on)
            shape (shape of data, e.g., (1, 28, 28))

        Returns:
            all x through the (predicted) reverse diffusion steps
        """

        # self.diff_model.eval()
        steps = []
        classes = classes.to(device)
        x_tm1 = th.randn((num_samples,) + shape).to(device)

        for t in reversed(range(0, self.diff_proc.num_diff_steps)):
            if self.verbose and (t + 1) % 100 == 0:
                print(f"Diffusion step {t+1}")
            t_tensor = th.full((x_tm1.shape[0],), t, device=device)
            # Use the model to predict noise and use the noise to step back
            pred_noise = self.diff_model(x_tm1, t_tensor)
            x_tm1 = self._sample_x_tm1_given_x_t(x_tm1, t, pred_noise, classes)
            steps.append(x_tm1.detach().cpu())

        return x_tm1, steps

    def _sample_x_tm1_given_x_t(self, x_t: th.Tensor, t: int, pred_noise: th.Tensor, classes: th.Tensor):
        """Denoise the input tensor at a given timestep using the predicted noise

        Args:
            x_t (any shape),
            t (timestep at which to denoise),
            predicted_noise (noise predicted at the timestep)

        Returns:
            x_tm1 (x[t-1] denoised sample by one step - x_t.shape)
        """
        b_t = extract(self.diff_proc.betas, t, x_t)
        a_t = extract(self.diff_proc.alphas, t, x_t)
        a_bar_t = extract(self.diff_proc.alphas_bar, t, x_t)
        post_var_t = extract(self.diff_proc.posterior_variance, t, x_t)

        if t > 0:
            z = th.randn_like(x_t)
        else:
            z = 0

        sigma_t = self.diff_proc.sigma_t(t, x_t)
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        class_score = self.guidance.grad(x_t, t_tensor, classes, pred_noise)
        m_tm1 = (x_t + b_t / (th.sqrt(1 - a_bar_t)) * (sigma_t * class_score - pred_noise)) / a_t.sqrt()
        noise = post_var_t.sqrt() * z
        xtm1 = m_tm1 + noise
        return xtm1


class ReconstructionGuidance(Guidance):
    """A class the computes the gradient used in reconstruction guidance"""

    def __init__(
        self, noise_pred: nn.Module, classifier: nn.Module, alpha_bars: th.Tensor, loss: nn.Module, lambda_: float = 1.0
    ):
        """
        @param noise_pred: eps_theta(x_t, t), noise prediction model
        @param classifier: Classifier model p(y|x_0)
        @param loss: Corresponding loss function for the classifier (e.g., CrossEntropy)
        @param lambda_: Magnitude of the gradient
        """
        super(ReconstructionGuidance, self).__init__(lambda_=lambda_)
        self.classifier = classifier
        self.noise_pred = noise_pred
        self.loss = loss
        self.alpha_bars = alpha_bars

    @th.no_grad()
    def grad(self, x_t, t, y, pred_noise):
        """Compute score function for the classifier

        Estimates the score grad_x_t log p(y | x_t) by mapping x_t to x_0
        and then evaluating the given likelihood p(y | x_0)
        """
        if self.lambda_ > 0.0:
            th.set_grad_enabled(True)
            # I do not know if this is correct, or even necessary.
            x_t = x_t.clone().detach().requires_grad_(True)
            x_0 = self._map_to_x_0(x_t, t, pred_noise)
            logits = self.classifier(x_0)
            loss = -self.loss(logits, y)
            grad_ = self.lambda_ * th.autograd.grad(loss, x_t, retain_graph=True)[0]
            th.set_grad_enabled(False)
        else:
            grad_ = th.zeros_like(x_t)
        return grad_

    def predict_x_0(self, x_t, t):
        t_tensor = th.full((x_t.shape[0],), t, device=x_t.device)
        pred_noise_t = self.noise_pred(x_t, t_tensor)
        return self._map_to_x_0(x_t, t_tensor, pred_noise_t)

    def _map_to_x_0(self, x_t, t, pred_noise_t):
        """Map x_t to x_0

        For reconstruction guidance, we assume that we only have access to the likelihood
        p(y | x_0); we approximate p(y | x_t) ~= p(y | x_0_hat(x_t))
        """
        return mean_x_0_given_x_t(x_t, pred_noise_t, self.alpha_bars[t])


def mean_x_0_given_x_t(x_t: th.Tensor, noise_pred_t: th.Tensor, a_bar_t: th.Tensor):
    """Compute E[x_0|x_t] using Tweedie's formula

    See Prop. 1 in https://arxiv.org/pdf/2209.14687.pdf

    NB: This uses the noise prediction function eps_theta, not the score function s_theta.
    """
    # TODO: Fix this
    assert all(a_bar_t == a_bar_t[0])
    _a = a_bar_t[0]
    return (x_t - th.sqrt(1.0 - _a) * noise_pred_t) / th.sqrt(_a)


class ReconstructionClassifier(pl.LightningModule):
    """Train classifier for reconstruction guidance."""

    def __init__(self, model: nn.Module, loss_f: Callable):
        super().__init__()
        self.model = model
        self.loss_f = loss_f

        # Default Initialization
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.i_batch_train = 0
        self.i_batch_val = 0
        self.i_epoch = 0

    def training_step(self, batch, _):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device).float()
        y = batch["label"].to(self.device).long()
        predicted_y = self.model(x)
        loss = self.loss_f(predicted_y, y)
        self.log("train_loss", loss)
        self.train_loss += loss.detach().cpu().item()
        self.i_batch_train += 1
        return loss

    def on_train_epoch_end(self):
        print(" {}. Train Loss: {}".format(self.i_epoch, self.train_loss / self.i_batch_train))
        self.train_loss = 0.0
        self.i_batch_train = 0
        self.i_epoch += 1

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return [optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):
        batch_size = batch["pixel_values"].shape[0]
        x = batch["pixel_values"].to(self.device)
        y = batch["label"].to(self.device)

        # Unsure why/if this is needed.
        rng_state = th.get_rng_state()
        th.manual_seed(self.i_batch_val)
        th.set_rng_state(rng_state)

        predicted_y = self.model(x)

        loss = self.loss_f(predicted_y, y)
        self.log("val_loss", loss)
        self.val_loss += loss.detach().cpu().item()
        self.i_batch_val += 1
        return loss

    def on_validation_epoch_end(self):
        print(" {}. Validation Loss: {}".format(self.i_epoch, self.val_loss / self.i_batch_val))
        self.val_loss = 0.0
        self.i_batch_val = 0
