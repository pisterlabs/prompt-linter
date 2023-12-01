#!/usr/bin/env python3
import warnings

from collections import namedtuple

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import matplotlib.pyplot as plt

from torchdiffeq import odeint

from toy_plot import SDE, analytical_log_likelihood
from toy_configs import register_configs
from toy_train_config import SampleConfig, get_model_path, get_classifier_path, ExampleConfig, \
    GaussianExampleConfig, BrownianMotionExampleConfig, BrownianMotionDiffExampleConfig, TestType, IntegratorType
from models.toy_sampler import AbstractSampler, interpolate_schedule
from toy_likelihoods import Likelihood, ClassifierLikelihood, GeneralDistLikelihood
from models.toy_temporal import TemporalTransformerUnet, TemporalClassifier, TemporalNNet
from models.toy_diffusion_models_config import GuidanceType


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SampleOutput = namedtuple('SampleOutput', 'samples fevals')

# TODO: FIX THIS UP
from generative_model import generate_trajectory
from linear_gaussian_prob_prog import get_linear_gaussian_variables, JointVariables


SDEConfig = namedtuple('SDEConfig', 'drift diffusion sde_steps end_time')
DiffusionConfig = namedtuple('DiffusionConfig', 'f g')


def create_table(A, Q, C, R, mu_0, Q_0, dim):
    table = {}
    table[dim] = {}
    table[dim]['A'] = A
    table[dim]['Q'] = Q
    table[dim]['C'] = C
    table[dim]['R'] = R
    table[dim]['mu_0'] = mu_0
    table[dim]['Q_0'] = Q_0
    return table

def compute_diffusion_step(
        sde: SDEConfig,
        diffusion: DiffusionConfig,
        diffusion_time: torch.tensor
):
    A = torch.tensor([[1.]], device=device)
    Q = sde.diffusion * torch.tensor(1. / sde.sde_steps, device=device).reshape(1, 1)
    C = diffusion.f(diffusion_time).reshape(1, -1)
    R = diffusion.g(diffusion_time).reshape(1, -1) ** 2
    mu_0 = torch.tensor([0.], device=device)
    Q_0 = Q

    table = create_table(
        A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0, dim=1
    )

    ys, _, _, _ = generate_trajectory(
        num_steps=sde.sde_steps, A=A, Q=Q, C=C, R=R, mu_0=mu_0, Q_0=Q_0
    )

    lgv = get_linear_gaussian_variables(dim=1, num_obs=sde.sde_steps, table=table)
    jvs = JointVariables(lgv.ys)
    return jvs

#########################
#########################
def suppresswarning():
    warnings.warn("user", UserWarning)


class ToyEvaluator:
    def __init__(self, cfg: SampleConfig):
        self.cfg = cfg

        d_model = torch.tensor(1)
        self.sampler = hydra.utils.instantiate(cfg.sampler)
        self.diffusion_model = hydra.utils.instantiate(cfg.diffusion, d_model=d_model, device=device).to(device)
        self.diffusion_model.eval()
        self.likelihood = hydra.utils.instantiate(cfg.likelihood)
        self.example = OmegaConf.to_object(cfg.example)

        self.cond = torch.tensor([self.cfg.cond], device=device) if self.cfg.cond is not None and self.cfg.cond >= 0. else None

        self.load_model()

    def load_model(self):
        model_path = get_model_path(self.cfg)
        try:
            # load softmax model
            print('attempting to load diffusion model: {}'.format(model_path))
            self.diffusion_model.load_state_dict(torch.load('{}'.format(model_path)))
            print('successfully loaded diffusion model')
        except Exception as e:
            print('FAILED to load model: {} because {}\ncreating it...'.format(model_path, e))

    def grad_log_lik(self, xt, t, cond, model_output, cond_traj):
        x0_hat = self.sampler.predict_xstart(xt, model_output, t)
        if type(self.diffusion_model) == TemporalTransformerUnet:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, cond_traj)
        elif type(self.likelihood) in [ClassifierLikelihood, GeneralDistLikelihood]:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat, t)
        else:
            return self.likelihood.grad_log_lik(cond, xt, x0_hat)

    def viz_trajs(self, traj, end_time, idx, clf=True, clr='green'):
        full_state_pred = traj.detach().squeeze(0).cpu().numpy()

        plt.plot(torch.linspace(0, end_time, full_state_pred.shape[0]), full_state_pred, color=clr)

        plt.savefig('figs/sample_{}.pdf'.format(idx))

        if clf:
            plt.clf()


class DiscreteEvaluator(ToyEvaluator):
    def sample_trajectories(self, cond_traj=None):
        x = self.get_x_min()

        samples = [x]
        for t in torch.arange(self.sampler.diffusion_timesteps-1, -1, -1, device=device):
            if t % 100 == 0:
                print(x[0, 0])
            time = t.reshape(-1)
            if type(self.diffusion_model) == TemporalTransformerUnet:
                unconditional_output = self.diffusion_model(x, time, None, None)
            else:
                unconditional_output = self.diffusion_model(x, time, None)
            if self.cfg.guidance == GuidanceType.Classifier:
                if self.cond is not None:
                    with torch.enable_grad():
                        xt = x.detach().clone().requires_grad_(True)
                        grad_log_lik = self.grad_log_lik(xt, time, self.cond, unconditional_output, cond_traj)
                else:
                    grad_log_lik = torch.tensor(0.)
                x = self.sampler.classifier_guided_reverse_sample(
                    xt=x, unconditional_output=unconditional_output,
                    t=t.item(), grad_log_lik=grad_log_lik
                )
            elif self.cfg.guidance == GuidanceType.ClassifierFree:
                if self.cond is None or self.cond < 0.:
                    conditional_output = unconditional_output
                else:
                    if type(self.diffusion_model) == TemporalTransformerUnet:
                        conditional_output = self.diffusion_model(x, time, cond_traj, self.cond)
                    else:
                        conditional_output = self.diffusion_model(x, time, self.cond)
                x = self.sampler.classifier_free_reverse_sample(
                    xt=x, unconditional_output=unconditional_output,
                    conditional_output=conditional_output, t=t.item()
                )
            else:
                print('Unknown guidance: {}... defaulting to unconditional sampling'.format(self.cfg.guidance))
                posterior_mean = self.sampler.get_posterior_mean(x, unconditional_output, time)
                x = self.sampler.reverse_sample(
                    x, t.item(), posterior_mean,
                )
            samples.append(x)
        return SampleOutput(samples=samples, fevals=-1)

    @torch.no_grad()
    def ode_log_likelihood(self, x, extras=None, atol=1e-4, rtol=1e-4):
        """ THIS PROBABLY SHOULDN'T BE USED """
        extras = {} if extras is None else extras
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                diffusion_time = t.reshape(-1) * self.sampler.diffusion_timesteps
                model_output = self.diffusion_model(x, diffusion_time, **extra_args)
                sf_est = self.sampler.get_sf_estimator(model_output, xt=x, t=diffusion_time)
                coef = -0.5 * interpolate_schedule(diffusion_time, self.sampler.betas)
                dx_dt = coef * (x + sf_est)
                fevals += 1
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        x_min = x, x.new_zeros([x.shape[0]])
        times = torch.tensor([self.sampler.t_eps, 1.], device=x.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='dopri5')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


class ContinuousEvaluator(ToyEvaluator):
    def get_score_function(self, t, x):
        if self.cfg.test == TestType.Gaussian:
            return self.analytical_gaussian_score(
                t=t,
                x=x,
            )
        elif self.cfg.test == TestType.BrownianMotion:
            return self.analytical_brownian_motion_score(t=t, x=x)
        elif self.cfg.test == TestType.BrownianMotionDiff:
            return self.analytical_brownian_motion_diff_score(t=t, x=x)
        elif self.cfg.test == TestType.Test:
            model_output = self.diffusion_model(x=x, time=t.reshape(-1), cond=None)
            # return self.sampler.get_sf_estimator(model_output, xt=x, t=t)
            bad_s = self.sampler.get_sf_estimator(model_output, xt=x, t=t)
            true_s = self.analytical_gaussian_score(
                t=t,
                x=x,
            )
            return bad_s
        else:
            raise NotImplementedError

    def get_dx_dt(self, t, x):
        time = t.reshape(-1)
        sf_est = self.get_score_function(t=time, x=x)
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def get_gaussian_dx_dt(self, t, x):
        time = t.reshape(-1)
        sf_est = self.analytical_gaussian_score(
            t=t,
            x=x,
        )
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def get_brownian_dx_dt(self, t, x):
        time = t.reshape(-1)
        sf_est = self.analytical_brownian_motion_score(t=t, x=x)
        dx_dt = self.sampler.probability_flow_ode(x, time, sf_est)
        return dx_dt

    def analytical_gaussian_score(self, t, x):
        '''
        Compute the analytical marginal score of p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0 = N(mu_0, sigma_0) and p_1 = N(0, 1)
        '''
        _, lmc, std = self.sampler.marginal_prob(x=x, t=t)
        f = lmc.exp()
        var = self.cfg.example.sigma ** 2 * f ** 2 + std ** 2
        score = (f * self.cfg.example.mu - x) / var
        return score

    def analytical_brownian_motion_score(self, t, x):
        '''
        Compute the analytical score p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0(x, s) = N(0, d(s)\sqrt(s)) and p_1(x, s) = N(0, 1)
        '''
        sde = SDEConfig(self.cfg.sde_drift, self.cfg.sde_diffusion, self.cfg.sde_steps-1, 1.)

        f = lambda t : self.sampler.marginal_prob(x, t)[1].exp()[:, 0, :]
        g = lambda t : self.sampler.marginal_prob(x, t)[2][:, 0, :]
        diffusion = DiffusionConfig(f, g)

        jvs = compute_diffusion_step(sde, diffusion, diffusion_time=t)
        y_dist = jvs.dist.dist
        score = torch.linalg.solve(y_dist.covariance_matrix, (y_dist.loc.reshape(-1, 1) - x))

        return score

    def analytical_brownian_motion_diff_score(self, t, x):
        '''
        Compute the analytical score p_t for t \in (0, 1)
        given the SDE formulation from Song et al. in the case that
        p_0(x, s) = N(0, d(s)\sqrt(s)) and p_1(x, s) = N(0, 1)
        where we consider sequential differences dX_t \equiv X_t - X_{t-1}
        and where X_t is Brownian Motion so X_t \sim N(X_{t-1}, \sqrt{dt})
        '''
        f = self.sampler.marginal_prob(x, t)[1].exp()[:, 0, :]
        g = self.sampler.marginal_prob(x, t)[2][:, 0, :]

        dt = 1. / (self.cfg.sde_steps-1)

        var = f ** 2 * dt + g ** 2

        score = -x / var

        return score

    def get_x_min(self):
        if type(self.example) == GaussianExampleConfig:
            x_min = self.sampler.prior_sampling(device).sample([
                    self.cfg.num_samples, 1, 1
            ])
        elif type(self.example) == BrownianMotionExampleConfig:
            x_min = self.sampler.prior_sampling(device).sample([
                self.cfg.num_samples,
                self.cfg.sde_steps-1,
                1,
            ])
        elif type(self.example) == BrownianMotionDiffExampleConfig:
            x_min = self.sampler.prior_sampling(device).sample([
                self.cfg.num_samples,
                self.cfg.sde_steps-1,
                1,
            ])
        else:
            raise NotImplementedError
        return x_min

    def sample_trajectories_euler_maruyama(self, steps=torch.tensor(1000)):
        x_min = self.get_x_min()
        x = x_min.clone()

        steps = steps.to(x.device)
        for time in torch.linspace(1., 0., steps, device=x.device):
            sf_est = self.get_score_function(t=time, x=x)
            x, _ = self.sampler.reverse_sde(x=x, t=time, score=sf_est, steps=steps)

        return SampleOutput(samples=[x_min, x], fevals=steps)

    def sample_trajectories_probability_flow(self, atol=1e-5, rtol=1e-5):
        x_min = self.get_x_min()

        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            dx_dt = self.get_dx_dt(t, x)
            return dx_dt

        times = torch.tensor([1., self.sampler.t_eps], device=x_min.device)
        # times = torch.arange(1., -0.001, -0.001, device=x_min.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        return SampleOutput(samples=sol, fevals=fevals)

    def sample_trajectories(self):
        if self.cfg.integrator_type == IntegratorType.ProbabilityFlow:
            sample_out = self.sample_trajectories_probability_flow()
        elif self.cfg.integrator_type == IntegratorType.EulerMaruyama:
            sample_out = self.sample_trajectories_euler_maruyama()
        else:
            raise NotImplementedError
        return sample_out

    @torch.no_grad()
    def ode_log_likelihood(self, x, extras=None, atol=1e-5, rtol=1e-5):
        extras = {} if extras is None else extras
        # hutchinson's trick
        v = torch.randint_like(x, 2) * 2 - 1
        fevals = 0
        def ode_fn(t, x):
            nonlocal fevals
            fevals += 1
            with torch.enable_grad():
                x = x[0].detach().requires_grad_()
                dx_dt = self.get_dx_dt(t, x)
                grad = torch.autograd.grad((dx_dt * v).sum(), x)[0]
                d_ll = (v * grad).flatten(1).sum(1)
            return torch.cat([dx_dt.reshape(-1), d_ll.reshape(-1)])
        x_min = x, x.new_zeros([x.shape[0]])
        # times = torch.tensor([self.sampler.t_eps, 1.], device=x.device)
        times = torch.tensor([self.sampler.t_eps, 1.], device=x.device)
        sol = odeint(ode_fn, x_min, times, atol=atol, rtol=rtol, method='rk4')
        latent, delta_ll = sol[0][-1], sol[1][-1]
        ll_prior = self.sampler.prior_logp(latent, device=device).flatten(1).sum(1)
        # compute log(p(0)) = log(p(T)) + Tr(df/dx) where dx/dt = f
        return ll_prior + delta_ll, {'fevals': fevals}


def plt_llk(traj, lik, plot_type='scatter', ax=None):
    full_state_pred = traj.detach().squeeze().cpu().numpy()
    full_state_lik = lik.detach().squeeze().cpu().numpy()

    if plot_type == 'scatter':
        plt.scatter(full_state_pred, full_state_lik, color='blue')
    elif plot_type == 'line':
        idx = full_state_pred.argsort()
        sorted_state = full_state_pred[idx]
        sorted_lik = full_state_lik[idx]
        plt.plot(sorted_state, sorted_lik, color='red')
    elif plot_type == '3d_scatter':
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        for i, x in enumerate(torch.linspace(0., 1., full_state_pred.shape[-1])):
            xs = x.repeat(full_state_pred.shape[0]) if len(full_state_pred.shape) > 1 else x
            ys = full_state_pred[:, i] if len(full_state_pred.shape) > 1 else full_state_pred[i]
            zs = full_state_lik
            ax.scatter(xs=xs, ys=ys, zs=zs, color='blue')
        return ax

    elif plot_type == '3d_line':
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        idx = full_state_pred.argsort(0)
        xs = torch.linspace(0., 1., full_state_pred.shape[-1])
        for i, lik in enumerate(full_state_lik):
            ys = full_state_pred[i, :]
            zs = np.array(lik).repeat(full_state_pred.shape[1])
            ax.plot(xs=xs, ys=ys, zs=zs, color='red')

    plt.savefig('figs/scatter.pdf')

def test_gaussian(end_time, cfg, sample_trajs, std):
    analytical_llk = torch.distributions.Normal(
        cfg.example.mu, cfg.example.sigma
    ).log_prob(sample_trajs)
    print('analytical_llk: {}'.format(analytical_llk.squeeze()))
    ode_llk = std.ode_log_likelihood(sample_trajs)
    print('\node_llk: {}\node evals: {}'.format(ode_llk, ode_llk[1]))
    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
    plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_brownian_motion(end_time, cfg, sample_trajs, std):
    dt = end_time / (cfg.sde_steps-1)
    analytical_trajs = torch.cat([
        torch.zeros(sample_trajs.shape[0], 1, 1, device=sample_trajs.device),
        sample_trajs
    ], dim=1)

    analytical_llk = analytical_log_likelihood(analytical_trajs, SDE(cfg.sde_drift, cfg.sde_diffusion), dt)
    print('analytical_llk: {}'.format(analytical_llk))

    ode_llk = std.ode_log_likelihood(sample_trajs)
    print('\node_llk: {}'.format(ode_llk))

    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    if sample_trajs.shape[1] > 1:
        ax = plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='3d_scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='3d_line', ax=ax)
    else:
        plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()

def test_brownian_motion_diff(end_time, cfg, sample_trajs, std):
    dt = end_time / (cfg.sde_steps-1)
    analytical_trajs = torch.cat([
        torch.zeros(sample_trajs.shape[0], 1, 1, device=sample_trajs.device),
        sample_trajs.cumsum(dim=-2)
    ], dim=1)

    analytical_llk = analytical_log_likelihood(analytical_trajs, SDE(cfg.sde_drift, cfg.sde_diffusion), dt)
    print('analytical_llk: {}'.format(analytical_llk))

    ode_llk = std.ode_log_likelihood(sample_trajs)
    print('\node_llk: {}'.format(ode_llk))

    mse_llk = torch.nn.MSELoss()(analytical_llk.squeeze(), ode_llk[0])
    print('\nmse_llk: {}'.format(mse_llk))

    plt.clf()
    if sample_trajs.shape[1] > 1:
        ax = plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='3d_scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='3d_line', ax=ax)
    else:
        plt_llk(sample_trajs, ode_llk[0].exp(), plot_type='scatter')
        plt_llk(sample_trajs, analytical_llk.exp(), plot_type='line')
    import pdb; pdb.set_trace()


def test(end_time, cfg, out_trajs, std):
    if type(std.example) == GaussianExampleConfig:
        test_gaussian(end_time, cfg, out_trajs, std)
    elif type(std.example) == BrownianMotionExampleConfig:
        test_brownian_motion(end_time, cfg, out_trajs, std)
    elif type(std.example) == BrownianMotionDiffExampleConfig:
        test_brownian_motion_diff(end_time, cfg, out_trajs, std)
    else:
        raise NotImplementedError

def viz_trajs(cfg, std, out_trajs, end_time):
    if type(cfg.example) == BrownianMotionDiffExampleConfig:
        undiffed_trajs = out_trajs.cumsum(dim=-2)
        out_trajs = torch.cat([
            torch.zeros(undiffed_trajs.shape[0], 1, 1, device=undiffed_trajs.device),
            undiffed_trajs
        ], dim=1)
    for idx, out_traj in enumerate(out_trajs):
        std.viz_trajs(out_traj, end_time, idx, clf=False)


@hydra.main(version_base=None, config_path="conf", config_name="continuous_sample_config")
def sample(cfg):
    std = ContinuousEvaluator(cfg=cfg)
    end_time = torch.tensor(1., device=device)

    # cond_traj = None
    # rare_traj_file = 'rare_traj.pt'
    # rare_traj = torch.load(rare_traj_file).to(device)
    # std.viz_trajs(rare_traj, end_time, 100, clf=False, clr='red')
    # cond_traj = rare_traj.diff(dim=-1).reshape(1, -1, 1)

    sample_traj_out = std.sample_trajectories()

    print('fevals: {}'.format(sample_traj_out.fevals))
    sample_trajs = sample_traj_out.samples
    trajs = sample_trajs[-1]
    out_trajs = trajs

    viz_trajs(cfg, std, out_trajs, end_time)

    test(end_time, cfg, out_trajs, std)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        suppresswarning()

    cs = ConfigStore.instance()
    cs.store(name="vpsde_sample_config", node=SampleConfig)
    register_configs()

    with torch.no_grad():
        sample()
