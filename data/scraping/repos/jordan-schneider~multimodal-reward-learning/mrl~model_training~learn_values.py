import logging
import pickle as pkl
import re
from copy import deepcopy
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, cast

import fire  # type: ignore
import GPUtil  # type: ignore
import numpy as np  # type: ignore
import torch
from gym3 import ExtractDictObWrapper  # type: ignore
from mrl.dataset.offline_buffer import RlDataset
from mrl.dataset.roller import procgen_rollout
from linear_procgen.util import ENV_NAMES, make_env
from mrl.model_training.online_batcher import BatchGenerator
from mrl.model_training.writer import SequentialWriter
from mrl.util import find_best_gpu, find_policy_path, reinit, setup_logging
from phasic_policy_gradient.ppg import PhasicValueModel
from procgen import ProcgenGym3Env
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange  # type: ignore


class Checkpointer:
    def __init__(self, path: Path, name: str, extension: str) -> None:
        self.path = Path(path)
        self.name = name
        self.extension = extension

    def save(self, model: torch.nn.Module, counter: int) -> None:
        torch.save(model, self.path / f"{self.name}.{counter}.{self.extension}")

    def load(self, overwrite: bool = False) -> Tuple[Optional[torch.nn.Module], int]:
        models = list(self.path.glob(f"{self.name}.[0-9]*.{self.extension}"))
        if len(models) == 0 or overwrite:
            return None, 0

        model_iters = [
            re.search(f"{self.name}\.([0-9]+)\.jd", model.name).group(1)  # type: ignore
            for model in models
        ]
        model_iter = max(model_iters)
        model_path = models[np.argmax(model_iters)]
        logging.info(f"Loading Q model from {model_path}")
        latest_model = cast(torch.nn.Module, torch.load(model_path))

        return latest_model, int(model_iter)


class QNetwork(torch.nn.Module):
    def __init__(
        self,
        policy: PhasicValueModel,
        n_actions: int,
        discount_rate: float,
        activation: Literal["relu", "leaky", "elu"] = "relu",
        value_init: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert discount_rate >= 0.0 and discount_rate <= 1.0
        self.discount_rate = discount_rate

        self.enc = deepcopy(policy.get_encoder(policy.true_vf_key))
        self.enc.cnn.set_activation(activation)
        if not value_init:
            reinit(self.enc)

        self.head = self.add_action_heads(
            n_actions=n_actions,
            value_head=policy.get_vhead(policy.true_vf_key),
            copy_weights=value_init,
        )

        self.device = policy.device if device is None else device
        self.to(device=self.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device)
        action = action.to(device=self.device)
        q_values = self.get_action_values(obs)
        if q_values.shape[1] == 1:
            # Gather freaks out if the length of the dim you're gathering along is 1
            return q_values
        out = q_values.gather(dim=1, index=action.view(-1, 1)).reshape(-1)
        return out

    def get_action_values(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[1:] == (64, 64, 3)

        # ImpalaEncoder expects (batch, time, h, w, c)
        obs = obs.reshape((1, *obs.shape))

        z = self.enc.stateless_forward(obs)
        q_values = self.head(z)[0]
        return q_values

    def state_value(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(device=self.device)
        q_values = self.get_action_values(obs)
        v, _ = q_values.max(dim=1)
        return v

    @staticmethod
    def add_action_heads(
        n_actions: int, value_head: torch.nn.Linear, copy_weights: bool = False
    ) -> torch.nn.Linear:
        """Takes a state value head and copies it to n_action state-action value heads.

        Args:
            n_actions (int): Size of the action space
            value_head (torch.nn.Linear): An (-1, 1) shaped linear layer.

        Returns:
            torch.nn.Linear: An (-1, n_actions) shaped linear layer with copied weights.
        """
        assert value_head.out_features == 1

        out = type(value_head)(value_head.in_features, n_actions)
        if copy_weights:
            out.weight.data[:] = value_head.weight

        return out


def train_q(
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RlDataset,
    val_period: int,
    checkpoint_period: int,
    checkpointer: Checkpointer,
    target_mixing_coeff: float,
    q: QNetwork,
    optim: torch.optim.Optimizer,
    writer: SequentialWriter,
) -> QNetwork:
    q_target = deepcopy(q)
    q_target.eval()

    val_counter = 0
    val_log_step = 0
    for batch_counter in trange(n_train_steps // batch_size):
        batch = batch_gen.make_sars_batch(timesteps=batch_size)

        states, actions, rewards, next_states = batch.make_sars()
        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        loss = torch.zeros(1)

        if n > 0:
            q_pred = q.forward(torch.tensor(states), torch.tensor(actions)).cpu()
            writer.add_histogram("train/q_pred", q_pred)

            with torch.no_grad():
                v_next = q_target.state_value(torch.tensor(next_states)).cpu()
                q_targ = rewards + q.discount_rate * v_next
                writer.add_histogram("train/q_targ", q_targ)
            assert q_pred.shape == (n,), f"q_pred={q_pred.shape} not expected ({n})"
            assert q_targ.shape == (n,), f"q_targ={q_targ.shape} not expected ({n})"
            loss += torch.sum((q_pred - q_targ) ** 2)

        # TODO: Update to use firsts intsead of dones
        final_states = batch.states[:-1][batch.dones]
        if len(final_states) > 0:

            if __debug__:
                pos_states = torch.sum(final_states[:, 0, 0, 0] > 0)
                neg_states = torch.sum(final_states[:, 0, 0, 0] < 0)
                assert pos_states + neg_states == len(final_states)

            q_pred_final = q.state_value(final_states).cpu()
            writer.add_histogram("train/q_pred_final", q_pred_final)
            # TODO: Update to use firsts intsead of dones
            q_targ_final = batch.rewards[:-1][batch.dones]
            writer.add_histogram("train/q_targ_final", q_targ_final)
            loss += torch.sum((q_pred_final - q_targ_final) ** 2)

        writer.add_scalar("train/loss", loss)

        loss.backward()
        optim.step()

        val_counter += len(batch.states)

        if val_counter > val_period:
            val_counter = 0
            val_loss = eval_q_rmse(
                q_fn=q.forward,
                data=val_data,
                discount_rate=q.discount_rate,
                writer=writer,
            )
            writer.add_scalar("val/rmse", val_loss)
            val_log_step += 1

        with torch.no_grad():
            # Stolen from OpenAI spinning up SAC implementation
            for p, p_targ in zip(q.parameters(), q_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(target_mixing_coeff)
                p_targ.data.add_((1 - target_mixing_coeff) * p.data)

        if batch_counter % checkpoint_period == 0:
            checkpointer.save(model=q, counter=batch_counter)

    return q


def train_q_trunc_returns(
    horizon: int,
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RlDataset,
    val_period: int,
    q: QNetwork,
    optim: torch.optim.Optimizer,
    writer: SequentialWriter,
) -> QNetwork:
    val_counter = 0
    n_batches = n_train_steps // batch_size
    for _ in trange(n_batches):
        states, actions, partial_returns = batch_gen.make_trunc_return_batch(
            timesteps=batch_size, horizon=horizon, discount_rate=q.discount_rate
        )

        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        q_pred = q.forward(
            torch.tensor(states).to(device=q.device),
            torch.tensor(actions).to(device=q.device),
        ).cpu()
        assert q_pred.shape == (n,), f"q_pred={q_pred.shape} not expected ({n})"
        loss = torch.sum((q_pred - partial_returns) ** 2)

        writer.add_scalar("train/loss", loss)

        loss.backward()
        optim.step()

        val_counter += n

        if val_counter > val_period:
            val_counter = 0
            val_loss = eval_q_partial_rmse(
                q_fn=q.forward,
                data=val_data,
                k=horizon,
                discount_rate=q.discount_rate,
                device=q.device,
            )
            writer.add_scalar("val/rmse", val_loss)

    return q


def get_rollouts(
    env: ProcgenGym3Env,
    val_env_steps: int,
    policy: PhasicValueModel,
    datadir: Path,
    overwrite: bool,
) -> RlDataset:
    val_rollouts_path = datadir / "val_rollouts.pkl"

    val_data: Optional[RlDataset] = None

    if not overwrite and val_rollouts_path.exists():
        val_data = cast(RlDataset, pkl.load(val_rollouts_path.open("rb")))

        val_missing = val_env_steps - len(val_data)
    else:
        val_missing = val_env_steps

    if val_missing > 0:
        states, actions, rewards, firsts = procgen_rollout(
            env, policy, val_missing, tqdm=True
        )
        if val_data is not None:
            val_data.append_gym3(
                states=states, actions=actions, rewards=rewards, firsts=firsts
            )
        else:
            val_data = RlDataset.from_gym3(
                states=states, actions=actions, rewards=rewards, firsts=firsts
            )

        pkl.dump(val_data, open(datadir / "val_rollouts.pkl", "wb"))

    assert val_data is not None

    return val_data


def learn_q(
    indir: Path,
    outdir: Path,
    env_name: ENV_NAMES = "miner",
    value_init: bool = False,
    lr: float = 10e-3,
    discount_rate: float = 0.999,
    decay_rate: float = 0.01,
    batch_size: int = 64,
    train_env_steps: int = 10_000_000,
    val_env_steps: int = 100_000,
    val_period: int = 2000 * 10,
    checkpoint_period: int = 10_000,
    target_mixing_coeff: float = 0.999,
    activation: Literal["relu", "leaky", "elu"] = "relu",
    trunc_returns: bool = False,
    trunc_horizon: Optional[int] = None,
    overwrite_validation: bool = False,
    overwrite_model: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
) -> None:
    if trunc_returns:
        assert (
            trunc_horizon is not None
        ), f"Must specify a truncation horizon to use truncated returns."

    setup_logging(level=verbosity)

    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_path, policy_iter = find_policy_path(indir / "models")
    device = find_best_gpu()
    policy = torch.load(policy_path, map_location=device)
    policy.device = device

    model_outdir = outdir / "value"
    model_outdir.mkdir(parents=True, exist_ok=True)
    model_name = (
        f"q_model_{policy_iter}"
        if not trunc_returns
        else f"q_model_trunc_{policy_iter}"
    )
    checkpointer = Checkpointer(path=model_outdir, name=model_name, extension="jd")
    model_path = model_outdir / (model_name + ".jd")

    env = make_env(env_name, num=1)

    if not overwrite_model and model_path.exists():
        q = cast(QNetwork, checkpointer.load()[0])
        assert (
            q is not None
        ), f"Cannot load q network despite something being at model_path={model_path}"
    else:
        q = QNetwork(
            policy,
            n_actions=env.ac_space.eltype.n,
            discount_rate=discount_rate,
            value_init=value_init,
            activation=activation,
        )

    val_data = get_rollouts(
        env=env,
        val_env_steps=val_env_steps,
        policy=policy,
        datadir=outdir,
        overwrite=overwrite_validation,
    )

    optim = torch.optim.Adam(q.parameters(), lr=lr, weight_decay=decay_rate)

    writer = SequentialWriter(SummaryWriter(log_dir=outdir / "logs" / "refine_q"))

    if trunc_returns:
        assert (
            trunc_horizon is not None
        ), "Must specify a truncation horizon if using truncated returns"
        q = train_q_trunc_returns(
            horizon=trunc_horizon,
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            q=q,
            optim=optim,
            writer=writer,
        )
    else:
        q = train_q(
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            checkpoint_period=checkpoint_period,
            checkpointer=checkpointer,
            target_mixing_coeff=target_mixing_coeff,
            q=q,
            optim=optim,
            writer=writer,
        )

    torch.save(q, model_path)


def compute_returns(
    rewards: np.ndarray, discount_rate: float, use_conv: bool = False
) -> np.ndarray:
    assert discount_rate >= 0.0 and discount_rate <= 1.0

    # TODO(joschnei): Benchmark this
    if use_conv:
        discounts = np.array([discount_rate ** i for i in range(len(rewards))])
        # TODO(joschnei): There must be a better way to do a 1d vector convolution
        returns = np.array([rewards[i:] @ discounts[:-i] for i in range(len(rewards))])
    else:
        returns = np.empty(len(rewards))
        current_return = 0
        for i, reward in enumerate(reversed(rewards)):  # type: ignore
            current_return = current_return * discount_rate + reward
            returns[-i] = current_return

    return returns


@torch.no_grad()
def eval_q_rmse(
    q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: RlDataset,
    discount_rate: float,
    writer: SequentialWriter,
) -> float:
    loss = 0.0
    for traj in data.trajs(include_last=False):
        assert (
            traj.states is not None
            and traj.actions is not None
            and traj.rewards is not None
        )
        assert len(traj.states) > 0, "0 states in this traj"
        assert len(traj.actions) > 0, "0 actions in this traj"
        assert len(traj.rewards) > 0, "0 rewards in this traj"
        values = (
            q_fn(torch.tensor(traj.states), torch.tensor(traj.actions)).detach().cpu()
        )
        writer.add_histogram("val/q_pred", values)
        returns = compute_returns(traj.rewards, discount_rate)
        writer.add_histogram("val/returns", returns)

        errors = values - returns
        writer.add_histogram("val/q_error", errors)
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
        if not np.isfinite(loss):
            logging.warning(
                f"NaN validation loss. values={values}, returns={returns}, errors={errors}, rewards{traj.rewards}, actions={traj.actions}"
            )
            raise ValueError("NaN validation loss")
    return loss


@torch.no_grad()
def eval_q_partial_rmse(
    q_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    data: RlDataset,
    k: int,
    discount_rate: float,
    device: torch.device,
) -> float:
    states, actions, partial_returns = data.truncated_returns(
        horizon=k, discount_rate=discount_rate
    )

    loss = 0.0
    for state_batch, action_batch, return_batch in zip(
        np.array_split(states, len(states) // 100),
        np.array_split(actions, len(actions) // 100),
        np.array_split(partial_returns, len(partial_returns) // 100),
    ):
        values = (
            q_fn(state_batch.to(device=device), action_batch.to(device=device))
            .detach()
            .cpu()
        )
        errors = values - return_batch
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
    return loss


def eval_q(
    datadir: Path, discount_rate: float = 0.999, env_interactions: int = 1_000_000
) -> None:
    datadir = Path(datadir)
    policy_path, iter = find_policy_path(datadir / "models")
    q_path = datadir / f"q_model_{iter}.jd"

    policy = cast(PhasicValueModel, torch.load(policy_path))
    q = cast(QNetwork, torch.load(q_path))

    env = ProcgenGym3Env(1, "miner")
    env = ExtractDictObWrapper(env, "rgb")

    writer = SequentialWriter(SummaryWriter(log_dir=datadir / "logs" / "eval_q"))

    logging.info("Gathering environment interactions")
    data = RlDataset.from_gym3(
        *procgen_rollout(env, policy, env_interactions, tqdm=True)
    )
    pkl.dump(data, open(datadir / "eval_rollouts.pkl", "wb"))

    logging.info("Evaluating loss")
    loss = eval_q_rmse(q.forward, data, discount_rate, writer=writer)

    logging.info(f"Loss={loss} over {env_interactions} env timesteps")


def refine_v(
    indir: Path,
    outdir: Path,
    env_name: ENV_NAMES = "miner",
    lr: float = 10e-3,
    discount_rate: float = 0.999,
    batch_size: int = 64,
    train_env_steps: int = 10_000_000,
    val_env_steps: int = 100_000,
    val_period: int = 2000 * 10,
    trunc_returns: bool = False,
    trunc_horizon: Optional[int] = None,
    overwrite: bool = False,
    verbosity: Literal["INFO", "DEBUG"] = "INFO",
):
    if trunc_returns:
        assert (
            trunc_horizon is not None
        ), f"Must specify a truncation horizon to use truncated returns."

    setup_logging(level=verbosity)

    indir = Path(indir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    policy_path, policy_iter = find_policy_path(indir / "models")
    device_id = GPUtil.getFirstAvailable(order="load")[0]
    device = torch.device(f"cuda:{device_id}")
    policy = torch.load(policy_path, map_location=device)
    policy.device = device

    # Freeze the non-value parameters
    for param in policy.get_encoder("pi").cnn.parameters():
        param.requires_grad = False
    for param in policy.pi_head.parameters():
        param.requires_grad = False

    model_outdir = outdir / "value"
    model_outdir.mkdir(parents=True, exist_ok=True)
    model_outname = (
        f"v_model_{policy_iter}.jd"
        if not trunc_returns
        else f"v_model_trunc_{policy_iter}.jd"
    )
    outpath = model_outdir / model_outname

    env = make_env(env_name, 1)

    val_data = get_rollouts(
        env=env,
        val_env_steps=val_env_steps,
        policy=policy,
        datadir=model_outdir,
        overwrite=overwrite,
    )

    optim = torch.optim.Adam(policy.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=outdir / "logs")

    if trunc_returns:
        assert (
            trunc_horizon is not None
        ), "Must specify a truncation horizon if using truncated returns"
        policy = train_v_trunc_returns(
            horizon=trunc_horizon,
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            v=policy,
            discount_rate=discount_rate,
            optim=optim,
            writer=writer,
        )
    else:
        policy = train_v(
            batch_gen=BatchGenerator(env=env, policy=policy),
            n_train_steps=train_env_steps,
            batch_size=batch_size,
            val_data=val_data,
            val_period=val_period,
            v=policy,
            discount_rate=discount_rate,
            optim=optim,
            writer=writer,
        )

    torch.save(policy, outpath)


def train_v_trunc_returns(
    horizon: int,
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RlDataset,
    val_period: int,
    v: PhasicValueModel,
    discount_rate: float,
    optim: torch.optim.Optimizer,
    writer: SummaryWriter,
) -> PhasicValueModel:
    val_counter = 0
    val_step = 0
    train_step = 0
    n_batches = n_train_steps // batch_size
    for _ in trange(n_batches):
        states, _, partial_returns = batch_gen.make_trunc_return_batch(
            timesteps=batch_size, horizon=horizon, discount_rate=discount_rate
        )

        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        v_pred = v.value(torch.tensor(states).to(device=v.device)).cpu()
        assert v_pred.shape == (n,), f"v_pred={v_pred.shape} not expected ({n})"
        loss = torch.sum((v_pred - partial_returns) ** 2)

        writer.add_scalar("train/loss", loss, global_step=train_step)

        loss.backward()
        optim.step()

        train_step += 1
        val_counter += n

        if val_counter > val_period:
            val_counter = 0
            val_loss = eval_v_partial_rmse(
                v_fn=v.value,
                data=val_data,
                k=horizon,
                discount_rate=discount_rate,
                device=v.device,
            )
            writer.add_scalar("val/rmse", val_loss, global_step=val_step)
            val_step += 1

    return v


@torch.no_grad()
def eval_v_partial_rmse(
    v_fn: Callable[[torch.Tensor], torch.Tensor],
    data: RlDataset,
    k: int,
    discount_rate: float,
    device: torch.device,
) -> float:
    states, _, partial_returns = data.truncated_returns(
        horizon=k, discount_rate=discount_rate
    )

    loss = 0.0
    for state_batch, return_batch in zip(
        np.array_split(states, len(states) // 100),
        np.array_split(partial_returns, len(partial_returns) // 100),
    ):
        values = v_fn(state_batch.to(device=device)).detach().cpu()
        errors = values - return_batch
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
    return loss


def train_v(
    batch_gen: BatchGenerator,
    n_train_steps: int,
    batch_size: int,
    val_data: RlDataset,
    val_period: int,
    v: PhasicValueModel,
    discount_rate: float,
    optim: torch.optim.Optimizer,
    writer: SummaryWriter,
) -> PhasicValueModel:
    val_counter = 0
    val_step = 0
    train_step = 0
    n_batches = n_train_steps // batch_size
    for _ in trange(n_batches):
        batch = batch_gen.make_sars_batch(timesteps=batch_size)

        states, _, rewards, next_states = batch.make_sars()

        # n is not batch_size because batch_size actions generate batch_size - # dones - 1
        # usable transitions
        n = len(states)

        optim.zero_grad()
        v_pred = v.value(torch.tensor(states)).cpu()

        with torch.no_grad():
            v_next = v.value(torch.tensor(next_states)).cpu()
        v_targ = rewards + discount_rate * v_next
        assert v_pred.shape == (n,), f"v_pred={v_pred.shape} not expected ({n})"
        assert v_targ.shape == (n,), f"v_targ={v_targ.shape} not expected ({n})"
        loss = torch.sum((v_pred - v_targ) ** 2)

        # TODO: Update to use firsts intsead of dones
        v_pred_final = v.value(batch.states[batch.dones]).cpu()
        v_targ_final = batch.rewards[batch.dones].cpu()
        loss += torch.sum((v_pred_final - v_targ_final) ** 2)

        writer.add_scalar("train/loss", loss, global_step=train_step)

        loss.backward()
        optim.step()

        train_step += 1
        val_counter += n

        if val_counter > val_period:
            val_counter = 0
            val_loss = eval_v_rmse(
                v_fn=v.value,
                data=val_data,
                discount_rate=discount_rate,
                device=v.device,
            )
            writer.add_scalar("val/rmse", val_loss, global_step=val_step)
            val_step += 1

    return v


@torch.no_grad()
def eval_v_rmse(
    v_fn: Callable[[torch.Tensor], torch.Tensor],
    data: RlDataset,
    discount_rate: float,
    device: torch.device,
) -> float:
    loss = 0.0
    for traj in data.trajs(include_last=False):
        assert traj.states is not None and traj.rewards is not None
        values = v_fn(torch.tensor(traj.states[:-1]).to(device=device)).detach().cpu()
        returns = compute_returns(traj.rewards, discount_rate)[:-1]

        errors = values - returns
        loss += torch.sqrt(torch.mean(errors ** 2)).item()
    return loss


if __name__ == "__main__":
    fire.Fire({"q": learn_q, "v": refine_v, "eval": eval_q})
