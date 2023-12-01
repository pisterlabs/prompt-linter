from typing import Optional

import distrax
import haiku as hk
import jax
import jax.numpy as jnp


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return distrax.Categorical(logits=logits), value


class ContinuousValueHead(hk.Module):
    """Network head that produces a continuous distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
        mean_activation: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.mean_action = mean_activation
        self._mean_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=False,
        )
        self._scale_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Orthogonal(0.01),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Orthogonal(1.0),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        if self.mean_action == "sigmoid":
            means = jax.nn.sigmoid(self._mean_layer(inputs))
        else:
            means = self._mean_layer(inputs)
        scales = self._scale_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        scales = jnp.maximum(scales, 0.01)
        return (
            distrax.MultivariateNormalDiag(loc=means, scale_diag=scales),
            value,
        )


class CNN(hk.Module):
    def __init__(self, args):
        super().__init__(name="CNN")
        output_channels = args.ppo.output_channels
        kernel_shape = args.ppo.kernel_shape
        self.conv_a_0 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.conv_a_1 = hk.Conv2D(
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=1,
            padding="SAME",
        )
        self.linear_a_0 = hk.Linear(output_channels)

        self.flatten = hk.Flatten()

    def __call__(self, inputs: jnp.ndarray):
        # Actor and Critic
        x = self.conv_a_0(inputs)
        x = jax.nn.relu(x)
        x = self.conv_a_1(x)
        x = jax.nn.relu(x)
        x = self.flatten(x)
        x = self.linear_a_0(x)
        x = jax.nn.relu(x)

        return x


def make_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                CategoricalValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_rice_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                ContinuousValueHead(num_values=num_actions),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_coingame_network(num_actions: int, args):
    def forward_fn(inputs):
        layers = []
        if args.ppo.with_cnn:
            cnn = CNN(args)
            cvh = CategoricalValueHead(num_values=num_actions)
            layers.extend([cnn, cvh])
        else:
            layers.extend(
                [
                    hk.nets.MLP(
                        [64, 64],
                        w_init=hk.initializers.Orthogonal(jnp.sqrt(2)),
                        b_init=hk.initializers.Constant(0),
                        activate_final=True,
                    ),
                    CategoricalValueHead(num_values=num_actions),
                ]
            )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network
