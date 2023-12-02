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
            w_init=hk.initializers.Constant(0),
            # w_init=hk.initializers.RandomNormal(),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0),
            # w_init=hk.initializers.RandomNormal(),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = self._logit_layer(inputs)
        # logits = jax.nn.sigmoid(self._logit_layer(inputs))
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Categorical(logits=logits), value)


class BernoulliValueHead(hk.Module):
    """Network head that produces a categorical distribution and value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Constant(0),
            with_bias=False,
        )
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        # matching the way that they do it.
        logits = jnp.squeeze(self._logit_layer(inputs), axis=-1)
        probs = jax.nn.sigmoid(logits)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (distrax.Bernoulli(probs=1 - probs), value)


class PolicyHead(hk.Module):
    """Network head that produces a categorical distribution."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._logit_layer = hk.Linear(
            num_values,
            w_init=hk.initializers.Constant(0),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):
        logits = jnp.squeeze(self._logit_layer(inputs), axis=-1)
        probs = jax.nn.sigmoid(logits)
        return distrax.Bernoulli(probs=1 - probs)


class ValueHead(hk.Module):
    """Network head that produces a value."""

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._value_layer = hk.Linear(
            1,
            w_init=hk.initializers.Constant(0),
            with_bias=False,
        )

    def __call__(self, inputs: jnp.ndarray):

        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return value


def make_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                CategoricalValueHead(num_values=num_actions),
                # BernoulliValueHead(num_values=1),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_policy_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                PolicyHead(num_values=1),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network


def make_value_network(num_actions: int):
    """Creates a hk network using the baseline hyperparameters from OpenAI"""

    def forward_fn(inputs):
        layers = []
        layers.extend(
            [
                ValueHead(num_values=1),
            ]
        )
        policy_value_network = hk.Sequential(layers)
        return policy_value_network(inputs)

    network = hk.without_apply_rng(hk.transform(forward_fn))
    return network
