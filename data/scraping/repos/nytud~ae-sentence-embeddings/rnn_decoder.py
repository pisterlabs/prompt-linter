"""A module for RNN decoder layers as an alternative of a GPT decoder
This module also contains the complex `AeTransformerGRUDecoder` layer
that combines Transformer and GRU layers
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
from copy import deepcopy
from types import MappingProxyType

import tensorflow as tf
from tensorflow.keras import layers as tfl
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.activations import serialize as act_serialize
from transformers import OpenAIGPTConfig

from ae_sentence_embeddings.layers.transformer_ae_layers import AeTransformerDecoder
from ae_sentence_embeddings.argument_handling import RnnLayerArgs


class AeGRUDecoder(tfl.Layer):
    """A GRU-based decoder"""

    def __init__(
            self,
            num_rnn_layers: int = 2,
            hidden_size: int = 768,
            layernorm_eps: float = 1e-12,
            dropout_rate: float = 0.1,
            **kwargs
    ) -> None:
        """Initialize the GRU decoder.

        Args:
            num_rnn_layers: Number of RNN layers. Defaults to `2`.
            hidden_size: Hidden size in the RNN and dense layers. Defaults to `768`.
            layernorm_eps: Layer normalization epsilon parameter. Defaults to `1e-12`.
            dropout_rate: A dropout rate between 0 and 1. `Defaults to 0.1`.
            **kwargs: Parent class keyword arguments.
        """
        super().__init__(**kwargs)
        self._num_rnn_layers = num_rnn_layers
        self._hidden_size = hidden_size
        self._layernorm_eps = layernorm_eps
        self._dropout_rate = dropout_rate
        self._decoder_dense = tfl.Dense(self._hidden_size, activation="tanh",
                                        name="ae_decoder_dense")
        self._rnn = [tfl.GRU(
            units=self._hidden_size,
            recurrent_dropout=self._dropout_rate,
            return_sequences=True,
            return_state=True,
            name=f"ae_GRU_{i}"
        ) for i in range(self._num_rnn_layers)]
        self._dropout = tfl.Dropout(self._dropout_rate)
        self._layernorm_dense = tfl.LayerNormalization(epsilon=self._layernorm_eps)
        self._layernorm_out = tfl.LayerNormalization(epsilon=self._layernorm_eps)

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the model

        Args:
            inputs: A tuple of two tensors: the initial hidden state tensor of size `(batch_size, hidden_size)`
                and an embedded input tensor of shape `(batch_size, sequence_length, hidden_size)`.
            training: Specify whether the layer is being used in training mode.

        Returns:
            A tensor of shape `(batch_size, sequence_length, hidden_size)`.

        """
        hidden_state, embeddings = inputs
        embeddings = self._layernorm_dense(self._decoder_dense(embeddings))
        hidden_states = [hidden_state]
        for rnn_layer in self._rnn:
            embeddings, hidden_state = rnn_layer(
                inputs=embeddings,
                initial_state=tf.reduce_mean(hidden_states, axis=0),
                training=training
            )
            embeddings = self._layernorm_dense(embeddings)
            hidden_states.append(hidden_state)
        embeddings = self._dropout(embeddings, training=training)
        return self._layernorm_out(embeddings)

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "num_rnn_layers": self._num_rnn_layers,
            "hidden_size": self._hidden_size,
            "layernorm_eps": self._layernorm_eps,
            "dropout_rate": self._dropout_rate
        }

    @property
    def hidden_size(self) -> int:
        return self._hidden_size


class AeGRUCellDecoder(tfl.Layer):
    """An GRU-based AE decoder that does not use teacher forcing"""

    def __init__(
            self,
            num_rnn_layers: int = 2,
            hidden_size: int = 768,
            layernorm_eps: float = 1e-12,
            dropout_rate: float = 0.1,
            **kwargs) -> None:
        """

        Args:
            num_rnn_layers: Number of RNN layers. Defaults to 2
            hidden_size: Hidden size in the RNN and dense layers. Defaults to 768
            layernorm_eps: Layer normalization epsilon parameter. Defaults to 1e-12
            dropout_rate: A dropout rate between 0 and 1. that will be applied to the outputs of
                          the dense layer. Defaults to 0.1
            **kwargs: Parent class keyword arguments
        """
        super().__init__(**kwargs)
        if num_rnn_layers < 1:
            raise ValueError("At least 1 RNN layer must be used!")
        self._num_rnn_layers = num_rnn_layers
        self._hidden_size = hidden_size
        self.layernorm_eps = layernorm_eps
        self.dropout_rate = dropout_rate
        self.rnn_layers = [tfl.GRUCell(self._hidden_size) for _ in range(self._num_rnn_layers)]
        # `GRUCell` is mutable, `[tfl.GRUCell(...)] * self._num_rnn_layers` should NOT be used!
        self.dense_layers = [tfl.Dense(
            units=self._hidden_size,
            activation="tanh",
            kernel_initializer="glorot_uniform",
            input_shape=(None, None, self._hidden_size)
        ) for _ in range(self._num_rnn_layers)]
        self.dropout = tfl.Dropout(self.dropout_rate)
        self.layernorm = tfl.LayerNormalization(epsilon=self.layernorm_eps)

    def _process_sequence(
            self,
            rnn_cell: tfl.GRUCell,
            dense: tfl.Dense,
            pre_inputs: tf.Tensor,
            hidden_states: tf.Tensor,
            training: Optional[bool] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Helper function: process a time series with an RNN cell without teacher forcing

        Args:
            rnn_cell: An RNN cell
            dense: A dense layer
            pre_inputs: A tensor of shape `(batch_size, timesteps, hidden_size)`. These inputs will be added
                        to the RNN cell output of time step `t-1` to get the RNN input of time step `t`
            hidden_states: Hidden state tensor of shape `(batch_size, hidden_size)`
            training: Specify whether dropout is being used in training mode. Defaults to `None`

        Returns:
            RNN output tensor of shape `(batch_size, timesteps, hidden_size)` and a hidden state tensor
            of shape `(batch_size, hidden_size)`
        """
        num_timesteps = tf.shape(pre_inputs)[1]
        outputs = []
        step_output = self.dropout(dense(hidden_states), training=training)
        for i in tf.range(num_timesteps):
            step_input = tf.add(step_output, pre_inputs[i])
            pre_step_output, hidden_states = rnn_cell(step_input, hidden_states)
            step_output = self.dropout(dense(pre_step_output), training=training)
            outputs.append(step_output)
        outputs = self.layernorm(tf.transpose(outputs, [1, 0, 2]))
        return outputs, hidden_states

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor],
             training: Optional[bool] = None) -> tf.Tensor:
        """Call the layer

        Args:
            inputs: An input embedding tensor of shape `(batch_size, hidden_size)` and
                    a tensor of shape `(batch_size, timesteps, hidden_size)` which will be used
                    as initial RNN input tensor. This can be a zero tensor.
            training: Specify whether the model is being used in training mode

        Returns:
            Activations of shape `(batch_size, timesteps, hidden_size)`
        """
        hidden_states, rnn_x = inputs
        for rnn, dense in zip(self.rnn_layers, self.dense_layers):
            rnn_x, hidden_states = self._process_sequence(rnn, dense, rnn_x, hidden_states,
                                                          training=training)
        return rnn_x

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        return {
            **base_config,
            "num_rnn_layers": self._num_rnn_layers,
            "hidden_size": self._hidden_size,
            "layernorm_eps": self.layernorm_eps,
            "dropout_rate": self.dropout_rate
        }


class AeTransformerGRUDecoder(tfl.Layer):
    """A Transformer + GRU decoder"""

    def __init__(
            self,
            transformer_config: OpenAIGPTConfig,
            gru_config: RnnLayerArgs,
            num_transformer2gru: int = 2,
            **kwargs
    ) -> None:
        """Initialize the layer

        Args:
            transformer_config: Transformer configuration arguments passed to `AeTransformerGRUDecoder`
            gru_config: GRU configuration arguments passed to `AeGRUDecoder`
            num_transformer2gru: Number of dense layers between the Transformer and
                the GRU. Defaults to `2`
            **kwargs: Parent class keyword arguments
        """
        if not isinstance(num_transformer2gru, int) or num_transformer2gru <= 0:
            raise ValueError(
                f"Argument `num_transformer2gru` must be a positive integer, got {num_transformer2gru}")
        super().__init__(**kwargs)
        self._transformer_config = deepcopy(transformer_config)
        self._gru_config = deepcopy(gru_config)
        self._num_transformer2gru = num_transformer2gru
        self._transformer = AeTransformerDecoder(self._transformer_config)
        self._gru = AeGRUDecoder(**gru_config.to_dict())
        self._transformer2gru = [
            tfl.Dense(
                self._gru_config.hidden_size,
                kernel_initializer=TruncatedNormal(stddev=self._transformer_config.initializer_range),
                activation=self._transformer_config.afn,
                name=f"transformer2gru_dense_{i}"
            ) for i in range(num_transformer2gru)
        ]
        self._intermediate_layernorm = tfl.LayerNormalization(
            epsilon=self._transformer_config.layer_norm_epsilon)
        self._intermediate_dropout = tfl.Dropout(self._transformer_config.resid_pdrop)

    # noinspection PyCallingNonCallable
    def call(
            self,
            inputs: Tuple[tf.Tensor, tf.Tensor],
            training: Optional[bool] = None
    ) -> tf.Tensor:
        """Call the layer

        Args:
            inputs: A sentence embedding tensor of shape `(batch_size, gru_hidden_size)` and
                an embedding tensor of shape `(batch_size, sequence_length, transformer_hidden_size)`.
                Note that no attention mask is required because the transformer layers perform causal masking.
            training: specifies whether the model is being used in training mode

        Returns:
            A tensor of shape `(batch_size, sequence_length, gru_hidden_size)`
        """
        sent_embeddings, tok_embeddings = inputs
        tok_embeddings = self._transformer((tok_embeddings, None), training=training)
        for dense_layer in self._transformer2gru:
            tok_embeddings = dense_layer(tok_embeddings)
        tok_embeddings = self._intermediate_layernorm(
            self._intermediate_dropout(tok_embeddings, training=training))
        return self._gru((sent_embeddings, tok_embeddings))

    def get_config(self) -> Dict[str, Any]:
        """Get config for Keras serialization"""
        base_config = super().get_config()
        return {
            **base_config,
            "transformer_config_dict": self._transformer_config.to_dict(),
            "gru_config_dict": self._gru_config.to_dict(),
            "num_transformer2gru": self._num_transformer2gru
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> AeTransformerGRUDecoder:
        """Initialize the object from a config dict"""
        config = deepcopy(config)
        transformer_config = OpenAIGPTConfig(**config.pop("transformer_config_dict"))
        gru_config = RnnLayerArgs(**config.pop("gru_config_dict"))
        return cls(transformer_config, gru_config, **config)

    @property
    def transformer_config(self) -> MappingProxyType:
        return MappingProxyType(self._transformer_config.to_dict())

    @property
    def gru_config(self) -> MappingProxyType:
        return MappingProxyType(self._gru_config.to_dict())

    @property
    def intermediate_mapping_config(self) -> MappingProxyType:
        """Get info on the dense layers between the transformer and the GRU"""
        return MappingProxyType({
            "num_layers": self._num_transformer2gru,
            "dropout_rate": self._intermediate_dropout.rate,
            "layernorm_eps": self._intermediate_layernorm.epsilon,
            "initializer_dev": self._transformer2gru[0].kernel_initializer.stddev,
            "activation": act_serialize(self._transformer2gru[0].activation)
        })

    @property
    def num_transformer2gru(self) -> int:
        return self._num_transformer2gru
