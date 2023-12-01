"""Network interface implementation using the TF v1 framework."""

import warnings

import gin
import numpy as np
import tensorflow as tf

from alpacka.networks import core


class TFMetaGraphNetwork(core.Network):
    """Fixed network loaded from the TF v1 MetaGraph checkpoint."""

    def __init__(
        self,
        network_signature,
        model_path=gin.REQUIRED,
        x_name='ppo2_model/Ob:0',
        y_name='ppo2_model/pi_1/add:0',
        device='/device:CPU:*',
    ):
        """Initialize TF session from MetaGraph.

        Args:
            network_signature (NetworkSignature): Network signature.
            model_path (str): Path to a saved model. It's a common part of
                the three files with extensions: .meta, .index, .data.
            x_name (str): Name of the input placeholder.
                Default for PPO2 from OpenAI Baselines.
            y_name (str): Name of the output tensor.
                Default for PPO2 from OpenAI Baselines.
            device (str or None): Device to move all ops in the graph to. By
                default, move to CPU. If None, keep them where they are.
        """
        super().__init__(network_signature)

        # Recall to legacy execution engine and create a tf.Session.
        tf.compat.v1.disable_eager_execution()
        self._sess = tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(self._sess)

        # Import meta graph and restore checkpoint.
        self._saver = tf.compat.v1.train.import_meta_graph(model_path + '.meta')

        graph = tf.compat.v1.get_default_graph()
        if device is not None:
            for op in graph.get_operations():
                # Dirty hack, as TF doesn't support moving already-existing ops
                # between devices in the public API.
                op._set_device(device)

        self._saver.restore(self._sess, model_path)

        # Get input and output ops.
        self._x = graph.get_tensor_by_name(x_name)
        self._y = graph.get_tensor_by_name(y_name)
        self._batch_size = self._x.shape[0]

        # Test the restored model compliance with the network signature.
        # Disregard the datatype - it's easy to have false positives here.
        # dtype assertion.
        assert self._x.shape[1:] == network_signature.input.shape
        assert self._y.shape[1:] == network_signature.output.shape

        self._output_dtype = network_signature.output.dtype

        if self._batch_size is not None:
            warnings.warn(
                'The input batch dimension has fixed size ({}), you should save'
                ' your graph with the batch dimension set to None.'.format(
                    self._batch_size))

    def predict(self, inputs):
        inputs = inputs.astype(self._x.dtype.as_numpy_dtype())
        batch_size = inputs.shape[0]
        if self._batch_size is not None and batch_size < self._batch_size:
            # Handle an input batch size lower than the model fixed batch size.
            inputs = np.resize(inputs, (self._batch_size, ) + inputs.shape[1:])

        outputs = self._sess.run(self._y, feed_dict={self._x: inputs})
        # Trim the output batch if it's too big.
        return outputs[:batch_size].astype(self._output_dtype)

    @property
    def params(self):
        return self._sess.run(tf.compat.v1.trainable_variables())

    @params.setter
    def params(self, new_params):
        for t, v in zip(tf.compat.v1.trainable_variables(), new_params):
            tf.compat.v1.keras.backend.set_value(t, v)

    def save(self, checkpoint_path):
        # No-op: we can't train this network, so it doesn't make sense to save
        # another copy of the checkpoint.
        pass

    def restore(self, checkpoint_path):
        # No-op: we do this in __init__.
        pass
