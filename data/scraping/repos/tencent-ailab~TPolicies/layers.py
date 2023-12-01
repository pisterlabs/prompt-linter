"""layers extension in the style of tf.layers/slim.layers"""
from collections import namedtuple
from functools import partial

import tensorflow as tf
from tensorflow.contrib.framework import add_arg_scope
import tensorflow.contrib.layers as tfc_layers
from tensorflow.contrib.layers.python.layers import utils as lutils
from tensorflow.contrib.layers import xavier_initializer

import tpolicies.ops as tp_ops
from tpolicies import ops as tp_ops
from tpolicies.ops import INF, cat_sample_from_logits, ortho_init
from tpolicies.ops import one_step_lstm_op
from tpolicies.utils.distributions import CategoricalPdType, BernoulliPdType
from tpolicies.utils.distributions import MaskSeqCategoricalPdType
from tpolicies.utils.distributions import DiagGaussianPdType


@add_arg_scope
def identity_layer(inputs, outputs_collections=None, scope=None):
  """Identity layer.

  Args:
    inputs: A Tensor
    outputs_collections:
    scope:

  Returns:
    A outputs `Tensor`.

  """
  with tf.variable_scope(scope, default_name='identity_layer') as sc:
    outputs = tf.identity(inputs)
  return lutils.collect_named_outputs(outputs_collections, sc.name, outputs)


@add_arg_scope
def glu(inputs, context, output_size, outputs_collections=None, scope=None):
  """Gated Linear Units.

  The impl follows the GLU described in the Supplementary Material of AlphaStar
  Nature paper.

  Args:
    inputs: (bs, M), the input `Tensor`
    context: (bs, N), the context `Tensor`
    output_size: int, output size
    outputs_collections:
    scope:

  Returns:
    An output `Tensor`.
  """
  inputs_shape = inputs.get_shape().as_list()
  assert len(inputs_shape) == 2
  inputs_size = inputs_shape[1]
  with tf.variable_scope(scope, default_name='glu') as sc:
    # NOTE(pengsun): activation_fn must be None
    gate = tf.nn.sigmoid(
      tfc_layers.fully_connected(context, inputs_size, activation_fn=None)
    )
    gated_inputs = tf.math.multiply(gate, inputs)  # elementwise times
    # NOTE(pengsun): activation_fn must be None
    outputs = tfc_layers.fully_connected(gated_inputs, output_size,
                                         activation_fn=None)
    return lutils.collect_named_outputs(outputs_collections, sc.name, outputs)


# sparse embedding stuff
@add_arg_scope
def linear_embed(inputs,
                 vocab_size,
                 enc_size,
                 inverse_embed=False,
                 outputs_collections=None,
                 weights_initializer=ortho_init(scale=1.0),
                 weights_regularizer=None,
                 scope=None):
  """Linear embedding layer, simply tf.nn.embedding_lookup or inverse embedding.

  In the case of linear embedding, the inputs is a Tensor of index (discrete to
  dense embedding) when inverse_embed=True; For inverse embedding, the inputs is
   a Tensor (dense to ense embedding) when inverse_embed=False.

  Args:
    inputs: when inverse_embed=False, (bs, d1, ...), each is an index in
      range(vocab_size); when inverse_embed=True, (bs, enc_size)
    vocab_size:
    enc_size:
    inverse_embed: True: a "enc_size -> vocab_size" query; False: a
      "vocab_size -> enc_size" embedding
    weights_initializer:
    weights_regularizer:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  with tf.variable_scope(scope, default_name='linear_embed') as sc:
    weights = tf.get_variable('weights', (vocab_size, enc_size),
                              initializer=weights_initializer,
                              regularizer=weights_regularizer)
    if not inverse_embed:
      assert inputs.dtype in [tf.int32, tf.int64], 'inputs must be index'
      outputs = tf.nn.embedding_lookup(weights, inputs)
      outputs_alias = sc.name
    else:
      assert inputs.dtype in [tf.float16, tf.float32, tf.float64], (
        'inputs must be a dense tensor')
      outputs = tf.matmul(inputs, weights, transpose_b=True)
      outputs_alias = sc.name + '_inverse'
    return lutils.collect_named_outputs(outputs_collections, outputs_alias,
                                        outputs)


# normalization stuff
@add_arg_scope
def ln(inputs,
       epsilon=1e-8,
       begin_norm_axis=-1,
       activation_fn=None,
       enable_openai_impl=False,
       scope=None):
  """Applies layer normalization.

  See https://arxiv.org/abs/1607.06450.
  CAUTION: presume the last dim (shape[-1]) being the feature dim!!
  TODO(pengsun): doc from where this impl is borrowed

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension
      has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision
      Error.
    begin_norm_axis: beginning dim
    activation_fn: activation function. None means no activation.
    enable_openai_impl:
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope(scope, default_name="ln"):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    inputs_rank = inputs_shape.ndims
    if begin_norm_axis < 0:
      begin_norm_axis = inputs_rank + begin_norm_axis
    if begin_norm_axis >= inputs_rank:
      raise ValueError('begin_norm_axis (%d) must be < rank(inputs) (%d)' %
                       (begin_norm_axis, inputs_rank))
    norm_axes = list(range(begin_norm_axis, inputs_rank))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
    beta = tf.get_variable("beta", params_shape,
                           initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape,
                            initializer=tf.ones_initializer())
    if enable_openai_impl:
      normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
      outputs = normalized * gamma + beta
    else:
      normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
      outputs = gamma * normalized + beta
    if activation_fn is not None:
      outputs = activation_fn(outputs)

  return outputs


@add_arg_scope
def inst_ln(inputs,
            epsilon=1e-8,
            enable_openai_impl=False,
            activation_fn=None,
            scope=None):
  """Applies Instance normalization.

  See https://arxiv.org/pdf/1607.08022.pdf.
  CAUTION: presume the last dim (shape[-1]) being the feature dim!!

  Args:
    inputs: A tensor with 2 or more dimensions, where the first dimension
      has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision
      Error.
    activation_fn: activation function. None means no activation.
    enable_openai_impl:
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor with the same shape and data dtype as `inputs`.
  """
  with tf.variable_scope(scope, default_name="inst_ln"):
    inputs_shape = inputs.get_shape()
    params_shape = inputs_shape[-1:]
    inputs_rank = inputs_shape.ndims
    norm_axes = list(range(1, inputs_rank - 1))
    mean, variance = tf.nn.moments(inputs, norm_axes, keep_dims=True)
    beta = tf.get_variable("beta", params_shape,
                           initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", params_shape,
                            initializer=tf.ones_initializer())
    if enable_openai_impl:
      normalized = (inputs - mean) / tf.sqrt(variance + epsilon)
      outputs = normalized * gamma + beta
    else:
      normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
      outputs = gamma * normalized + beta
    if activation_fn is not None:
      outputs = activation_fn(outputs)

  return outputs


# dense-/res- net stuff
@add_arg_scope
def dense_sum_blocks(inputs, n, enc_dim, layer_norm: bool = True,
                     outputs_collections=None, scope=None):
  """Dense-sum blocks with fully connected layers.

  Args:
    inputs:
    n: Number of blocks. Each blocks has exactly one fully connected layer.
    enc_dim: output encoding dim.
    layer_norm: Whether to use layer norm
    outputs_collections:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  with tf.variable_scope(scope, default_name='densesum_blks') as sc:
    embed = inputs
    pre_embeds_sum = None
    for i in range(n):
      embed = tfc_layers.fully_connected(embed, enc_dim)
      if i == 0:
        pre_embeds_sum = embed
      else:
        pre_embeds_sum += embed
      embed = pre_embeds_sum
      if layer_norm:
        embed = ln(embed, epsilon=1e-8, scope='ln_'+str(i))
  return lutils.collect_named_outputs(outputs_collections, sc.name, embed)


@add_arg_scope
def dense_sum_conv_blocks(inputs, n, ch_dim, k_size,
                          mode: str = '2d',
                          layer_norm: bool = True,
                          outputs_collections=None,
                          scope=None):
  """Dense-sum blocks with 1D or 2D convolutional layers.

  Args:
    inputs:
    n: Number of blocks. Each blocks has exactly one fully connected layer.
    ch_dim: output channel dim.
    k_size: int or tuple, kernel size
    mode: str, '1d' or '2d'
    layer_norm: Whether to use layer norm
    outputs_collections:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  with tf.variable_scope(
      scope,
      default_name='densesum_conv{}_blks'.format(mode)) as sc:
    embed = inputs
    pre_embeds_sum = None
    if mode == '2d':
      conv_layer = tfc_layers.conv2d
      kernel_size = [k_size, k_size]
    elif mode == '1d':
      conv_layer = tfc_layers.conv1d
      kernel_size = k_size
    else:
      raise ValueError('Unknown mode {}.'.format(mode))

    for i in range(n):
      embed = conv_layer(embed, ch_dim, kernel_size)
      if i == 0:
        pre_embeds_sum = embed
      else:
        pre_embeds_sum += embed
      embed = pre_embeds_sum
      if layer_norm:
        embed = ln(embed, epsilon=1e-8, scope='ln_'+str(i))
  return lutils.collect_named_outputs(outputs_collections, sc.name, embed)


@add_arg_scope
def res_sum_blocks(inputs,
                   n_blk: int,
                   n_skip: int,
                   enc_dim: int,
                   layer_norm: bool = False,
                   relu_input: bool = False,
                   outputs_collections=None,
                   scope=None):
  """Residual sum blocks with fully connected layers.

  The res blocks are more usual with conv layers, some refs are:
  ref1: https://github.com/deepmind/scalable_agent/blob/master/experiment.py#L152-L173
  ref2: https://github.com/google-research/tf-slim/blob/master/tf_slim/nets/resnet_v1.py#L107-L126
  Our impl here is similar to [ref2], where we add a "shortcut connection" with
  the "residual connection" whose enc_dim are the same. [ref1] uses an
  extra layer to enforce the same end_dim and then takes the sum.

  Args:
    inputs: a Tensor, (batch_size, inputs_dim)
    n_blk: int, how many block
    n_skip: int, how many conv layers to skip inside a block
    enc_dim: int, output encoding dim
    layer_norm: Whether to use layer norm
    relu_input: Whether to relu the inputs
    outputs_collections: str, outputs collections
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, enc_dim)
  """
  embed = inputs
  with tf.variable_scope(scope, default_name='res_sum_blks') as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as feature dim
      # shortcut connection
      shortcut_scope = 'blk{}_shortcut'.format(i)
      if blk_inputs_dim == enc_dim:
        shortcut = identity_layer(embed, scope=shortcut_scope)
      else:
        shortcut = tfc_layers.fully_connected(embed, enc_dim,
                                              activation_fn=None,
                                              scope=shortcut_scope)
      # residual connection
      if relu_input:
        embed = tf.nn.relu(embed)
      for j in range(n_skip-1):
        embed = tfc_layers.fully_connected(embed, enc_dim,
                                           activation_fn=tf.nn.relu,
                                           scope='blk{}_fc{}'.format(i, j))
      embed = tfc_layers.fully_connected(embed, enc_dim, activation_fn=None,
                                         scope='blk{}_fc{}'.format(
                                           i, n_skip - 1)
                                         )
      # shortcut + residual
      combined = shortcut + embed
      if layer_norm:
        combined = ln(combined, epsilon=1e-8, scope='ln_'+str(i))
      embed = tf.nn.relu(combined)
  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


@add_arg_scope
def res_sum_blocks_v2(inputs,
                      n_blk: int,
                      n_skip: int,
                      enc_dim: int,
                      layer_norm: bool = False,
                      outputs_collections=None,
                      scope=None):
  """Residual sum blocks with fully connected layers, v2.

  Our impl is much like:
  https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L102-L131
  and the cited papers therein.
  In our impl, the basic block looks:
  when n_skip = 1,
  - relu - weight -
  then input adds to output.
  when n_skip = 2,
  - relu - weight - relu - weight -
  then input adds to output, etc.
  An optional layer_norm can be inserted *BEFORE* relu.
  NOTE: add a leading layer WITHOUT activation and normalization to enforce the
    channel size, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L213-L215
  NOTE: add normalization + relu to the outputs when used as last layer, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L221-L223

  Args:
    inputs: a Tensor, (batch_size, inputs_dim)
    n_blk: int, how many blocks
    n_skip: int, how many weight layers to skip inside a block (i.e., how many
      weight layers in the residual branch)
    enc_dim: int, output encoding dim
    layer_norm: Whether to use layer norm
    outputs_collections: str, outputs collections
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, enc_dim)
  """
  embed = inputs
  with tf.variable_scope(scope, default_name='res_sum_blks_v2') as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as feature dim

      # shortcut connection (simply checking the channel dim)
      shortcut = identity_layer(embed, scope='blk{}_shortcut'.format(i))
      assert blk_inputs_dim == enc_dim, """
      input dim {} must == enc dim {}. Otherwise, use a preceding layer WITHOUT 
      activation to enforce this.
      """.format(blk_inputs_dim, enc_dim)

      # residual connection
      for j in range(n_skip):
        if layer_norm:
          embed = ln(embed, epsilon=1e-8, scope='ln_blk{}_fc{}'.format(i, j))
        embed = tf.nn.relu(embed)
        embed = tfc_layers.fully_connected(
          embed, enc_dim,
          activation_fn=None,
          biases_initializer=None if layer_norm else tf.zeros_initializer(),
          scope='blk{}_fc{}'.format(i, j)
        )
      # combine: shortcut + residual
      embed = shortcut + embed

  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


@add_arg_scope
def res_sum_conv_blocks(inputs,
                        n_blk: int,
                        n_skip: int,
                        ch_dim: int,
                        k_size: int,
                        mode: str = '2d',
                        layer_norm: bool = False,
                        outputs_collections=None,
                        scope=None):
  """Residual sum blocks with 1D or 2D convolutional layer.

  ref1: https://github.com/deepmind/scalable_agent/blob/master/experiment.py#L152-L173
  ref2: https://github.com/google-research/tf-slim/blob/master/tf_slim/nets/resnet_v1.py#L107-L126
  Our impl here is similar to [ref2], where we add a "shortcut connection" with
  the "residual connection" whose ch_dim are the same. [ref1] uses an
  extra layer to enforce the same ch_dim and then takes the sum.

  Args:
    inputs: a Tensor, NHWC format (batch_size, H, W, inputs_dim)
    n_blk: int, how many blocks
    n_skip: int, how many conv layers to skip inside a block
    ch_dim: int, channel dim
    k_size: int, kerner size for 1D or 2D conv
    mode: str, '2d' or '3d'
    layer_norm: Whether to use layer norm
    outputs_collections: str, outputs collection
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, H, W, ch_dim)
  """
  embed = inputs
  # Note the tfc_layers.convXd padding defaults to SAME
  if mode == '2d':
    conv_layer = tfc_layers.conv2d
    shortcut_k_size = [1, 1]
    k_size = [k_size, k_size]
  elif mode == '1d':
    conv_layer = tfc_layers.conv1d
    shortcut_k_size = 1
    k_size = k_size
  else:
    raise ValueError('Unknown mode {}'.format(mode))

  with tf.variable_scope(scope,
                         default_name='res_conv{}_blks'.format(mode)) as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as channel dim
      # shortcut connection
      shortcut_scope = 'blk{}_shortcut'.format(i)
      if blk_inputs_dim == ch_dim:
        shortcut = identity_layer(embed, scope=shortcut_scope)
      else:
        shortcut = conv_layer(embed, ch_dim, shortcut_k_size,
                              activation_fn=None, scope=shortcut_scope)
      # residual connection
      for j in range(n_skip-1):
        embed = conv_layer(embed, ch_dim, k_size, activation_fn=tf.nn.relu,
                           scope='blk{}_conv{}'.format(i, j))
      embed = conv_layer(embed, ch_dim, k_size, activation_fn=None,
                         scope='blk{}_conv{}'.format(i, n_skip - 1))
      # shortcut + residual
      combined = shortcut + embed
      if layer_norm:
        combined = ln(combined, epsilon=1e-8, scope='ln_'+str(i))
      embed = tf.nn.relu(combined)
  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


@add_arg_scope
def res_sum_bottleneck_blocks(inputs,
                              n_blk: int,
                              n_skip: int,
                              ch_dim: int,
                              bottleneck_ch_dim: int,
                              k_size: int,
                              mode: str = '2d',
                              layer_norm: bool = False,
                              layer_norm_type='inst_ln',
                              outputs_collections=None,
                              scope=None):
  """Residual sum blocks with 1D or 2D convolutional bottleneck layer.

  ref1: https://github.com/deepmind/scalable_agent/blob/master/experiment.py#L152-L173
  ref2: https://github.com/google-research/tf-slim/blob/master/tf_slim/nets/resnet_v1.py#L107-L126
  Our impl here is similar to [ref2], where we add a "shortcut connection" with
  the "residual connection" whose ch_dim are the same. [ref1] uses an
  extra layer to enforce the same ch_dim and then takes the sum.

  Args:
    inputs: a Tensor, NHWC format (batch_size, H, W, inputs_dim)
    n_blk: int, how many bottlenecks
    n_skip: int, how many conv layers to skip inside a bottlenecks
    ch_dim: int, channel dim
    bottleneck_ch_dim: int, bottleneck channel dim, usually < ch_dim
    k_size: int, kerner size for 1D or 2D conv
    mode: str, '2d' or '3d'
    layer_norm: Whether to use layer norm
    layer_norm_type: str, yype of layer norm
    outputs_collections: str, outputs collection
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, H, W, ch_dim)
  """
  embed = inputs
  # Note the tfc_layers.convXd padding defaults to SAME
  if mode == '2d':
    conv_layer = tfc_layers.conv2d
    one_size = [1, 1]
    k_size = [k_size, k_size]
  elif mode == '1d':
    conv_layer = tfc_layers.conv1d
    one_size = 1
    k_size = k_size
  else:
    raise ValueError('Unknown mode {}'.format(mode))

  with tf.variable_scope(scope,
                         default_name='res_conv{}_blks'.format(mode)) as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as channel dim
      # shortcut connection
      shortcut_scope = 'blk{}_shortcut'.format(i)
      if blk_inputs_dim == ch_dim:
        shortcut = identity_layer(embed, scope=shortcut_scope)
      else:
        shortcut = conv_layer(embed, ch_dim, one_size,
                              activation_fn=None, scope=shortcut_scope)
      # residual connection
      embed = conv_layer(embed, bottleneck_ch_dim, one_size,
                         activation_fn=tf.nn.relu,
                         scope='blk{}_conv{}'.format(i, 0))
      for j in range(n_skip):
        embed = conv_layer(embed, bottleneck_ch_dim, k_size,
                           activation_fn=tf.nn.relu,
                           scope='blk{}_conv{}'.format(i, j+1))
      embed = conv_layer(embed, ch_dim, one_size, activation_fn=None,
                         scope='blk{}_conv{}'.format(i, n_skip + 1))
      # shortcut + residual
      combined = shortcut + embed
      if layer_norm:
        if layer_norm_type == 'ln':
          combined = ln(combined, begin_norm_axis=1, epsilon=1e-8,
                        scope='ln_'+str(i))
        elif layer_norm_type == 'inst_ln':
          combined = inst_ln(combined, epsilon=1e-8, scope='inst_ln' + str(i))
        else:
          raise KeyError('Unknown layer_norm_type {}'.format(layer_norm_type))
      embed = tf.nn.relu(combined)
  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


def res_sum_bottleneck_blocks_v2(inputs,
                                 n_blk: int,
                                 n_skip: int,
                                 ch_dim: int,
                                 bottleneck_ch_dim: int,
                                 k_size: int,
                                 mode: str = '2d',
                                 layer_norm_type='inst_ln',
                                 outputs_collections=None,
                                 scope=None):
  """Residual sum blocks with 1D or 2D convolutional bottleneck layer, v2.

  Our impl is much like:
  https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L102-L131
  which is the so-called "pre-activation res block". See also the cited papers
  therein.
  In our impl, the basic block looks:
  when n_skip = 1
  - relu - weight -
  then input adds to output.
  when n_skip = 2
  - relu - weight - relu - weight -
  then input adds to output, etc.
  *EACH* weight layer should be understood as a bottleneck structure that
  expands to a narrow-wide-wide three-conv-layer, i.e.,
  - weight -
  means
  - 1x1conv_narrow - kxkconv_narrow - 1x1conv_wide
  An optional layer_norm can be inserted *BEFORE* relu.
  NOTE: add a leading layer WITHOUT activation and normalization to enforce the
    channel size, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L213-L215
  NOTE: add normalization + relu to the outputs when used as last layer, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L221-L223

  Args:
    inputs: a Tensor, NHWC format (batch_size, H, W, inputs_dim)
    n_blk: int, how many blocks
    n_skip: int, how many weight layers to skip inside a block (i.e., how
      many weight layers in the residual branch)
    ch_dim: int, channel dim
    bottleneck_ch_dim: int, bottleneck channel dim, usually < ch_dim
    k_size: int, kernel size for 1D or 2D conv
    mode: str, '2d' or '3d'
    layer_norm_type: str, type of layer norm. None means no layer norm.
    outputs_collections: str, outputs collection
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, H, W, ch_dim)
  """
  embed = inputs
  # Note the tfc_layers.convXd padding defaults to SAME
  if mode == '2d':
    conv_layer = tfc_layers.conv2d
    k_size_one = [1, 1]
    k_size = [k_size, k_size]
  elif mode == '1d':
    conv_layer = tfc_layers.conv1d
    k_size_one = 1
    k_size = k_size
  else:
    raise ValueError('Unknown mode {}'.format(mode))

  with tf.variable_scope(
      scope,
      default_name='res_sum_bottleneck{}_blks_v2'.format(mode)) as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as channel dim

      # shortcut connection (simply checking the channel dim)
      shortcut = identity_layer(embed, scope='blk{}_shortcut'.format(i))
      assert blk_inputs_dim == ch_dim, """
      input dim {} must == ch dim {}. Otherwise, use a preceding layer WITHOUT 
      activation to enforce this.
      """.format(blk_inputs_dim, ch_dim)

      # residual connection
      for j in range(n_skip):
        # (bs, H, W, C)
        if layer_norm_type is not None:
          # NOTE(pengsun): a single layer_norm should suffice here
          if layer_norm_type == 'ln':
            embed = ln(embed, begin_norm_axis=1, epsilon=1e-8,
                       scope='ln_blk{}_{}'.format(i, j))
          elif layer_norm_type == 'inst_ln':
            embed = inst_ln(embed, epsilon=1e-8,
                            scope='inst_ln_blk{}_{}'.format(i, j))
          else:
            raise KeyError('Unknown layer_norm_type {}'.format(layer_norm_type))
        embed = tf.nn.relu(embed)
        # (bs, H, W, C)
        embed = conv_layer(embed, bottleneck_ch_dim, k_size_one,
                           activation_fn=tf.nn.relu,
                           scope='blk{}_conv{}_0'.format(i, j))
        # (bs, H, W, BC)
        embed = conv_layer(embed, bottleneck_ch_dim, k_size,
                           activation_fn=tf.nn.relu,
                           scope='blk{}_conv{}_1'.format(i, j))
        # (bs H, W, BC)
        embed = conv_layer(
          embed, ch_dim, k_size_one,
          activation_fn=None,
          biases_initializer=(None if layer_norm_type else
                              tf.zeros_initializer()),
          scope='blk{}_conv{}_2'.format(i, j)
        )
        # (bs, H, W, C)

      # combine: shortcut + residual
      embed = shortcut + embed
  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


def res_sum_bottleneck_blocks_v3(inputs,
                                 n_blk: int,
                                 n_skip: int,
                                 ch_dim: int,
                                 bottleneck_ch_dim: int,
                                 k_size: int,
                                 mode: str = '2d',
                                 layer_norm_type='inst_ln',
                                 outputs_collections=None,
                                 scope=None):
  """Residual sum blocks with 1D or 2D convolutional bottleneck layer, v3.

  Our impl is much like:
  https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L102-L131
  which is the so-called "pre-activation res block". See also the cited papers
  therein.
  In our impl, the basic block looks:
  when n_skip = 1
  - relu - weight -
  then input adds to output.
  when n_skip = 2
  - relu - weight - relu - weight -
  then input adds to output, etc.
  *EACH* weight layer should be understood as a bottleneck structure that
  expands to a narrow-wide-wide three-conv-layer, i.e.,
  - weight -
  means
  - 1x1conv_narrow - kxkconv_narrow - 1x1conv_wide
  An optional layer_norm can be inserted *BEFORE* relu.
  NOTE: add a leading layer WITHOUT activation and normalization to enforce the
    channel size, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L213-L215
  NOTE: add normalization + relu to the outputs when used as last layer, see:
    https://github.com/google-research/tf-slim/blob/8f0215e924996d7287392241bc8d8b1133d0c5ca/tf_slim/nets/resnet_v2.py#L221-L223
  NOTE: normalization after EACH bottleneck conv weight, different from v2

  Args:
    inputs: a Tensor, NHWC format (batch_size, H, W, inputs_dim)
    n_blk: int, how many blocks
    n_skip: int, how many weight layers to skip inside a block (i.e., how
      many weight layers in the residual branch)
    ch_dim: int, channel dim
    bottleneck_ch_dim: int, bottleneck channel dim, usually < ch_dim
    k_size: int, kernel size for 1D or 2D conv
    mode: str, '2d' or '3d'
    layer_norm_type: str, type of layer norm. None means no layer norm.
    outputs_collections: str, outputs collection
    scope: scope or scope

  Returns:
    A Tensor, the outputs, (batch_size, H, W, ch_dim)
  """
  embed = inputs
  # Note the tfc_layers.convXd padding defaults to SAME
  if mode == '2d':
    conv_layer = tfc_layers.conv2d
    k_size_one = [1, 1]
    k_size = [k_size, k_size]
  elif mode == '1d':
    conv_layer = tfc_layers.conv1d
    k_size_one = 1
    k_size = k_size
  else:
    raise ValueError('Unknown mode {}'.format(mode))

  with tf.variable_scope(
      scope,
      default_name='res_sum_bottleneck{}_blks_v3'.format(mode)) as sc:
    for i in range(n_blk):
      blk_inputs_dim = embed.shape[-1].value  # last dim as channel dim

      # shortcut connection (simply checking the channel dim)
      shortcut = identity_layer(embed, scope='blk{}_shortcut'.format(i))
      assert blk_inputs_dim == ch_dim, """
      input dim {} must == ch dim {}. Otherwise, use a preceding layer WITHOUT 
      activation to enforce this.
      """.format(blk_inputs_dim, ch_dim)

      # residual connection
      conv_norm = None
      if layer_norm_type is not None:
        # TODO(pengsun): refactor the code, combine with pre-act norm stuff
        if layer_norm_type == 'ln':
          conv_norm = ln
        elif layer_norm_type == 'inst_ln':
          conv_norm = inst_ln
        else:
          raise KeyError('Unknown layer_norm_type {}'.format(layer_norm_type))
      for j in range(n_skip):
        # (bs, H, W, C)
        if layer_norm_type is not None:
          # pre-activation normalization if any
          if layer_norm_type == 'ln':
            embed = ln(embed, begin_norm_axis=1, epsilon=1e-8,
                       scope='ln_blk{}_{}'.format(i, j))
          elif layer_norm_type == 'inst_ln':
            embed = inst_ln(embed, epsilon=1e-8,
                            scope='inst_ln_blk{}_{}'.format(i, j))
          else:
            raise KeyError('Unknown layer_norm_type {}'.format(layer_norm_type))
        embed = tf.nn.relu(embed)
        # (bs, H, W, C)
        embed = conv_layer(embed, bottleneck_ch_dim, k_size_one,
                           activation_fn=tf.nn.relu,
                           normalizer_fn=conv_norm,
                           scope='blk{}_conv{}_0'.format(i, j))
        # (bs, H, W, BC)
        embed = conv_layer(embed, bottleneck_ch_dim, k_size,
                           activation_fn=tf.nn.relu,
                           normalizer_fn=conv_norm,
                           scope='blk{}_conv{}_1'.format(i, j))
        # (bs H, W, BC)
        embed = conv_layer(
          embed, ch_dim, k_size_one,
          activation_fn=None,
          normalizer_fn=None,
          biases_initializer=(None if layer_norm_type else
                              tf.zeros_initializer()),
          scope='blk{}_conv{}_2'.format(i, j)
        )
        # (bs, H, W, C)

      # combine: shortcut + residual
      embed = shortcut + embed
  return lutils.collect_named_outputs(outputs_collections,
                                      sc.original_name_scope, embed)


# transformer stuff
def trans_mask(inputs, queries=None, keys=None, mtype=None):
  """Masks paddings on keys or queries to inputs.

  TODO: doc where it is from

  e.g.,
  >> queries = tf.constant([[[1.],
                      [2.],
                      [0.]]], tf.float32) # (1, 3, 1)
  >> keys = tf.constant([[[4.],
                   [0.]]], tf.float32)  # (1, 2, 1)
  >> inputs = tf.constant([[[4., 0.],
                             [8., 0.],
                             [0., 0.]]], tf.float32)
  >> mask(inputs, queries, keys, "key")
  array([[[ 4.0000000e+00, -4.2949673e+09],
      [ 8.0000000e+00, -4.2949673e+09],
      [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
  >> inputs = tf.constant([[[1., 0.],
                           [1., 0.],
                            [1., 0.]]], tf.float32)
  >> mask(inputs, queries, keys, "query")
  array([[[1., 0.],
      [1., 0.],
      [0., 0.]]], dtype=float32)

  Args:
    inputs: 3d tensor. (N, T_q, T_k)
    queries: 3d tensor. (N, T_q, d)
    keys: 3d tensor. (N, T_k, d)
    mtype: str

  Returns:
    A `Tensor` representing the output mask.
  """
  padding_num = -2 ** 32 + 1
  if mtype in ("k", "key", "keys"):
    # Generate masks
    masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1))  # (N, T_k)
    masks = tf.expand_dims(masks, 1)  # (N, 1, T_k)
    masks = tf.tile(masks, [1, tf.shape(queries)[1], 1])  # (N, T_q, T_k)

    # Apply masks to inputs
    paddings = tf.ones_like(inputs) * padding_num
    outputs = tf.where(tf.equal(masks, 0), paddings, inputs)  # (N, T_q, T_k)
  elif mtype in ("q", "query", "queries"):
    # Generate masks
    masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))  # (N, T_q)
    masks = tf.expand_dims(masks, -1)  # (N, T_q, 1)
    masks = tf.tile(masks, [1, 1, tf.shape(keys)[1]])  # (N, T_q, T_k)

    # Apply masks to inputs
    outputs = inputs * masks
  elif mtype in ("f", "future", "right"):
    diag_vals = tf.ones_like(inputs[0, :, :])  # (T_q, T_k)
    tril = tf.linalg.LinearOperatorLowerTriangular(
      diag_vals).to_dense()  # (T_q, T_k)
    masks = tf.tile(tf.expand_dims(tril, 0),
                    [tf.shape(inputs)[0], 1, 1])  # (N, T_q, T_k)

    paddings = tf.ones_like(masks) * padding_num
    outputs = tf.where(tf.equal(masks, 0), paddings, inputs)
  else:
    print("Check if you entered mtype correctly!")
    raise ValueError('wtf mtype {}?'.format(mtype))

  return outputs


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 pointer=False,
                                 scope=None):
  """ Scaled dot product attention, See 3.2.1.

  TODO: doc where it is from

  Args:—
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
  """
  with tf.variable_scope(scope, default_name="scaled_dot_product_attention"):
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # key masking
    outputs = trans_mask(outputs, Q, K, mtype="key")

    # causality or future blinding masking
    if causality:
      outputs = trans_mask(outputs, mtype="future")

    # softmax
    output_logits = outputs
    outputs = tf.nn.softmax(outputs)  # default axis=-1
    if pointer:
      return output_logits, outputs
    attention = tf.transpose(outputs, [0, 2, 1])  # only used for tf.summary
    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # query masking
    outputs = trans_mask(outputs, Q, K, mtype="query")

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

  return outputs


def scaled_dot_product_attention_v2(Q, K, V, matrix_mask,
                                    dropout_rate=0.0,
                                    scope=None):
  """ Simplified from scaled_dot_product_attention
  Args:
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
  """
  with tf.variable_scope(scope, default_name="scaled_dot_product_attention"):
    d_k = Q.get_shape().as_list()[-1]

    # dot product
    outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

    # scale
    outputs /= d_k ** 0.5

    # key masking
    outputs = trans_mask(outputs, Q, K, mtype="key")

    # entry_mask
    outputs = tp_ops.mask_logits(outputs, matrix_mask)

    # softmax
    outputs = tf.nn.softmax(outputs)  # default axis=-1
    # attention = tf.transpose(outputs, [0, 2, 1])  # only used for tf.summary
    # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

    # query masking
    outputs = trans_mask(outputs, Q, K, mtype="query")

    # dropout
    outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=True)

    # weighted sum (context vectors)
    outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

  return outputs


def multihead_attention(queries, keys, values,
                        num_heads=8,
                        dropout_rate=0.0,
                        training=True,
                        causality=False,
                        pointer=False,
                        scope=None):
  """Applies multihead attention. See 3.2.2

  Original Reference: A TensorFlow Implementation of the Transformer: Attention Is All You Need
  https://github.com/Kyubyong/transformer
  Kyubyong Park's implementation, a highly starred repository besides google's

  Args:
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with shape of (N, T_q, C)
  """
  d_model = queries.get_shape().as_list()[-1]
  with tf.variable_scope(scope, default_name='multihead_attention'):
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)

    # Attention
    if pointer:
      # do not use multi-head
      logits, pd = scaled_dot_product_attention(Q, K, V, causality,
                                                dropout_rate, training, pointer)
      return logits, pd
    else:
      outputs = scaled_dot_product_attention(Q_, K_, V_, causality,
                                             dropout_rate, training, pointer)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                        axis=2)  # (N, T_q, d_model)

    # Residual connection
    outputs += queries

    # Normalize
    outputs = ln(outputs)

  return outputs


def multihead_attention_v2(queries, keys, values, entry_mask,
                           num_heads=8,
                           dropout_rate=0.0,
                           scope=None):
   """Simplified from multihead_attention
   Args:
     queries: A 3d tensor with shape of [N, T_q, d_model].
     keys: A 3d tensor with shape of [N, T_k, d_model].
     values: A 3d tensor with shape of [N, T_k, d_model].
     num_heads: An int. Number of heads.
     dropout_rate: A floating point number.
     training: Boolean. Controller of mechanism for dropout.
     causality: Boolean. If true, units that reference the future are masked.
     scope: Optional scope for `variable_scope`.
   Returns:
     A 3d tensor with shape of (N, T_q, C)
   """
   d_model = queries.get_shape().as_list()[-1]
   with tf.variable_scope(scope, default_name='multihead_attention'):
     # Linear projections
     Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
     K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
     V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)
     # Split and concat
     Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                    axis=0)  # (h*N, T_q, d_model/h)
     K_ = tf.concat(tf.split(K, num_heads, axis=2),
                    axis=0)  # (h*N, T_k, d_model/h)
     V_ = tf.concat(tf.split(V, num_heads, axis=2),
                    axis=0)  # (h*N, T_k, d_model/h)
     # Duplicate mask
     matrix_mask = tf.tile(tf.expand_dims(entry_mask, axis=1), [1, entry_mask.shape[-1], 1])
     matrix_mask = tf.math.logical_and(
       matrix_mask,
       tf.tile(tf.expand_dims(entry_mask, axis=-1), [1, 1, entry_mask.shape[-1]])
     )
     matrix_mask = tf.tile(matrix_mask, [num_heads, 1, 1])
     outputs = scaled_dot_product_attention_v2(Q_, K_, V_, matrix_mask, dropout_rate)
     # Restore shape
     outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                         axis=2)  # (N, T_q, d_model)
     # Residual connection
     outputs += queries
     # Normalize
     outputs = ln(outputs)
   return outputs


def multihead_attention_v3(queries, keys, values, entry_mask,
                           num_heads=8, enc_dim=128,
                           dropout_rate=0.0, scope=None):
  """ Identical to AStar
  Args:
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with shape of (N, T_q, C)
  """
  # d_model = queries.get_shape().as_list()[-1]
  d_model = enc_dim
  with tf.variable_scope(scope, default_name='multihead_attention'):
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)

    # Duplicate mask
    matrix_mask = tf.tile(tf.expand_dims(entry_mask, axis=1), [1, entry_mask.shape[-1], 1])
    matrix_mask = tf.math.logical_and(
      matrix_mask,
      tf.tile(tf.expand_dims(entry_mask, axis=-1), [1, 1, entry_mask.shape[-1]])
    )
    matrix_mask = tf.tile(matrix_mask, [num_heads, 1, 1])

    outputs = scaled_dot_product_attention_v2(Q_, K_, V_, matrix_mask, dropout_rate)

    # fc to double the number of channels to 256 and
    # the head results are summed and passed through
    # a 2-layer MLP with hidden size 1024 and output
    # size 256
    outputs = tfc_layers.fully_connected(outputs, 256)
    outputs = tf.add_n(tf.split(outputs, num_heads, axis=0))
    outputs = tfc_layers.fully_connected(outputs, 1024)
    outputs = tfc_layers.fully_connected(outputs, 256, activation_fn=None)

    # # Residual connection
    # outputs += queries
    # # Normalize
    # outputs = ln(outputs)

  return outputs


def self_attention_ffsum(queries, keys, values, entry_mask,
                         num_heads=8, enc_dim=128,
                         dropout_rate=0.0, scope=None):
  """ Self Attention + Feed Forward using Sum.

  Simplified from multihead_attention and follows the AStar paper impl.

  Args:
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    enc_dim:
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with shape of (N, T_q, C)
  """
  # d_model = queries.get_shape().as_list()[-1]
  d_model = enc_dim
  with tf.variable_scope(scope, default_name='self_attention_ffsum'):
    # The Self Attention Block
    # Linear projections
    Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
    K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
    V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

    # Split to num_heads
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                   axis=0)  # (h*N, T_q, d_model/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2),
                   axis=0)  # (h*N, T_k, d_model/h)

    # Duplicate mask
    matrix_mask = tf.tile(tf.expand_dims(entry_mask, axis=1),
                          [1, entry_mask.shape[-1], 1])
    matrix_mask = tf.math.logical_and(
      matrix_mask,
      tf.tile(tf.expand_dims(entry_mask, axis=-1), [1, 1, entry_mask.shape[-1]])
    )
    matrix_mask = tf.tile(matrix_mask, [num_heads, 1, 1])

    # Attention
    outputs = scaled_dot_product_attention_v2(Q_, K_, V_, matrix_mask,
                                              dropout_rate)

    outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                        axis=2)  # Restore shape: (N, T_q, d_model)
    outputs += queries  # Residual connection
    outputs = ln(outputs, scope='attention_ln')  # normalize

    # The ff Block, AStar paper goes:
    #  In each layer, each self-attention head uses keys, queries, and values of
    #  size 128, then passes the aggregated values through a Conv1D with kernel
    #  size 1 to double the number of channels (to 256). The head results are
    #  summed and passed through a 2-layer MLP with hidden size 1024 and output
    #  size 256.
    skip = outputs
    outputs_split = tf.split(outputs, num_heads, axis=-1)
    for i in range(num_heads):
      # Note(pengsun): equivalent to conv1d with kernel size 1
      outputs_split[i] = tfc_layers.fully_connected(outputs_split[i], 256)
    outputs = tf.add_n(outputs_split)
    outputs = tfc_layers.fully_connected(outputs, 1024)
    outputs = tfc_layers.fully_connected(outputs, 256, activation_fn=None)
    outputs += skip  # Residual connection
    outputs = ln(outputs, scope='ff_ln')  # normalize

  return outputs


def ff(inputs, num_units, scope=None):
  """position-wise feed forward net. See 3.3

  Args:
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

  Returns:
    A 3d tensor with the same shape and dtype as inputs
  """
  with tf.variable_scope(scope, default_name="positionwise_feedforward"):
    # Inner layer
    outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

    # Outer layer
    outputs = tf.layers.dense(outputs, num_units[1])

    # Residual connection
    outputs += inputs

    # Normalize
    outputs = ln(outputs)

  return outputs


# rnn stuff
@add_arg_scope
def lstm(inputs_x_seq: list,
         inputs_terminal_mask_seq: list,
         inputs_state,
         nh,
         forget_bias=1.0,
         weights_initializer=ortho_init(1.0),
         biases_initializer=tf.constant_initializer(0.0),
         weights_regularizer=None,
         biases_regularizer=None,
         use_layer_norm=False,
         scope=None):
  """ An lstm layer (cell) tailored for RL task.

  It includes a mask that indicates the terminal time step of an unroll.
  Borrowed and modified from openai/baselines.

  Args:
    inputs_x_seq: list of rollout_len Tensors, each sized (nrollout, dim)
    inputs_terminal_mask_seq: list of rollout_len Tensors, each sized
     (nrollout, 1). A  mask that indicates whether it is terminal of an
     unroll.
    inputs_state: Tensor, (nrollout, 2*nh), initial hidden state of the input
     rollout
    nh: int, number of hiddent units.
    forget_bias:
    weights_initializer=ortho_init(1.0),
    biases_initializer=tf.constant_initializer(0.0),
    scope=None

  Returns:
    A list of outputs
    A Tensor, the updated hidden state
  """
  # shorter names
  xs, ms, s = inputs_x_seq, inputs_terminal_mask_seq, inputs_state
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  with tf.variable_scope(scope, default_name='lstm'):
    # weights & biases
    # Use xavier_initializer for wx per qingwei's verification
    wx = tf.get_variable("wx", [nin, nh * 4], initializer=xavier_initializer(),
                         regularizer=weights_regularizer)
    wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer,
                         regularizer=weights_regularizer)
    b = tf.get_variable("b", [nh * 4], initializer=biases_initializer,
                        regularizer=biases_regularizer)
    # normalization function
    x_nf, h_nf, c_nf = None, None, None
    if use_layer_norm:
      with tf.variable_scope('x_ln', reuse=tf.AUTO_REUSE) as sc:
        x_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
      with tf.variable_scope('h_ln', reuse=tf.AUTO_REUSE) as sc:
        h_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
      with tf.variable_scope('c_ln', reuse=tf.AUTO_REUSE) as sc:
        c_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)

  c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
  for idx, (x, m) in enumerate(zip(xs, ms)):
    c = c * (1 - m)
    h = h * (1 - m)
    c, h = one_step_lstm_op(c, h, x, wx, wh, b, forget_bias, x_nf, h_nf, c_nf)
    xs[idx] = h
  s = tf.concat(axis=1, values=[c, h])
  return xs, s


@add_arg_scope
def k_lstm(inputs_x_seq: list,
           inputs_termial_mask_seq: list,
           inputs_state,
           nh,
           k,
           forget_bias=0.0,
           weights_initializer=ortho_init(1.0),
           biases_initializer=tf.constant_initializer(0.0),
           weights_regularizer=None,
           biases_regularizer=None,
           use_layer_norm=False,
           scope=None):
  """ An skip-k-step lstm layer (cell) tailored for RL task.

  It includes a mask that indicates the terminal time step of an unroll.
  Borrowed and modified from openai/baselines.

  Args:
    inputs_x_seq: list of rollout_len Tensors, each sized (nrollout, dim)
    inputs_terminal_mask_seq: list of rollout_len Tensors, each sized
     (nrollout, 1). A  mask that indicates whether it is terminal of an
     unroll.
    inputs_state: Tensor, (nrollout, 2*nh+1), initial hidden state of the
    input rollout. The last dim stores cyclic step count information in
    {0, 1, ... , k-1}
    nh:
    k: number of steps to skip
    forget_bias: float, forget bias
    weights_initializer: defaults to ortho_init(1.0),
    biases_initializer: defaults to tf.constant_initializer(0.0),
    scope:

  Returns
    A list of outputs.
    A Tensor, the updated hidden state.
  """
  # shorter names
  xs, ms, s = inputs_x_seq, inputs_termial_mask_seq, inputs_state
  nbatch, nin = [v.value for v in xs[0].get_shape()]
  with tf.variable_scope(scope, default_name='k_lstm'):
    # weights & biases
    wx = tf.get_variable("wx", [nin, nh * 4], initializer=weights_initializer,
                         regularizer=weights_regularizer)
    wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer,
                         regularizer=weights_regularizer)
    b = tf.get_variable("b", [nh * 4], initializer=biases_initializer,
                        regularizer=biases_regularizer)
    # normalization function
    x_nf, h_nf, c_nf = None, None, None
    if use_layer_norm:
      with tf.variable_scope('x_ln', reuse=tf.AUTO_REUSE) as sc:
        x_nf = partial(ln, scope=sc)
      with tf.variable_scope('h_ln', reuse=tf.AUTO_REUSE) as sc:
        h_nf = partial(ln, scope=sc)
      with tf.variable_scope('c_ln', reuse=tf.AUTO_REUSE) as sc:
        c_nf = partial(ln, scope=sc)

  nh = (s.shape[1].value - 1) // 2
  c, h, cyclic_step_count = tf.split(axis=1, value=s,
                                     num_or_size_splits=[nh, nh, 1])
  for idx, (x, m) in enumerate(zip(xs, ms)):
    c = c * (1 - m)
    h = h * (1 - m)
    c_lstm, h_lstm = one_step_lstm_op(c, h, x, wx, wh, b, forget_bias, x_nf,
                                      h_nf, c_nf)
    mod_mask = tf.equal(cyclic_step_count, 0)
    mod_mask = tf.cast(mod_mask, tf.float32)
    c = tf.multiply(mod_mask, c_lstm) + tf.multiply(1 - mod_mask, c)
    h = tf.multiply(mod_mask, h_lstm) + tf.multiply(1 - mod_mask, h)
    xs[idx] = h
    # prepare for the next time step
    cyclic_step_count = tf.mod(cyclic_step_count + 1, k)
  s = tf.concat(axis=1, values=[c, h, cyclic_step_count])
  return xs, s


# Action stuff
ActionHead = namedtuple('ActionHead', [
  'flatparam',
  'argmax',
  'sam',  # sampled action
  'neglogp',  # negative log-likelihood, i.e., -log(p)
  'pd',  # probabilistic distribution
  'ent'  # entropy
])


@add_arg_scope
def to_action_head(flatparam, pdtype_cls, temperature=1.0,
                   nseq=None, mask=None, labels=None, sample=None, scope=None):
  """Convert logits to ActionHead.

  Args:
    flatparam: (batch_size, pdtype_cls.param_shape())
    pdtype_cls: distribution type
    scope: for tf.variable_scope

  Returns:
    A ActionHead class instance.
  """
  if pdtype_cls == DiagGaussianPdType:
    n_actions = int(flatparam.shape[-1]//2)
    # logstd -> logstd + 0.5 * log(T)
    if temperature != 1.0:
      mean, logstd = tf.split(axis=-1, num_or_size_splits=2,
                              value=flatparam)
      flatparam = tf.concat(
        [mean, logstd + 0.5 * tf.log(float(temperature))], axis=-1)
  else:
    flatparam /= temperature
    n_actions = flatparam.shape[-1]
  if pdtype_cls == CategoricalPdType:
    pdtype = pdtype_cls(ncat=n_actions)
  elif pdtype_cls == BernoulliPdType:
    pdtype = pdtype_cls(size=n_actions)
  elif pdtype_cls == MaskSeqCategoricalPdType:
    pdtype = pdtype_cls(nseq=nseq, ncat=n_actions, mask=mask, labels=labels)
    flatparam = tf.reshape(flatparam, shape=(-1, nseq*n_actions))
  elif pdtype_cls == DiagGaussianPdType:
    pdtype = pdtype_cls(size=n_actions)
  else:
    raise NotImplemented('Unknown pdtype_cls {}'.format(pdtype_cls))

  with tf.variable_scope(scope, default_name='to_action_head'):
    head_pd = pdtype.pdfromflat(flatparam)
    head_argmax = head_pd.mode()
    # Note(pengsun): we cannot write `head_sam = sample or head_pd.sample()`,
    # as it is interpreted as `Tensor or Tensor` and raises an error
    if sample is not None:
      head_sam = sample
    else:
      head_sam = head_pd.sample()
    head_neglogp = head_pd.neglogp(head_sam)
    head_entropy = head_pd.entropy()
  return ActionHead(flatparam, head_argmax, head_sam, head_neglogp, head_pd,
                    head_entropy)


@add_arg_scope
def discrete_action_head(inputs,
                         n_actions,
                         pdtype_cls,
                         mask=None,
                         enc_dim=None,
                         embed_scope=None,
                         temperature=1.0,
                         scope=None):
  """Layer that makes an action head.

  The embedding layer is created or reused by taking the embed_scope.

  Args:
    inputs:
    n_actions:
    pdtype_cls:
    mask:
    enc_dim:
    embed_scope:
    scope:

  Returns:
    A `Tensor` representing the logits.
  """
  with tf.variable_scope(scope, default_name='discrete_action_head'):
    head_logits = tfc_layers.fully_connected(inputs,
                                             n_actions,
                                             activation_fn=None,
                                             normalizer_fn=None,
                                             scope='logits')

    if enc_dim is not None and embed_scope is not None:
      # get the action embedding to do the "offset-add" (invented by lxhan)
      # TODO(pengsun): double-check the two-layer size, why the n_actions for
      #  the first layer?
      head_h = tfc_layers.fully_connected(inputs, n_actions, scope='bfc1')
      # [bs, n_actions]
      head_h_branch = tfc_layers.fully_connected(head_h,
                                                 enc_dim,
                                                 activation_fn=None,
                                                 normalizer_fn=None,
                                                 scope='bfc2')
      # [bs, enc_dim]
      offset = linear_embed(head_h_branch,
                            vocab_size=n_actions,
                            enc_size=enc_dim,
                            inverse_embed=True,
                            scope=embed_scope)
      # [bs, n_actions]
      # do the offset-adding
      head_logits += offset

  if mask is not None:
    head_logits = tp_ops.mask_logits(head_logits, mask)

  return to_action_head(head_logits, pdtype_cls, temperature=temperature)


@add_arg_scope
def discrete_action_head_v2(inputs,
                            n_actions,
                            pdtype_cls,
                            context=None,
                            mask=None,
                            temperature=1.0,
                            scope=None):
  """Layer that makes an action head, v2.

  Convert the inputs to logits. Can pass in an optional context for GLU, and an
  optional mask for available actions.

  Args:
    inputs: input Tensor
    n_actions: int, number of actions
    pdtype_cls:
    mask: (bs, n_actions), mask for available actions. Default None means "not
      use"
    context: (bs, M), context Tensor. For GLU. Default None means "not use"
    outputs_collections:
    scope:

  Returns:
    A `Tensor` representing the logits.
  """
  with tf.variable_scope(scope, default_name='discrete_action_head_v2'):
    if context is None:
      head_logits = tfc_layers.fully_connected(inputs,
                                               n_actions,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               scope='logits')
    else:
      head_logits = glu(inputs, context, n_actions, scope='gated_logits')
    if mask is not None:
      head_logits = tp_ops.mask_logits(head_logits, mask)

  return to_action_head(head_logits, pdtype_cls, temperature=temperature)


@add_arg_scope
def loc_action_head(inputs,
                    pdtype_cls,
                    mask=None,
                    temperature=1.0,
                    logits_mode='1x1',
                    scatter_ind=None,
                    scatter_bs=None,
                    scope=None):
  """Layer that makes a location action (one-hot of a 2D map) head.

  Args:
    inputs: [bs, H, W, C]
    pdtype_cls: distribtion
    mask: [bs, H, W]
    logits_mode: whether to perform scaling up output size x2
    scope:

  Returns:
    An action head for the flatten logits, [bs, HH*WW]. Cases are:
      logits_mode == '1x1': HH = H, WW = W
      logits_mode == '3x3up2': HH = 2*H, WW = 2*W
  """
  with tf.variable_scope(scope, default_name='loc_action_head'):
    # [bs, H, W, C]
    if logits_mode == '3x3up2':
      loc_logits = tfc_layers.conv2d_transpose(inputs, 1, [3, 3],
                                               stride=2,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               scope='3x3up2mapping')
    elif logits_mode == '1x1':
      loc_logits = tfc_layers.conv2d(inputs, 1, [1, 1],
                                     activation_fn=None,
                                     normalizer_fn=None,
                                     scope='1x1mapping')
    else:
      raise ValueError('Unknown logits_mode {}'.format(logits_mode))
    # [bs, HH, WW, 1]
    loc_logits = tf.squeeze(loc_logits, axis=-1)
    # [bs, HH, WW]
    if mask is not None:
      loc_logits = tp_ops.mask_logits(loc_logits, mask)
    # [bs, HH, WW]
    loc_logits_flat = tfc_layers.flatten(loc_logits)
    if scatter_ind is not None and scatter_bs is not None:
      loc_logits_flat = tf.scatter_nd(
        tf.expand_dims(scatter_ind, axis=-1),
        loc_logits_flat,
        shape=[scatter_bs] + loc_logits_flat.shape[1:]
      )
    # [bs, H*W]
    return to_action_head(loc_logits_flat, pdtype_cls, temperature=temperature)


def _ptr_decode(y, memory, num_dec_blocks, ff_dim, enc_dim, training=True):
  with tf.variable_scope("decoder"):
    dec_logits, dec_pd = [], []
    dec = y
    # Blocks
    for i in range(num_dec_blocks):
      with tf.variable_scope("num_blocks_{}".format(i)):
        if i < num_dec_blocks - 1:
          # Vanilla attention
          dec = multihead_attention(queries=dec,
                                    keys=memory,
                                    values=memory,
                                    dropout_rate=0.0,
                                    training=training,
                                    causality=False,
                                    scope="vanilla_attention")
          # Feed Forward
          dec = ff(dec, num_units=[ff_dim, enc_dim])
        else:
          # pointer attention
          dec_logits, dec_pd = multihead_attention(
            queries=dec,
            keys=memory,
            values=memory,
            dropout_rate=0.0,
            training=training,
            causality=False,
            pointer=True,
            scope="pointer_attention")
  return dec_logits, dec_pd


@add_arg_scope
def ptr_action_head(inputs_query,
                    inputs_ptr_mask,
                    inputs_entity_embed,
                    ptr_out_dim,
                    num_dec_blocks,
                    ff_dim,
                    enc_dim,
                    pdtype_cls,
                    temperature=1.0,
                    scatter_ind=None,
                    scatter_bs=None,
                    scope=None):
  """ Pointer-Network action head.

  Args:
    inputs_query: [bs, some dim]
    inputs_ptr_mask: [bs, 600]
    inputs_entity_embed: [bs, some dim]
    ptr_out_dim:
    num_dec_blocks:
    ff_dim:
    enc_dim:
    pdtype_cls:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  with tf.variable_scope(scope, default_name='ptr_head'):
    select_logits, select_prob = _ptr_decode(
      y=inputs_query,
      memory=inputs_entity_embed,
      num_dec_blocks=num_dec_blocks,
      ff_dim=ff_dim,
      enc_dim=enc_dim,
      training=False
    )
  select_logits = tf.reshape(select_logits, [-1, ptr_out_dim])
  select_logits = tp_ops.mask_logits(logits=select_logits, mask=inputs_ptr_mask)
  if scatter_ind is not None and scatter_bs is not None:
    select_logits = tf.scatter_nd(tf.expand_dims(scatter_ind, axis=-1), select_logits,
                                  shape=[scatter_bs] + select_logits.shape[1:])

  return to_action_head(select_logits, pdtype_cls, temperature=temperature)


@add_arg_scope
def ptr_action_head_v2(inputs_query,
                       inputs_ptr_mask,
                       inputs_entity_embed,
                       inputs_func_embed,
                       ptr_out_dim,
                       pdtype_cls,
                       temperature=1.0,
                       scatter_ind=None,
                       scatter_bs=None,
                       scope=None):
  """ Pointer-Network action head.

  Args:
    inputs_query: [bs, some dim]
    inputs_ptr_mask: [bs, 600]
    inputs_entity_embed: [bs, some dim]
    ptr_out_dim:
    num_dec_blocks:
    ff_dim:
    enc_dim:
    pdtype_cls:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  with tf.variable_scope(scope, default_name='ptr_head'):
    inputs_query = tfc_layers.fully_connected(inputs_query, 256, activation_fn=None)
    inputs_query += tf.expand_dims(inputs_func_embed, axis=1)
    inputs_query = tf.nn.relu(inputs_query)
    inputs_query = tfc_layers.fully_connected(inputs_query, 32, activation_fn=None)  # per AStar
    projected_keys = tfc_layers.fully_connected(inputs_entity_embed, 32, activation_fn=None)
    # attentions (= queries * keys) as logits
    tar_logits = tf.reduce_sum(inputs_query * projected_keys, axis=-1)
    tar_logits = tp_ops.mask_logits(logits=tar_logits, mask=inputs_ptr_mask)

    if scatter_ind is not None and scatter_bs is not None:
      tar_logits = tf.scatter_nd(tf.expand_dims(scatter_ind, axis=-1), tar_logits,
                                 shape=[scatter_bs] + tar_logits.shape[1:])

  return to_action_head(tar_logits, pdtype_cls, temperature=temperature)


@add_arg_scope
def multinomial_action_head(inputs,
                            inputs_select_mask,
                            temperature=1.0,
                            scatter_ind=None,
                            scatter_bs=None,
                            scope=None):
  """Multinominal action head.  (i.e., lxhan's make_multi_bi_head)

  Args:
    inputs: [bs, some dim]
    inputs_select_mask: [bs, 600]
    pdtype_cls:
    scope:

  Returns:
    An outputs `Tensor`.
  """
  n_action_states = 2  # whether the action is executed or not
  with tf.variable_scope(scope, default_name='multinomial_action_head'):
    # this code block should not be here
    # query_h = dense_sum_blocks(inputs=inputs_query, n=4, enc_dim=enc_dim,
    #                      scope='q_res_blk')
    # query_h = tf.expand_dims(query_h, axis=1)
    # query_h = tf.tile(query_h, multiples=[
    #   1, tf.shape(inputs_entity_embed)[1], 1])
    # head_h = tf.concat([inputs_entity_embed, query_h], axis=-1)
    # head_h = dense_sum_blocks(inputs=head_h, n=4, enc_dim=enc_dim,
    #                     scope='eq_res_blk')
    head_logits = tfc_layers.fully_connected(inputs,
                                             n_action_states,
                                             scope='logits',
                                             activation_fn=None,
                                             normalizer_fn=None)

  # modify the logits that unavailable position will be -inf
  neginf = tf.zeros_like(inputs_select_mask, dtype=tf.float32) - INF
  offset = tf.where(inputs_select_mask,
                    tf.zeros_like(inputs_select_mask, dtype=tf.float32),
                    neginf)
  offset = tf.expand_dims(offset, axis=-1)
  offset = tf.concat([tf.zeros_like(offset), offset], axis=-1)
  head_logits += offset

  # hack to flatten ms's logits and reduce_sum neglogp, entropy;
  # otherwise, should use MultiCategoricalPd, which, however, is
  # slow by creating multiply CategoricalPd instances
  ## TODO: hack for backward compatibility, remove this later
  head_logits = head_logits[:, :, 1] - head_logits[:, :, 0]
  if scatter_ind is not None and scatter_bs is not None:
    head_logits = tf.scatter_nd(tf.expand_dims(scatter_ind, axis=-1), head_logits,
                                shape=[scatter_bs] + head_logits.shape[1:])
  ms_head = to_action_head(head_logits, BernoulliPdType, temperature=temperature)
  # ms_head = ActionHead(
  #   logits=tf.reshape(ms_head.logits, [-1, tf.reduce_prod(tf.shape(ms_head.logits)[1:])]),
  #   argmax=ms_head.argmax,
  #   sam=ms_head.sam,
  #   neglogp=tf.reduce_sum(ms_head.neglogp, axis=-1),
  #   pd=ms_head.pd,
  #   ent=tf.reduce_sum(ms_head.ent, axis=-1))
  return ms_head


@add_arg_scope
def sequential_selection_head(inputs,
                              inputs_select_mask,
                              input_keys,
                              input_selections,
                              max_num=64,
                              temperature=1.0,
                              forget_bias=0.0,
                              weights_initializer=ortho_init(1.0),
                              biases_initializer=tf.constant_initializer(0.0),
                              weights_regularizer=None,
                              biases_regularizer=None,
                              use_layer_norm=False,
                              scope=None):
  """Sequential Selection head using lstm and pointer network.

  Args:
    inputs: [bs, some dim], the input embeddings
    inputs_select_mask: [bs, unit_num], the last item is end_selection
    input_keys: [bs, unit_num, key_dim]
    input_selections: outer-fed selection samples, i.e., the labels or ground
      truths

  Returns:
    A head structure.
    An updated `selected units embedding` with the same shape of the inputs.
  """
  # make pre embedding
  nbatch, nin = inputs.get_shape().as_list()
  n_embed = 256
  _, unit_num, nh = input_keys.get_shape().as_list()
  # unit_num serve as the End Of Selection <EOS> token
  # nh is the dim of key and lstm's hidden state
  expand_mask = tf.cast(tf.expand_dims(inputs_select_mask, -1), tf.float32)
  mean_key = (tf.reduce_sum(input_keys * expand_mask, axis=[1])
              / (tf.reduce_sum(expand_mask, axis=[1]) + 1e-8))
  # make keys with end key
  end_key = tf.get_variable("end_key", [1, 1, nh],
                            initializer=tf.constant_initializer(0.2),
                            regularizer=None)
  input_keys = tf.concat([input_keys,
                          tf.tile(end_key, [nbatch, 1, 1])], axis=1)
  # make mask with terminal state
  inputs_select_mask = tf.concat(
    [inputs_select_mask, tf.constant([[True]] * nbatch, tf.bool)], axis=1)
  with tf.variable_scope(scope, default_name='sequential_selection_head'):
    with tf.variable_scope(scope, default_name='lstm'):
      # weights & biases
      wx = tf.get_variable("wx", [n_embed, nh * 4], initializer=weights_initializer,
                           regularizer=weights_regularizer)
      wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer,
                           regularizer=weights_regularizer)
      b = tf.get_variable("b", [nh * 4], initializer=biases_initializer,
                          regularizer=biases_regularizer)
      wkey = tf.get_variable("wkey", [nh, nin], initializer=weights_initializer,
                             regularizer=weights_regularizer)
      with tf.variable_scope('embed', reuse=tf.AUTO_REUSE) as sc_embed:
        pass
      # normalization function
      x_nf, h_nf, c_nf = None, None, None
      if use_layer_norm:
        with tf.variable_scope('x_ln', reuse=tf.AUTO_REUSE) as sc:
          x_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
        with tf.variable_scope('h_ln', reuse=tf.AUTO_REUSE) as sc:
          h_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
        with tf.variable_scope('c_ln', reuse=tf.AUTO_REUSE) as sc:
          c_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
    c = tf.constant(0.0, shape=[nbatch, nh], dtype=tf.float32)
    h = tf.constant(0.0, shape=[nbatch, nh], dtype=tf.float32)
    samples = []
    logits_list = []
    outer_fed_selection = input_selections is not None
    inputs_list = [inputs]
    select_masks = []
    for idx in range(max_num):
      select_masks.append(inputs_select_mask)
      s_embed = tfc_layers.fully_connected(inputs, n_embed, scope=sc_embed)
      c, h = one_step_lstm_op(c, h, s_embed, wx, wh, b,
                              forget_bias, x_nf, h_nf, c_nf)
      # attentions (= queries * keys) as logits
      logits = tf.reduce_sum(tf.expand_dims(h, axis=1) * input_keys, axis=-1)
      masked_logits = tp_ops.mask_logits(logits, inputs_select_mask)/temperature
      logits_list.append(masked_logits)
      if outer_fed_selection:
        sample = input_selections[:, idx]
      else:
        # sample from logits
        sample = cat_sample_from_logits(masked_logits)
      samples.append(sample)
      # update the input embedding
      index = tf.stack([tf.range(nbatch), sample], axis=1)
      inputs += tf.matmul(tf.gather_nd(input_keys, index) - mean_key, wkey)
      inputs_list.append(inputs)
      # update the selection mask
      inputs_select_mask = tf.tensor_scatter_nd_update(inputs_select_mask,
                                                       index, [False]*nbatch)
      # Mask all the units except <EOS> if the selection already ends
      with tf.xla.experimental.jit_scope(compile_ops=False):
        end_ind = tf.cast(tf.where(tf.equal(sample, unit_num)), tf.int32)
        inputs_select_mask = tf.tensor_scatter_update(
          inputs_select_mask, end_ind,
          tf.concat([tf.zeros_like(end_ind, dtype=tf.bool)] * unit_num
                    + [tf.ones_like(end_ind, dtype=tf.bool)], axis=1)
        )
    samples = tf.stack(samples, axis=1)
    select_masks = tf.stack(select_masks, axis=1)
    # finding the first <EOS> unit
    mask = tf.not_equal(samples, tf.constant(unit_num, tf.int32))
    end_indices = tf.math.minimum(
      tf.reduce_sum(tf.cast(mask, tf.int32), axis=1), max_num-1)
    loss_mask = tf.tensor_scatter_nd_update(
      mask, tf.stack([tf.range(nbatch), end_indices], axis=1), [True]*nbatch)
    # update the input embedding using the output selected units
    embed = tf.gather_nd(inputs_list,
                         tf.stack([end_indices, tf.range(nbatch)], axis=1))
    logits = tf.stack(logits_list, axis=1)
    labels = None
    if outer_fed_selection:
      x = tf.one_hot(samples, unit_num+1)
      s = tf.reduce_sum(x, axis=1, keepdims=True)
      select_labels = tf.concat([s[:, :, :-1], tf.zeros([nbatch, 1, 1])],
                                axis=-1) * tf.cast(select_masks, tf.float32)
      end_labels = tf.concat([tf.zeros([nbatch, max_num, unit_num]),
                              tf.ones([nbatch, max_num, 1])], axis=-1)
      labels = tf.where_v2(tf.expand_dims(mask, axis=-1),
                           select_labels, end_labels)
      labels = labels / tf.reduce_sum(labels, axis=-1, keepdims=True)
      samples = None
    head = to_action_head(logits, MaskSeqCategoricalPdType, nseq=max_num,
                          mask=loss_mask, labels=labels, sample=samples)
    return head, embed


@add_arg_scope
def sequential_selection_head_v2(inputs,
                                 inputs_select_mask,
                                 input_keys,
                                 input_selections,
                                 input_func_embed,
                                 max_num=64,
                                 temperature=1.0,
                                 forget_bias=0.0,
                                 weights_initializer=ortho_init(1.0),
                                 biases_initializer=tf.constant_initializer(0.0),
                                 weights_regularizer=None,
                                 biases_regularizer=None,
                                 use_layer_norm=False,
                                 scope=None):
  """Sequential Selection head using lstm and pointer network.

  Args:
    inputs: [bs, some dim], the input embeddings
    inputs_select_mask: [bs, unit_num], the last item is end_selection
    input_keys: [bs, unit_num, key_dim]
    input_selections: outer-fed selection samples, i.e., the labels or ground
      truths

  Returns:
    A head structure.
    An updated `selected units embedding` with the same shape of the inputs.
  """
  # make pre embedding
  nbatch, nin = inputs.get_shape().as_list()
  n_embed = 256
  _, unit_num, nh = input_keys.get_shape().as_list()
  # unit_num serve as the End Of Selection <EOS> token
  # nh is the dim of key and lstm's hidden state
  expand_mask = tf.cast(tf.expand_dims(inputs_select_mask, -1), tf.float32)
  mean_key = (tf.reduce_sum(input_keys * expand_mask, axis=[1])
              / (tf.reduce_sum(expand_mask, axis=[1]) + 1e-8))
  # make keys with end key
  end_key = tf.get_variable("end_key", [1, 1, nh],
                            initializer=tf.constant_initializer(0.2),
                            regularizer=None)
  input_keys = tf.concat([input_keys,
                          tf.tile(end_key, [nbatch, 1, 1])], axis=1)
  # make mask with terminal state
  inputs_select_mask = tf.concat(
    [inputs_select_mask, tf.constant([[True]] * nbatch, tf.bool)], axis=1)
  with tf.variable_scope(scope, default_name='sequential_selection_head'):
    with tf.variable_scope(scope, default_name='lstm'):
      # weights & biases
      wx = tf.get_variable("wx", [32, nh * 4], initializer=weights_initializer,
                           regularizer=weights_regularizer)
      wh = tf.get_variable("wh", [nh, nh * 4], initializer=weights_initializer,
                           regularizer=weights_regularizer)
      b = tf.get_variable("b", [nh * 4], initializer=biases_initializer,
                          regularizer=biases_regularizer)
      wkey = tf.get_variable("wkey", [nh, nin], initializer=weights_initializer,
                             regularizer=weights_regularizer)
      with tf.variable_scope('embed_fc1', reuse=tf.AUTO_REUSE) as sc_embed_fc1:
        pass
      with tf.variable_scope('embed_fc2', reuse=tf.AUTO_REUSE) as sc_embed_fc2:
        pass
      # normalization function
      x_nf, h_nf, c_nf = None, None, None
      if use_layer_norm:
        with tf.variable_scope('x_ln', reuse=tf.AUTO_REUSE) as sc:
          x_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
        with tf.variable_scope('h_ln', reuse=tf.AUTO_REUSE) as sc:
          h_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
        with tf.variable_scope('c_ln', reuse=tf.AUTO_REUSE) as sc:
          c_nf = partial(ln, epsilon=1e-5, enable_openai_impl=True, scope=sc)
    c = tf.constant(0.0, shape=[nbatch, nh], dtype=tf.float32)
    h = tf.constant(0.0, shape=[nbatch, nh], dtype=tf.float32)
    samples = []
    logits_list = []
    outer_fed_selection = input_selections is not None
    inputs_list = [inputs]
    select_masks = []
    for idx in range(max_num):
      select_masks.append(inputs_select_mask)
      s_embed = tfc_layers.fully_connected(inputs, n_embed, activation_fn=None, scope=sc_embed_fc1)
      s_embed += input_func_embed
      s_embed = tf.nn.relu(s_embed)
      s_embed = tfc_layers.fully_connected(s_embed, 32, activation_fn=None, scope=sc_embed_fc2)  # per AStar
      c, h = one_step_lstm_op(c, h, s_embed, wx, wh, b,
                              forget_bias, x_nf, h_nf, c_nf)
      # attentions (= queries * keys) as logits
      logits = tf.reduce_sum(tf.expand_dims(h, axis=1) * input_keys, axis=-1)
      masked_logits = tp_ops.mask_logits(logits, inputs_select_mask)/temperature
      logits_list.append(masked_logits)
      if outer_fed_selection:
        sample = input_selections[:, idx]
      else:
        # sample from logits
        sample = cat_sample_from_logits(masked_logits)
      samples.append(sample)
      # update the input embedding
      index = tf.stack([tf.range(nbatch), sample], axis=1)
      inputs += tf.matmul(tf.gather_nd(input_keys, index) - mean_key, wkey)
      inputs_list.append(inputs)
      # update the selection mask
      inputs_select_mask = tf.tensor_scatter_nd_update(inputs_select_mask,
                                                       index, [False]*nbatch)
      # Mask all the units except <EOS> if the selection already ends
      with tf.xla.experimental.jit_scope(compile_ops=False):
        end_ind = tf.cast(tf.where(tf.equal(sample, unit_num)), tf.int32)
        inputs_select_mask = tf.tensor_scatter_update(
          inputs_select_mask, end_ind,
          tf.concat([tf.zeros_like(end_ind, dtype=tf.bool)] * unit_num
                    + [tf.ones_like(end_ind, dtype=tf.bool)], axis=1)
        )
    samples = tf.stack(samples, axis=1)
    select_masks = tf.stack(select_masks, axis=1)
    # finding the first <EOS> unit
    mask = tf.not_equal(samples, tf.constant(unit_num, tf.int32))
    end_indices = tf.math.minimum(
      tf.reduce_sum(tf.cast(mask, tf.int32), axis=1), max_num-1)
    loss_mask = tf.tensor_scatter_nd_update(
      mask, tf.stack([tf.range(nbatch), end_indices], axis=1), [True]*nbatch)
    # update the input embedding using the output selected units
    embed = tf.gather_nd(inputs_list,
                         tf.stack([end_indices, tf.range(nbatch)], axis=1))
    logits = tf.stack(logits_list, axis=1)
    labels = None
    if outer_fed_selection:
      x = tf.one_hot(samples, unit_num+1)
      s = tf.reduce_sum(x, axis=1, keepdims=True)
      select_labels = tf.concat([s[:, :, :-1], tf.zeros([nbatch, 1, 1])],
                                axis=-1) * tf.cast(select_masks, tf.float32)
      end_labels = tf.concat([tf.zeros([nbatch, max_num, unit_num]),
                              tf.ones([nbatch, max_num, 1])], axis=-1)
      labels = tf.where_v2(tf.expand_dims(mask, axis=-1),
                           select_labels, end_labels)
      labels = labels / tf.reduce_sum(labels, axis=-1, keepdims=True)
      samples = None
    head = to_action_head(logits, MaskSeqCategoricalPdType, nseq=max_num,
                          mask=loss_mask, labels=labels, sample=samples)
    return head, embed


@add_arg_scope
def dot_prod_attention(values, query, mask):
  a = tf.stack([tf.reduce_sum(tf.multiply(v, query), axis=-1)
                for v in values], -1)
  w = tf.nn.softmax(tp_ops.mask_logits(a, mask))
  ws = tf.unstack(w, axis=-1)
  res = tf.add_n([tf.multiply(v, tf.expand_dims(ww, axis=-1))
                  for v, ww in zip(values, ws)])  # add_n'n indicates num of values
  return res


@add_arg_scope
def lstm_embed_block(inputs_x, inputs_hs, inputs_mask, nc,
                     outputs_collections=None):
  """ lstm embedding block.

  Args
    inputs_x: current state - (nrollout*rollout_len, input_dim)
    inputs_hs: hidden state - (nrollout*rollout_len, hs_len), NOTE: it's the
    states at every time steps of the rollout.
    inputs_mask: hidden state mask - (nrollout*rollout_len,)
    nc:

  Returns
    A Tensor, the lstm embedding outputs - (nrollout*rollout_len, out_idm)
    A Tensor, the new hidden state - (nrollout, hs_len), NOTE: it's the state at
     a single time step.
  """
  def consist_seq_dropout(input_seq):
    assert isinstance(input_seq, list)
    dropout_mask = tf.nn.dropout(tf.ones(shape=[nc.nrollout,
                                                input_seq[0].shape[-1]],
                                         dtype=tf.float32),
                                 keep_prob=1 - nc.lstm_dropout_rate)
    return [x * dropout_mask for x in input_seq]

  with tf.variable_scope('lstm_embed') as sc:
    # to list sequence and call the lstm cell
    x_seq = tp_ops.batch_to_seq(inputs_x, nc.nrollout, nc.rollout_len)
    # add dropout before LSTM cell TODO(pengsun): use tf.layers.dropout?
    if 1 > nc.lstm_dropout_rate > 0 and not nc.test:
      x_seq = consist_seq_dropout(x_seq)
    hsm_seq = tp_ops.batch_to_seq(tp_ops.to_float32(inputs_mask),
                                  nc.nrollout, nc.rollout_len)
    inputs_hs = tf.reshape(inputs_hs, [nc.nrollout, nc.rollout_len, nc.hs_len])
    initial_hs = inputs_hs[:, 0, :]
    if nc.lstm_cell_type == 'lstm':
      lstm_embed, hs_new = lstm(inputs_x_seq=x_seq,
                                inputs_terminal_mask_seq=hsm_seq,
                                inputs_state=initial_hs,
                                nh=nc.nlstm,
                                forget_bias=nc.forget_bias,
                                use_layer_norm=nc.lstm_layer_norm,
                                scope='lstm')
    elif nc.lstm_cell_type == 'k_lstm':
      lstm_embed, hs_new = k_lstm(inputs_x_seq=x_seq,
                                  inputs_termial_mask_seq=hsm_seq,
                                  inputs_state=initial_hs,
                                  nh=nc.nlstm,
                                  k=nc.lstm_duration,
                                  forget_bias=nc.forget_bias,
                                  use_layer_norm=nc.lstm_layer_norm,
                                  scope='k_lstm')
    else:
      raise NotImplementedError('unknown cell_type {}'.format(nc.lstm_cell_type))

    # add dropout after LSTM cell
    if 1 > nc.lstm_dropout_rate > 0 and not nc.test:
      lstm_embed = consist_seq_dropout(lstm_embed)
    lstm_embed = tp_ops.seq_to_batch(lstm_embed)

    return (
      lutils.collect_named_outputs(outputs_collections, sc.name + '_out',
                                   lstm_embed),
      lutils.collect_named_outputs(outputs_collections, sc.name + '_hs', hs_new)
    )
