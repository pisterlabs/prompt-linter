import tensorflow as tf

from tensorflow import orthogonal_initializer, constant_initializer


# from OpenAI baseline
# https://github.com/openai/baselines/blob/master/baselines/a2c/utils.py
def lstm(xs, ms, s, nh, init_scale=1.0):
    nbatch, nin = [v.value for v in xs[0].get_shape()]
    wx = tf.get_variable("wx", [nin, nh*4], initializer=orthogonal_initializer(init_scale))
    wh = tf.get_variable("wh", [nh, nh*4], initializer=orthogonal_initializer(init_scale))
    b = tf.get_variable("b", [nh*4], initializer=constant_initializer(0.0))

    c, h = tf.split(axis=1, num_or_size_splits=2, value=s)
    for idx, (x, m) in enumerate(zip(xs, ms)):
        c = c*(1-m)
        h = h*(1-m)
        z = tf.matmul(x, wx) + tf.matmul(h, wh) + b
        i, f, o, u = tf.split(axis=1, num_or_size_splits=4, value=z)
        i = tf.nn.sigmoid(i)
        f = tf.nn.sigmoid(f)
        o = tf.nn.sigmoid(o)
        u = tf.tanh(u)
        c = f*c + i*u
        h = o*tf.tanh(c)
        xs[idx] = h
    s = tf.concat(axis=1, values=[c, h])
    return xs, s

# (batch_size * step_size, dim) -> (batch_size, step_size, dim) -> (step_size, batch_size, dim)
def batch_to_seq(batch, batch_size, step_size):
    seq = tf.reshape(batch, [batch_size, step_size, int(batch.shape[1])])
    seq = [tf.squeeze(v, axis=1) for v in tf.split(seq, num_or_size_splits=step_size, axis=1)]
    return seq

# (step_size, batch_size, dim) -> (batch_size, step_size, dim) -> (batch_size * step_size, dim)
def seq_to_batch(seq, batch_size, step_size):
    seq = tf.concat(seq, axis=1)
    seq = tf.reshape(seq, [step_size, batch_size, -1])
    seq = tf.transpose(seq, [1, 0, 2])
    batch = tf.reshape(seq, [batch_size * step_size, -1])
    return batch
