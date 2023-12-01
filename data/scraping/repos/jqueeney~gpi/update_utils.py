"""Helper functions used by algorithms for policy updates."""
import numpy as np
import tensorflow as tf

def cg(f_Ax, b, cg_iters=20, residual_tol=1e-10):
    """Conjugate gradient sub-routine sourced from OpenAI Baselines."""
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for _ in range(cg_iters):
        z = f_Ax(p).numpy()
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x.astype('float32')

def make_F(actor,s_all,weights_all,sub,damp=0.0):
    """Creates matrix-vector product function for average FIM."""

    s_sub = s_all[::sub]
    weights_sub = weights_all[::sub]

    kl_info_ref = actor.get_kl_info(s_sub)

    def F(x):
        with tf.GradientTape() as outtape:
            with tf.GradientTape() as intape:
                kl = actor.kl(s_sub,kl_info_ref)
                kl_loss = tf.reduce_mean(weights_sub*kl)
            grads = intape.gradient(kl_loss,actor.trainable)
            grad_flat = tf.concat([tf.reshape(grad,[-1]) for grad in grads],-1)
            output = tf.reduce_sum(grad_flat * x)
        result = outtape.gradient(output,actor.trainable)
        result_flat = tf.concat([tf.reshape(grad,[-1]) for grad in result],-1)
        result_flat += damp * x

        return result_flat

    return F