"""
baseline.py

This is the vanilla implementation of transformer network.

23.09.2019 - @yashbonde
"""

import tensorflow as tf
from types import SimpleNamespace

from .ops_util import get_opt
from .tf_layers import embed_sequence, positions_for, ff, noam_scheme, label_smoothing,\
    get_sine_cosine_embedding_matrix, decoder_block, encoder_block


def prepare_decoder_function(target, pad=None):
    decoder_input = shift_right_2d(target, pad)
    return decoder_input


def decoder_fn(config, dec_out, enc_out, encoder_pad_mask, decoder_pad_mask):
    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        for layer_idx in range(config.num_layers):
            dec_out = decoder_block(q=dec_out, k=enc_out, v=enc_out, enc_mask=encoder_pad_mask,
                                    dec_mask=decoder_pad_mask, scope='layer_{}'.format(layer_idx), config=config)
    return dec_out


def encoder_fn(config, enc_out, encoder_pad_mask):
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        for layer_idx in range(config.num_layers):
            enc_out = encoder_block(q=enc_out, ext_mask=encoder_pad_mask, scope='layer_{}'.format(layer_idx),
                                    config=config)
    return enc_out


def transformer(config, encoder_placeholder, target_placeholder=None, training=False):
    """
    Function for making and training baseline model. This style of code is heavily
    inspired from OpenAI's code.

    encoder_placeholder: placeholder to encoder stacks, tensor at the bottom of encoder
        [batch_size, input_length]
    target_placeholder: placeholder with target values, tensor at the top of decoder
        [batch_size, target_length]
    """
    encoder_inp = encoder_placeholder
    # decoder_inp = prepare_decoder_function(target_placeholder)  # TODO: Fix this
    decoder_inp = target_placeholder[:, :-1]

    with tf.variable_scope('embed'):
        # get context embedding matrix and position embedding matrix
        con_emb_matrix = tf.get_variable(
            'context_matrix', [config.vocab_size, config.embedding_dim])
        pos_emb_matrix = get_sine_cosine_embedding_matrix(
            max(config.cntx_len, config.max_decode_length),
            config.embedding_dim
        )

        enc_con_emb = tf.gather(con_emb_matrix, encoder_inp)  # encoder context
        dec_con_emb = tf.gather(con_emb_matrix, decoder_inp)  # decoder context
        enc_pos_emb = tf.gather(
            pos_emb_matrix,
            positions_for(encoder_inp, past_length=0))
        dec_pos_emb = tf.gather(
            pos_emb_matrix,
            positions_for(decoder_inp, past_length=0))

        # appropriate value normalisation and getting masks
        enc_con_emb *= config.embedding_dim ** 0.5
        dec_con_emb *= config.embedding_dim ** 0.5
        encoder_pad_mask = tf.math.equal(
            encoder_inp, config.pad_id, name='encoder_pad_masking')
        decoder_pad_mask = tf.math.equal(
            decoder_inp, config.pad_id, name='decoder_pad_masking')

        print('encoder_pad_maskencoder_pad_maskencoder_pad_mask: {}'.format(
            encoder_pad_mask))
        print('decoder_pad_maskdecoder_pad_maskdecoder_pad_mask: {}'.format(
            decoder_pad_mask))

        enc_out = enc_con_emb + enc_pos_emb
        dec_out = dec_con_emb + dec_pos_emb
        enc_out = tf.layers.dropout(enc_out, 0.3, training=training)
        dec_out = tf.layers.dropout(dec_out, 0.3, training=training)
        print('main_model > enc_out: {}'.format(enc_out))
        print('main_model > dec_out: {}'.format(dec_out))

    # now we make the model, this is simple matter of calling the layers --> calling encoder and decoder functions
    enc_out = encoder_fn(config=config, enc_out=enc_out,
                         encoder_pad_mask=encoder_pad_mask)
    dec_out = decoder_fn(config=config, dec_out=dec_out, enc_out=enc_out,
                         encoder_pad_mask=encoder_pad_mask, decoder_pad_mask=decoder_pad_mask)

    if config.use_inverse_embedding:
        # use the same embedding for input and output
        pred_logits = tf.matmul(dec_out, con_emb_matrix, transpose_b=True)
        fproj_w, fproj_b = con_emb_matrix, None
    else:
        # use different embedding
        pred_logits, fproj_w, fproj_b = ff(
            dec_out, 'final_projection', config.vocab_size, return_param=True)

    pred_seq = tf.argmax(pred_logits, axis=2)  # [bs, seqlen]

    if training:
        # change loss function so we can ignore the padding in loss calculation
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(target_placeholder[:, 1:], config.vocab_size),
            logits=pred_logits
        )
        non_padding = tf.cast(tf.math.logical_not(
            decoder_pad_mask), tf.float32)
        loss = tf.reduce_sum(cross_entropy * non_padding) / \
            (tf.reduce_sum(non_padding) + 1e-7)

        # train ops with gradient clipping + noam learning scheme
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(config.lr, global_step, 1000)
        # opt = get_opt(config.opt)(lr)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in opt.compute_gradients(loss)]
        # train_step = opt.apply_gradients(capped_gvs, global_step)
        train_step = tf.train.AdamOptimizer( lr, 0.9, 0.98, 1e-9).minimize(loss, global_step)
        tf.summary.scalar('lr',  lr)

        # summary
        tf.summary.scalar('loss', loss)

        ret = SimpleNamespace(
            context_embedding=con_emb_matrix,
            position_embedding=pos_emb_matrix,
            pred_seq=pred_seq,
            pred_logits=pred_logits,
            loss=loss,
            train_step=train_step,
            encoder_embedding=enc_out,
            encoder_pad_mask=encoder_pad_mask,
            decoder_pad_mask=decoder_pad_mask,
            fproj_w=fproj_w,
            fproj_b=fproj_b
        )

        return ret

    else:
        return pred_logits, pred_seq
