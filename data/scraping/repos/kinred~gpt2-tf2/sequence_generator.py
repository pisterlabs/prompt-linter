import tensorflow as tf
import sentencepiece as spm
from gpt2_model import Gpt2
import json
import click
import os

def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1]

    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )

    # Nucleas Sampling (https://arxiv.org/pdf/1904.09751.pdf)


def top_p_logits(logits, p):
    """Took from OpenAI GPT-2 Implememtation"""
    batch = tf.shape(logits)[0]
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
    indices = tf.stack([
        tf.range(0, batch),
        tf.maximum(tf.reduce_sum(tf.cast(cumulative_probs <= p, tf.int32), axis=-1) - 1, 0),
    ], axis=-1)
    min_values = tf.gather_nd(sorted_logits, indices)
    return tf.where(
        logits < min_values,
        tf.ones_like(logits) * -1e10,
        logits,
    )

class SequenceGenerator:

    def __init__(self, model_dir, vocab_file):
        self.sp = None
        self.model = None
        self.model_dir = model_dir
        self.vocab_file = vocab_file

    def load_model(self):

        self.model = Gpt2.create_from_params(self.model_dir)

        ckpt = tf.train.Checkpoint(model=self.model)

        ckpt_manager = tf.train.CheckpointManager(ckpt, self.model_dir, max_to_keep=100)

        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
        print('Model weights loaded into memory')

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.vocab_file)

    def sample_sequence(self,
                        context=None,
                        seq_len=512,
                        bos=3,
                        eos=4,
                        temperature=1,
                        top_k=0,
                        top_p=0,
                        nucleus_sampling=True):

        if context == None:
            print("Give some context to model.................")
            return
        context = tf.expand_dims(([bos] + self.sp.encode_as_ids(context)), 0)
        prev = context
        output = context
        pasts = None
        for i in range(seq_len):
            logits, pasts = self.model(prev, training=False, pasts=pasts)
            # print("LOGITS: " + str(logits))
            logits = logits[:, -1, :] / tf.cast(temperature, tf.float32)
            # print("PAST: " + str(tf.shape(past)))
            logits = top_k_logits(logits, k=top_k)
            # print(logits)
            if nucleus_sampling:
                logits = top_p_logits(logits, p=top_p)

            samples = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
            # print(samples)
            if tf.equal(samples, eos):
                print("Predicted end of sequence.")
                break

            output = tf.concat([output, samples], axis=1)
            prev = output
            # print(tf.shape(output))
            # print(output)

        # print("--------------------------")
        result = tf.squeeze(output, axis=0)
        pred = [int(i) for i in result]
        generated_seq = self.sp.decode_ids(pred[1:])
        generated_seq = generated_seq.replace("[SEP]", "").strip()
        generated_seq = ' '.join(generated_seq.split())
        return generated_seq

@click.command()
@click.option('--model-dir', type=str, default="./model", show_default=True, help="Model Path")
@click.option('--vocab', type=str, default="./data/bpe_model.model", show_default=True, help="Vocab")
@click.option('--seq-len', type=int, default=512, show_default=True, help="seq_len")
@click.option('--temperature', type=float, default=1.0, show_default=True, help="seq_len")
@click.option('--top-k', type=int, default=8, show_default=True, help="seq_len")
@click.option('--top-p', type=float, default=0.9, show_default=True, help="seq_len")
@click.option('--nucleus_sampling', type=bool, default=True, show_default=True, help="seq_len")
@click.option('--context', type=str, default="", show_default=True, help="Context given to model")
def seq_gen(model_dir, vocab, seq_len, temperature, top_k, top_p, nucleus_sampling, context):
    model_dir = os.path.abspath(model_dir)
    sg = SequenceGenerator(model_dir, vocab)
    sg.load_model()
    generated_seq = sg.sample_sequence(context,
                                     seq_len=seq_len,
                                     temperature=temperature,
                                     top_k=top_k,
                                     top_p=top_p,
                                     nucleus_sampling=nucleus_sampling)

    print(generated_seq)


if __name__ == "__main__":
    seq_gen()
