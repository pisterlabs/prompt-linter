#coding:utf-8
import os
import _pickle as cPickle
import argparse

import numpy as np
import tensorflow as tf
from collections import defaultdict

from data_structure import get_batches
from hntm import HierarchicalNeuralTopicModel
from coherence import compute_coherence

def compute_freq_tokens(sess, model, bow_idxs, idx_to_word, topic_freq_tokens=None, parent_idx=0, depth=0, n_token=10, verbose=False):
    if depth == 0:
        topics_freq_indices = np.argsort(sess.run(model.topic_bow), 1)[:, ::-1][:, :n_token]
        topics_freq_idxs = bow_idxs[topics_freq_indices]
        topic_freq_tokens = {topic_idx: [idx_to_word[idx] for idx in topic_freq_idxs] for topic_idx, topic_freq_idxs in zip(model.topic_idxs, topics_freq_idxs)}
        
        # print root
        freq_tokens = topic_freq_tokens[parent_idx]
        if verbose: print(parent_idx, ' '.join(freq_tokens))
    
    child_idxs = model.tree_idxs[parent_idx]
    depth += 1
    for child_idx in child_idxs:
        freq_tokens = topic_freq_tokens[child_idx]
        if verbose: print('  '*depth, child_idx, ' '.join(freq_tokens))
        
        if child_idx in model.tree_idxs: 
            compute_freq_tokens(sess, model, bow_idxs, idx_to_word, topic_freq_tokens=topic_freq_tokens, parent_idx=child_idx, depth=depth, n_token=n_token, verbose=verbose)
            
    return topic_freq_tokens

def compute_perplexity(sess, model, batches, verbose=False):
    ppl_list = []
    for batch in batches:
        feed_dict = model.get_feed_dict(batch, mode='test')
        ppls_batch = sess.run(model.topic_ppls, feed_dict = feed_dict)
        ppl_list += list(ppls_batch)
    ppl_mean = np.exp(np.mean(ppl_list))
    if verbose: print('Perplexity = %.1f' % ppl_mean)
    return ppl_mean
                        
def compute_topic_specialization(sess, model, instances, verbose=False):
    topics_vec = sess.run(tf.nn.l2_normalize(model.topic_bow, 1))
    norm_bow = np.sum([instance.bow for instance in instances], 0)
    norm_vec = norm_bow / np.linalg.norm(norm_bow)

    topics_spec = 1 - topics_vec.dot(norm_vec)

    depth_topic_idxs = defaultdict(list)
    for topic_idx, depth in model.tree_depth.items():
        depth_topic_idxs[depth].append(topic_idx)

    depth_specs = {}
    if verbose: print('Topic Specialization:', end=' ')
    for depth, topic_idxs in depth_topic_idxs.items():
        topic_indices = np.array([model.topic_idxs.index(topic_idx) for topic_idx in topic_idxs])
        depth_spec = np.mean(topics_spec[topic_indices])
        depth_specs[depth] = depth_spec
        if verbose: print('depth %i = %.2f' % (depth, depth_spec), end=', ')
    print('')
    
    return depth_specs

def compute_hierarchical_affinity(sess, model, verbose=False):
    def get_cos_sim(parent_to_child_idxs):
        parent_child_bows = {parent_idx: np.concatenate([normed_tree_topic_bow[child_idx] for child_idx in child_idxs], 0) for parent_idx, child_idxs in parent_to_child_idxs.items()}
        cos_sim = np.mean([np.mean(normed_tree_topic_bow[parent_idx].dot(child_bows.T)) for parent_idx, child_bows in parent_child_bows.items()])
        return cos_sim    
    
    tree_topic_bow = {topic_idx: tf.nn.l2_normalize(topic_bow) for topic_idx, topic_bow in model.tree_topic_bow.items()}
    normed_tree_topic_bow = sess.run(tree_topic_bow)

    third_child_idxs = [child_idx for child_idx, depth in model.tree_depth.items() if depth==3]
    second_parent_to_child_idxs = {parent_idx:child_idxs for parent_idx, child_idxs in model.tree_idxs.items() if model.tree_depth[parent_idx] == 2}
    second_parent_to_unchild_idxs = {parent_idx: [child_idx for child_idx in third_child_idxs if child_idx not in child_idxs] for parent_idx, child_idxs in second_parent_to_child_idxs.items()}
    
    if sum(len(unchilds) for unchilds in second_parent_to_unchild_idxs.values()) > 0:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = get_cos_sim(second_parent_to_unchild_idxs)
    else:
        child_cos_sim = get_cos_sim(second_parent_to_child_idxs)
        unchild_cos_sim = 0
    
    if verbose: print('Hierarchical Affinity: child = %.2f, non-child = %.2f'%(child_cos_sim, unchild_cos_sim))
    return child_cos_sim, unchild_cos_sim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default='0')
    parser.add_argument('-path_data', type=str, default='data/bags/instances.pkl')
    parser.add_argument('-path_model', type=str, default='model/bags/checkpoint_stable/model-385000')
    parser.add_argument('-dir_corpus', type=str, default='corpus/bags')
    parser.add_argument('-batch_size', type=int, default=64)
    
    option = parser.parse_args()
    option.path_config = option.path_model + '.config'
    if os.path.split(option.dir_corpus)[-1] == '20news':
        option.window_size = 0
    else:
        option.window_size = 20
    
    os.environ["CUDA_VISIBLE_DEVICES"] = option.gpu

    # load data
    _, _, test_instances, word_to_idx, idx_to_word, bow_idxs = cPickle.load(open(option.path_data,'rb'))
    test_batches = get_batches(test_instances, option.batch_size)
    
    # load model
    print('loading %s...' % option.path_model)
    config = cPickle.load(open(option.path_config, 'rb'))
    model = HierarchicalNeuralTopicModel(config)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, option.path_model)

        freq_tokens = compute_freq_tokens(sess, model, bow_idxs, idx_to_word, verbose=True)
        compute_coherence(freq_tokens.values(), option.dir_corpus, topns=[5, 10], window_size=option.window_size, verbose=True)
        compute_perplexity(sess, model, test_batches, verbose=True)
        compute_topic_specialization(sess, model, test_instances, verbose=True)
        compute_hierarchical_affinity(sess, model, verbose=True)

        sess.close()
