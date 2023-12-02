# -*- coding: utf-8 -*-
import time
from Model import *
from coherenceScore import cal_coherence


def train_CS_BTM():
    print("===== Run BTM, Topic Number=" + str(K) + ", alpha=" + str(alpha) + ", beta=" +
          str(beta) + ", n_iter=" + str(n_iter) + ", save_step=" + str(save_step) + "=====")

    clock_start = time.process_time()
    model = Model(K, alpha, beta, n_iter, save_step)
    model.train(doc_pt, output_dir)
    clock_end = time.process_time()

    print("procedure time : %f seconds" % (clock_end - clock_start))

    return model


def display_biterm(bs, vocal):
    voc = {}
    for i, line in enumerate(open(vocal).readlines()):
        wid, word = line.strip().split()
        voc[i] = word

    for bi in bs:
        w1 = bi.get_wi()    # 词对中的一个词序号
        w2 = bi.get_wj()    # 词对中的第二个词序号
        print("%s\t%s\t%d" % (voc[w1], voc[w2], bi.get_z()))


if __name__ == "__main__":
    K = 7
    alpha = 0.1
    beta = 0.01
    n_iter = 30
    save_step = 10

    output_dir = "../output/"
    doc_pt = "../../data/test_tweet_1w.dat"                   # input documents
    model_dir = "../output/model/"                            # dictionary to save model
    vocabulary_path = output_dir + "vocabulary.txt"     # generated vocabulary

    print("\n\n================ Topic Learning =============")
    my_model = train_CS_BTM()
    # display_biterm(my_model.bs, vocabulary_path)

    print("\n\n================ Topic Inference =============")
    # my_model = Model(K, alpha, beta, n_iter, save_step)
    topic_dict = my_model.infer(doc_pt, model_dir, vocabulary_path)

    cal_coherence(topic_dict, my_model.topic_words, my_model.indexToWord)
