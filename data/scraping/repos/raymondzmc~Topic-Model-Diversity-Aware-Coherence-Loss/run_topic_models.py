import os
import random
import argparse
import itertools
from os.path import join as pjoin

import torch
import numpy as np
import gensim
import gensim.downloader as api
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm


from contextualized_topic_models.models.ctm import CombinedTM, ZeroShotTM
from contextualized_topic_models.models.pytorchavitm.avitm.avitm_model import AVITM_model
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.data_preparation import bert_embeddings_from_file
from contextualized_topic_models.evaluation.measures import TopicDiversity, CoherenceNPMI, CoherenceCV, CoherenceWordEmbeddings, InvertedRBO
from contextualized_topic_models.utils.visualize import save_word_dist_plot, save_histogram
from composite_activations import composite_activations

import pdb

# Disable tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(0)

def load_dataset(text_file, bow_file):
    input_text = []
    input_bow = []
    vocab = set()

    with open(text_file, 'r') as f:
        for line in f:
            input_text.append(line.rstrip('\n'))

    with open(bow_file, 'r') as f:
        for line in f:
            input_bow.append(line.rstrip('\n'))
            vocab.update(line.split())

    assert len(input_text) == len(input_bow), \
        f"The number of lines in \"{text_file}\" ({len(input_text)}) does not match the number of lines in {bow_file} ({len(input_bow)})!"

    print(f"Successfully read {len(input_text)} documents with a vocab of {len(vocab)}.")
    return input_text, input_bow


def evaluate(topics, texts, embeddings_path=None):
    texts = [doc.split() for doc in texts]

    npmi_metric = CoherenceNPMI(texts=texts, topics=topics)
    npmi_score = npmi_metric.score()
    # cv_metric = CoherenceCV(texts=texts, topics=topics)
    # cv_score = cv_metric.score()
    we_metric = CoherenceWordEmbeddings(topics=topics)
    we_score = we_metric.score()
    irbo_metric = InvertedRBO(topics=topics)
    irbo_score = irbo_metric.score()
    td_metric = TopicDiversity(topics=topics)
    td_score = td_metric.score(topk=10)
    return npmi_score, we_score, irbo_score, td_score


def main(args):
    dataset_name = os.path.basename(args.text_file).split('.')[0].split('_')[0]
    model_name = args.model_type

    text_for_contextual, text_for_bow = load_dataset(args.text_file, args.bow_file)
    bow_corpus = [doc.split() for doc in text_for_bow]

    qt = TopicModelDataPreparation("all-mpnet-base-v2", device=args.device)
    
    
    dataset_cache_file = pjoin(args.cache_path, f"{dataset_name}-{args.input_embeddings}.pt")
    if os.path.isfile(dataset_cache_file):
        training_dataset = torch.load(dataset_cache_file)
        print(f"Loaded processed dataset at \"{dataset_cache_file}\".")
    else:
        # Use custom embeddings for GoogleNews
        if dataset_name == 'GoogleNews':
            custom_embeddings_path = 'contextualized_topic_models/data/gnews/bert_embeddings_gnews'
            custom_embeddings = np.load(custom_embeddings_path, allow_pickle=True)
            training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow, custom_embeddings=custom_embeddings)
        else:
            training_dataset = qt.fit(text_for_contextual=text_for_contextual, text_for_bow=text_for_bow)
        print(f"Finished processed dataset!")
        torch.save(training_dataset, dataset_cache_file)

    vocab_mask, npmi_matrix, word_vectors = None, None, None
    if args.use_npmi_loss:
        model_name += '-npmiloss'
        npmi_cache_file = pjoin(args.cache_path, f"{dataset_name}_npmi_matrix.pt")
        if os.path.isfile(npmi_cache_file):
            npmi_matrix = torch.load(npmi_cache_file)
            print(f"Loaded NPMI matrix at \"{npmi_cache_file}\"")
        else:
            print("Computing NPMI matrix...")
            
            vocab_size = len(training_dataset.vocab)
            npmi_matrix = np.zeros((vocab_size, vocab_size))
            dictionary = Dictionary()
            dictionary.id2token = training_dataset.idx2token
            dictionary.token2id = {v:k for k, v in training_dataset.idx2token.items()}
            topics = [list(training_dataset.idx2token.values())]
            npmi = CoherenceModel(topics=topics, texts=bow_corpus, dictionary=dictionary, coherence='c_npmi', topn=len(topics[0]))
            segmented_topics = npmi.measure.seg(npmi.topics)
            accumulator = npmi.estimate_probabilities(segmented_topics)
            num_docs = accumulator.num_docs
            eps = 1e-12
            for w1, w2 in tqdm(segmented_topics[0]):
                w1_count = accumulator[w1]
                w2_count = accumulator[w2]
                co_occur_count = accumulator[w1, w2]
                
                p_w1_w2 = co_occur_count / num_docs
                p_w1 = (w1_count / num_docs)
                p_w2 = (w2_count / num_docs)
                npmi_matrix[w1, w2] = np.log((p_w1_w2 + eps) / (p_w1 * p_w2)) / -np.log(p_w1_w2  + eps)
            torch.save(npmi_matrix, npmi_cache_file)

            # input_occurrence = (training_dataset.X_bow > 0).astype(int)
            # occurrence_matrix = input_occurrence.sum(0).T.dot(input_occurrence.sum(0))
            # cooccurrence_matrix = input_occurrence.T.dot(input_occurrence).todense()
            # # num_docs = 
            # sim_ratio = (cooccurrence_matrix / 2000) / (occurrence_matrix / 2000**2)
            # sim_ratio = np.where(sim_ratio > 0, np.log2(sim_ratio), 0)
            # sim_ratio[sim_ratio > 0] = np.log2(sim_ratio[sim_ratio > 0])
            # np.fill_diagonal(cooccurrence_matrix, 0)
            # token2idx = {v:k for k, v in training_dataset.idx2token.items()}
            # vocab_size = len(training_dataset.vocab)
            # cooccurrence_matrix = torch.zeros(vocab_size, vocab_size)
            # occurrence_matrix = torch.zeros(vocab_size)
            # for doc in tqdm(text_for_bow):
            #     word_indices = [token2idx[x] for x in doc.split()]
            #     occurrence_matrix[word_indices] += 1
            #     for idx1, idx2 in itertools.combinations(word_indices, 2):
            #         cooccurrence_matrix[idx1, idx2] += 1
            #         cooccurrence_matrix[idx2, idx1] += 1
        
        # pdb.set_trace()
        # cooccurr_matrix = cooccurrence_matrix /\
        #     torch.matmul(occurrence_matrix.unsqueeze(1), occurrence_matrix.unsqueeze(0))
        
        # dist_cache_file = pjoin(args.cache_path, f"{dataset_name}_dist_matrix.pt")
        # if os.path.isfile(dist_cache_file):
        #     dist_matrix, vocab_mask = torch.load(dist_cache_file)
        #     print(f"Loaded embedding pairwise distance matrix at \"{dist_cache_file}\"")
        # else:
        #     print("Computing embedding distance matrix...")
        #     wv = api.load('word2vec-google-news-300')
        #     word_vectors = np.zeros((len(training_dataset.idx2token), wv.vector_size))
        #     missing_indices = []
        #     for idx, token in training_dataset.idx2token.items():
        #         if wv.has_index_for(token):
        #             word_vectors[idx] = wv.get_vector(token)
        #         else:
        #             missing_indices.append(idx)
            
        #     # Use the mean vector for OOV tokens
        #     vocab_mask = np.ones(len(word_vectors), dtype=bool)
        #     vocab_mask[missing_indices] = False
        #     # word_vectors[missing_mask] = word_vectors[~missing_mask].mean(axis=0)
        
        #     # Compute normalized distance matrix
        #     dist_matrix = cosine_distances(word_vectors)
        #     for row_idx, row in enumerate(dist_matrix):
        #         dist_matrix[row_idx] = (row - row.min()) / (row.max() - row.min())
        #     torch.save((dist_matrix, vocab_mask), dist_cache_file)
        # dist_matrix = (dist_matrix - dist_matrix.min()) / (dist_matrix.max() - dist_matrix.min())
    elif args.use_glove_loss:
        model_name += f'-gloveloss{args.weight_lambda}'
        glove_path = 'contextualized_topic_models/data/glove.6B.50d.w2vformat.txt'
        wv = gensim.models.KeyedVectors.load_word2vec_format(glove_path)
        word_vectors = np.zeros((len(training_dataset.idx2token), wv.vector_size))
        missing_indices = []
        for idx, token in training_dataset.idx2token.items():
            if wv.has_index_for(token):
                word_vectors[idx] = wv.get_vector(token)
            else:
                missing_indices.append(idx)
    
    if args.use_diversity_loss:
        model_name += f'-diversityloss{args.weight_alpha}'

    
    topics = [25, 50, 75, 100, 150]
    for num_topics in topics:
        npmi_scores, we_scores, irbo_scores, td_scores = [], [], [], []

        # Results path for the current run
        results_path = pjoin(args.results_path, f"{dataset_name}-{model_name}", f'{num_topics}topics')
        if args.use_mdkp:
            npmi_scores_mdkp, we_scores_mdkp, irbo_scores_mdkp, td_scores_mdkp = [], [], [], []
        os.makedirs(results_path, exist_ok=True)
        scores_file = pjoin(results_path, 'evaluation_scores.out')
        fp = open(scores_file, 'a+')

        for seed in range(args.num_seeds):
            set_random_seed(seed)

            # Concatenate BoW input with embeddings in CombinedTM (https://aclanthology.org/2021.acl-short.96.pdf)
            if args.model_type == 'combined':
                tm = CombinedTM(
                    bow_size=len(training_dataset.idx2token),
                    contextual_size=768,
                    n_components=num_topics,
                    num_epochs=args.num_epochs,
                    device=args.device,
                    use_npmi_loss=args.use_npmi_loss,
                    npmi_matrix=npmi_matrix,
                    vocab_mask=vocab_mask,
                    use_diversity_loss=args.use_diversity_loss,
                    use_glove_loss=args.use_glove_loss,
                    word_vectors=word_vectors,
                    loss_weights={"lambda": args.weight_lambda, "beta": 1, "alpha": args.weight_alpha},
                )

            # Use only contextualized embeddings in ZeroShotTM (https://aclanthology.org/2021.eacl-main.143.pdf)
            elif args.model_type == 'zeroshot':
                tm = ZeroShotTM(
                    bow_size=len(training_dataset.idx2token),
                    contextual_size=768,
                    n_components=num_topics,
                    num_epochs=args.num_epochs,
                    device=args.device,
                    use_npmi_loss=args.use_npmi_loss,
                    npmi_matrix=npmi_matrix,
                    vocab_mask=vocab_mask,
                    use_diversity_loss=args.use_diversity_loss,
                    use_glove_loss=args.use_glove_loss,
                    word_vectors=word_vectors,
                    loss_weights={"lambda": args.weight_lambda, "beta": 1, "alpha": args.weight_alpha},
                )
            
            elif args.model_type == 'prodlda':
                tm = AVITM_model(
                    input_size=len(training_dataset.idx2token),
                    num_topics=num_topics,
                    num_epochs=args.num_epochs,
                    device=args.device,
                    use_npmi_loss=args.use_npmi_loss,
                    npmi_matrix=npmi_matrix,
                    vocab_mask=vocab_mask,
                    use_diversity_loss=args.use_diversity_loss,
                    use_glove_loss=args.use_glove_loss,
                    word_vectors=word_vectors,
                    loss_weight={"lambda": args.weight_lambda, "beta": 1, "alpha": args.weight_alpha},
                )
            else:
                raise Exception("Not implemented")

            

            
            output_path = pjoin(results_path, f'seed-{seed}-model_output.pt')
            if os.path.exists(output_path):
                model_output = torch.load(output_path)
                topics = model_output['topics']
            else:
                tm.fit(training_dataset)
                topics = [v for _, v in tm.get_topics(10).items()]
                model_output = {
                    'topics': topics,
                    'topic-document-matrix': tm.get_doc_topic_distribution(training_dataset),
                    'topic-word-matrix': tm.model.beta.detach().cpu().numpy().T,
                }
                torch.save(model_output, output_path)
    
            scores = evaluate(topics, text_for_bow, embeddings_path=None)
            print("Scores:", scores)
            npmi_scores.append(scores[0])
            # cv_scores.append(scores[1])
            we_scores.append(scores[1])
            irbo_scores.append(scores[2])
            td_scores.append(scores[3])
            
            fp.write(f'scores: {scores}\n')
            
            if args.use_mdkp and num_topics <= 75:
                optimized_topics = composite_activations(
                    model_output['topic-word-matrix'],
                    model_output['topic-document-matrix'],
                    training_dataset.vocab,
                    bow_corpus,
                    num_topics,
                )
                mdkp_scores = evaluate(optimized_topics, text_for_bow, embeddings_path=None)
                print("MDKP optimized Scores:", mdkp_scores)
                npmi_scores_mdkp.append(mdkp_scores[0])
                we_scores_mdkp.append(mdkp_scores[1])
                irbo_scores_mdkp.append(mdkp_scores[2])
                td_scores_mdkp.append(mdkp_scores[3])
                fp.write(f'mdkp optimized scores: {mdkp_scores}\n')
                
            
            

            if args.plot_word_dist:
                img_path = pjoin(results_path, 'plots')
                for topic_idx, beta in enumerate(tm.model.beta):
                    if topic_idx >= 10:
                        break
                    save_word_dist_plot(
                        torch.softmax(beta, 0), training_dataset.vocab, 
                        pjoin(img_path, f"topic-{topic_idx}.jpg"),
                        top_n=100)
            # pdb.set_trace()
                    
        print(f"[{num_topics}-Topics]")
        print("Average Scores (NPMI, WE, I-RBO, TD):")
        print(f"{np.mean(npmi_scores)}\t{np.mean(we_scores)}\t{np.mean(irbo_scores)}\t{np.mean(td_scores)}")
        fp.write(f"\nAverage scores over {args.num_seeds} seeds (NPMI, WE, I-RBO, TD):\n")
        fp.write(f"{np.mean(npmi_scores)}\t{np.mean(we_scores)}\t{np.mean(irbo_scores)}\t{np.mean(td_scores)}\n")
        
        if args.use_mdkp:
            print("Average MDKP Optimized Scores (NPMI, WE, I-RBO, TD):")
            print(f"{np.mean(npmi_scores_mdkp)}\t{np.mean(we_scores_mdkp)}\t{np.mean(irbo_scores_mdkp)}\t{np.mean(td_scores_mdkp)}")
            fp.write(f"\nAverage MDKP optimized scores over {args.num_seeds} seeds (NPMI, WE, I-RBO, TD):\n")
            fp.write(f"{np.mean(npmi_scores_mdkp)}\t{np.mean(we_scores_mdkp)}\t{np.mean(irbo_scores_mdkp)}\t{np.mean(td_scores_mdkp)}\n")
        fp.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--text_file", help="Unprocessed text file path", default=pjoin('resources', '20news_unprep.txt'), type=str)
    parser.add_argument("--bow_file", help="Processed bag-of-words file path", default=pjoin('resources', '20news_prep.txt'), type=str)

    parser.add_argument("--model_type", help="Type of model to run", default='zeroshot', choices=['prodlda', 'zeroshot', 'combined'])
    parser.add_argument("--input_embeddings", help="Name for pretrained embedding model for dense input", default="all-mpnet-base-v2", type=str)
    parser.add_argument("--device", help="Device for model training/inference", default=None)

    # Experiments
    parser.add_argument("--cache_path", help="Path for caching", type=str, default='.cache')
    parser.add_argument("--num_seeds", help="Number of random seeds for each topic number", type=int, default=10)
    parser.add_argument("--results_path", help="Path for saving results", type=str, default='results')
    parser.add_argument("--plot_word_dist", help="Visualize topic-word distribution over vocab",  action='store_true')

    # CTM hyperparameters (default to implementation from orginal paper)
    parser.add_argument("--hidden_sizes", help="Hidden size", default=(100, 100), type=tuple)
    parser.add_argument("--activation", help="Activation function", default='softplus', type=str, choices=['softplus', 'relu'])
    parser.add_argument("--dropout", help="Dropout rate", default=0.2, type=float)
    parser.add_argument("--batch_size", help="Batch size for training", default=100, type=int)
    parser.add_argument("--lr", help="Learning rate", default=2e-3, type=float)
    parser.add_argument("--momentum", help="Momentum for optimizer", default=0.99, type=float)
    parser.add_argument("--num_epochs", help="Number of epochs for training", default=100, type=int)

    parser.add_argument("--use_npmi_loss", help="Use NPMI loss", action='store_true')
    parser.add_argument("--use_diversity_loss", help="Use Diversity loss", action='store_true')
    parser.add_argument("--weight_alpha", help="Weight for diversity loss", type=float, default=0.5)
    parser.add_argument("--weight_lambda", help="Weight for distance loss", type=float, default=100)
    parser.add_argument("--divergence_loss", help="Use topic divergence loss", action='store_true')
    parser.add_argument("--contextualize_beta", help="Model beta as a function of input", action='store_true')
    
    parser.add_argument("--use_glove_loss", help="Use Glove Embeddings for distance loss", action='store_true')
    parser.add_argument("--use_mdkp", help="Use Multi-Dimensional Knapsack Problem algorithm for optimizing NPMI post training", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.cache_path):
        os.makedirs(args.cache_path)
    
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    try:
        args.device = torch.device(f'cuda:{int(args.device)}')
    except:
        args.device = 'cpu'
    main(args)
