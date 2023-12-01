"""
Helpers for tweet extraction/processing
"""
import csv
import gensim
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from globals import ROOT_DIR
import numpy as np
import string
import gensim.corpora as corpora
from gensim.models import CoherenceModel

TWEET_SENTENCE_SIZE = 17  # 17 is average tweet token length


def split_data(x, y, seed, verbose=True):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
    total = len(x_train) + len(x_valid) + len(x_test)
    if verbose:
        print("[Sizes] Training set: {:.2f}%, Validation set: {:.2f}%, Test set: {:.2f}%".format(
            len(x_train) / float(total) * 100,
            len(x_valid) / float(total) * 100,
            len(x_test) / float(total) * 100))

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def aggregate(start, end, file_names):
    aggregate_data = {}
    for i in range(start, end):
        path_name = os.path.join(ROOT_DIR, 'data/{}_{}.npz'.format(file_names, i))
        results = np.load(path_name, allow_pickle=True)
        print("Downloading {}, Processed {} / {}".format(path_name, i+1, end - start))
        results = results['a']
        results = results[()]
        aggregate_data = {**results, **aggregate_data}
    return aggregate_data


def extract_labels(filename):
    print("=== Extracting annotations ===")
    data = {}
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['tweet_id']] = row['maj_label']
    return data


def generate_random_embedding(embed_dim):
    return np.random.normal(scale=0.6, size=(embed_dim,))


def extract_tweets(label_data, data, experiment_flag):
    print("=== Processing tweet data maps from JSON ===")
    labels = []
    labels_map = {'hateful': 0, 'abusive': 1, 'normal': 2, 'spam': 3}
    error_count = 0
    outputs = {}
    replies = np.load(os.path.join(ROOT_DIR, 'data/reply_data.npy'), allow_pickle=True)
    replies = replies[()]

    for j, (key, value) in enumerate(data.items()):

        if int(value['id_str']) not in label_data:
            error_count += 1
            continue

        output = {}
        output['id'] = value['id_str']
        output['tweet'] = value['text']
        output['label'] = labels_map[label_data[int(value['id_str'])]]
        labels.append(output['label'])
        output['retweeted'] = int(value['retweeted'])
        output['in_reply_to_status_id'] = value['in_reply_to_status_id'] if value[
                                                                      'in_reply_to_status_id'] is not None else -1
        output['user_id'] = value['user']['id']
        output['retweet_count'] = 0 if value['retweet_count'] == 0 else np.log(value['retweet_count'])
        output['favorite_count'] = 0 if value['favorite_count'] == 0 else np.log(value['favorite_count'])
        output['label_string'] = label_data[int(value['id_str'])]

        # add context tweet
        status_id = str(output['in_reply_to_status_id'])
        if status_id in replies:
            output['context_tweet'] = replies[status_id]
        else:
            output['context_tweet'] = None

        #  tokenize / clean
        output['tokens'] = output['tweet'].translate(str.maketrans('', '', string.punctuation)).lower()
        output['tokens'] = output['tokens'].split(' ')
        if experiment_flag == 2:
            output['context_tokens'] = output['context_tweet'].translate(
                str.maketrans('', '', string.punctuation)).lower() if output['context_tweet'] else None
            output['context_tokens'] = output['context_tokens'].split() if output['context_tokens'] else None

        outputs[output['id']] = output
    return outputs, labels


def prepare_output_file(filename, output=None, file_action_key='a+', aggregate=False):
    """

    :param filename:
    :param output: dictionary to write to csv
    :param clean_flag: bool to delete existing dictionary
    :param file_action_key: w to write or a+ to append to file
    :return:
    """
    file_exists = os.path.isfile(filename)

    if output is None or output == []:
        raise ValueError("Please specify output list to write to output file.")
    with open(filename, file_action_key) as csvfile:
        fieldnames = sorted(list(output[0].keys())) # to make sure new dictionaries in diff order work okay
        if aggregate:
            fieldnames = ['title', 'epoch', 'test_f_score', 'test_f_score_hateful',
                          'num_experiments', 'test_acc', 'test_f_score_abusive', 'test_recall_hateful',
                          'test_precision', 'valid_precision', 'train_recall', 'test_recall', 'train_precision',
                          'valid_recall', 'test_recall_normal',
                          'test_loss', 'test_f_score_spam', 'test_recall_normal',
                          'test_precision_spam', 'test_precision_abusive',
                          'test_recall_spam', 'test_f_score_normal', 'test_recall_abusive',
                          'test_precision_normal', 'test_precision_hateful',
                          'train_recall_abusive', 'train_f_score_hateful', 'valid_precision_normal',
                          'train_f_score_spam',
                          'valid_f_score', 'valid_precision_abusive', 'learning_rate', 'valid_recall_normal',
                          'train_precision_spam',
                          'train_f_score_normal', 'valid_recall_abusive', 'valid_loss', 'valid_acc',
                          'train_precision_hateful',
                          'train_recall_spam', 'valid_f_score_normal', 'train_recall_normal', 'valid_recall_spam',
                          'valid_recall_hateful', 'train_loss', 'train_f_score', 'train_acc', 'train_recall_hateful',
                          'valid_precision_spam', 'train_precision_abusive', 'train_f_score_abusive',
                          'valid_f_score_spam',
                          'valid_f_score_hateful', 'valid_precision_hateful', 'valid_f_score_abusive',
                          'train_precision_normal', 'num_epochs'
                          ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists or file_action_key == 'w' or os.path.getsize(filename) == 0:
            writer.writeheader()

        for entry in output:
            writer.writerow(entry)


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def get_scores(docs, nlp, seed):
    try:
        data_lemmatized = lemmatization(docs, nlp, allowed_postags=['NOUN'])

        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)

        # Create Corpus
        texts = data_lemmatized

        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        lda_model = gensim.models.ldamulticore.LdaMulticore(corpus=corpus,
                                                            id2word=id2word,
                                                            num_topics=20,
                                                            random_state=seed,
                                                            passes=3,
                                                            workers=3)
        # # Print the Keyword in the 10 topics
        dominant_keywords = []
        for i, row_list in enumerate(lda_model[corpus]):
            row = row_list[0] if lda_model.per_word_topics else row_list
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = lda_model.show_topic(topic_num)
                    dominant_keywords.extend([word for word, prop in wp])
                else:
                    break
        topic_words = Counter([keyword for keyword in dominant_keywords if keyword not in ['-PRON-']]).most_common(10)

        # Compute Coherence Score
        coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        return lda_model.log_perplexity(corpus), coherence_lda, [word for (word, count) in topic_words]
    except:
        return None, None, None


if __name__ == "__main__":
    print('testing...')
    merge_dict1 = dict([('test_f_score_hateful', 0.2728), ('test_recall_normal', 0.6829), ('test_precision_abusive', 0.6549), ('test_recall', 0.6872), ('test_f_score', 0.7074), ('test_f_score_spam', 0.5285), ('test_precision', 0.7661), ('test_f_score_normal', 0.7728), ('test_recall_spam', 0.7737), ('test_acc', 0.6872), ('test_precision_hateful', 0.2568), ('test_precision_normal', 0.89), ('test_f_score_abusive', 0.6908), ('test_recall_abusive', 0.7315), ('test_precision_spam', 0.4015), ('test_loss', 0.7477769), ('test_recall_hateful', 0.295), ('valid_precision_abusive', 0.6506), ('train_recall_spam', 0.8113), ('valid_precision_spam', 0.3965), ('valid_f_score_spam', 0.5265), ('valid_f_score_abusive', 0.6715), ('train_loss', 0.62086684), ('valid_recall', 0.685), ('train_precision_hateful', 0.7927), ('train_f_score_spam', 0.7776), ('train_f_score', 0.7512), ('train_precision_normal', 0.6447), ('valid_f_score_hateful', 0.2707), ('valid_precision', 0.7628), ('train_f_score_hateful', 0.8066), ('train_f_score_abusive', 0.805), ('valid_f_score', 0.7047), ('learning_rate', 0.0009), ('train_precision_spam', 0.7471), ('valid_precision_hateful', 0.2642), ('valid_recall_abusive', 0.6943), ('valid_f_score_normal', 0.7728), ('train_recall_abusive', 0.7932), ('train_recall_normal', 0.5853), ('valid_recall_normal', 0.6848), ('train_precision', 0.7515), ('valid_precision_normal', 0.8869), ('train_acc', 0.7531), ('epoch', 16), ('train_recall', 0.7531), ('valid_loss', 0.76342475), ('valid_recall_spam', 0.7864), ('train_f_score_normal', 0.613), ('valid_recall_hateful', 0.2826), ('train_precision_abusive', 0.8178), ('train_recall_hateful', 0.8214), ('valid_acc', 0.685), ('seed', 28), ('title', 'CNN_test_twitter_word'), ('num_epochs', 100)])
    print(merge_dict1)
    prepare_output_file(filename='results.csv',
                        output=[merge_dict1])

    merge_dict2 = dict([('test_precision_abusive', 0.7019), ('test_precision_spam', 0.4204), ('test_recall', 0.6929), ('test_f_score_normal', 0.7814), ('test_recall_normal', 0.7078), ('test_recall_spam', 0.7012), ('test_f_score', 0.7135), ('test_precision_normal', 0.8725), ('test_loss', 0.7417837), ('test_precision_hateful', 0.23), ('test_f_score_spam', 0.5253), ('test_f_score_abusive', 0.6926), ('test_acc', 0.6929), ('test_f_score_hateful', 0.293), ('test_recall_hateful', 0.4055), ('test_recall_abusive', 0.6842), ('test_precision', 0.7595), ('train_precision', 0.76), ('valid_precision_abusive', 0.714), ('valid_precision_hateful', 0.2209), ('learning_rate', 0.0009), ('valid_f_score_hateful', 0.2806), ('valid_recall_spam', 0.7153), ('train_f_score_abusive', 0.8132), ('valid_recall', 0.6906), ('valid_f_score_abusive', 0.6968), ('epoch', 16), ('train_precision_spam', 0.7551), ('valid_acc', 0.6906), ('train_recall_hateful', 0.8455), ('train_f_score_hateful', 0.8236), ('valid_recall_abusive', 0.6806), ('train_precision_hateful', 0.8031), ('train_loss', 0.6023533), ('train_recall_spam', 0.781), ('valid_recall_normal', 0.704), ('valid_precision_spam', 0.4159), ('train_f_score', 0.7598), ('valid_f_score', 0.7122), ('train_recall_normal', 0.6189), ('train_precision_normal', 0.6514), ('train_acc', 0.7608), ('valid_precision', 0.7611), ('valid_loss', 0.7475463), ('valid_f_score_spam', 0.5256), ('train_f_score_normal', 0.6344), ('train_f_score_spam', 0.7676), ('valid_recall_hateful', 0.3863), ('valid_precision_normal', 0.8755), ('valid_f_score_normal', 0.7804), ('train_recall_abusive', 0.7993), ('train_precision_abusive', 0.8283), ('train_recall', 0.7608), ('seed', 27), ('title', 'CNN_test_twitter_word'), ('num_epochs', 100)])

    prepare_output_file(filename='results.csv',
                        output=[merge_dict2])