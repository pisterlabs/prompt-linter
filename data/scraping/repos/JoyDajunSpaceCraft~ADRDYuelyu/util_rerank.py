import os
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import json
from scipy.interpolate import make_interp_spline, BSpline, interp1d

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import numpy as np


def use_langchain_rerank(post):
    llm = OpenAI(temperature=0.9, openai_api_key="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")
    prompt = PromptTemplate(
        input_variables=["post"],
        template="Image you are a alzheimer's expert, here is the post from Reddit, can you help me find the information want in this post?"
                 "{post}",
    )

    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain only specifying the input variable.
    print(chain.run(post))


def find_long_memoryloss():
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/new_all_adrd.csv")
    post = df.selftext.values.tolist()
    idxs = df.id.values.tolist()
    title = df.title.values.tolist()

    keywords = ["memory loss", "forget", "forgot", "forgotten"]

    with open("long_text", "w") as f:

        for index, i in enumerate(post):
            if len(sent_tokenize(i)) > 20:
                for keyword in keywords:
                    if keyword in i:
                        f.write(idxs[index] + "\n")
                        f.write(title[index] + "\n")
                        f.write(post[index] + "\n")
                        print("matched index is: ", idxs[index])
                        print("matched title is: ", title[index])
                        print("matched post is: ", post[index])
                        continue


def get_new_order_longpost():
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/new_order.csv")
    pre_new_doc_ids = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/id_comparation.csv")

    old_ids = pre_new_doc_ids.Old_ids.values.tolist()
    shorten_post = df.shorten_post.values.tolist()
    idx = df.reddit_id.values.tolist()
    new_long_idx = []
    for index, i in enumerate(shorten_post):
        if len(sent_tokenize(i)) > 20:
            new_long_idx.append(idx[index])
    print(set(new_long_idx) & set(old_ids))


def find_old_order():
    old_post = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/old_order.csv")
    pre_new_doc_ids = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/id_comparation.csv")
    new_post = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/new_order.csv")
    id_memory_loss = pre_new_doc_ids.Ismemoryloss.values.tolist()

    old_id = old_post.reddit_id.values.tolist()
    old_posts = old_post.shorten_post.values.tolist()
    # new_top50_ids = pre_new_doc_ids.New_ids.values.tolist()
    new_ids = new_post.reddit_id.values.tolist()
    for idx, i in enumerate(new_ids):
        res = find_shorten()
        # print(res)
        if i in old_id:
            # if i in old_id and i in res:
            post_length = len(sent_tokenize(old_post.loc[old_post.reddit_id == i].shorten_post.values[0]))
            # if post_length>15 and post_length < 20:
            print("the index over lap is ", i)
            print("the new rank is ", idx + 1)
            print("the old rank is ", old_post.loc[old_post.reddit_id == i].index.values[0] + 1)
            if i in pre_new_doc_ids.Old_ids.values.tolist():
                is_memory_loss = pre_new_doc_ids.loc[pre_new_doc_ids.Old_ids == i].Ismemoryloss.values[0]
                print("Is from memory loss", is_memory_loss)
            else:
                print("This post is not in the old id")


def find_shorten():
    all_shorten = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/all_shorthen_post.csv")
    hiws = all_shorten.hiw.values.tolist()
    ids = all_shorten.reddit_id.values.tolist()
    shortens = []
    for id, hiw in zip(ids, hiws):
        if type(hiw) != float:
            shortens.append(id)
    # print(len(shortens))
    return shortens


def precision_at_k_sklearn(df: pd.DataFrame, k: int = 2, y_test: str = 'y_actual') -> float:
    y_true = df.head(k + 1)[y_test]
    y_pred = [1] * (k + 1)
    i = 0
    while i < len(y_true):
        # replace hardik with shardul
        if y_true[i] == 2:
            y_true[i] = 1

        # replace pant with ishan
        if y_true[i] == 3:
            y_true[i] = 1
        i += 1

    return precision_score(y_true, y_pred)


def recall_at_k_sklearn(df: pd.DataFrame, k: int = 2, y_test: str = 'y_actual') -> float:
    y_true = df.head(k + 1)[y_test]
    i = 0
    while i < len(y_true):
        # replace hardik with shardul
        if y_true[i] == 2:
            y_true[i] = 1

        # replace pant with ishan
        if y_true[i] == 3:
            y_true[i] = 1
        i += 1

    y_pred = [1] * (k + 1)
    # print("k in the recall", k+1)
    # print("true value in the recall is", y_true)
    # print("predict value in the recall is", y_pred)

    return recall_score(y_true, y_pred)


def precision_at_k(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
    """
    Function to compute precision@k for an input boolean dataframe

    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output

    Output:
        Floating-point number of precision value for k items
    """
    # check we have a valid entry for k
    if k <= 0:
        raise ValueError('Value of k should be greater than 1, read in as: {}'.format(k))
    # check y_test & y_pred columns are in df
    if y_test not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_test))
    if y_pred not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_pred))

    # extract the k rows
    dfK = df.head(k)
    # compute number of recommended items @k
    denominator = dfK[y_pred].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None


def recall_at_k(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual', y_pred: str = 'y_recommended') -> float:
    """
    Function to compute recall@k for an input boolean dataframe

    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output

    Output:
        Floating-point number of recall value for k items
    """
    # check we have a valid entry for k
    if k <= 0:
        raise ValueError('Value of k should be greater than 1, read in as: {}'.format(k))
    # check y_test & y_pred columns are in df
    if y_test not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_test))
    if y_pred not in df.columns:
        raise ValueError('Input dataframe does not have a column named: {}'.format(y_pred))

    # extract the k rows
    dfK = df.head(k)
    # compute number of all relevant items

    denominator = df[y_test].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None


def precision_at_k_new(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual') -> float:
    # extract the k rows
    dfK = df.head(k)
    # compute number of recommended items @k
    denominator = k
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None


def recall_at_k_new(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual') -> float:
    # extract the k rows
    dfK = df.head(k)
    # compute number of all relevant items
    count = 0

    for j in dfK[y_test].values.tolist():
        if j == 1:
            count += 1
    denominator = count
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator / denominator
    else:
        return None


def average_precision(df: pd.DataFrame, y_test: str = 'y_actual', y_pred: str = 'y_recommended'):
    precision_list = 0

    for k in range(50):
        # extract the k rows
        dfK = df.head(k)
        # print(dfK[y_test])
        # compute number of recommended items @k
        denominator = dfK[y_test].sum()
        # print("k is ", k)
        # print("denominator",denominator)
        precision_list += denominator / (k + 1)
        # print("precision_list",precision_list)

    k = 50
    dfK = df.head(k)
    true_number = dfK[y_pred].sum()

    # print("ap", precision_list / true_number)
    return precision_list / true_number


def ap_for_k(df: pd.DataFrame, k: int = 3, y_test: str = 'y_actual'):
    precision_list = 0
    # extract the k rows

    # print(dfK[y_test])
    # compute number of recommended items @k
    count = 0
    for i in range(k):
        dfi = df.head(i)
        for j in dfi[y_test].values.tolist():
            if j == 1:
                count += 1
        precision_list += count / (i + 1)

    print("ap", (precision_list / ((k + 1) * 10)))

    return (precision_list / ((k + 1) * 10))


def match_result_id_with_50():
    # gpt_df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/gpt_new_order.csv")
    bart_df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/bart_new_order.csv")
    old_post = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/new_Plot_basd.csv")
    old_ids = old_post.Old_ids.values.tolist()
    true_false = old_post.oldIsmemoryloss.values.tolist()
    bart_true_false = []
    # gpt_true_false = []
    bart_rank = bart_df.reddit_id.values.tolist()
    # gpt_rank = bart_df.reddit_id.values.tolist()
    # for i in gpt_rank:
    #     gpt_true_false.append(true_false[old_ids.index(i)])

    for i in bart_rank:
        bart_true_false.append(true_false[old_ids.index(i)])

    old_post["bart_id"] = bart_rank[:50]
    old_post["bartIsmemoryloss_top50"] = bart_true_false[:50]
    # old_post["gpt_id"] = gpt_rank[:50]
    # old_post["gptIsmemoryloss_top50"] = gpt_true_false[:50]

    old_post.to_csv("new_Plot_basd.csv", index=False)


def plot_main():
    # df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/Plot_basd.csv")
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/new_Plot_basd.csv")
    old_r = []
    old_p = []
    old_f1 = []
    old_ap = []
    new_r = []
    new_p = []
    new_f1 = []
    new_ap = []
    bart_ap = []
    gpt_ap = []
    # plt.style.use('_mpl-gallery')
    for i in range(50):
        old_rk = recall_at_k(df, i + 1, "oldIsmemoryloss", "old_predict")
        old_r.append(old_rk)
        old_pk = precision_at_k(df, i + 1, "oldIsmemoryloss", "old_predict")
        old_p.append(old_pk)
        new_rk = recall_at_k(df, i + 1, "newIsmemoryloss_top50", "new_predict")
        new_r.append(new_rk)
        new_pk = precision_at_k(df, i + 1, "newIsmemoryloss_top50", "new_predict")
        new_p.append(new_pk)
        # new_f1.append(2*new_pk*new_rk/(new_pk+new_rk))
        # old_f1.append(2*old_pk*old_rk/(old_pk+old_rk))

    old_ap = average_precision(df, "oldIsmemoryloss", "old_predict").round(2)
    new_ap = average_precision(df, "newIsmemoryloss_top50", "new_predict").round(2)
    bart_ap = average_precision(df, "bartIsmemoryloss_top50", "new_predict").round(2)
    gpt_ap = average_precision(df, "gptIsmemoryloss_top50", "new_predict").round(2)

    x_row = [i for i in range(1, 51)]

    # print("The average in new is {} and index is {}".format(sum(old_ap) / 50, new_f1.index(max(new_f1))))
    # print("The average in old is {} and index is {}".format(sum(new_p) / 50, old_f1.index(max(old_f1))))

    # plt.plot(x1, y1, label="old recall")
    # plt.plot(x2, y2, label="old precision")
    # plt.plot(x3, y3, label="new recall")
    # plt.plot(x4, y4, label="new precision")
    # print("The maximun in new is {} and index is {}".format(sum(new_f1)/50, new_f1.index(max(new_f1))))
    # print("The maximun in old is {} and index is {}".format(sum(old_f1)/50, old_f1.index(max(old_f1))))
    # plt.plot(x_row, new_f1,label="new F1")
    # plt.plot(x_row, old_f1,label="old F1")

    # plt.plot(x_row,new_ap, label="question mark average precision")
    # plt.plot(x_row,old_ap, label="old average precision")
    # plt.plot(x_row,bart_ap, label="bart average precision")
    # plt.plot(x_row,gpt_ap, label="gpt average precision")
    # print("The average in new is {} and index is {}".format(sum(new_ap)/50, new_ap.index(max(new_ap))))
    # print("The average in old is {} and index is {}".format(sum(old_ap)/50, old_ap.index(max(old_ap))))
    # print(new_ap)
    # print(old_ap)
    # print(sum(bart_ap)/len(bart_ap))
    # print(sum(gpt_ap)/len(gpt_ap))
    # plt.plot(x_row, old_r, label="old recall")
    # plt.plot(x_row, old_p, label="old precision")
    # plt.plot(x_row, new_r, label="new recall")
    # plt.plot(x_row, new_p, label="new precision")

    plt.legend()

    plt.title("Curve of Average Precision in top 50")
    # plt.title("Curve of Precision and Recall in top 50")
    plt.xlabel("Kth hit")
    # plt.ylabel("Precision and Recall")
    plt.ylabel("AP score")
    plt.savefig("AP.jpg")
    plt.show()


def rank_1760():
    old_ = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/old_order_1760.csv"
    new_ = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/new_order_1760.csv"
    bart_ = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/bart_1760_new_order.csv"
    bart_quesition = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/bart_question_new_order.csv"
    old_df = pd.read_csv(old_)
    new_df = pd.read_csv(new_)
    bart_df = pd.read_csv(bart_)
    bart_question_df = pd.read_csv(bart_quesition)

    old_r = []
    old_p = []
    old_ap = []

    new_r = []
    new_p = []
    new_ap = []

    bart_r = []
    bart_p = []
    bart_ap = []

    bart_question_p = []
    bart_question_ap = []

    for i in range(50):
        old_ap.append(ap_for_k(old_df, i, "memory_loss"))
        # old_r.append(recall_at_k_sklearn(old_df, i, "memory_loss"))
        old_p.append(precision_at_k_sklearn(old_df, i, "memory_loss"))

        new_ap.append(ap_for_k(new_df, i, "memory_loss"))
        # new_r.append(recall_at_k_sklearn(new_df, i, "memory_loss"))
        new_p.append(precision_at_k_sklearn(new_df, i, "memory_loss"))

        bart_ap.append(ap_for_k(bart_df, i, "memory_loss"))
        # bart_r.append(recall_at_k_sklearn(bart_df,i, "memory_loss"))
        bart_p.append(precision_at_k_sklearn(bart_df, i, "memory_loss"))

        bart_question_p.append(precision_at_k_sklearn(bart_question_df, i, "memory_loss"))

    x_row = [i for i in range(1, 51)]

    # plt.plot(x_row,old_ap, label="full post average precision")
    # plt.plot(x_row,new_ap, label="question mark average precision")
    # plt.plot(x_row,bart_ap, label="bart average precision")

    print("average precision in plot old", sum(old_p) / 50)
    print("average precision in plot new", sum(new_p) / 50)
    print("average precision in plot bart", sum(bart_p) / 50)
    print("average precision in plot bart question", sum(bart_question_p) / 50)

    plt.plot(x_row, old_p, label="full post precision@K")
    plt.plot(x_row, new_p, label="question mark precision@K")
    plt.plot(x_row, bart_p, label="bart precision@K")
    plt.plot(x_row, bart_question_p, label="bart question precision@K")

    # plt.plot(x_row, old_r, label="full post recall@K")
    # plt.plot(x_row, new_r, label="question mark recall@K")
    # plt.plot(x_row, bart_r, label="bart recall@K")
    plt.legend()
    # plt.title("Curve of Average Precision in top 50")
    plt.title("Curve of Precision and Recall in top 50")
    plt.xlabel("Kth hit")
    plt.ylabel("Precision and Recall")
    # plt.ylabel("AP score")
    plt.savefig("precision_recall_for_1760.jpg")
    plt.show()


def find_overlap():
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/id_comparation.csv")
    old = df.Old_ids.values.tolist()
    new = df.New_ids.values.tolist()
    seen_post = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/DrXieSeenPosts.csv"
    seen_df = pd.read_csv(seen_post)

    list = set(new) - (set(old) & set(new))
    for i in list:
        if i in seen_df.id.values.tolist():
            print(i)


# def use_


def futher_filterinformationwant():
    csv_path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/gpt_extraction.csv"
    df = pd.read_csv(csv_path)
    posts = df.shorten_post.values.tolist()
    ids = df.reddit_id.values.tolist()

    GPT_HIW = []
    question_words = ["how", "what", "why"]

    for idx, post in zip(ids, posts):
        sent_post = sent_tokenize(post)
        information_want = []
        for sp in sent_post:
            sp = str.lower(sp)
            if any(qw in sp for qw in question_words):
                information_want.append(sp)
        if information_want != []:
            GPT_HIW.append("\t".join(information_want))
        else:
            GPT_HIW.append("")
    df.GPThiw = GPT_HIW
    df.to_csv("filter_new_gpt.csv", index=False)


def use_langchain():
    from langchain import OpenAI, SerpAPIWrapper
    from langchain.agents import initialize_agent, Tool
    from langchain.prompts import PromptTemplate
    llm = OpenAI(temperature=0, openai_api_key="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")
    template = "Find the sentences have information want from the post {post}. "
    prompt = PromptTemplate(
        input_variables=["post"],
        template=template,
    )
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    df = pd.read_csv("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/top_50_round1.csv")
    post = df.Post.values.tolist()
    ids = df.Id.values.tolist()
    # Run the chain only specifying the input variable.
    res = {}
    for idx, i in zip(ids, post):
        res[idx] = chain.run(i)
    import json
    with open("/Users/yuelyu/PycharmProjects/ADRD/ADRDtask15/gpt_res.json", "w") as f:
        json.dump(res, f)


def compliment_question_mask():
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/refine_middle_1760_bart.json"
    original_question_mark_hiw = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask16/refine_middle_1760_.json"

    with open(original_question_mark_hiw, "r") as f:
        original_json = json.load(f)

    with open(path, "r") as f:
        json_ = json.load(f)

    orignal_hiws = list(original_json.keys())
    new_hiws = list(json_.keys())
    # print(len(set(orignal_hiws))) # 535
    # print(len(set(new_hiws))) # 1684

    mismatch_ids = list(set(new_hiws) - set(orignal_hiws))
    # here we compliment the gap with question mask post

    # find the missing ids and merge the question mark and bart as the new dataset
    for key, value in json_.items():
        if key in mismatch_ids and value["hiw"] != []:
            max_index = None
            count_context = 0
            for hiw, context in value["valid_context"].items():
                if len(context) > count_context:
                    count_context = len(context)
                    max_index = hiw  # here find which context have the maximun length
            # print("inserted hiw", max_index)
            # print("insert key", key)
            # print("inserted valid_context", max_index)
            original_json[key] = {"hiw": max_index, "valid_context": {max_index: value["valid_context"][max_index]}}

    with open("refine_middle_1760_bart_question.json", "w") as f:
        json.dump(original_json, f)


def find_1000_posts():
    import random
    path = "/Users/yuelyu/PycharmProjects/ADRD/ADRDtask14/new_all_adrd.csv"
    df = pd.read_csv(path)

    post = df.selftext.values.tolist()
    ids = df.id.values.tolist()
    random_numbers = random.sample(ids, 1200)
    # print(random_numbers)
    dicts = {"id": [], "title": [], "post": []}
    count = 0
    for i in random_numbers:
        if count == 1100:
            break
        else:
            if df.loc[df.id == i].selftext.values[0] == "[delete]" or len(df.loc[df.id == i].selftext.values[0]) <=10\
                     or len(df.loc[df.id == i].selftext.values[0]) > 3000 :
                continue
            else:
                dicts["id"].append(i)
                dicts["title"].append(df.loc[df.id == i].title.values[0])
                dicts["post"].append(df.loc[df.id == i].selftext.values[0])
                count += 1
    df2 = pd.DataFrame(dicts)
    df2.to_csv("/Users/yuelyu/Downloads/1000_chatgpt.csv", index=False)


def get_topic():
    path = "/Users/yuelyu/Downloads/annotated_100_post.csv"
    df = pd.read_csv(path)
    import gensim
    from gensim import corpora

    # preprocess the text data by cleaning and tokenizing the sentences
    text_data = df.post.values.tolist()

    # convert the text data into a list of lists of tokens
    tokens = [[token for token in sentence.split()] for sentence in text_data]

    # create a dictionary mapping tokens to unique integer IDs
    dictionary = corpora.Dictionary(tokens)

    # convert the tokenized sentences into bag-of-words vectors
    bow_corpus = [dictionary.doc2bow(token) for token in tokens]

    # train an LDA model with 2 topics
    lda_model = gensim.models.ldamodel.LdaModel(corpus=bow_corpus, id2word=dictionary, num_topics=2)

    # get the topic distributions for each sentence in the corpus
    doc_topics = [lda_model.get_document_topics(bow) for bow in bow_corpus]
    print(doc_topics)
    # assign each sentence to its corresponding topic based on the topic probabilities
    topic_assignments = []
    for i, topics in enumerate(doc_topics):
        max_prob = max(topics, key=lambda x: x[1])
        topic_assignments.append((i, max_prob[0]))

    # print the topic assignments for each sentence
    for i, topic in topic_assignments:
        print(f"Sentence {i + 1} belongs to topic {topic + 1}")


if __name__ == "__main__":
    # use_langchain_rerank(
    #     "How To Deal With Her Not Recognizing Her House my grandmother is mid stage, and in the past month or two, something new that has come up, is a couple times a week she will get upset and insist she's not in her house, despite living there for 51 years.  this will go on for hours, sometimes the whole day.  she gets very depressed and upset over it.  she remembers her address, she remembers the city, but she insists she is living in an apartment in 'a town i don't know the name of' and goes on and on about how much she hates it there and wants to go home.  telling her it is her home, and trying to get her to recognize some of her belongings doesn't help, it just makes her very mad and she starts yelling that she's not crazy.  telling her she will be going home next week, and everything will be back to normal then doesn't help either.  she remains depressed and upset and lamenting about wanting to go home right away.  trying to redirect the conversation to another topic doesn't help either, she goes right back to talking about wanting to go home.  we're not sure *what* to say to her on the days she gets like this.  is this one of those things where there is nothing to do?  we just have to deal with her being upset/depressed on these days?")
    # find_long_memoryloss()
    # get_new_order_longpost()
    # find_old_order()
    # match_result_id_with_50()
    # plot_main()
    # find_overlap()
    # use_langchain()
    # futher_filterinformationwant()
    # rank_1760()
    # compliment_question_mask()
    # find_1000_posts()
    get_topic()