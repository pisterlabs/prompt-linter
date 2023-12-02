import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from datetime import date
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import logging
import warnings
from lxml import etree
from tqdm import tqdm
from gensim.matutils import cossim
from math import sqrt

LDA_MODEL_NAME = 'analys/trained_lda_model'
GEN_CHART_NAME = 'GenerateDistributionOfParagraphWordCountsChart'
PATH_TO_PARAPHRASES = r'C:\Users\kiva0319\IdeaProjects\hrdmd1803\Strong-Paraphrase-Generation-2020\processed\paraphrases.xml'
data_words = []
bad_paragraphs = 0

root = etree.parse(PATH_TO_PARAPHRASES)
root = root.getroot()


def safe_list_get(l, idx):
    try:
        return l[idx][0]
    except IndexError:
        return -1
    except TypeError:
        return -2


def safe_get_str(l, idx):
    try:
        return str(l[idx][0])
    except IndexError:
        return '-1'
    except TypeError:
        return '-2'


def extact_paragraphs(element_paragraphs_1):
    bad_paragraphs = 0
    for paragraph in element_paragraphs_1:
        if int(paragraph.attrib.get("words")) >= 5:
            words = []
            for word in paragraph.text.split(";"):
                if word.isalpha():
                    words.append(word)
            data_words.append(words)
        else:
            bad_paragraphs += 1
    return bad_paragraphs


lda_model = LdaModel.load(LDA_MODEL_NAME)

result_xml = etree.Element('raw_data')
result_doc = etree.ElementTree(result_xml)
corpus_info = etree.SubElement(result_xml, 'head')
etree.SubElement(corpus_info, 'description').text = "—"
element_size = etree.SubElement(corpus_info, 'size')
etree.SubElement(corpus_info, 'date').text = str(date.today())
topics = etree.SubElement(corpus_info, 'topics')
topics_info = etree.SubElement(corpus_info, 'topics_short_info')

print("Write topics:")
for i in tqdm(range(lda_model.num_topics)):
    key_words = lda_model.show_topic(topicid=i)
    cur_topic = etree.SubElement(topics, 'topic', id=str(i))
    for (words, relevance) in key_words:
        etree.SubElement(cur_topic, 'topic', relevance=str(relevance)).text = words
print()
print("Write topics short info:")
for i in tqdm(range(0, lda_model.num_topics - 1)):
    etree.SubElement(topics_info, 'topic', id=str(i)).text = lda_model.print_topic(i)
print()
print("Write paragraphs topics:")
articles_list = etree.SubElement(result_xml, 'corpus')
bad_paragraphs = 0
count = 0
id2word = lda_model.id2word

metrics_element = etree.SubElement(corpus_info, 'metrics')
class_negative_element = etree.SubElement(metrics_element, 'class', type="-1")
cne_top_1 = etree.SubElement(class_negative_element, 'num_common_top_1_topics')
cne_top_2 = etree.SubElement(class_negative_element, 'num_common_top_2_topics')
cne_top_3 = etree.SubElement(class_negative_element, 'num_common_top_3_topics')

class_neutral_element = etree.SubElement(metrics_element, 'class', type="0")
cnne_top_1 = etree.SubElement(class_neutral_element, 'num_common_top_1_topics')
cnne_top_2 = etree.SubElement(class_neutral_element, 'num_common_top_2_topics')
cnne_top_3 = etree.SubElement(class_neutral_element, 'num_common_top_3_topics')

class_positive_element = etree.SubElement(metrics_element, 'class', type="+1")
cpe_top_1 = etree.SubElement(class_positive_element, 'num_common_top_1_topics')
cpe_top_2 = etree.SubElement(class_positive_element, 'num_common_top_2_topics')
cpe_top_3 = etree.SubElement(class_positive_element, 'num_common_top_3_topics')

class_not_negative_element = etree.SubElement(metrics_element, 'class', type="0+1")
cnnne_top_1 = etree.SubElement(class_not_negative_element, 'num_common_top_1_topics')
cnnne_top_2 = etree.SubElement(class_not_negative_element, 'num_common_top_2_topics')
cnnne_top_3 = etree.SubElement(class_not_negative_element, 'num_common_top_3_topics')


def create_element_with_topics_info(source_text, attr, parant_element, element_name, element_text):
    text = source_text.split(";")
    corpus = [id2word.doc2bow(text)]
    row = lda_model[corpus][0]
    row = sorted(row, key=lambda x: (x[1]), reverse=True)
    topic_num_1 = safe_list_get(row, 0)
    topic_num_2 = safe_list_get(row, 1)
    topic_num_3 = safe_list_get(row, 2)
    element = etree.SubElement(parant_element, element_name)
    element.text = element_text
    for name in attr:
        element.set(name, attr[name])
    element.set('topic_1', str(topic_num_1))
    element.set('topic_2', str(topic_num_2))
    element.set('topic_3', str(topic_num_3))
    keywords = lda_model.show_topic(topic_num_1, topn=3)
    element.set('topic_word_1', str(safe_get_str(keywords, 0)))
    element.set('topic_word_2', str(safe_get_str(keywords, 1)))
    element.set('topic_word_3', str(safe_get_str(keywords, 2)))
    return topic_num_1, topic_num_2, topic_num_3


metrics = {
    'num_common_top_1_topics': {-1: [], 0: [], 1: []},
    'num_common_top_2_topics': {-1: [], 0: [], 1: []},
    'num_common_top_3_topics': {-1: [], 0: [], 1: []},
    'its': {-1: [], 0: [], 1: []}
}


class ITSCalculator():
    def __init__(self, lda_model, id2word, cluster_creation_tolerance=0.6, cluster_boxing_tolerance=0.6,
                 topic_value_importance=0.9):
        self.cluster_creation_tolerance = cluster_creation_tolerance
        self.cluster_boxing_tolerance = cluster_boxing_tolerance
        self.topic_value_importance = topic_value_importance
        self.lda_model = lda_model
        self.id2word = id2word
        self.statistic = {-1: [], 0: [], 1: [], 2: []}
        self.values = []

    def show_statistic_and_add_to(self, parent_element):
        print("_________ ITS ________")
        for clazz in self.statistic:
            self.statistic[clazz].sort()
            # print(self.statistic[clazz])
            print("[ " + str(clazz) + " ]", end=' ')
            # self.statistic[clazz] = self.statistic[clazz][10:-10]
            # print(self.statistic[clazz])
            print("average :", np.average(self.statistic[clazz]), end=' ')
            print("median :", np.median(self.statistic[clazz]), end=' ')
            print("mean :", np.mean(self.statistic[clazz]), end=' ')
            print("var :", np.var(self.statistic[clazz]), end=' ')
            print("sqrt(var) :", sqrt(np.var(self.statistic[clazz])), end=' ')
            print("std :", np.std(self.statistic[clazz]), end='')
            print("variation :",
                  float(np.var(self.statistic[clazz])) / max(float(np.mean(self.statistic[clazz])), 0.00001), end='\n')
        print()
        articles = etree.SubElement(parent_element, "its")
        etree.SubElement(articles, 'negative').text = str(np.average(self.statistic[-1]))
        etree.SubElement(articles, 'neutral').text = str(np.average(self.statistic[0]))
        etree.SubElement(articles, 'positive').text = str(np.average(self.statistic[1]))

    def calc(self, element_paragraphs_from, element_paragraphs_to, clas):
        paragraph_to_class = dict()
        paragraph_to_lda_vec = dict()
        count = 0
        for i, paragraph in enumerate(element_paragraphs_from):
            paragraph_to_class.update({count: 0})
            doc = lda_model[id2word.doc2bow(paragraph.text.split(";"))]
            paragraph_to_lda_vec.update({count: doc})
            count += 1
        for i, paragraph in enumerate(element_paragraphs_to):
            paragraph_to_class.update({count: 1})
            doc = self.lda_model[self.id2word.doc2bow(paragraph.text.split(";"))]
            paragraph_to_lda_vec.update({count: doc})
            count += 1
        # print("paragraph_to_class =", paragraph_to_class)
        # print("paragraph_to_lda_vec =", paragraph_to_lda_vec)
        pairToDist = dict()
        for p1 in paragraph_to_class:
            for p2 in paragraph_to_class:
                if p1 < p2 and paragraph_to_class.get(p1) != paragraph_to_class.get(p2):
                    # print("new pair", p1, p2)
                    pairToDist.update(
                        {(p1, p2): cossim(paragraph_to_lda_vec.get(p1), paragraph_to_lda_vec.get(p2))})
        # print("pairToDist =", pairToDist)
        # print("len(pairToDist) =", len(pairToDist))
        clusters = dict()
        clusters_count = 0
        stack = []
        visited_paragraphs = []
        for startParagraph in range(count):
            if startParagraph in visited_paragraphs:
                continue
            clusters.update({clusters_count: set()})
            stack.append(startParagraph)
            while stack:
                paragraph = stack.pop()
                if paragraph in visited_paragraphs:
                    continue
                visited_paragraphs.append(paragraph)
                clusters[clusters_count].add(paragraph)
                for key in pairToDist:
                    if pairToDist[key] > self.cluster_creation_tolerance:
                        if paragraph == key[0]:
                            stack.append((key[1]))
                        else:
                            stack.append((key[0]))
            clusters_count += 1
        # print("len(clusters) =", len(clusters))
        # print("clusters =", clusters)
        free_paragraphs = []
        big_clusters = dict()
        big_clusters_count = 0
        for cluster_num in clusters:
            if len(clusters[cluster_num]) < 2:
                free_paragraphs.append(clusters[cluster_num].pop())
            else:
                big_clusters.update({big_clusters_count: clusters[cluster_num]})
                big_clusters_count += 1
        boxed_big_clusters = dict()
        for paragraph_new in free_paragraphs:
            for cluster in big_clusters:
                boxed_big_clusters.update({cluster: set()})
                distances = [0]
                for paragraph in big_clusters[cluster]:
                    if (paragraph_to_class.get(paragraph_new) != paragraph_to_class.get(paragraph)):
                        distances.append(pairToDist[(min(paragraph_new, paragraph), max(paragraph_new, paragraph))])
                if max(distances) > self.cluster_boxing_tolerance:
                    boxed_big_clusters[cluster].add(paragraph_new)
        for cluster_num in big_clusters:
            big_clusters[cluster_num] = big_clusters[cluster_num].union(boxed_big_clusters[cluster_num])
        # print("free_paragraphs =", free_paragraphs)
        # print("clusters =", big_clusters)
        if len(big_clusters) < 1:
            self.statistic[clas].append(0)
            if str(clas) != "-1":
                self.statistic[2].append(0)
            return str(0)
        else:
            global_dists = []
            for cluster_num in big_clusters:
                dists = []
                for p1 in big_clusters[cluster_num]:
                    for p2 in big_clusters[cluster_num]:
                        if p1 < p2 and paragraph_to_class.get(p1) != paragraph_to_class.get(p2):
                            dists.append(pairToDist[(p1, p2)])
                dist = sum(dists) / len(dists)
                global_dists.append(dist)
                # print("dists =", dists)
            global_dist = sum(global_dists) / len(global_dists)
            global_dist_normalized = (global_dist + 1.0) / 2.0
            # print("global_dist =", global_dist)
            # print("global_dist_normalized =", global_dist_normalized)
            self.statistic[clas].append(global_dist_normalized)
            if str(clas) != "-1":
                self.statistic[2].append(global_dist_normalized)
            return str(global_dist_normalized)


its_calculator = ITSCalculator(lda_model, id2word)

for element in tqdm(root[1]):
    id = element[0].text
    old_id = element[1].text
    id_1 = element[2].text
    id_2 = element[3].text
    title_1 = element[4].text
    title_2 = element[5].text
    text_1 = element[6].text
    text_2 = element[7].text
    words_title_1 = int(element[8].text)
    words_title_2 = int(element[9].text)
    words_article_1 = int(element[10].text)
    words_article_2 = int(element[11].text)
    num_of_paragraphs_1 = int(element[12].text)
    num_of_paragraphs_2 = int(element[13].text)
    element_paragraphs_1 = element[14]
    element_paragraphs_2 = element[15]
    jaccard = element[16].text
    clas = int(element[17].text)

    paraphrase = etree.SubElement(articles_list, 'paraphrase')
    etree.SubElement(paraphrase, 'value', name="id").text = str(count)
    etree.SubElement(paraphrase, 'value', name="old_id").text = str(old_id)
    etree.SubElement(paraphrase, 'value', name="id_1").text = str(id_1)
    etree.SubElement(paraphrase, 'value', name="id_2").text = str(id_2)
    etree.SubElement(paraphrase, 'value', name="title_1").text = str(title_1)
    etree.SubElement(paraphrase, 'value', name="title_2").text = str(title_2)

    # words_title_1_element
    create_element_with_topics_info(title_1, {'name': 'words_title_2'}, paraphrase, 'value', str(words_title_1))
    # words_title_2_element
    create_element_with_topics_info(title_2, {'name': 'words_title_2'}, paraphrase, 'value', str(words_title_2))
    # words_article_1_element
    create_element_with_topics_info(text_1, {'name': 'words_article_1'}, paraphrase, 'value', str(words_article_1))
    # words_article_2_element
    create_element_with_topics_info(text_2, {'name': 'words_article_2'}, paraphrase, 'value', str(words_article_2))

    etree.SubElement(paraphrase, 'value', name="num_of_paragraphs_1").text = str(num_of_paragraphs_1)
    etree.SubElement(paraphrase, 'value', name="num_of_paragraphs_2").text = str(num_of_paragraphs_2)

    paragraphs_1 = etree.SubElement(paraphrase, 'value', name="paragraphs_1")
    p1_1 = set()
    p1_2 = set()
    p1_3 = set()
    p2_1 = set()
    p2_2 = set()
    p2_3 = set()

    lda_vecs_1 = []
    lda_vecs_2 = []

    for paragraph in element_paragraphs_1:
        if paragraph.attrib.get("words") and int(paragraph.attrib.get("words")) >= 5:
            t1, t2, t3 = create_element_with_topics_info(paragraph.text, {}, paragraphs_1, 'paragraph',
                                                         str(paragraph.text))
            p1_1.add(t1)
            p1_2.update([t1, t2])
            p1_3.update([t1, t2, t3])
        else:
            bad_paragraphs += 1

    paragraphs_2 = etree.SubElement(paraphrase, 'value', name="paragraphs_2")
    for paragraph in element_paragraphs_2:
        if paragraph.attrib.get("words") and int(paragraph.attrib.get("words")) >= 5:
            t1, t2, t3 = create_element_with_topics_info(paragraph.text, {}, paragraphs_2, 'paragraph',
                                                         str(paragraph.text))
            p2_1.add(t1)
            p2_2.update([t1, t2])
            p2_3.update([t1, t2, t3])
        else:
            bad_paragraphs += 1
    its_tag = etree.SubElement(paraphrase, 'ITS')
    # its_tag.text = str(its_calculator.calc(element_paragraphs_1, element_paragraphs_2, clas))
    try:
        its_tag.text = str(its_calculator.calc(element_paragraphs_1, element_paragraphs_2, clas))
    except Exception:
        print(str(id))
        continue
    tops1_1 = list(p1_1)
    tops1_1.sort()
    tops2_1 = list(p1_2)
    tops2_1.sort()
    tops3_1 = list(p1_3)
    tops3_1.sort()
    etree.SubElement(paraphrase, 'p1_1').text = (";".join(str(x) for x in tops1_1))
    etree.SubElement(paraphrase, 'p1_2').text = (";".join(str(x) for x in tops2_1))
    etree.SubElement(paraphrase, 'p1_3').text = (";".join(str(x) for x in tops3_1))
    tops1_2 = list(p2_1)
    tops1_2.sort()
    tops2_2 = list(p2_2)
    tops2_2.sort()
    tops3_2 = list(p2_3)
    tops3_2.sort()
    etree.SubElement(paraphrase, 'p2_1').text = (";".join(str(x) for x in tops1_2))
    etree.SubElement(paraphrase, 'p2_2').text = (";".join(str(x) for x in tops2_2))
    etree.SubElement(paraphrase, 'p2_3').text = (";".join(str(x) for x in tops3_2))

    intersection1 = list(p1_1.intersection(p2_1))
    intersection1.sort()
    etree.SubElement(paraphrase, 'intersection_1', size=str(len(intersection1))).text = (
        ";".join(str(x) for x in intersection1))
    intersection2 = list(p1_2.intersection(p2_2))
    intersection2.sort()
    etree.SubElement(paraphrase, 'intersection_2', size=str(len(intersection2))).text = (
        ";".join(str(x) for x in intersection2))
    intersection3 = list(p1_3.intersection(p2_3))
    intersection3.sort()
    etree.SubElement(paraphrase, 'intersection_3', size=str(len(intersection3))).text = (
        ";".join(str(x) for x in intersection3))

    metrics['num_common_top_1_topics'][clas].append(len(intersection1))
    metrics['num_common_top_2_topics'][clas].append(len(intersection2))
    metrics['num_common_top_3_topics'][clas].append(len(intersection3))

    etree.SubElement(paraphrase, 'value', name="clas").text = str(clas)
    count += 1

m1 = metrics['num_common_top_1_topics']
m2 = metrics['num_common_top_2_topics']
m3 = metrics['num_common_top_3_topics']

cne_top_1.text = str(sum(m1[-1]) / len(m1[-1]))
cne_top_2.text = str(sum(m2[-1]) / len(m2[-1]))
cne_top_3.text = str(sum(m3[-1]) / len(m3[-1]))

cnne_top_1.text = str(sum(m1[0]) / len(m1[0]))
cnne_top_2.text = str(sum(m2[0]) / len(m2[0]))
cnne_top_3.text = str(sum(m3[0]) / len(m3[0]))

cpe_top_1.text = str(sum(m1[1]) / len(m1[1]))
cpe_top_2.text = str(sum(m2[1]) / len(m3[1]))
cpe_top_3.text = str(sum(m3[1]) / len(m3[1]))

cnnne_top_1.text = str((sum(m1[0]) + sum(m1[1])) / (len(m1[0]) + len(m1[1])))
cnnne_top_2.text = str((sum(m2[0]) + sum(m2[1])) / (len(m2[0]) + len(m2[1])))
cnnne_top_3.text = str((sum(m3[0]) + sum(m3[1])) / (len(m3[0]) + len(m3[1])))

print("Number of th bad paragraphs:", bad_paragraphs)
its_calculator.show_statistic_and_add_to(result_xml)
print("Save xml:")
outFile = open("processed/topics_with_its_2.xml", 'wb')
result_doc.write(outFile, xml_declaration=True, encoding='utf-8', pretty_print=True)
print("XML successfully saved!")
