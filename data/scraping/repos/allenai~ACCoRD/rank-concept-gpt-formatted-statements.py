import pickle
import pandas as pd
from operator import itemgetter
import numpy as np
import openai
from rouge_score import rouge_scorer
import spacy
import re

#--------------------------------------------------------
openai.api_key = "sk-NzQrkRfqE5lnJPubH7faej1ZcDuz0s40qCkTTeFt" 
pd.set_option('display.max_colwidth', None)
#--------------------------------------------------------
# unpickle concept dictionary
with open('/net/nfs2.s2-research/soniam/concept-rel/resources/forecite/forecite_concept_dict.pickle', 'rb') as handle:
    concept_dict = pickle.load(handle)
    print("...unpickled forecite concept dictionary")

print(len(concept_dict))
# Initialize N
N = 100000
# N largest values in dictionary
# Using sorted() + itemgetter() + items()
res = dict(sorted(concept_dict.items(), key = itemgetter(1), reverse = True)[:N])
# printing result
# print("The top N value pairs are " + str(res))
concept_list = list(res.keys())

#--------------------------------------------------------
# CONCEPT LIST
#--------------------------------------------------------
selected_nlp_concepts = ['adversarial training', 'beam search', 'bert', 'elmo', 'gpt', 'glove', 'word2vec', 'resnet', 'domain shift', 'ulmfit', 'newsqa', 'squad', 'imagenet', 'lstm', 'roberta', 'variational autoencoder', 'dropout', 'fasttext', 'hierarchical softmax', 'distant supervision']
#--------------------------------------------------------
df_nlp_concepts = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/nlp_concepts.csv")
nlp_concept_scores = []
for index, row in df_nlp_concepts.iterrows():
    # get forecite score for each nlp concept
    nlp_concept_scores.append(concept_dict[row['concept']])

df_nlp_concepts['score'] = nlp_concept_scores
df_nlp_concepts = df_nlp_concepts[~df_nlp_concepts.concept.isin(['bert model', 'pre - train bert model', 'pre - train bert', 'moses', 'cho et al', 'dagan', 'yarowsky', 'hochreiter', 'turney', 'ney', 'och', 'grosz', 'steedman', 'well translation'])]

# # top 150 nlp concepts by score
# top150_nlp_concepts = df_nlp_concepts.sort_values(by='score', ascending=False)[:150]['concept'].tolist()
# top 150 nlp concepts by num_papers
top150_nlp_concepts = df_nlp_concepts.sort_values(by='num_papers', ascending=False)[:150]['concept'].tolist()
#--------------------------------------------------------
#--------------------------------------------------------

relations = ['compare', 'isa']
version = "v1-4class"
concept_set = "nlp-concepts"
df_all = pd.DataFrame()

for concept in top150_nlp_concepts[:20]:
    our_extractions_top6 = []
    our_generations_top6 = []
    df_concept = pd.DataFrame()

    for relation in relations:
        print(concept, relation)
        # load data
        df_concept_relation = pd.read_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/%s/gpt-formatted-statements-all-s2orc-%s-%s.csv" % (version, concept_set, concept, relation))
        df_concept_relation = df_concept_relation.loc[:, ~df_concept_relation.columns.str.contains('^Unnamed')]
        df_concept_relation = df_concept_relation.dropna()
        if len(df_concept_relation) != 0:
            # heuristic #1: remove rows with "our" in the text because we don't want references to things that can't be resolved
            df_concept_relation = df_concept_relation[~df_concept_relation['gpt_generation'].str.contains("our")]

            # heuristic #2: remove rows with "et al" in the concept b because we don't want authors names as the concept b
            df_concept_relation = df_concept_relation[~df_concept_relation['concept_b'].str.contains("et al")]

            # heuristic #3: remove rows where the conceptA occurs more than once in the generation because we don't want to explain the concept in terms of itself
            for index, row in df_concept_relation.iterrows():
                if row['gpt_generation'].count(concept) > 1:
                    df_concept_relation = df_concept_relation.drop(index)

            #--------------------------------------------------------
            # append this concept-relation specific df to a df for the concept so we can select top 6 by multilabel pred score later
            df_concept = df_concept.append(df_concept_relation)

            basic_ranking_top6 = df_concept_relation.sort_values(by=['max_multilabel_pred_score'])['gpt_generation'][:6]
            #--------------------------------------------------------

            # select rows that have conceptB == forecite concept
            df_concept_relation = df_concept_relation[df_concept_relation['is_conceptb_forecite']==1]

            # count prevalence of concept Bs
            df_sorted_by_conceptB_counts = df_concept_relation.groupby(['concept_b'])['concept_b'].count().to_frame(name='count').sort_values(['count'], ascending=False).apply(lambda x: x)
            # get frequency of concept Bs
            df_sorted_by_conceptB_frequency = df_concept_relation['concept_b'].value_counts(normalize=True).to_frame(name='concept_b_frequency').sort_values(['concept_b_frequency'], ascending=False).apply(lambda x: x)
            df_sorted_by_conceptB_frequency = df_sorted_by_conceptB_frequency.reset_index().rename(columns={'index':'concept_b'})
            # print(df_sorted_by_conceptB_frequency)

            # group by conceptB --> select the rows with the max multilabel pred score
            df_concept_relation_groupby_conceptB_maxpredscore = df_concept_relation.loc[df_concept_relation.groupby("concept_b")['%s_pred_score' % relation].idxmax()]
            # print(df_concept_relation_groupby_conceptB_maxpredscore)
            # merge with concept_b frequencies df on concept_b
            result = pd.merge(df_sorted_by_conceptB_frequency, df_concept_relation_groupby_conceptB_maxpredscore, on="concept_b")
            
            # #save all ranked-filtered descriptions
            # result.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s-%s.csv" % (version, concept_set, concept, relation))
            # result.to_json("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s-%s.json" % (version, concept_set, concept, relation), orient="records")
            #--------------------------------------------------------
            # save top 3
            if len(result) < 3:
                top = len(result)
            else: 
                top = 3
            result = result[:top]
            print(result)

            # result.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s-%s-top.csv" % (version, concept_set, concept, relation))
            # result.to_json("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s-%s-top.json" % (version, concept_set, concept, relation), orient="records")
            
            df_all = df_all.append(result)
            #--------------------------------------------------------
            # get top 3 descriptions for this concept-relation pair
            our_generations_top3 = result['gpt_generation'][:top]
            our_extractions_top3 = result['original_sentence'][:top]

            # add it to the set of top descriptions for this concept
            our_generations_top6.extend(our_generations_top3)
            our_extractions_top6.extend(our_extractions_top3)
            #--------------------------------------------------------

    for i in range(len(our_generations_top6)):
        print("extraction: %s" % our_extractions_top6[i])
        print("generation: %s" % our_generations_top6[i])
        print("")
    
    # # FOR USER STUDY
    # # sort the df that includes the rows for all the relations by the max_multilabel_pred_score and select the top 6 generations
    # df_concept = df_concept.sort_values(by=['max_multilabel_pred_score'])['gpt_generation'][:6]
    # df_survey = pd.DataFrame(list(zip(our_extractions_top6, basic_ranking_top6, our_generations_top6)), columns =['Set A', 'Set B', 'Set C'])
    # df_survey.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/v1-4class/top5-sets/%s.csv" % concept)

# df_all.reset_index(drop=True, inplace=True)
# print(df_all)
# fn = "top-150-nlp-concepts"
# df_all.to_csv("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s.csv" % (version, concept_set, fn))
# df_all.to_json("/net/nfs2.s2-research/soniam/concept-rel/abstractive-summarization/gpt-generations/top-n-forecite-concepts-gpt-generations/%s/ranked-filtered/%s/%s.json" % (version, concept_set, fn), orient="records")
