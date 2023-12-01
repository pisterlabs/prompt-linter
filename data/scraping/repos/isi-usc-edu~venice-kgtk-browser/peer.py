from whoosh.index import open_dir
from whoosh.fields import *
from whoosh import qparser
from whoosh.qparser import QueryParser
from whoosh import scoring

from nltk import word_tokenize

import tiktoken
import openai
import json
import os


MAX_TOKENS = 4096 - 500
KG_DOWNWEIGHT_SCORE = 0.1


# read openai key from the environment variable
openai_key = os.environ.get('OPENAI_KEY')
openai.api_key = openai_key


def clean_up_query(query):
    return " ".join(word_tokenize(query))


def custom_scoring(searcher, fieldname, text, matcher):
    # frequency = scoring.Frequency().scorer(searcher, fieldname, text).score(matcher)
    # tfidf =  scoring.TF_IDF().scorer(searcher, fieldname, text).score(matcher)
    bm25 = scoring.BM25F().scorer(searcher, fieldname, text).score(matcher)
    bm25 *= KG_DOWNWEIGHT_SCORE if searcher.document()['input_type'] == 'kg' else 1
    return bm25


def num_tokens_from_string(string: str, encoding_name: str="gpt2") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_facts(prompt=None, token_allowance=MAX_TOKENS):
    ix = open_dir("indexdir")

    causal_claims_intro = "We have extracted causal claims. They are:\n"
    survey_intro = "We have conducted a survey. The responses are presented in a tabular form with columns separated by |. The columns are: %s. The responses are:\n"
    acled_intro = "Violent events have been recorded. The events are presented in a tabular form with columns separated by |. The columns are: %s. The events are:\n"
    kg_intro = "We have collected messages discussing Cabo Delgado from Telegram and tagged them with [moral foundations] and [emotions]. Those messages are:\n"

    len_tokens = num_tokens_from_string(causal_claims_intro) +\
        num_tokens_from_string(survey_intro) +\
        num_tokens_from_string(acled_intro) +\
        num_tokens_from_string(kg_intro)

    causal_claims = []
    surveys = []
    acleds = []
    kgs = []
    idset = {"causal_claims": set([]), "acled": set([]), "surveys": set([]), "kg": set([])}
    survey_schema = None
    acled_schema = None

    my_weighting = scoring.FunctionWeighting(custom_scoring)
    with ix.searcher(weighting=my_weighting) as searcher:
    # with ix.searcher() as searcher:
        if prompt is not None:
            # query = QueryParser("text", ix.schema, group=qparser.OrGroup.factory(0.9)).parse(clean_up_query(prompt))
            query = QueryParser("text", ix.schema, group=qparser.OrGroup).parse(clean_up_query(prompt))
            results = searcher.search(query, limit=None)
        else:
            results = searcher.documents()

        for r in results:
            added = False
            if r['input_type'] == 'causal claim' and r['content'] not in causal_claims:
                newlen = num_tokens_from_string(r['content'])
                if len_tokens + newlen < token_allowance:
                    len_tokens += newlen
                    causal_claims.append(r['content'])
                    idset['causal_claims'].add(r['participant_id'])
                    added = True

            if r['input_type'].find("survey:") != -1:
                newlen = num_tokens_from_string(r['content'])
                if len_tokens + newlen < token_allowance:
                    len_tokens += newlen
                    surveys.append(r['content'])
                    idset['surveys'].add(r['participant_id'])
                    added = True
                if survey_schema is None:
                    survey_schema = r['input_type'].split(": ")[1]
                    len_tokens += num_tokens_from_string(survey_schema)

            if r['input_type'].find("acled:") != -1:
                newlen = num_tokens_from_string(r['content'])
                if len_tokens + newlen < token_allowance:
                    len_tokens += newlen
                    idset['acled'].add(r['participant_id'])
                    acleds.append(r['content'])
                    added = True

                if acled_schema is None:
                    acled_schema = r['input_type'].split(": ")[1]
                    len_tokens += num_tokens_from_string(acled_schema)

            if r['input_type'] == 'kg':
                newlen = num_tokens_from_string(r['content'])
                if len_tokens + newlen < token_allowance:
                    len_tokens += newlen
                    idset['kg'].add(r['participant_id'])
                    kgs.append(r['content'])
                    added = True

            # if we've reached the limit, duck out.
            if not added:
                break

    # flip the responses due to recency bias
    causal_claims.reverse()
    surveys.reverse()
    acleds.reverse()
    print("LENGTHS:", len(causal_claims), len(surveys), len(acleds), len(kgs))
    claims = "".join((
        causal_claims_intro,
        "".join(causal_claims), #CCs have the newline baked in
        "\n\n",
        survey_intro % survey_schema,
        "\n".join(surveys),
        "\n\n",
        acled_intro % acled_schema,
        "\n".join(acleds),
        kg_intro,
        "\n".join(kgs)
    ))

    return claims, idset


def get_response(system, prompt, preamble=None):

    preamble = open('test_preamble.txt').read()
    resp = json.load(open('test_resp.json'))
    idset = {'causal_claims': {'491', '492', '678', '252', '605', '587', '355', '588', '827', '339', '586', '794', '128', '604', '933', '575'}, 'acled': set(), 'surveys': set(), 'kg': {'Q00_sentence_id53862', 'Q00_sentence_id15654', 'Q00_sentence_id48125', 'Q00_sentence_id31768', 'Q00_sentence_id35164', 'Q00_sentence_id56913', 'Q00_sentence_id47967', 'Q00_sentence_id53221', 'Q00_sentence_id39314', 'Q00_sentence_id28917', 'Q00_sentence_id15725', 'Q00_sentence_id24487', 'Q00_sentence_id52619', 'Q00_sentence_id27415', 'Q00_sentence_id26517', 'Q00_sentence_id39892', 'Q00_sentence_id53856', 'Q00_sentence_id53836', 'Q00_sentence_id16961', 'Q00_sentence_id35175', 'Q00_sentence_id26875', 'Q00_sentence_id56245'}}

    return (preamble, resp, None, idset)

#    tokens_provided = num_tokens_from_string(preamble if preamble is not None else "", "gpt2")\
#        + num_tokens_from_string(system, "gpt2")\
#        + num_tokens_from_string(prompt, "gpt2")
#    print("We got %d tokens" % tokens_provided)
#    if tokens_provided >= MAX_TOKENS:
#        return (preamble, None, "Error: Preamble and Prompt total %d tokens, exceeding the %d token limit!" %
#                (tokens_provided, MAX_TOKENS), None)
#
#    token_allowance = MAX_TOKENS - tokens_provided
#
#    if preamble is None:
#        preamble, idset = get_facts(prompt, token_allowance=token_allowance)
#    else:
#        idset = {}
#
#
#    messages = [{"role": "system", "content": system},
#                {"role": "user", "content": preamble},
#                {"role": "user", "content": prompt}]
#
#    #resp = openai.ChatCompletion.create(
#    #    model="gpt-3.5-turbo",
#    #    messages= messages,
#    #    temperature = 0
#    #)
#
#    #of = open('test_data.json', 'w')
#    #json.dump(resp, of, indent=4)
#    #of.close()
#
#    preamble = open('test_preamble.txt').read()
#    resp = json.load(open('test_resp.json'))
#    idset = {'causal_claims': {'491', '492', '678', '252', '605', '587', '355', '588', '827', '339', '586', '794', '128', '604', '933', '575'}, 'acled': set(), 'surveys': set(), 'kg': {'Q00_sentence_id53862', 'Q00_sentence_id15654', 'Q00_sentence_id48125', 'Q00_sentence_id31768', 'Q00_sentence_id35164', 'Q00_sentence_id56913', 'Q00_sentence_id47967', 'Q00_sentence_id53221', 'Q00_sentence_id39314', 'Q00_sentence_id28917', 'Q00_sentence_id15725', 'Q00_sentence_id24487', 'Q00_sentence_id52619', 'Q00_sentence_id27415', 'Q00_sentence_id26517', 'Q00_sentence_id39892', 'Q00_sentence_id53856', 'Q00_sentence_id53836', 'Q00_sentence_id16961', 'Q00_sentence_id35175', 'Q00_sentence_id26875', 'Q00_sentence_id56245'}}
#
#    return (preamble, resp, None, idset)
