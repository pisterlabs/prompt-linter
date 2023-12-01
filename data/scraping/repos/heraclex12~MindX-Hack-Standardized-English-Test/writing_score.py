from flask import jsonify, request
# from . import routes
from models.database import *
import numpy as np
import tensorflow as tf
import json
from models.database import EnglishMongoDB
from flask_restful import Resource, reqparse
from writing.compute_spell_grammar_score import compute_spell_grammar_score
from writing.encode_doc import encode_doc
from writing.BasicScoring import basic_score
from writing.CoherenceScoring import coherence_score
from writing.RelativeScoring import  prompt_rel_score, prompts


def get_score(logit):
    """
        Normalize score by range 0->10
    :param logit: score by range 0->1
    :return: overall_score
    :rtype: float
    """
    min_value = 0
    max_value = 10
    overall_score = round(logit * (max_value - min_value) + min_value)
    return overall_score



class WritingScoring(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        question_id = json_data['question_id']
        question_content = json_data['question_content']
        anwser_content = json_data['answer_content']

        # check grammar and spelling
        spellmar_score, errors = compute_spell_grammar_score(anwser_content)

        # extract embeddings
        embeddings = encode_doc(anwser_content, question_id)
        embeddings = np.expand_dims(embeddings, 0)

        # compute scores
        co_score = coherence_score.predict(embeddings)
        se_score = basic_score.predict(embeddings)
        re_score = prompt_rel_score.predict(embeddings, prompts[int(question_id)])

        return {'score': spellmar_score, 'coherence_score': get_score(co_score), 'semantic_score': get_score(se_score), 'relative_score': get_score(re_score), 'error': errors}, 200
