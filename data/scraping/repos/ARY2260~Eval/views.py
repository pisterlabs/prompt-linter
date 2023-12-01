from app import app
import models.answers as answers
from flask import request, jsonify, after_this_request
from middlewares.is_admin import is_admin
from dotenv import load_dotenv
from flask_api import status
import pandas as pd
import cohere
from annoy import AnnoyIndex
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity
from cohere.classify import Example
from copy import deepcopy


warnings.filterwarnings('ignore')

api_key = os.getenv('COHERE_API')
load_dotenv()
co = cohere.Client(api_key)
class EvaluateMarks():
    """
    """

    def __init__(self, question_no, total_marks, query, gen_grammar=True, gen_trials=3, custom_toxic=True):
        """
        """
        self.question_no = question_no
        self.total_marks = total_marks
        self.query = query
        self.gen_grammar = gen_grammar
        self.gen_trails = gen_trials
        self.custom_toxic = custom_toxic
        self.toxic_examples = [
            Example("you are hot trash", "Toxic"),
            Example("go to hell", "Toxic"),
            Example("get rekt moron", "Toxic"),
            Example("get a brain and use it", "Toxic"),
            Example("say what you mean, you jerk.", "Toxic"),
            Example("Are you really this stupid", "Toxic"),
            Example("I will honestly kill you", "Toxic"),
            Example("yo how are you", "Benign"),
            Example("I'm curious, how did that happen", "Benign"),
            Example("Try that again", "Benign"),
            Example("Hello everyone, excited to be here", "Benign"),
            Example("I think I saw it first", "Benign"),
            Example("That is an interesting point", "Benign"),
            Example("I love this", "Benign"),
            Example("We should try that sometime", "Benign"),
            Example("You should go for it", "Benign"),
            Example("people are not good.", "Benign")
        ]

    def _semantic_check(self):
        """
        """
        answersAll = answers.get_all()
        database = pd.DataFrame(answersAll)
        embeds = co.embed(texts=database[database.q_category == self.question_no][['answer1', 'answer2', 'answer3', 'answer4', 'answer5']].values.flatten().tolist(),
                          model='large',
                          truncate='LEFT').embeddings

        # Create the search index, pass the size of embedding
        search_index = AnnoyIndex(len(embeds[0]), 'angular')
        # Add all the vectors to the search index
        for i in range(len(embeds)):
            search_index.add_item(i, embeds[i])
        search_index.build(10)  # 10 trees
        search_index.save('temp2.ann')

        # process user input
        user_input_embeds = co.embed(texts=[self.query],
                                     model='large',
                                     truncate='LEFT').embeddings
        similar_item_ids = search_index.get_nns_by_vector(user_input_embeds[0], 10,
                                                          include_distances=True)

        min_distance = min(similar_item_ids[1])
        return min_distance

    def _gen_grammar_check(self):
        """
        """
        similarity_values = []
        for i in range(self.gen_trails):
            response = co.generate(
                model='xlarge',
                prompt=f'This is a spell check generator that checks for grammar and corrects it. This also capitalizes the first letter of the sentence.\n\nSample: I would like a peice of pie.\nCorrect: I would like a piece of the pie.\n\nSample: my coworker said he used a financial planner to help choose his stocks so he wouldn\'t loose money.\nCorrect: My coworker said he used a financial planner to help him choose his stocks so he wouldn\'t lose money.\n\nSample: I ordered pizza, I also ordered garlic knots.\nCorrect: I ordered pizza; I also ordered garlic knots.\n\nSample: i bought winning lottery ticket the corner store\nCorrect: I bought my winning lottery ticket at the corner store.\n\nSample: try to reread your work to ensure you haven\'t left out any small words\nCorrect: Try to reread your work to ensure you haven\'t left out any small words.\n\nSample: I went to the movies with my sister. We will see the new comedy about dancing dogs.\nCorrect: I went to the movies with my sister. We saw the new comedy about dancing dogs.\n\nSample: the boy took their turn on the field.\nCorrect: The boy took his turn on the field.\n--\nSample: I could of won the race if I trained more.\nCorrect: I could have won the race if I had trained more.\n--\nSample: I went to the office, than i started my meeting.\nCorrect: I went to the office, then I started my meeting.\n--\nSample: {self.query}\nCorrect:',
                max_tokens=100,
                temperature=1.2,
                k=0,
                p=0.75,
                frequency_penalty=0,
                presence_penalty=0,
                stop_sequences=["--"],
                return_likelihoods='NONE')

            output = response.generations[0].text.rstrip("(\n)--")
            output = output.lstrip(" ")
            output = output.lower()

            embeds = co.embed(texts=[self.query.lower(), output],
                              model='large',
                              truncate='LEFT').embeddings

            sim_value = cosine_similarity(
                X=[embeds[0]], Y=[embeds[1]], dense_output=True)
            similarity_values.append(sim_value)
        similarity_score = max(similarity_values)
        return 1.0 if similarity_score > 0.997 else 0.0

    def _class_grammar_check(self):
        """
        """
        response = co.classify(
            model='cdb39157-6b82-4cb4-92c5-9e6037623d79-ft',
            inputs=[f"{self.query}"])
        return(float(response.classifications[0].prediction))

    def _grammar_check(self):
        """
        """
        if self.gen_grammar:
            return self._gen_grammar_check()
        else:
            return self._class_grammar_check()

    def _default_toxic_check(self):
        """
        """
        sentences = self.query.lower().rstrip('. ').split('.')
        for i in sentences:
            response = co.classify(
                model='large',
                inputs=[f"{i}"],
                examples=self.toxic_examples)
            if response.classifications[0].prediction == 'Toxic':
                return 1.0
        return 0.0

    def _custom_toxic_check(self):
        """
        """
        sentences = self.query.lower().rstrip('. ').split('.')
        for i in sentences:
            response = co.classify(
                model='8cec2377-0f7f-4557-81a4-7abc7dea3828-ft',
                inputs=[f"{i}"])
            if float(response.classifications[0].prediction) == 1.0:
                return 1.0
        return 0.0

    def _toxic_check(self):
        """
        """
        if self.custom_toxic:
            return self._custom_toxic_check()
        else:
            return self._default_toxic_check()

    def _jaccard_similarity(self, doc1, doc2):
        """
        """
        if(doc1 == '' and doc2 == ''):
            return 0.0

        # List the unique words in a document
        words_doc1 = set(doc1.lower().split())
        words_doc2 = set(doc2.lower().split())

        # Find the intersection of words list of doc1 & doc2
        intersection = words_doc1.intersection(words_doc2)

        # Find the union of words list of doc1 & doc2
        union = words_doc1.union(words_doc2)

        # Calculate Jaccard similarity score
        # using length of intersection set divided by length of union set
        return float(len(intersection)) / len(union)

    def _duplication_check(self):
        """
        """
        sentences = self.query.lower().rstrip('. ').split('.')
        similarities = []
        for i in range(len(sentences)):
            rest = deepcopy(sentences)
            rest.pop(i)
            rest = "".join(rest)
            score = self._jaccard_similarity(sentences[i], rest)
            similarities.append(score)
        duplication_ratio = sum(similarities)/(len(sentences)*0.08)

        if duplication_ratio > 2.0:
            dup_score = 2
        elif duplication_ratio > 1.0:
            dup_score = 1
        else:
            dup_score = 0
        return dup_score

    def run_checks(self):
        """
        """
        if hasattr(self,'check_results'):
            return self.check_results
        s_score = self._semantic_check()
        g_score = self._grammar_check()
        t_score = self._toxic_check()
        d_score = self._duplication_check()
        self.check_results = [s_score, g_score, t_score, d_score]
        return self.check_results

    def evaluate(self):
        """
        """
        if hasattr(self,'check_results'):
            semantic_score, grammar_score, toxic_score, duplication_score = self.check_results
        else:
            semantic_score, grammar_score, toxic_score, duplication_score = self.run_checks()

        if toxic_score == 1.0:
            return {'score': 0, 'tag': 'Toxic'}

        if duplication_score == 2:
            semantic_score += 0.2

        elif duplication_score == 1:
            semantic_score += 0.1

        if semantic_score < 0.4:
            scored_marks = self.total_marks
        elif semantic_score < 0.75:
            scored_marks = self.total_marks*(2/3)
        elif semantic_score < 0.1:
            scored_marks = self.total_marks*(1/3)
        else:
            scored_marks = 0

        if grammar_score == 0.0:
            scored_marks = scored_marks-1 if scored_marks > 2 else scored_marks-0.5

        scored_marks = max(0, scored_marks)
        return {'score': scored_marks, 'tag': 'Not Toxic'}

@app.route('/answers', methods=['POST'])
@is_admin()
def add_answer():
  answer1 = request.json.get('answer1')
  answer2 = request.json.get('answer2')
  answer3 = request.json.get('answer3')
  answer4 = request.json.get('answer4')
  answer5 = request.json.get('answer5')
  question = request.json.get('question')
  if not (answer1 and question):
    return jsonify({'message':
                    "Insufficient data"}), status.HTTP_400_BAD_REQUEST
  answers.create(
    answer1=answer1,
    answer2=answer2,
    answer3=answer3,
    answer4=answer4,
    answer5=answer5,

    question=question,
  )
  return jsonify({}), status.HTTP_201_CREATED


@app.route('/answers', methods=['GET'])
def get_answers():

  @after_this_request
  def add_header(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

  answersAll = answers.get_all()

  return jsonify(answersAll), status.HTTP_200_OK

@app.route('/evaluate', methods=['POST'])
def evaluate_answer():
  answerQuery = request.json.get('answer')
  categoryQuery = request.json.get('category')
  marks = request.json.get('marks')
  eval = EvaluateMarks(int(categoryQuery), int(marks), query=answerQuery)
  marks = eval.evaluate()
  # print(marks)
  # print(eval.run_checks())
  return marks

@app.route('/run_checks', methods=['POST'])
def run_checks():
  answerQuery = request.json.get('answer')
  categoryQuery = request.json.get('category')
  marks = request.json.get('marks')
  eval = EvaluateMarks(int(categoryQuery), int(marks), query=answerQuery)
  checks = eval.run_checks()
  return checks
  
# @app.route('/evaluate', methods=['POST'])
# def evaluate_answer():
#   answerQuery = request.json.get('answer')
#   categoryQuery = request.json.get('category')

#   @after_this_request
#   def add_header(response):
#     response.headers.add('Access-Control-Allow-Origin', '*')
#     return response

#   answersAll = answers.get_all()

#   # testing prediction part
#   df = pd.DataFrame(answersAll)
#   df.drop('id', axis=1, inplace=True)
#   print(df)
#   co = cohere.Client(api_key)
#   embeds = co.embed(texts=list(df['answer']), model='large',
#                     truncate='LEFT').embeddings
#   search_index = AnnoyIndex(len(embeds[0]), 'angular')

#   # Add all the vectors to the search index
#   for i in range(len(embeds)):
#     search_index.add_item(i, embeds[i])
#   search_index.build(10)  # 10 trees
#   search_index.save('test.ann')

#   user_input = [answerQuery]
#   user_input_embeds = co.embed(texts=user_input,
#                                model='large',
#                                truncate='LEFT').embeddings

#   similar_item_ids = search_index.get_nns_by_vector(user_input_embeds[0],
#                                                     10,
#                                                     include_distances=True)

#   # Format and print the text and distances
#   results = pd.DataFrame(
#     data={
#       'texts': df.iloc[similar_item_ids[0]]['answer'],
#       'label': df.iloc[similar_item_ids[0]]['category'],
#       'distance': similar_item_ids[1]
#     })
#   print(f"Question:'{user_input}'\nNearest neighbors:")

#   mean_distance = results[results['label'] == int(
#     categoryQuery)]['distance'].mean()
#   if mean_distance < 1:
#     return "The answer is correct " + str(mean_distance), status.HTTP_200_OK
#   return "The answer is incorrect"
#   # end testing
