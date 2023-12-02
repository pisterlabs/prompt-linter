import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.docstore.document import Document
from langchain.llms import OpenAI
import pinecone
from myapp.database.models import Question, Answer, Quiz, QuizQuestion, QuizParticipant, UserResponse
from myapp.database.db import db


class Handler:
    def __init__(self, pinecone_env, index_name):
        """
        A base class to handle interactions with Pinecone.

        Args:
            pinecone_env (str): The environment to use in Pinecone.
            index_name (str): The name of the Pinecone index to use.
        """
        self.pinecone_env = pinecone_env
        pinecone.init(
            api_key=os.environ['PINECONE_API_KEY'], environment=self.pinecone_env)
        self.index_name = index_name


class Search(Handler):
    def __init__(self, pinecone_env, index_name):
        """
        Search class inheriting from Handler, primarily responsible for performing semantic searches on the index.

        Args:
            pinecone_env (str): The environment to use in Pinecone.
            index_name (str): The name of the Pinecone index to use.
        """
        super().__init__(pinecone_env, index_name)
        self.embeddings = OpenAIEmbeddings()

    def get_question_ids(self, course, topic, num, score=0.3):
        """
        Fetches the IDs of questions that match a given topic and course from the Pinecone index.

        Args:
            course (str): The course to search within.
            topic (str): The topic to search for.
            num (int): The number of results to return.
            score (float, optional): The minimum similarity score for results. Defaults to 0.3.

        Returns:
            list: A list of question IDs that match the search criteria.
        """
        cone = Pinecone.from_existing_index(self.index_name, self.embeddings)
        docs = cone.similarity_search_with_score(
            topic, num, {"course": course})
        print("Docs returned:", docs)
        num = min(num, len(docs))
        ids = [docs[i][0].metadata["question_id"]
            for i in range(num) if docs[i][1] > score]
        return ids


class Predict(Handler):
    def __init__(self, pinecone_env, index_name):
        from run import app
        super().__init__(pinecone_env, index_name)
        self.embeddings = OpenAIEmbeddings()
        self.app = app

    def get_message(self, participation_id):
        """
        Generates a formatted message with the course, questions, correct answers, and user's answers based on a given quiz participation ID.

        Args:
            participation_id (int): The ID of the quiz participation to get the message for.

        Returns:
            str: A formatted message containing details about the quiz and user's answers.
        """
        quiz_manager = QuizManager(self.app)
        response_data = quiz_manager.get_responses(participation_id)
        course = response_data[0]['question'].course
        text = f"i just took a quiz on {course} and below is the question, correct answer and my answer. would you please provide feedback on the quiz? please start with 'Here's some feedback on your quiz:'. describe the overall performance and then comment only on the questions with wrong answers. base your rating on the comparison of my answers and the correct answers provided. make your feedback concise and no more than 100 words.\n\n"
        for res in response_data:
            text += f"question: {res['question'].question}\ncorrect answer: {next(filter(lambda answer: answer.is_correct, res['answers']), None).answer}\nmy answer: {res['response'].answer}\n\n"
        return text

class QuizManager:
    def __init__(self, app):
        self.app = app

    def get_quizzes(self, user_id):
        """
        Fetches all quizzes created by a specific user.

        Args:
            user_id (str): The ID of the user to get quizzes for.

        Returns:
            list: A list of Quiz objects created by the user.
        """
        with self.app.app_context():
            quizzes = Quiz.query.filter_by(
                user_id=user_id).order_by(Quiz.quiz_id.desc()).all()
            return quizzes

    def get_questions_answers(self, quiz_id):
        """
        Fetches all questions and their corresponding answers in a specific quiz.

        Args:
            quiz_id (int): The ID of the quiz to get questions and answers for.

        Returns:
            list: A list of dictionaries, each containing a Question object and a list of its corresponding Answer objects.
        """
        with self.app.app_context():
            quiz_questions = QuizQuestion.query.filter_by(
                quiz_id=quiz_id).all()
            questions_answers = []

            for quiz_question in quiz_questions:
                question = quiz_question.question
                answers = Answer.query.filter_by(
                    question_id=question.question_id).all()
                questions_answers.append({
                    'question': question,
                    'answers': answers
                })

            return questions_answers

    def get_participation(self, quiz_id):
        """
        Fetches all participations in a specific quiz.

        Args:
            quiz_id (int): The ID of the quiz to get participations for.

        Returns:
            list: A list of QuizParticipant objects for the quiz.
        """
        with self.app.app_context():
            participation = QuizParticipant.query.filter_by(
                quiz_id=quiz_id).all()
            return participation

    def get_responses(self, participation_id):
        """
        Fetches all user responses for a specific quiz participation.

        Args:
            participation_id (int): The ID of the quiz participation to get responses for.

        Returns:
            list: A list of dictionaries, each containing a Question object, the selected Answer object, and a list of all Answer objects for the question.
        """
        with self.app.app_context():
            user_responses = UserResponse.query.filter_by(
                participation_id=participation_id).all()
            response_data = []

            for user_response in user_responses:
                question = Question.query.filter_by(
                    question_id=user_response.question_id).first()
                answers = Answer.query.filter_by(
                    question_id=question.question_id).all()
                selected_answer = Answer.query.filter_by(
                    answer_id=user_response.answer_id).first()
                response_data.append({
                    'question': question,
                    'response': selected_answer,
                    'answers': answers
                })

            return response_data

class Ingest(Handler):
    def __init__(self, pinecone_env, index_name):
        from run import app
        super().__init__(pinecone_env, index_name)
        self.embeddings = OpenAIEmbeddings()
        self.app = app

    def ingest_questions(self, question_ids):
        """
        Ingests questions into the Pinecone index.

        Args:
            question_ids (list): A list of question IDs to ingest.
        """
        documents = []

        with self.app.app_context():
            for question_id in question_ids:
                question = db.session.get(Question, question_id)
                answers = Answer.query.filter_by(question_id=question_id).all()

                if not question or not answers:
                    print(
                        f"No question or answers found for question_id: {question_id}")
                    continue

                combined = f"{question.question}\n"
                for answer in answers:
                    combined += f"- {answer.answer}\n"

                document = Document(
                    page_content=combined,
                    metadata={
                        "course": question.course,
                        "question_id": question_id,
                    },
                )

                documents.append(document)
                print("Documents to be sent:", documents)

        Pinecone.from_documents(
            documents, self.embeddings, index_name=self.index_name)
        print(
            f"Successfully sent {len(documents)} questions and answers to Pinecone.")


class Index(Handler):
    def __init__(self, pinecone_env, index_name):
        super().__init__(pinecone_env, index_name)

    def view_stats(self):
        """
        Fetches statistics about the Pinecone index.

        Returns:
            dict: A dictionary containing statistics about the index.
        """
        index = pinecone.Index(self.index_name)
        stats = index.describe_index_stats()
        return stats
