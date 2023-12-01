from flask import make_response, jsonify, request, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_restx import Resource, Namespace
from http import HTTPStatus
import openai
from moview.config.container.container_config import ContainerConfig
from moview.config.loggers.mongo_logger import *
from moview.decorator.timing_decorator import api_timing_decorator
from moview.decorator.validation_decorator import validate_char_count
from moview.controller.constants.answer_contants import MAX_INTERVIEW_QUESTION_LENGTH, MAX_INTERVIEW_ANSWER_LENGTH
from moview.exception.retry_execution_error import RetryExecutionError

api = Namespace('answer', description='answer api')


@api.route('/answer')
class AnswerConstructor(Resource):

    @jwt_required()
    @api_timing_decorator
    @validate_char_count({
        'question_content': MAX_INTERVIEW_QUESTION_LENGTH,
        'answer_content': MAX_INTERVIEW_ANSWER_LENGTH
    })
    @jwt_required()
    def post(self):
        user_id = str(get_jwt_identity())
        g.user_id = user_id
        request_body = request.get_json()

        interview_id = request_body['interview_id']
        g.interview_id = interview_id
        question_id = request_body['question_id']
        question_content = request_body['question_content']
        answer_content = request_body['answer_content']

        interview_service = ContainerConfig().interview_service
        answer_service = ContainerConfig().answer_service

        try:
            interview_dict = interview_service.find_interview(user_id=user_id, interview_id=interview_id)

            interview_service.add_latest_question_into_interview(interview_id=interview_id,
                                                                 interview_dict=interview_dict,
                                                                 question_id=question_id,
                                                                 question_content=question_content)

            chosen_question, saved_id = answer_service.maybe_give_followup_question_about_latest_answer(
                interview_id=interview_id,
                question_id=question_id,
                question_content=question_content,
                answer_content=answer_content)

        except RetryExecutionError as e:
            error_logger(msg="RETRY EXECUTION ERROR")
            raise e

        except openai.error.RateLimitError as e:
            error_logger(msg="RATE LIMIT ERROR")
            raise e

        except Exception as e:
            error_logger(msg="UNKNOWN ERROR", error=e)
            raise e

        execution_trace_logger("ANSWER CONTROLLER: POST",
                               question_id=question_id,
                               question_content=question_content,
                               answer_content=answer_content,
                               chosen_question=chosen_question,
                               saved_id=saved_id)

        return make_response(jsonify(
            {'message': {
                'question_content': chosen_question,
                'question_id': saved_id
            }}
        ), HTTPStatus.OK)
