import json
import uuid
import openai
from openai import OpenAIError
import os
from database_util.redis.redis_client import api_key_manager
from config.global_logger import logger
from openai_util.function_call.funcation_invoke import get_invoke_method_info_by_name, do_step_by_step, \
    get_function_result_from_openai_response
from openai_util.chat import openai_chat_completions, insert_ai_response_record


def push_message(message):
    json_item = json.dumps(message)  # Convert dictionary to JSON string
    api_key_manager.r.lpush('AUDIO_CHAT_HISTORY', json_item)
    api_key_manager.r.ltrim('AUDIO_CHAT_HISTORY', 0, 5)


def get_recent_chat_history():
    byte_list = api_key_manager.r.lrange('AUDIO_CHAT_HISTORY', 0, -1)
    return [json.loads(item.decode('utf-8')) for item in byte_list]


def create_chat_completion(user_content, function_msg, functions=None, function_call=None, message_id='0',
                           parent_id='0'):
    try:
        openai.api_key = api_key_manager.get_openai_key()
        use_model = os.getenv('DEFAULT_CHAT_MODEL', 'gpt-3.5-turbo')
        messages = get_recent_chat_history() + [{"role": "user", "content": user_content}]
        if function_msg:
            messages.append(function_msg)
            use_model = os.getenv('GET_METHOD_ARGUMENTS_MODEL', 'gpt-3.5-turbo-0613')
        if not function_call:
            function_call = 'none'
        if not functions:
            # return openai.ChatCompletion.create(
            #     model=use_model,
            #     messages=messages,
            #     temperature=0
            # )
            return openai_chat_completions(params={"model": use_model,
                                                   "messages": messages,
                                                   "temperature": 0}, message_id=message_id,
                                           parent_id=parent_id, user_name=os.getenv('MY_NAME', 'default'),
                                           ip="127.0.0.1")
        use_model = os.getenv('GET_INVOKE_METHOD_MODEL', 'gpt-3.5-turbo-16k')
        # return openai.ChatCompletion.create(
        #     model=use_model,
        #     messages=messages,
        #     functions=functions,
        #     function_call=function_call,
        #     temperature=0
        # )
        return openai_chat_completions(params={"model": use_model,
                                               "messages": messages, "functions": functions,
                                               "function_call": function_call,
                                               "temperature": 0}, message_id=message_id,
                                       parent_id=parent_id, user_name=os.getenv('MY_NAME', 'default'),
                                       ip="127.0.0.1")
    except OpenAIError as e:
        logger.debug(f"user_content: {user_content}, function_msg: {function_msg}, functions: {functions}")
        logger.error(e.message)
        return None


# def run_conversation_v1(user_content):
#     selection_response = create_chat_completion(user_content, None,
#                                                 [get_invoke_method_info_by_name("get_invoke_method_info")],
#                                                 "auto")
#     logger.debug("1:" + json.dumps(selection_response))
#     function_name, function_result = get_function_result_from_openai_response(selection_response)
#     if not function_name:
#         return selection_response
#     execution_response = create_chat_completion(user_content, None, [function_result], "auto")
#     logger.debug("2:" + json.dumps(execution_response))
#     function_name, function_result = get_function_result_from_openai_response(execution_response)
#     if not function_name:
#         return execution_response
#     final_response = create_chat_completion(user_content, {"role": "function", "name": function_name,
#                                                            "content": json.dumps(function_result), },
#                                             None,
#                                             "auto")
#     logger.debug("3:" + json.dumps(final_response))
#     return final_response


def create_chat_completion_with_msg(new_msg, functions, func_name=None):
    try:
        if func_name:
            functon_call = {"name": func_name}
        else:
            functon_call = "auto"
        if not functions:
            openai.api_key = api_key_manager.get_openai_key()
            return openai.ChatCompletion.create(
                model=os.getenv('GET_METHOD_ARGUMENTS_MODEL', 'gpt-3.5-turbo-0613'),
                messages=new_msg,
                temperature=0
            )
        openai.api_key = api_key_manager.get_openai_key()
        return openai.ChatCompletion.create(
            model=os.getenv('GET_METHOD_ARGUMENTS_MODEL', 'gpt-3.5-turbo-0613'),
            messages=new_msg,
            functions=functions,
            function_call=functon_call,
            temperature=0
        )
    except OpenAIError as e:
        logger.error(e, exc_info=True)
        return None


def run_single_step_chat(index, messages, functions, func_name):
    try:
        messages.append({"role": "user", "content": "now you are in step {}".format(index)})
        new_msg = list(messages)
        get_arguments_response = create_chat_completion_with_msg(new_msg, functions, func_name)
        # logger.debug("get_arguments_response:{}".format(json.dumps(get_arguments_response)))
        function_name, function_result = get_function_result_from_openai_response(get_arguments_response)
        logger.debug("function_name:{}, function_result:{}".format(function_name, function_result))
        if not function_name:
            return get_arguments_response, json.dumps(function_result)
        new_msg.append(get_arguments_response["choices"][0]["message"])
        new_msg.append({"role": "function", "name": function_name,
                        "content": json.dumps(function_result)})
        final_response = create_chat_completion_with_msg(new_msg,
                                                         None)
        logger.debug("final_response:{}".format(json.dumps(final_response)))
        messages.append(final_response["choices"][0]["message"])
        messages.append(
            {"role": "assistant", "content": "the result of step {} is: {}".format(index, json.dumps(function_result))})
        return final_response, json.dumps(function_result)
    except OpenAIError as e:
        logger.exception(e)
        return None


def run_conversation_v2(user_content, parent_id):
    # Analyze the user's instructions into detailed operation steps through chatgpt
    message_id = 'audio_chat-' + str(uuid.uuid4())
    step_response = create_chat_completion(user_content, None,
                                           [do_step_by_step()],
                                           "auto", message_id, parent_id)
    logger.debug([do_step_by_step()])
    message = step_response["choices"][0]["message"]
    if not message.get("function_call"):
        logger.debug("response:{}".format(step_response["choices"][0]["message"]['content']))
        insert_ai_response_record(message['content'], step_response["id"], message_id, '127.0.0.1',
                                  os.getenv('MY_NAME', 'default'))
        return message['content'], step_response["id"]
    function_args = message["function_call"]["arguments"]
    logger.debug("response:{}".format(json.loads(function_args)))
    steps = json.loads(function_args)['steps']
    # order by step_order asc
    steps.sort(key=lambda x: x['step_order'])
    messages = [{"role": "system",
                 "content": "You are an advanced robot, and you can do almost anything that humans ask you to do." +
                            "Please answer the user with Chinese"},
                {"role": "user", "content": user_content},
                {"role": "user", "content": "Follow the steps below:" + json.dumps(steps)}]
    # loop through each step
    order_step_response = None
    for index, step in enumerate(steps):
        order_step_response, function_result = run_single_step_chat(index, messages, [
            get_invoke_method_info_by_name(step['step_method'])], step['step_method'])
        logger.debug(
            "order:{}, response:{}".format(index, order_step_response["choices"][0]["message"]['content']))
    if order_step_response:
        insert_ai_response_record(order_step_response["choices"][0]["message"]['content'], order_step_response["id"],
                                  message_id, '127.0.0.1', os.getenv('MY_NAME', 'default')
                                  )
        return order_step_response["choices"][0]["message"]['content'], order_step_response["id"]
    else:
        return '出错了', None

# print(run_conversation_v2("Navigate to Hangzhou"))
# print(run_conversation_v2("If the weather in Hangzhou is fine, call gongqi"))
# print(run_conversation_v2(
#     "https://images.unsplash.com/photo-1618843479313-40f8afb4b4d8?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=2070&q=80  What is the brand of the car in this picture"))
# print(run_conversation_v2("play a song of 'Welcome Home, Son'"))
