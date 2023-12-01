import json

import openai

from config.global_logger import logger
from database_util.redis.redis_client import api_key_manager
from openai_util.function_call.funcation_invoke import get_invoke_method_info_by_name, \
    get_function_result_from_openai_response
from openai_util.function_call.openaifunc_decorator import openai_func


@openai_func
def save_method_code(code_str, skill_name):
    """
    Save the method code to the file.
    :param code_str: The code string to be saved.
    :param skill_name: The name of the skill to be learned.
    """
    print_learn_code_prompt = f'''
                Generate a Python code file according to the following content and save it under the learn_skill path. This file should  contain one utility function: {skill_name}.
                 ```
                 {code_str}
                 ```
                 The written method needs to add the annotation @openai_func, and generate method annotations to ensure that the code is indented, similar to the following:
                 from openai_util.function_call.openaifunc_decorator import openai_func\
                 @openai_func\
                 def function(param1: str, param2: str):\
                     \'''\
                     the method description\
                     :param param1: the description of param1\
                     :param param2: the description of param2\
                     \'''\
                     pass\
                '''
    params = {

        "messages": [{"role": "user", "content": print_learn_code_prompt}],
        "model": 'gpt-4',
        "temperature": 0,
        "functions": [get_invoke_method_info_by_name("run_python_code")],
        "function_call": {"name": "run_python_code"}
    }
    response2 = openai.ChatCompletion.create(**params)
    get_function_result_from_openai_response(response2)
    try:
        import learn_skill
    except Exception as e:
        logger.error(e)
        return 'Failed: ' + str(e)
    return 'Success'


last_error = None


@openai_func
def learn_and_save_as_skill(skill_name: str):
    logger.debug('learn_and_save_as_skill')
    """
    Learn a new skill and save it as a skill.
    :param skill_name: The name of the skill to be learned.
    """
    openai.api_key = api_key_manager.get_openai_key()
    method_code_prompt = f'''
    Generate python code that contains one utility function: {skill_name}.
You can use Google to query the required information and repair errors, 
use 'install_modules' to install the modules that code required, 
use 'run_python_code' to check whether the code is correct, 
and use 'save_method_code' to save the final generated code'''
    learn_skill_messages = [{"role": "user", "content": method_code_prompt}]
    params = {
        "messages": learn_skill_messages,
        "model": 'gpt-4',
        "temperature": 0,
        "functions": [get_invoke_method_info_by_name("query_info_from_google"),
                      get_invoke_method_info_by_name("install_modules"),
                      get_invoke_method_info_by_name("run_python_code"),
                      get_invoke_method_info_by_name("save_method_code")],
        "function_call": "auto"
    }
    retry_count = 0
    while True:
        global last_error
        response = openai.ChatCompletion.create(**params)
        function_name, function_result = get_function_result_from_openai_response(response)
        if (function_name == 'save_method_code' and function_result == 'Success') or retry_count >= 20:
            break
        else:
            retry_count += 1
        if function_result and 'Error' in function_result:
            if function_result == last_error:
                function_result = function_result + \
                                  ' (This is the second time this error occurs, \
                                  please think carefully and modify the code from another way)'
            last_error = function_result
        learn_skill_messages.append(response['choices'][0]['message'])
        if function_name:
            learn_skill_messages.append({"role": "function", "name": function_name,
                                         "content": json.dumps(function_result)})
        # print('messages: ', json.dumps(learn_skill_messages))
    return f"""lean and save as skill successfully after try {retry_count} times."""


# print(learn_and_save_as_skill('Get the current lunar date representation'))