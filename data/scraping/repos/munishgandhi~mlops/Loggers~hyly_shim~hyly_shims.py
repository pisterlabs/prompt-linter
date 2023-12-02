import openai
from typing import Any, Dict, List, Optional
from datetime import datetime
import requests
import time
import os
import asyncio
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt
import json

openai.organization = "org-kf0CNDw9Dvl65UAapwwoDpRF"
openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]



# Ashish - important!! Install mysql-connector-python instead of mysql-connector
import mysql.connector as SQLC


class Hyly_Shim(object):
    def __init__(self):
        self.MODEL_COST_PER_1K_TOKENS = {
            "gpt-4": 0.03,
            "gpt-4-0314": 0.03,
            "gpt-4-completion": 0.06,
            "gpt-4-0314-completion": 0.06,
            "gpt-4-32k": 0.06,
            "gpt-4-32k-0314": 0.06,
            "gpt-4-32k-completion": 0.12,
            "gpt-4-32k-0314-completion": 0.12,
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-instruct": 0.002,
            "gpt-3.5-turbo-0301": 0.002,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-0613-completion": 0.002,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-3.5-turbo-16k-0613-completion": 0.004,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-16k-completion": 0.004,
            "text-ada-001": 0.0004,
            "text-embedding-ada-002": 0.0001,
            "ada": 0.0004,
            "text-babbage-001": 0.0005,
            "babbage": 0.0005,
            "text-curie-001": 0.002,
            "curie": 0.002,
            "text-davinci-003": 0.02,
            "text-davinci-002": 0.02,
            "code-davinci-002": 0.02,
        }

    def retry_on_error(max_retries=3, wait_time=2):
        def decorator(func):
            def wrapper(*args, **kwargs):
                retries = 0
                while retries < max_retries:
                    try:
                        return func(*args, **kwargs)
                    except (openai.error.ServiceUnavailableError, openai.error.APIError) as e:
                        print(f"Error occurred: {str(e)}")
                        retries += 1
                        print(f"Retrying... (Attempt {retries}/{max_retries})")
                        time.sleep(wait_time)
                raise Exception(f"Failed after {max_retries} retries.")

            return wrapper

        return decorator

    def get_openai_token_cost_for_model(self,
                                        model_name: str, num_tokens: int, is_completion: bool = False
                                        ) -> float:
        suffix = "-completion" if is_completion and (
                model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5-turbo-16k-0613") or model_name.startswith("gpt-3.5-turbo-0613") or model_name.startswith("gpt-3.5-turbo-16k") ) else ""
        model = model_name.lower() + suffix
        if model not in self.MODEL_COST_PER_1K_TOKENS:
            raise ValueError(
                f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(self.MODEL_COST_PER_1K_TOKENS.keys())
            )
        return round(self.MODEL_COST_PER_1K_TOKENS[model] * num_tokens / 10 , 2)

    def calculate_total_time(self, start_time, end_time):
        total_time_seconds = end_time - start_time

        if total_time_seconds < 0:
            total_time = f"{total_time_seconds:.2f}s"
        elif total_time_seconds < 60:
            total_time = f"{total_time_seconds:.2f}s"
        else:
            total_time = f"{total_time_seconds / 60:.2f} minutes"

        return total_time

    def record_into_mysql(self, model_name, prompt_tokens, completion_tokens, total_tokens, who_is_using="",
                          using_for=""):
        prompt_cost = self.get_openai_token_cost_for_model(model_name=model_name, num_tokens=prompt_tokens)
        completion_cost = self.get_openai_token_cost_for_model(model_name=model_name,
                                                               num_tokens=completion_tokens,
                                                               is_completion=True)
        cost = prompt_cost + completion_cost
        cost = str(cost)

        url = 'https://qa.hyly.us/graphql'
        headers = {'Authorization': 'userid___1709542253522009420'}
        body = f"""
           mutation {{
            createOpenaiHistory(
            input: {{
                gptModelName: "{model_name}",
                calledBy: "{who_is_using}",
                toolName: "{using_for}",
                totalPromptTokens: "{prompt_tokens}",
                totalCompletionTokens: "{completion_tokens}",
                totalTokens: "{total_tokens}",
                totalCost: "{cost}"
            }}
            ) {{
                openAiHistory {{
                    gptModelName
                    totalPromptTokens
                    totalCompletionTokens
                    totalTokens
                    totalCost
                    calledBy
                    toolName
                    promptTexts
                    responseTexts
                }}
            }}
            }}

        """
        response = requests.post(url=url, json={"query": body}, headers=headers)
        return prompt_cost, completion_cost, cost

    def update_model_name(self, old_model_name, token_limit, new_model_name, user_prompt=""):
        if user_prompt != "":
            encoding = tiktoken.get_encoding("cl100k_base")
            total_token_len = len(encoding.encode(user_prompt)) + 512
            if total_token_len > token_limit:
                return new_model_name
            else:
                return old_model_name
        else:
            return old_model_name

    @retry_on_error()
    async def hyly_openai_completion(self, message, functions, model_name, temperature, max_tokens, who_is_using,
                                     app_category, app, app_field, system_prompt="", user_prompt=""):

        if model_name == 'gpt-3.5-turbo-instruct':
            start = time.time()
            url = "https://api.openai.com/v1/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"  # Replace with your actual API key
            }

            payload = {
                "model": model_name,
                "prompt": message,
                "temperature": temperature,
                "max_tokens": max_tokens
            }



            session = requests.Session()
            response = session.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                print(data)

                response = data["choices"][0]["text"]
                usage = data["usage"]

                prompt_tokens = usage['prompt_tokens']
                completion_tokens = usage['completion_tokens']
                total_tokens = usage['total_tokens']
                using_for = app
                prompt_cost, completion_cost, total_cost = self.record_into_mysql(model_name=model_name,
                                                                                  prompt_tokens=prompt_tokens,
                                                                                  completion_tokens=completion_tokens,
                                                                                  total_tokens=total_tokens,
                                                                                  who_is_using=who_is_using,
                                                                                  using_for=using_for)
                end = time.time()
                total_time = self.calculate_total_time(start, end)
                log_data = {"who_is_using": who_is_using,
                            "app_category": app_category,
                            "app": app,
                            "app_field": app_field,
                            "input": user_prompt,
                            "response": response,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "prompt_cost": prompt_cost,
                            "completion_cost": completion_cost,
                            "total_cost": total_cost,
                            "total_time": total_time
                            }

                return log_data, response
        elif model_name != 'text-embedding-ada-002':
            start = time.time()
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"  # Replace with your actual API key
            }

            model_name = self.update_model_name(old_model_name=model_name, token_limit=4096, new_model_name="gpt-3.5-turbo-16k-0613", user_prompt=user_prompt)

            payload = {
                "model": model_name,
                "messages": message,
                "temperature": temperature,
                "max_tokens": max_tokens
                      }

            if functions is not None and len(functions) > 0:
                payload["functions"] = functions
                payload["function_call"] = "auto"

            session = requests.Session()
            response = session.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if model_name == 'gpt-3.5-turbo-instruct':
                    response = data["choices"][0].text
                    usage = data.usage
                else:
                    response = data["choices"][0]["message"]
                    usage = data["usage"]

                prompt_tokens = usage['prompt_tokens']
                completion_tokens = usage['completion_tokens']
                total_tokens = usage['total_tokens']
                using_for = app
                prompt_cost, completion_cost, total_cost = self.record_into_mysql(model_name=model_name,
                                                                                  prompt_tokens=prompt_tokens,
                                                                                  completion_tokens=completion_tokens,
                                                                                  total_tokens=total_tokens,
                                                                                  who_is_using=who_is_using,
                                                                                  using_for=using_for)
                end = time.time()
                total_time = self.calculate_total_time(start, end)
                log_data = {"who_is_using": who_is_using,
                            "app_category": app_category,
                            "app": app,
                            "app_field": app_field,
                            "input": user_prompt,
                            "response": response['content'],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "prompt_cost": prompt_cost,
                            "completion_cost": completion_cost,
                            "total_cost": total_cost,
                            "total_time": total_time
                            }

                return log_data, response

        else:
            start = time.time()
            embeddings = openai.Embedding.create(input=[user_prompt], model=model_name)
            prompt_cost, completion_cost, total_cost = self.record_into_mysql(model_name=model_name,
                                                                              prompt_tokens=embeddings['usage']['total_tokens'],
                                                                              completion_tokens=0,
                                                                              total_tokens=embeddings['usage']['total_tokens'],
                                                                              who_is_using=who_is_using,
                                                                              using_for=app)
            end = time.time()
            total_time = self.calculate_total_time(start, end)
            log_data = {"who_is_using": who_is_using,
                        "app_category": app_category,
                        "app": app,
                        "app_field": app_field,
                        "input": user_prompt,
                        "response": embeddings['data'][0]['embedding'],
                        "prompt_tokens": embeddings['usage']['total_tokens'],
                        "completion_tokens": 0,
                        "total_tokens": embeddings['usage']['total_tokens'],
                        "prompt_cost": prompt_cost,
                        "completion_cost": completion_cost,
                        "total_cost": total_cost,
                        "total_time": total_time
                        }
            return log_data, embeddings['data'][0]['embedding']

    @retry_on_error()
    async def HylyOpenAICompletion(self, message, functions, model_name, temperature, max_tokens,
                                   who_is_using, app_category, app, app_field, system_prompt="", user_prompt=""):
        task = asyncio.create_task(
            self.hyly_openai_completion(message=message,
                                        functions=functions,
                                        model_name=model_name,
                                        temperature=temperature,
                                        max_tokens=max_tokens,
                                        who_is_using=who_is_using,
                                        app_category=app_category,
                                        app=app,
                                        app_field=app_field,
                                        system_prompt=system_prompt,
                                        user_prompt=user_prompt)
        )

        try:
            async with asyncio.timeout(60):
                return await task
        except Exception as e:
            print(e)
            response = 'Sorry, could not get response'
            log_data = {"who_is_using": who_is_using,
                        "app_category": app_category,
                        "app": app,
                        "app_field": app_field,
                        "input": user_prompt,
                        "response": response,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "prompt_cost": 0,
                        "completion_cost": 0,
                        "total_cost": 0,
                        "total_time": 0
                        }
        return log_data, response


if __name__ == "__main__":
    hs = Hyly_Shim()
