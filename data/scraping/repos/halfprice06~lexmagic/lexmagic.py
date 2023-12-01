import openai
import yaml
import os
import json
import time
import llm_functions
import vector_search_cc
from itertools import groupby
from vector_search_cc import *
from llm_functions import *

os.system('cls' if os.name == 'nt' else 'clear')

class LawBot:
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file

        # Load the configuration from the YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set API keys from the config file
        self.api_key_openai = config['api_keys']['openai']
                
        # Set default settings from the config file
        self.model = config['settings']['model']
        self.persona = config['settings']['persona']
        self.voice = config['settings']['voice']

        openai.api_key = self.api_key_openai
        
    def chat_completion(self, prompt: str, extra_context_from_user: str = ""):

        print(f"User's prompt: \n '{prompt}'")
        
        functions_used = []

        top_10_results = []  # Initialize top_10_results here
                    
        classifier_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "system", "content": "The user is going to ask you a question about Louisiana law. Decide whether you need more information to answer the question first. If the user just sends a greeting, you need more info. Always assume the question is about Louisiana law unless the user mentions otherwise."},
                        {"role": "user", "content": f"{prompt} | Extra context from user: {extra_context_from_user}"}],
            temperature=0,
            functions=FUNCTIONS,
            function_call={"name": "classify_question"}
        )

        classifier_response = classifier_completion.choices[0].message

        if classifier_response.get("function_call"):
            function_name = classifier_response["function_call"]["name"]
            function_args = json.loads(classifier_response["function_call"]["arguments"])
            function_to_call = getattr(llm_functions, function_name, None)

            if function_to_call is None:
                print("Function not found.")

            else:
                classifier_function_response = function_to_call(**function_args)
                    
            functions_used.append({"name": function_name, "arguments": function_args})

            print("Do we need more information from user?")

            print(classifier_function_response)
            
            if classifier_function_response == "No":
                print("We have all the information we need to answer the question.")

            else:
                follow_up_classifier_completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "system", "content": "The user is going to ask you a question about Louisiana law, but you need more information. Ask the user for the specific information you need to answer the question."},
                                {"role": "user", "content": f"{prompt}"}],
                    temperature=0,
                )

                # ...
                follow_up_classifier_response = follow_up_classifier_completion.choices[0].message

                print(follow_up_classifier_response)                

                follow_up_classifier_message = "NEED_MORE_INFO"

                return follow_up_classifier_response.content,  follow_up_classifier_message

        raw_completion = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "system", "content": "The user is going to ask you a question about Louisiana law. Answer completely."},
                        {"role": "user", "content": "Original question:" + prompt + "Additional context:" + extra_context_from_user}],
            temperature=0,
        )

        raw_response = raw_completion.choices[0].message

        print("GPT4's raw attempt at answering the question: \n")
        print(raw_response)

        internet_completion = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=1000,
            functions=FUNCTIONS,
            messages=[{"role": "system", "content": "The user is going to ask you a question about Louisiana law. Use an internet search to try and gather the answer."},
                        {"role": "user", "content": f"{prompt} + {extra_context_from_user} + Louisiana law"}],
            temperature=0,
            function_call={"name": "search"}
        )

        internet_response = internet_completion.choices[0].message

        if internet_response.get("function_call"):
            function_name = internet_response["function_call"]["name"]
            function_args = json.loads(internet_response["function_call"]["arguments"])
            function_to_call = getattr(llm_functions, function_name, None)

            if function_to_call is None:
                print("Function not found.")

            else:
                function_response = function_to_call(**function_args)
                    
            functions_used.append({"name": function_name, "arguments": function_args})

            # Summarize the function response using llm
            summary_completion = openai.ChatCompletion.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "system", "content": "The user is going to ask you a question about Louisiana law. Take the internet serp results and snippets and try to answer the question."},
                            {"role": "user", "content": f"User Question: {prompt} + {extra_context_from_user} \n Internet Serp Results: \n {function_response}]"}],                            
                temperature=0
                )

            summary_response = summary_completion.choices[0].message

        internet_reply = summary_response['content']

        print("GPT4's summary of internet search results: \n")
        print(internet_reply)

        messages = [{"role": "user", "content": f"A previous GPT4 instance answered the user's question ({prompt} + {extra_context_from_user}) as: \n\n {raw_response}. Another instance summarized the internet's answer to the question as: \n\n {internet_reply}. Write a query for a vector embeddings search that answers the question in the form of a hypothetical sentence that might appear in a Louisiana statute or case excerpt. Use complete sentences. Always use Louisiana terminology, i.e, 'parish' instead of 'county', 'immovable property' instead of 'real estate', etc."}]

        completion = openai.ChatCompletion.create(
            model=self.model,
            max_tokens=1000,
            messages=messages,
            temperature=0,
            functions=FUNCTIONS,
            function_call={"name": "vector_search_civil_code"},
        )

        response = completion.choices[0].message
        
        if response.get("function_call"):
            function_name = response["function_call"]["name"]
            function_args = json.loads(response["function_call"]["arguments"])
            function_to_call = getattr(vector_search_cc, function_name, None)

            if function_to_call is None:
                print("Function not found.")

            else:
                function_response = function_to_call(**function_args)
                
                # If the function called is vector_search_civil_code, assign the second element of the response to top_10_results
                if function_name == "vector_search_civil_code":
                    vector_output, top_10_results = function_response
                    
            functions_used.append({"name": function_name, "arguments": function_args})

        messages = [
            {"role": "system", "content": f"""You are Atticus, a legal answering service. Here are your instructions:

                        1. You will be given context from GPT4, the internet, and a semantic vector embeddings search of certain sources (e.g., the Civil Code, the Code of Procedure, the Criminal Code of Procedure etc.).
                        2. Use the context to answer the user's question: \n {prompt} + {extra_context_from_user} \n.
                        3. ALWAYS GIVE THE CIVIL CODE ARTICLES, STATUTES, and INTERNET RESULTS THE MOST WEIGHT WHEN ANSWERING THE QUESTION.
                        4. The raw GPT4 response may include hallucinations.
                        5. If the user asks you a question that is not about the law, do not answer.
                        6. If the answer does not appear to be in the vector embeddings results, you don't have to mention any specific code articles.
                        7. DO NOT ASSUME THAT JUST BECAUSE YOU ARE GIVEN A LAW FOR CONTEXT IT IS RELEVANT TO THE USER'S QUESTION.
                        8. Always use inline citations in the following format 'Explanatory sentence. La. C.C. art. 123' Don't put your citations in parentheses.
                        9. Do not cite to the raw GPT4 response, it often contains incorrect statements. Its only provided for reference.
                        10. Return your response in markdown formatting. Add the end of your response create a list of your citations.""" },

            {"role": "assistant", "content": f"Raw ChatPGT Response without help of vector embedding: \n\n {raw_response} \n\n Internet Results: \n\n {internet_reply}  \n\n Vector Results: \n\n {vector_output}"}
        ]

        print(f"\n\nFinal Prompt sent to GPT4: \n\n: {messages} \n\n")

        final_completion = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,
            )
        
        final_reply = final_completion.choices[0].message['content']

        print("Final reply: \n")
        print(final_reply)
        
        for func_info in functions_used:
            print('\033[92m' + "Function used: " + func_info["name"] + '\033[0m')
            print('\033[92m' + "Function arguments: " + str(func_info["arguments"]) + '\033[0m' + "\n")

        return final_reply, top_10_results
