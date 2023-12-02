import os
import openai
import re
import json
from utils.krippendorff_alpha import krippendorff 
from sklearn.metrics import cohen_kappa_score
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import csv

class RaLLM():
    def __init__(self, api_key):
        """
        Initialize the RaLLM class with the OpenAI API key.

        Parameter:
        - api_key (str): Your OpenAI API key.
        """
        openai.api_key = api_key

    def codebook2prompt(codebook, format, num_of_examples,language = 'en', has_context=False):
        """
        This function converts a codebook in tabular format to a natural language prompt.

        Parameters:
        - codebook (pandas.DataFrame): The codebook as a pandas DataFrame containing the codes, descriptions, examples, and contexts.
        - format (str): The format of prompt to generate, either 'example' or 'codebook'.
        - num_of_examples (int): The number of examples to include in the generated prompt.
        - language (str, optional, default='en'): The language of the generated prompt, either 'en' (English) or 'fr' (French).
        - has_context (bool, optional, default=False): Indicates whether the examples in the codebook have context information.

        Returns:
        - tuple: A tuple containing the generated prompt and the list of codes.
            - prompt (str): The generated natural language prompt.
            - code_set (list): A list of codes from the codebook.
        """
        code_set = []
        instruction = ''
        examples = ''
        if language =='fr':
            context_text = " dans le contexte de "
        elif language =='ch':
            context_text = " 在以下情景下 "
        else:
            context_text = " in the context of "

        for n in range(num_of_examples):
            if has_context:
                examples += codebook['example_'+str(n+1)]+ context_text + codebook['context_'+str(n+1)]+ '; '
            else:
                examples += codebook['example_'+str(n+1)]+ '; '

        codebook['examples'] = examples
        if format=='example':
            if language =='fr':
                for index, row in codebook.iterrows():
                    instruction += str(row['examples'])+ "est un exemple de " + str(row['code'])+ ",car "+ row['description']+ ";"
                    code_set.append(str(row['code']))
            elif language =='ch':
                for index, row in codebook.iterrows():
                    instruction += "这段文本的编码是" + str(row['examples'])+ "" + str(row['code'])+ ",因为"+ row['description']+ ";"
                    code_set.append(str(row['code']))
            else:
                for index, row in codebook.iterrows():
                    instruction += str(row['examples'])+ "is an example of " + str(row['code'])+ ",becasue "+ row['description']+ ";"
                    code_set.append(str(row['code']))
        else:
            if language =='fr':
                for index, row in codebook.iterrows():
                    instruction += "CODE: " + str(row['code'])+ "; LA DESCRIPTION: "+ row['description']+ "; EXAMPLES: "+ row['examples']+ ";"
                    code_set.append(str(row['code']))
            elif language =='ch':
                for index, row in codebook.iterrows():
                    instruction += "\n编码: "+ str(row['code'])+ " 描述: " + str(row['description'])+ "; 例子: "+ row['examples']+ ";"
                    code_set.append(str(row['code']))
            else:
                for index, row in codebook.iterrows():
                    instruction += "CODE: " + str(row['code'])+ "; DESCRIPTION: "+ row['description']+ "; EXAMPLES: "+ row['examples']
                    code_set.append(str(row['code']))
        return instruction, code_set

    def prompt_writer(data, context, codebook, code_set, meta_prompt, none_code, language='en',cot = 0):
        """
        This function generates a complete natural language prompt for coding based on the given parameters.

        Parameters:
        - data (str): The input sentence to be classified.
        - context (str): The context information for the coding task.
        - codebook (str): The codebook description in natural language.
        - code_set (list): A list of codes from the codebook.
        - identity_modifier (str): A modifier to specify the identity of the person performing the coding task.
        - context_description (str): A brief description of the context for the coding task.
        - none_code (bool): Indicates whether to include 'none' as a coding option if none of the codes apply.
        - language (str, optional, default='en'): The language of the generated prompt, either 'en' (English) or 'fr' (French).

        Returns:
        - str: The generated complete natural language prompt.
        """

        candidates = ''
        for code in code_set:
            if candidates :
                candidates += ' or '+str(code)
            else:
                candidates += str(code)

        if language == 'fr':
            instruction = """
                Décidez si une [texte] est 
                """+ candidates + '. Basé sur le livre de codes suivant qui comprend la description de chaque code et quelques exemples. Notez que le code est uniquement pour le [texte] et non le contexte, mais le contexte doit être pris en compte lors de la prise de décision.'
            if cot:
                last_instruction = 'Choisissez parmi les candidats suivants : ' + candidates + " Avant de prendre une décision, réfléchissons étape par étape avec le contexte et expliquons."
            else:
                last_instruction = 'Choisissez parmi les candidats suivants : ' + candidates + " \n"
            codebook_label = 'Livre de codes: '
            sentence_label = 'Texte: ['
            context_label = 'Contexte: '
        elif language == 'ch':
            instruction = """
                根据下列结构性编码本对以下[文本]编码, 注意编码仅针对[]内文本而不是上下文语境, 但分析时要结合上下文语境。 
                """
            if cot:
                last_instruction = '从下列编码中选择 ' + candidates + " 在做出决定之前，根据编码步骤，让我们结合语境一步一步地思考并解释原因。\n"
            else:
                last_instruction = '从下列编码中选择 ' + candidates + " \n"
            codebook_label = '编码本: '
            context_label = '上文语境: '
            sentence_label = '文本: ['
        else:
            instruction = """
                Classify the [text] into 
                """+ candidates + '. Based on the following codebook that includes the description of each code and a few examples. Note that the code is only for the [text] not the context, but the context should be considered when making the decision.'
            if cot:
                last_instruction = 'Choose from the following candidates: ' + candidates + " Before making a decision, let's think through step by step with the context and explain."
            else:
                last_instruction = 'Choose from the following candidates: ' + candidates + " \n"
            codebook_label = 'Codebook: '
            sentence_label = 'Text: ['
            context_label = 'Context: '
        
        if none_code:
            if language == 'fr':
                last_instruction += " ou 'NA' si aucun de ces codes ne s'applique."
            elif language == 'ch':
                last_instruction += " 或者 'NA' 如果无任何标签匹配."
            else:
                last_instruction += " or 'NA' if none of these codes applies."

        complete_prompt = "\n".join([meta_prompt,instruction, codebook_label+codebook+'\n', context_label+context+'\n', sentence_label+data+']\n', last_instruction])
        return complete_prompt

    #this decorator is used to retry if the rate limits are exceeded
    @retry(
        reraise=True,
        stop=stop_after_attempt(1000),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=(retry_if_exception_type(openai.error.Timeout)
            | retry_if_exception_type(openai.error.APIError)
            | retry_if_exception_type(openai.error.APIConnectionError)
            | retry_if_exception_type(openai.error.RateLimitError)),
    )
    def coder(complete_prompt, engine="text-davinci-003", voter = 1):
        """
        This function uses the OpenAI API to code a given sentence based on the provided complete_prompt.

        Parameters:
        - complete_prompt (str): The generated natural language prompt to be sent to the OpenAI API.
        - engine (str, optional, default="text-davinci-003"): The OpenAI engine to be used for coding. Can be "text-davinci-003" or "gpt-3.5-turbo".

        Returns:
        - str: The coding result from the OpenAI API.
        """
        # See API document at https://beta.openai.com/docs/api-reference/completions/create
        # max tokens: 100 is enough for single question. 
        # temperature: 0 for greedy (argmax). 
        if engine=='text-davinci-003':
            response = openai.Completion.create(engine=engine, prompt=complete_prompt, suffix=None, max_tokens=500, temperature=0.0)
        else:
            response = openai.ChatCompletion.create(
            model = engine, 
                messages=[{"role": "user", "content": complete_prompt}], temperature=0.0, n = voter)
        return response
    
    def llm_translator(data, language1, language2, engine="text-davinci-003"):
        """
        This function uses the OpenAI API to translate a given sentence from one language to another.

        Parameters:
        - data (str): The input sentence to be translated.
        - language1 (str): The source language of the input sentence.
        - language2 (str): The target language to translate the input sentence to.
        - engine (str, optional, default="text-davinci-003"): The OpenAI engine to be used for translation. Defaults to "text-davinci-003".

        Returns:
        - str: The translated sentence.
        """
        instruction = """\
                translate the following sentence from """+ language1 + """ to """ + language2 + """.
                """

        # See API document at https://beta.openai.com/docs/api-reference/completions/create
        # max tokens: 100 is enough for single question. 
        # temperature: 0 for greedy (argmax). 
        response = openai.Completion.create(engine=engine, prompt="\n".join([instruction,'Sentence: '+data]), suffix=None, max_tokens=100, temperature=0.0)
        response = response["choices"][0]["text"].strip()
        return response

    def scale_taker(prompt, item, engine="gpt-3.5-turbo"):
        """
        This function is used to take the scale for a single item using the specified language model engine.
        
        Parameters:
        - prompt (str): The initial prompt for the language model.
        - item (str): The item for which the scale is to be determined.
        - engine (str, optional): The language model engine to be used. Default is "gpt-3.5-turbo".

        Returns:
        - str: The scale value for the provided item.
        """
        if engine=='text-davinci-003':
            response = openai.Completion.create(engine=engine, prompt="\n".join([prompt, item]), suffix=None, max_tokens=100, temperature=0.0)
            response = response["choices"][0]["text"].strip()
        else:
            completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": "\n".join([prompt, item])}], temperature=0.0)
            response = completion["choices"][0]['message']["content"].strip() 
        result = re.search(r'\d+', response).group()   
        return result
    
    def scale_taker_seq(prompt, item, engine="gpt-3.5-turbo"):
        """
        This function is used to take the scale for each item in a single conversation session with mutiple turns.
        It takes the prompt, and then adds the item one by one, asking the model to give the scale for each item one by one.

        Parameters:
        - prompt (str): The initial prompt for the language model.
        - item (list): A list of items for which the scale is to be determined.
        - engine (str, optional): The language model engine to be used. Default is "gpt-3.5-turbo". Only works for "gpt-3.5-turbo" and above ("gpt-4").

        Returns:
        - list: A list of scale values for the provided items.
        """
        response_list = []
        context = []
        context.append({'role': 'user', 'content': prompt})
        for i in item:
            context.append({'role': 'user', 'content': i})
            completion = openai.ChatCompletion.create(
            model = engine, 
                    messages=context
            ,temperature=0.0)
            response = completion["choices"][0]['message']["content"].strip()
            result = re.search(r'\d+', response).group()               
            context.append({'role': 'assistant', 'content': result})
            response_list.append(result)
        return response_list

    def item_constructor(items, num):
        """
        This function constructs a list of items with a specified number of elements.

        Parameters:
        - items (pandas DataFrame): A DataFrame containing the items to be processed.
        - num (int): The number of items to include in the resulting list.

        Returns:
        - list: A list of items with a specified number of elements, each ending with the string " [BLANK]".
        """

        item_list = []
        num_items = 0
        for index, row in items.iterrows():
            item = row['ITEM'] + " [BLANK]"
            item_list.append(item)
            num_items += 1
            if num_items >= num:
                break
        return item_list

    def cohens_kappa_measure(code, result):
        """
        This function is used to calculate the Cohen's Kappa measure.

        Parameters:
        - code (list): A list of coded items assigned by the first coder.
        - result (list): A list of coded items assigned by the second coder.

        Returns:
        - float: The Cohen's Kappa measure, a value between -1 and 1 representing the level of agreement between the two coders.
        """
        return cohen_kappa_score(code, result)
    
    def krippendorff_alpha_measure(code, result, code_set):
        """
        This function is used to calculate the Krippendorff's alpha measure.

        Parameters:
        - code (list): A list of coded items assigned by the first coder.
        - result (list): A list of coded items assigned by the second coder.
        - codeset (list): A list of codes from the codebook.

        Returns:
        - float: The Krippendorff's alpha measure, a value between -1 and 1 representing the level of agreement between the two coders.
        """
        code_converted = [code_set.index(i) for i in code]
        result_converted = [code_set.index(i) for i in result]
        return krippendorff.krippendorff([code_converted, result_converted], missing_items='')
    
    def code_clean(results, code_set):
        """
        Cleans the codes returned by the model to ensure that only valid codes from the code set are used. Non-existent codes are remained unchanged.

        Parameters:
            results (list): A list of results (codes) returned by the model.
            code_set (list): A list of valid codes.

        Returns:
            list: A cleaned list of results containing only valid codes from the code set.
        """
        for i in range(len(results)):
            for code in code_set:
                if code in results[i]:
                    results[i] = code
                    break
        return results