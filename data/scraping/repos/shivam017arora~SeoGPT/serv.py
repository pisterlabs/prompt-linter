from __future__ import annotations
from flask import Flask, request, make_response
import json

from typing import Any, Dict, List, Optional

from pydantic import Extra

from dotenv import load_dotenv
load_dotenv('.env')

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

import os
from dotenv import dotenv_values

config = {
    **dotenv_values(".env.shared"),  # load shared development variables
    **dotenv_values(".env.secret"),  # load sensitive variables
    **os.environ,  # override loaded values with environment variables
}

RAPID_API_KEY = config['RAPID_API_KEY'];

from flask_cors import CORS
app = Flask(__name__)
cors = CORS(app, origins='http://localhost:3000')

class MyCustomChain(Chain):
    """
    An example of a custom chain.
    """

    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: BaseLanguageModel
    output_key: str = "text"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = self.llm.generate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            run_manager.on_text(f"Chain Response: {response.generations[0][0].text}")

        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Your custom chain logic goes here
        # This is just an example that mimics LLMChain
        prompt_value = self.prompt.format_prompt(**inputs)

        # Whenever you call a language model, or another chain, you should pass
        # a callback manager to it. This allows the inner run to be tracked by
        # any callbacks that are registered on the outer run.
        # You can always obtain a callback manager for this by calling
        # `run_manager.get_child()` as shown below.
        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        # If you want to log something about this run, you can do so by calling
        # methods on the `run_manager`, as shown below. This will trigger any
        # callbacks that are registered for that event.
        if run_manager:
            await run_manager.on_text("Chain Response: ", response.generations[0][0].text)

        return {self.output_key: response.generations[0][0].text}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"
    
def get_topic(sentence):
    prompt = """ 

    You are SeoGPT. 
    Your task is to give the best topic to search to get the most SEO rich content. 

    Your Job:
    You will be given a sentence. 

    ____SENTENCE__START___
    {sentence}
    ____SENTENCE___END___

    Give only 2 words responses which describes this sentence best. Make sure your response is relevant to the sentence and is max 2 words. 
    Your job is to provide the best topic to get the most SEO rich content, don't forget that. 

    GO!
    """
    from langchain.callbacks.stdout import StdOutCallbackHandler
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.prompts.prompt import PromptTemplate


    chain = MyCustomChain(
        prompt=PromptTemplate.from_template(prompt),
        llm=ChatOpenAI(),
    )

    return chain.run({"sentence": sentence}, callbacks=[StdOutCallbackHandler()])
    
def get_top_keywords(sentence):
    import requests

    url = "https://twinword-keyword-suggestion-v1.p.rapidapi.com/suggest/"
    print("Topic: ", sentence)

    querystring = {"phrase":sentence,"lang":"en","loc":"US"}

    headers = {
        "X-RapidAPI-Key": RAPID_API_KEY, # RAPID_API_KEY
        "X-RapidAPI-Host": "twinword-keyword-suggestion-v1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.json())

    def sorting_key(keyword):
        volume = int(keyword['search volume'])
        cpc = float(keyword['cpc'])
        competition = float(keyword['paid competition'])
        return volume * cpc * competition

    # Convert the keyword dictionary into a list of dictionaries
    keywords = response.json()['keywords']
    keyword_list = [{'keyword': keyword, **metrics} for keyword, metrics in keywords.items()]

    # Sort the keyword list using the custom sorting key
    sorted_keywords = sorted(keyword_list, key=sorting_key, reverse=True)

    # Get the top 10 keywords
    top_10_keywords = sorted_keywords[:10]

    return_data = list()
    # Print the top 10 keywords
    for keyword in top_10_keywords:
        return_data.append((keyword['keyword'], keyword['search volume']))
        
    return return_data

def get_sugguestions(topic, query):
    promptForSEOSuggesstions = """ 
    You are SeoGPT. 
    You task is to give suggestion to change certain words in the sentence based on the keywords. Your suggestions can be the keywords or something you can think of. Your suggestions should be relevant to the sentence and should increase the likelihood of getting picked up by search engines. 

    Keywords and Search Count:
    {keywords}

    Sentence:
    {sentence}


    Your response should be in the following format in JSON:
    "
    <word in sentence> : <string>,
    <word in sentence> : <string>
    "

    Your changes should be gramatically correct and fit the sentence. Make sure the response is only in JSON.

    """
    from langchain.callbacks.stdout import StdOutCallbackHandler
    from langchain.chat_models.openai import ChatOpenAI
    from langchain.prompts.prompt import PromptTemplate


    chain = MyCustomChain(
        prompt=PromptTemplate.from_template(promptForSEOSuggesstions),
        llm=ChatOpenAI(),
    )

    keywords_list = get_top_keywords(topic)
    keywordsPrompt = ', '.join([str(t) for t in keywords_list])

    print("Keywords: \n: ", keywordsPrompt)

    res = chain.run({"sentence": query, "keywords": keywordsPrompt}, callbacks=[StdOutCallbackHandler()])

    # Clean up the string by replacing single quotes with double quotes and removing unnecessary characters
    clean_string = res.replace("'", '"').replace("\n", "")

    # Parse the cleaned string as JSON to convert it into a list of dictionaries
    keyword_list = json.loads(clean_string)

    dictionary = eval(str(keyword_list))

    # Print the resulting dictionary
    return (dictionary)


@app.route('/')
def seo():
    request_query = request.args.get('query')

    topic = get_topic(request_query)
    suggestions = get_sugguestions(topic, request_query)

    if request_query:
        response = make_response(suggestions, 200)
    else:
        response = make_response('Bad Request', 400)

    return response

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8912)
