from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Union
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import requests
from utilities.helper import LLMHelper
from utilities.openAI_helper import openAI_helper
from utilities.customprompt import mod_evaluate_instructions
from datetime import datetime


class bing():
    def __init__(self ):

        self.sites : str = None
        # Set up LLM Helper
        self.llm_helpers = LLMHelper()
        self.openai_helpers = openAI_helper()
        self.k: int = 10
        self.evaluate_step : True
        self.verbose : True
        self.history : str = ""
    

#    def useBingSerch(self, question, history):
#        if LLMHelper.use_bing is True:
#            self.bing_search = bing_search(k=10)
#            agent_tools.append(Tool(name="Online Search", func=self.agent_bing_search, description='useful for when you need to answer questions about current events from the internet'),)
#        else:
#            self.bing_search = None
    

    def bing_search(self, query):
        if self.llm_helpers.use_bing  :
            
            user_message = {"role":"user", "content":query}
          #  response = redis_helpers.redis_get(self.redis_conn, query, 'bing_search_response', verbose = self.verbose)
            response = '\n\n'.join(self.run(query))
            response = self.evaluate(query, response,True)
            #    redis_helpers.redis_set(self.redis_conn, query, 'bing_search_response', response, CONVERSATION_TTL_SECS, verbose = True)
            
            return  user_message, {"role": "assistant", "content": response}
        else:
            return ''

    def run(self, query: str) -> str:
        """Run query through BingSearch and parse result."""

        if self.sites is None:
            self.sites = ""
            arr = self.llm_helpers.list_of_comma_separated_urls.split(",")
            if len(arr) > 0:
                sites_v = ["site:"+site.strip() for site in arr]
                sites_v = " OR ".join(sites_v)
                sites_v = f"({sites_v})"
                self.sites = sites_v

            # print("Sites", self.sites)

        snippets = []
        try:
            results = self._bing_search_results(f"{self.sites} {query}", count=self.k)
        except Exception as e:
            print("Error in bing search", e)
            return snippets

        if len(results) == 0:
            return "No good Bing Search Result was found"
        for result in results:
            snippets.append('['+result["url"] + '] ' + result["snippet"])
        
        return snippets

        
    def _bing_search_results(self, search_term: str, count: int) -> List[dict]:
        headers = {"Ocp-Apim-Subscription-Key": self.llm_helpers.bing_subscription_key}
        params = {
            "q": search_term,
            "count": count,
            "textDecorations": False,
            "textFormat": "Raw",
            "safeSearch": "Strict",
            "setLang":"ko"
        }

        response = requests.get(
            self.llm_helpers.bing_search_url, headers=headers, params=params  # type: ignore
        )
        response.raise_for_status()
        search_results = response.json()

        return search_results["webPages"]["value"]




    def evaluate(self, query, context, verbose):
        comp_model = self.llm_helpers.comp_model
        response = ""
        completion_enc = openAI_helper.get_encoder(comp_model)
        max_comp_model_tokens = openAI_helper.get_model_max_tokens(comp_model)

        query_len = len(completion_enc.encode(query))
        empty_prompt = len(completion_enc.encode(mod_evaluate_instructions.format(context = "", question = "", todays_time="", history="")))
        allowance = max_comp_model_tokens - empty_prompt - self.llm_helpers.max_output_token - query_len
        if verbose: print("Evaluate Call Tokens:", len(completion_enc.encode(context)), allowance)
        context = completion_enc.decode(completion_enc.encode(context)[:allowance]) 
        prompt = mod_evaluate_instructions.format(context = context, question = query, todays_time=self.get_date(""), history=self.history)
        if verbose: print("Evaluate OAI Call")
        response = openAI_helper.contact_openai(prompt, comp_model, self.llm_helpers.max_output_token, verbose=verbose)
       # response = response.replace("<|im_end|>", '')
        self.history += query + ' '
   
     #   response = response.replace("<|im_end|>", '')
        return response 
            
            
    def get_date(self, query):
        return f"Today's date and time {datetime.now().strftime('%A %B %d, %Y %H:%M:%S')}. You can use this date to derive the day and date for any time-related questions, such as this afternoon, this evening, today, tomorrow, this weekend or next week."


