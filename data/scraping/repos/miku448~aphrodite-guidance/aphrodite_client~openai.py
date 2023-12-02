import guidance
import os
import time
import collections

class AphroditeOpenAIClient(guidance.llms.OpenAI):
    llm_name: str = "aphrodite_openai"

    def __init__(self, model=None, caching=True, max_retries=5, max_calls_per_min=60,
                 api_key=None, api_type="open_ai", api_base=None, api_version=None, deployment_id=None,
                 temperature=0.0, chat_mode="auto", organization=None, rest_call=False,
                 allowed_special_tokens=None,
                 token=None, endpoint=None, encoding_name=None, tokenizer=None):

        # map old param values
        # TODO: add deprecated warnings after some time
        if token is not None:    
            if api_key is None:
                api_key = token
        if endpoint is not None:
            if api_base is None:
                api_base = endpoint

        # fill in default model value
        if model is None:
            model = os.environ.get("OPENAI_MODEL", None)
        if model is None:
            try:
                with open(os.path.expanduser('~/.openai_model'), 'r') as file:
                    model = file.read().replace('\n', '')
            except:
                pass

        # fill in default deployment_id value
        if deployment_id is None:
            deployment_id = os.environ.get("OPENAI_DEPLOYMENT_ID", None)

        # auto detect chat completion mode
        chat_mode = False
        
        # fill special tokens with tokenizer
        allowed_special_tokens = tokenizer.special_tokens_map.values()
        
        # fill in default API key value
        if api_key is None: # get from environment variable
            api_key = os.environ.get("OPENAI_API_KEY", 'EMPTY')
        if api_key is not None and os.path.exists(os.path.expanduser(api_key)): # get from file
            with open(os.path.expanduser(api_key), 'r') as file:
                api_key = file.read().replace('\n', '')
        if api_key is None: # get from default file location
            try:
                with open(os.path.expanduser('~/.openai_api_key'), 'r') as file:
                    api_key = file.read().replace('\n', '')
            except:
                pass
        if organization is None:
            organization = os.environ.get("OPENAI_ORGANIZATION", None)
        # fill in default endpoint value
        if api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", None) or os.environ.get("OPENAI_ENDPOINT", None) # ENDPOINT is deprecated

        self._tokenizer = tokenizer
        self.chat_mode = chat_mode
        
        self.allowed_special_tokens = allowed_special_tokens
        self.model_name = model
        self.deployment_id = deployment_id
        self.caching = caching
        self.max_retries = max_retries
        self.max_calls_per_min = max_calls_per_min
        if isinstance(api_key, str):
            api_key = api_key.replace("Bearer ", "")
        self.api_key = api_key
        self.api_type = api_type
        self.api_base = api_base
        self.api_version = api_version
        self.current_time = time.time()
        self.call_history = collections.deque()
        self.temperature = temperature
        self.organization = organization
        self.rest_call = rest_call
        self.endpoint = endpoint

        if not self.rest_call:
            self.caller = self._library_call
        else:
            self.caller = self._rest_call
            self._rest_headers = {
                "Content-Type": "application/json"
            }

    # TODO: ADD SUPPORT FOR ROLES?
    # def role_start(self, role_name, **kwargs):
    #     assert self.chat_mode, "role_start() can only be used in chat mode"
    #     return "<|im_start|>"+role_name+"".join([f' {k}="{v}"' for k,v in kwargs.items()])+"\n"
    
    # def role_end(self, role=None):
    #     assert self.chat_mode, "role_end() can only be used in chat mode"
    #     return "<|im_end|>"
    
    def end_of_text(self):
        return self._tokenizer.eos_token

    def token_to_id(self, token):
        return self.encode(token)[-1]
    
    def encode(self, string):
      return self._tokenizer.encode(string)
