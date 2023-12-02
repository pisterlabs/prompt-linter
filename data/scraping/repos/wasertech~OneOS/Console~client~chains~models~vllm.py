import json
from typing import Any, Coroutine, List, Optional, Iterable
from requests import Response, post, get

from langchain.llms.base import LLM
from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult

class vLLM(LLM):
    host: str = "localhost"
    port: str = "5085"
    use_beam_search: bool = False
    n: int = 1
    temperature: float = 0.0
    max_tokens: int = 16
    streaming: bool = False
    client_name: str = "Console"
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def is_nlp_server_up(self):
        try:
            r = get(f"http://{self.host}:{self.port}")
            if r.status_code == 200:
                return True
            raise Exception("Server is not up.")
        except Exception as e:
            return False


    def post_http_request(self,
                          prompt: str,
                          api_url: str,
                          stop: List[str] | None = None,
                          n: int = 1,
                          stream: bool = False,
                          temperature: float = 0.0,
                          max_tokens: int = 16,
                          use_beam_search: bool = False
                        ) -> Response:
        headers = {"User-Agent": self.client_name}
        pload = {
            "prompt": prompt,
            "n": n,
            "use_beam_search": use_beam_search,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "stop": stop
        }
        response = post(api_url, headers=headers, json=pload, stream=stream)
        return response

    def get_streaming_response(self, response: Response) -> Iterable[List[str]]:
        for chunk in response.iter_lines(chunk_size=8192,
                                        decode_unicode=False,
                                        delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode("utf-8"))
                output = data["text"]
                yield output

    def get_response(self, response: Response) -> List[str]:
        try:
            data = json.loads(response.content)
            output = data["text"]
            return output
        except json.decoder.JSONDecodeError as e:
            print("The server was not able to reply to the request.")
            print(f"Failed with error: {e}")
            print("This can happen if the server was running before hibernation and is now unable to access the GPU.")
            print("To fix this issue, restart the system.")
            print("I'm sure we all are sorry for the inconvenience.")
            return [None]

    async def get_streaming_response_async(self, response: Response):
        _response = ""
        for resp in self.get_streaming_response(response):
            _response = resp[0]
        return _response

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        n = 1
        api_url = f"http://{self.host}:{self.port}/generate"
        stream = False
        use_beam_search = False
        response = self.post_http_request(prompt, api_url, stop, n, stream, self.temperature, self.max_tokens, use_beam_search)
        
        return self.get_response(response)[0]

    async def _acall(self, prompt: str, stop: List[str] | None = None, run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any) -> Coroutine[Any, Any, str]:
        n = self.n
        api_url = f"http://{self.host}:{self.port}/generate"
        stream = False
        response = self.post_http_request(prompt, api_url, stop, n, stream, self.temperature, self.max_tokens, self.use_beam_search)
        
        return await self.get_streaming_response_async(response)
    
    def _generate(self, prompts: List[str], stop: List[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> LLMResult:
        if self.streaming:
            n = 1
            api_url = f"http://{self.host}:{self.port}/generate"
            stream = False
            use_beam_search = False
            response = self.post_http_request(prompts[0], api_url, stop, n, stream, self.temperature, self.max_tokens, use_beam_search)
        
            last_line = ""
            for stream_resp in self.get_streaming_response(response):
                if run_manager:
                    for i, line in enumerate(stream_resp):
                        last_token = line.replace(last_line, "")
                        last_line = line
                        run_manager.on_llm_new_token(
                                last_token,
                                verbose=self.verbose,
                            )
            return LLMResult(generations=[[Generation(text=last_line)]], llm_output=json.loads(response.content))     
        else:
            return LLMResult(generations=[[Generation(text=self._call(p))] for p in prompts], llm_output={})
    
    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        if self.streaming:
            n = 1
            api_url = f"http://{self.host}:{self.port}/generate"
            stream = True
            use_beam_search = False
            response = self.post_http_request(prompts[0], api_url, stop, n, stream, self.temperature, self.max_tokens, use_beam_search)
        
            last_line = ""
            for stream_resp in self.get_streaming_response(response):
                if run_manager:
                    for i, line in enumerate(stream_resp):
                        last_token = line.replace(last_line, "")
                        last_line = line
                        await run_manager.on_llm_new_token(
                                last_token,
                                verbose=self.verbose,
                            )
            return LLMResult(generations=[[Generation(text=last_line)]], llm_output=json.loads(response.content))
        else:
            return LLMResult(generations=[[Generation(text=self._call(p))] for p in prompts], llm_output={})
    
    # @property
    # def _identifying_params(self) -> Mapping[str, Any]:
    #     """Get the identifying parameters."""
    #     return {"n": self.n}