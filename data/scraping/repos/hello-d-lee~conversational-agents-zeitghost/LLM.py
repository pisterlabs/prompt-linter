from typing import List, Optional
from zeitghost.vertex.Helpers import VertexModels, stop_string_at_first_occurrence
from langchain.llms.base import LLM
from vertexai.preview.language_models import TextGenerationModel


class VertexLLM(LLM):
    """
    A class to Vertex LLM model that fits in the langchain framework
    this extends the langchain.llms.base.LLM class
    """
    model: TextGenerationModel
    predict_kwargs: dict
    model_source: str
    stop: Optional[List[str]]
    strip: bool
    strip_chars: List[str]

    def __init__(self
                 , stop: Optional[List[str]]
                 , strip: bool = False
                 , strip_chars: List[str] = ['{','}','\n']
                 , model_source=VertexModels.MODEL_TEXT_BISON_001.value
                 , **predict_kwargs
                 ):
        """
        :param model_source: str
            Name of LLM model to interact with
        :param endpoint: str
            Endpoint information for HTTP calls
        :param project: str
            Google Cloud Project ID
        :param location: str
            Google Cloud Location
        """
        super().__init__(model=TextGenerationModel.from_pretrained(model_source)
                         , strip=strip
                         , strip_chars=strip_chars
                         , predict_kwargs=predict_kwargs
                         , model_source=VertexModels.MODEL_TEXT_BISON_001.value
                         )
        self.model = TextGenerationModel.from_pretrained(model_source)
        self.stop = stop
        self.model_source = model_source
        self.predict_kwargs = predict_kwargs
        self.strip = strip
        self.strip_chars = strip_chars

    @property
    def _llm_type(self):
        return 'vertex'

    @property
    def _identifying_params(self):
        return {}

    def _trim_output(self, raw_results: str) -> str:
        '''
        utility function to strip out brackets and other non useful info
        '''
        for char in self.strip_chars:
            raw_results = raw_results.replace(char, '')
        return raw_results
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Wrapper around predict.
        Has special handling for SQL response formatting.

        :param prompt:
        :return: str
        """
        stop = self.stop
        prompt = str(prompt)
        prompt = prompt[:7999] #trimming the first chars to avoid issue
        result = str(self.model.predict(prompt, **self.predict_kwargs))
        if stop is not None:
            result = str(stop_string_at_first_occurrence(result, self.stop)) #apply stopwords
        if self.strip:
            return str(self._trim_output(result))
        else:
            return str(result)

    def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = str(self.model.predict(prompt, **self.predict_kwargs))
        stop = self.stop
        if stop:
            result = str(stop_string_at_first_occurrence(result, self.stop)) #apply stopwords
        return str(result)
    