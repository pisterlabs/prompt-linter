# Wrap our custom fine-tuned model into a Langchain LLM class, and make it available for chainlit factory.

import cohere
from config import config
from langchain.llms.base import LLM

co = cohere.Client(config.COHERE_API_KEY)    # This is COHERE_API_KEY


class CodeGenAlpha(LLM):
    model: str = config.CODE_GEN_MODEL_ID   # The custom model we used.
    
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self,prompt: str,stop = None,run_manager = None,) -> str:
        """ This is where the main logic of the model we used. Since we used a custom model, we needed to implement it. """
        
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = co.generate(model=self.model, prompt=f'{prompt}')

        return f' ```python \n {response.generations[0].text} \n ``` '  # Mostly code, so output should be in this form.

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""
        
        return {"model_type": f'COHERE_CUSTOM_MODEL_ID-<*****{self.model[-5:-1]}>'}


""" Now, this thing can be used as a custom LLM. Use it in the LLM Chain."""
