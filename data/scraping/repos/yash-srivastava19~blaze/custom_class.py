# Wrap our custom fine-tuned model into a Langchain LLM class, and make it available for chainlit factory.
import chainlit as cl
import cohere
from langchain.llms.base import LLM
from metaphor_python import Metaphor

### Caution : This should not kept in open, and I'll fix this in upcoming commits.

metaphor = Metaphor("56d9ca28-e84b-43ce-8f68-75e1a8bb4dd3")   ## This is metaphor API Key
co = cohere.Client('P21OYiQsKcOVS9XeQsacEhMVodPbbbYccpX2XWsz')    # This is COHERE_API_KEY

class SourcesString:
    """[ Custom Class to beautify methaphor.search.result(currently, only top 3 results are shown)] """
    def __init__(self, results_list) -> None:
        self.res_list = results_list

    def get_parsed_string(self):

        return "### Additional Sources:\n"+"\n".join(f"{i+1}. [{self.res_list[i].title}]({self.res_list[i].url})" for i in range(len(self.res_list)))


class CustomLLM(LLM):
    """[Wrapper for Cohere Chat Model, which is then appended with Metaphor Search Results to mimic RAG. ] """
  
    @property
    def _llm_type(self) -> str:
        return "custom"

    def  _call(self,prompt: str,stop = None,run_manager = None,) -> str:
        """ Cohere Chat Model is not supported as of now, so we create a custom wrapper it. """

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        response = co.chat(f'{prompt}', model = 'command')

        results = metaphor.search(prompt, use_autoprompt=True, num_results=3)
        # print(results)
        
        print(response)

        sources_text = SourcesString(results.results).get_parsed_string().strip("```")    # pretty hacky, but we need to find a good fix for this.
        print(sources_text)

        return response.text + "\n\n\n" + sources_text

    @property
    def _identifying_params(self) :
        """Get the identifying parameters."""

        return {"model_type": f'COHERE_CHAT_MODEL'}
