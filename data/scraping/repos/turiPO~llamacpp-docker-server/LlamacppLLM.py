import requests
from pydantic import PrivateAttr

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class LlamacppLLM(LLM):
    # server_address: str
    _server_address: str = PrivateAttr()


    def __init__(self, server_address: str = "localhost:8200"):
        self._server_address = server_address
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "llamacpp"

    def _call(
        self,
        prompt: str,
        # stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text completion using the Llamacpp server.

        Args:
            prompt (str): Text prompt for the completion.
            run_manager (Optional[CallbackManagerForLLMRun]): Callback manager for LLM run. Not used in this implementation.
            **kwargs (Any): Additional server parameters for the completion. The available options are:
                - stop (Optional[List[str]]): List of stop words or phrases to indicate completion termination. These words will not be included in the completion, so make sure to add them to the prompt for the next iteration (default: []).
                - n_predict (int): Number of tokens to predict in the completion.
                - stream (bool): Boolean indicating whether to stream the completion response.
                - top_k (int): Number of top-k tokens to consider in the completion.
                - top_p (float): Top-p (nucleus) sampling threshold for the completion.
                - tfs_z (float): Temperature scaling factor for the top frequent sampling in the completion.
                - typical_p (float): Typicality parameter for the completion.
                - repeat_last_n (int): Number of tokens to repeat at the end of the prompt in the completion.
                - temperature (float): Temperature for sampling tokens during the completion.
                - repeat_penalty (float): Penalty applied for repeating tokens in the completion.
                - presence_penalty (float): Penalty applied for token presence in the completion.
                - frequency_penalty (float): Penalty applied based on token frequency in the completion.
                - mirostat (bool): Boolean indicating whether to use Mirostat sampling in the completion.
                - mirostat_tau (float): Temperature parameter for Mirostat sampling in the completion.
                - mirostat_eta (float): Eta parameter for Mirostat sampling in the completion.
                - penalize_nl (bool): Boolean indicating whether to penalize newline characters in the completion.
                - n_keep (int): Number of tokens to keep fixed in the completion.
                - seed (int): Random seed for the completion.

        Returns:
            str: Generated text completion.

        Raises:
            ValueError: If `stop` argument is provided.
            RuntimeError: If the LLM server request fails.
        """
        
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            **kwargs
        }
        
        # Make the API request to the llamacpp server
        url = f"http://{self._server_address}/completion"
        response = requests.post(url, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Extract the generated text from the response
            generated_text = result["content"]
            
            return generated_text
        else:
            raise RuntimeError("LLM server request failed with status code: " + str(response.status_code))

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

if __name__ == "__main__":
    # Create an instance of LlamacppLLM
    server_address = "localhost:8200"  # Replace with the actual server address
    llm = LlamacppLLM(server_address)

    # Example usage
    prompt = "Hello Mr."
    generated_text = llm(prompt, stop=['\n'], n_predict=3)
    print(generated_text)
