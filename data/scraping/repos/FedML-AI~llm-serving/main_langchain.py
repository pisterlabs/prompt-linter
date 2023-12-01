from fedml.serving import FedMLInferenceRunner
from langchain import ConversationChain
from langchain.llms import HuggingFacePipeline

from src.base_chatbot import BaseChatbot


class Chatbot(BaseChatbot):
    def __init__(self):
        super().__init__()

        self.llm = HuggingFacePipeline(
            pipeline=self.model_args.get_hf_pipeline(task="text-generation", trust_remote_code=True)
        )
        self.llm_chain = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=self.model_args.verbose,
            input_key=self.model_args.prompt_info.input_key
        )

        # this is required for conversation memory buffer
        self.tokenizer = self.llm.pipeline.tokenizer

    def __call__(self, question: str) -> str:
        if len(question) == 0:
            raise ValueError("Received empty input.")

        response_text = self.llm_chain.predict(input=question)

        # chat history should be tracked by the API caller
        self.clear_history()
        return response_text


if __name__ == "__main__":
    chatbot = Chatbot()
    fedml_inference_runner = FedMLInferenceRunner(chatbot)
    fedml_inference_runner.run()
