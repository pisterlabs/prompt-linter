# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 Emanuel Erben
# SPDX-FileCopyrightText: 2023 Felix Nützel
# SPDX-FileCopyrightText: 2023 Jesse Palarus
# SPDX-FileCopyrightText: 2023 Amela Pucic

from time import time

from huggingface_hub import hf_hub_download
from langchain import LlamaCpp, PromptTemplate

from QAChat.Common.deepL_translator import DeepLTranslator, Result
from QAChat.QA_Bot.stream_LLM_callback_handler import StreamLLMCallbackHandler

from typing import List

from QAChat.VectorDB.vector_store import VectorStore

print("Init Lock")
from threading import Lock
critical_function_lock = Lock()


class QABot:
    def __init__(
            self,
            model=None,
            translator=None,
            repo_id="TheBloke/WizardLM-13B-V1-1-SuperHOT-8K-GGML",
            filename="wizardlm-13b-v1.1-superhot-8k.ggmlv3.q4_1.bin",
    ):
        self.answer = None
        self.context = None
        self.vector_store = VectorStore()

        self.model = model
        if model is None:
            self.model = self.get_llama_model(repo_id=repo_id, filename=filename)
        self.translator = translator
        if translator is None:
            self.translator = DeepLTranslator()

    def get_llama_model(
            self,
            repo_id,
            filename,
            n_ctx=2048,
            max_tokens=512,
    ):
        path = hf_hub_download(repo_id=repo_id, filename=filename)

        return LlamaCpp(
            model_path=path,
            verbose=False,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            temperature=0,
            n_gpu_layers=100,
            repeat_penalty=0.9,
            n_batch=256,
        )

    def answer_question_with_context(
            self, question: str, context: List[str], handler=None
    ) -> str:
        """
        This method takes a question and a list of context strings as input, and attempts to answer the question using the provided context.

        The method uses the context to understand the scope and reference points for the given question and then formulates an answer based on that context.

        Parameters:
        question (str): The question that needs to be answered. This should be a string containing a clear, concise question.

        context (list[str]): A list of strings providing context for the question. The list should provide relevant information that can be used to answer the question.

        Returns:
        str: The method returns a string that contains the answer to the question, formulated based on the provided context.

        Example:
        >> ask_question("What is the color of the sky?", ["The sky is blue during a clear day."])
        'The sky is blue during a clear day.'
        """

        context_str = "\n\n".join(f"{x}" for i, x in enumerate(context))

        template = (
            "You are a chatbot, your primary task is to help people by answering their questions. Keep your responses short, precise, and directly related to the user's question, using the following context to guide your answer:\n\n"
            "---\n"
            "{context_str}\n"
            "---\n\n"
            "Try your best to answer based on the given context, and avoid creating new information. If the context does not provide enough details to formulate a response, or if you are unsure, kindly state that you can't provide a certain answer.\n"
            "\n\n"
            "USER: {question}\n"
            "ASSISTANT:"
        )
        prompt = PromptTemplate(
            template=template,
            input_variables=["question", "context_str"]
        )

        print(prompt.format_prompt(question=question, context_str=context_str).to_string())
        with critical_function_lock:
            answer = self.model.generate_prompt(
                [
                    prompt.format_prompt(question=question, context_str=context_str),
                ],
                stop=["</s>"],
                callbacks=None if handler is None else [handler],
            )
            return answer.generations[0][0].text.strip()

    def translate_text(self, question, language="EN-US") -> Result:
        return self.translator.translate_to(
            question, language, use_spacy_to_detect_lang_if_needed=False
        )

    def answer_question(self, question: str, handler: StreamLLMCallbackHandler | None):
        """
        This method takes a user's question as input and returns an appropriate answer.

        Parameters:
        question (str): The question that needs to be answered. This should be a string containing a clear, concise question.


        Returns:
        str: The method returns a string that contains the answer to the question.

        Example:
        >> answer_question("What is the color of the sky?")
        'The sky is blue during a clear day.'
        """

        print(f"Receive Question: {question}")

        # translation = self.translate_text(question)
        # if handler is not None:
        #    handler.lang = translation.detected_source_lang
        # translated_question = translation.text
        # print(f"Translation: {translated_question}")

        context = self.vector_store.sim_search(question)
        print("Content:" + str(context["content"]))

        links = []
        for metadata in context["metadata"]:
            if metadata["link"] in links:
                break
            links.append(metadata["link"])
        if handler is not None:
            handler.send_links(links)
        print(f"Links: {links}")

        answer = self.answer_question_with_context(
            question, context["content"], handler
        )
        print(f"Answer: {answer}")
        # if translation.detected_source_lang != "EN-US":
        #    answer = self.translate_text(answer, translation.detected_source_lang).text
        # print(f"Translated answer: {answer}")

        return {
            "answer": answer,
            "question": question,
            "context": context,
        }


if __name__ == "__main__":
    qa_bot = QABot()
    start = time()
    qa_bot.answer_question("What is the color of the sky?", None)
    print(f"Time: {time() - start}")
