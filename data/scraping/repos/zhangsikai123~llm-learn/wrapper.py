import inspect

from colorama import Fore
from colorama import init
from colorama import Style
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


init()


class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {
            k: v
            for k, v in local_vars.items()
            if k not in set(ignore + ["self"]) and not k.startswith("_")
        }
        for k, v in self.hparams.items():
            setattr(self, k, v)


class ChatChainWrapper(HyperParameters):
    def __init__(self, llm, retriever, memory=None, chinese=False):
        self.save_hyperparameters()

    def run(self):
        # template = """Use the following pieces of context to answer the question at the end.
        # If you don't know the answer, just say that you don't know,
        # don't try to make up an answer.
        # Use three sentences maximum and keep the answer as concise as possible.
        # Use Chinese to answer the question
        # {context}
        # Question: {question}
        # Helpful Answer:"""
        while True:
            # Prompt the user for the PDF URL
            role = input("Enter the my role (or 'q' to quit): ")

            if role.lower() == "q":
                break
            template = (
                "you are a"
                + role
                + """ and you are asked the following question:
                {context}
                Question: {question}
                Helpful Answer:"""
            )
            if self.chinese:
                template += " remember always ask your question in Chinese"
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.retriever,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                memory=self.memory,
            )

            while True:
                # Prompt the user for a question
                question = input(f"{Fore.LIGHTCYAN_EX}me{Style.RESET_ALL}: ")
                if question.lower() == "q":
                    break

                # Perform question-answering
                answer = qa_chain({"query": question})

                # Display the answer
                print(
                    f"{Fore.LIGHTGREEN_EX}{role}{Style.RESET_ALL}: {answer['result']}"
                )
