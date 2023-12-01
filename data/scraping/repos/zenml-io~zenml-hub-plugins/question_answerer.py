#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.


from langchain import OpenAI
from langchain.chains import ChatVectorDBChain, SequentialChain
from langchain.vectorstores import VectorStore
from zenml.steps import BaseParameters, step


class QAParameters(BaseParameters):
    """Params for the question answerer step.

    Attributes:
        question: Question to ask.
    """

    question: str


@step(enable_cache=False)
def question_answerer_step(
    vector_store: VectorStore, params: QAParameters
) -> str:
    """Answers a question using a langchain vector store.

    Args:
        vector_store: Langchain vector store used to answer the question.
        params: Parameters for the step.

    Returns:
        The generated answer.
    """
    import logging
    import warnings

    # Suppress all warnings and non-error logs.
    logging.getLogger().setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llm = OpenAI(
            temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo"
        )
        chatgpt_chain = ChatVectorDBChain.from_llm(
            llm=llm, vectorstore=vector_store
        )
        seq_chain = SequentialChain(
            chains=[chatgpt_chain],
            input_variables=["chat_history", "question"],
        )
        answer = seq_chain.run(
            chat_history=[],
            question=params.question,
            verbose=False,
        )

    print(f"Question: {params.question}")
    print(f"Answer: {answer}")
    return answer
