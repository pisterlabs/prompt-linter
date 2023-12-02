"""Together with what has been done in v2, this version adds the integration of
ChatGPT for which uses the similar docs that are found in the first step
to generate the answer for the questions, i.e. figuring out the disease.
"""

import numpy
import pandas
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .disease_finder_v2 import get_vectorstore


def disease_finder_v3():
    vector_db = get_vectorstore("data/train.csv")
    df = pandas.read_csv("data/test.csv")

    k = 5
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0)

    question_center = (
        "Given similar diseases in the context, what is the most likely disease "
        "this patient may be suffering from?"
    )

    retriever_all = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k, "include_metadata": True},
    )

    # Build prompt
    template = """Use only the following pieces of context to answer the question at the end.
        Don't use any other information. Try to use the labels in the context to help you answer the question.
        This won't be used for diagnosis, but to help you understand the question. 
        Keep your answer only the disease name, don't include any other information
        {context}
        Question: {question}
    """

    QA_CHAIN_PROMPT = ChatPromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever_all,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    cnt = 0
    for text, label in zip(df.text, df.label):
        question = f"{question_center}: {text}"
        result = qa_chain({"query": question})
        # print(result["result"])
        prediction = result["result"]
        # top_n_results = retriever_all.get_relevant_documents(text)
        #
        # pred_labels = [doc.metadata.get("label") for doc in top_n_results]
        print(f"{label=}, {prediction=}")
        cnt += int(label.lower() in prediction.lower())

        if not int(label.lower() in prediction.lower()):
            print(label.lower(), "----", prediction.lower())

    print(cnt / len(df))
    return cnt / len(df)
