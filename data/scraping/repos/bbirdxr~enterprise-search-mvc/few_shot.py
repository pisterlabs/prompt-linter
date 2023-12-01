from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
import datetime
import json
from few_shot_example import examples
from context_info import context_info
os.environ["OPENAI_API_KEY"] = "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"


template = """Scenario: {scenario}
Conversation Examples:{conversations}"""

#If documents contain some unrelated information, you should keep the original answer's information and
example_prompt = PromptTemplate(
    input_variables=["scenario", "conversations"],
    template=template,
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=2)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=
    """You are an Assistant having a conversation with a human. current time is {current_time}. You have the ability and access to cancel orders, query order history, refund items, provide web links and transfer to a human representative. The answer should be comprehensive, effective, clear, friendly and credible. Add some emoji in your answers to make them cute. 
Given the following related answer examples and context, assuming that you have all the access to the context information above. 
'context' is the information in the background database system, you need to use the 'context' as the information you know. 
As an assitant, you are required to answer the question in a regular way.Here are some conversation examples:
""",
    suffix="""
You should limit the length of each answer to 150 words. Must replace all the '[xxx]' with detailed information, if you can't replace it just remove it. Remember that '[' and ']'  SHOULD NOT exist in your answer!!
Meanwhile, you should suspect the user's account information to prevent misusing of policy as frequent refunding behavior.
Now lets start talk! 
{question}""",
    input_variables=["question", "current_time"])

embeddings = OpenAIEmbeddings()
vector_store = Chroma(
    persist_directory=
    "/Users/bbird/data/wiz/project/Enterprise Search MVP/coding/few-shot/openai",
    embedding_function=embeddings)


def do_wish_search(question: str):
    retriever = vector_store.as_retriever(search_type="mmr")
    docs = retriever.get_relevant_documents(question, top_k=1)
    page_content_list = [doc.page_content for doc in docs]
    return 'You can use the documents following to refine your answer. The preferrence level of conversation examples is higher than documents.\nRelated document:( If unuseful or unrelated, just ignore. If useful, provide the additional url)[' + ',\n'.join(
        page_content_list) + ']'


def get_few_shot_prompt(question: str):
    return few_shot_prompt.format(
        question=question,
        current_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()
from langchain.schema import (HumanMessage, SystemMessage)

question = input("Please input: \n")

messages = [
    SystemMessage(content="context:" + context_info),
    SystemMessage(content=get_few_shot_prompt(question)),
    HumanMessage(content=question+" my uid is: {u4600089016}"),
    SystemMessage(content=do_wish_search(question))
]
response = chat(messages)
messages.append(response)

print(messages[-1].content)
while True:
    question = input('Please input: \n')
    messages.append(HumanMessage(content=question))
    lastest_message_pair = "assistant:" + messages[
        -2].content[:100] + "..., user:" + messages[-1].content
    messages.append(
        SystemMessage(content=do_wish_search(lastest_message_pair)))

    messages[1].content = get_few_shot_prompt(lastest_message_pair)

    result = chat(
        messages[:2] +
        [item for item in messages if isinstance(item, HumanMessage)][0:-1] +
        messages[-3:])
    messages.append(result)
    print(messages[-1].content)
