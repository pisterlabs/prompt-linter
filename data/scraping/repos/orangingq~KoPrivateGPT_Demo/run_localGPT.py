from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import click

from utils import slice_stop_words
from vectorDB import DB
from dotenv import load_dotenv
from embed import Embedding
from model import load_model

examples = [
    {
        "sentence": "세종대왕은 책을 좋아했다.",
        "relationships": 
            """
            Nodes: (1-세종대왕), (2-책), (3-과거)
            Edges: (1-좋아하다): 1->2
            Infoedges: (1-에 (시간)): 1->3"""
    },{
        "sentence": "학생 A는 선생님께 과제를 냈다.",
        "relationships": 
            """
            Nodes: (1-학생 A), (2-선생님), (3-과제), (4-과거)
            Edges: (1-내다): 1->3
            Infoedges: (1-에 (시간)): 1->4, (2-에게): 1->2"""
    },{
        "sentence": "왕 씨는 지난 6일에 위층 베란다에 남자아이가 매달려 있는 상황을 목격했다.",
        "relationships": 
            """
            Nodes: (1-왕 씨), (2-지난 6일), (3-위층 베란다), (4-남자아이)
            Edges: (1-매달리다): 4->3, (2-목격하다): 1->4
            Infoedges: (1-에 (시간)): 1->2, (2-에 (시간)): 2->2"""
    },{
        "sentence": "왕 씨는 지난 6일에 위층 베란다에 남자아이가 매달려 있는 상황을 목격했다. 왕 씨는 지방 당국에 신고했다. 지방 당국은 왕씨 일가에게 용감한 시민상을 수여했다.",
        "relationships":
            """
            Nodes: (1-왕 씨), (2-지난 6일), (3-위층 베란다), (4-남자아이), (5-지방 당국), (6-용감한 시민상), (7-과거)
            Edges: (1-매달리다): 4->3, (2-목격하다): 1->4, (3-신고하다):1->4, (4-수여하다):5->6
            Infoedges: (1-에 (시간)): 1->2, (2-에 (시간)): 2->2, (3-에게): 3->5, (4-부터 (시간)): 3->2, (5-까지 (시간)): 3->7, (6-에게): 4->1, (7-부터 (시간)): 4->2, (8-까지 (시간)): 4->7"""
    }
]
        

def make_kg(llm, retriever):
      # 주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
      #       질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.

    example_prompt = PromptTemplate(input_variables=["sentence", "relationships"], template="Sentence: {sentence}\nRelationships: {relationships}")

    prompt_template = """
            Given an input sentence, extrapolate as many relationships as possible from the prompt and update the state. 
            You should never lie. Extrapolate only true relationships.
            A relationship has three types of entity. Nodes, Edges, and Infoedge. 

            1. Node
            Every node has a distinct id and label. 
            The format of the node is (id-label).
            The label of a node is a subject of the sentence.

            2. Edge
            Every edge has a set of 'from' node ids, and a set of 'to' node ids, its own id, and label.
            The format of the edge is (id-label): node id->node id.
            Every edge is directed from a set of nodes and to another set of nodes. 
            The label of an edge is a basic type verb of the sentence. Label should be precise and short. 

            3. Infoedge
            Infoedge adds additional information of the sentence to the edge such as time, location, duration, or from whom. 
            Every infoedge has a single 'base' edge id, a set of 'info' node ids, its own id, and label. 
            The format of the infoedge is (id-label): edge id->node id.
            The label of an infoedge should be one of '부터 (시간)', '까지 (시간)', '에서 (장소)', '에 (시간)', '에게', and '에게서'. 

            Sentence: {input}
            Relationships:
            """
    # prompt = PromptTemplate(template=prompt_template, input_variables=["context", "sentence"])
    prompt = FewShotPromptTemplate(
    examples=examples, 
    example_prompt=example_prompt, 
    suffix=prompt_template + "Sentence: {input}\nRelationships: \n", 
    input_variables=["input"]
)
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs)
    return qa


def get_kg(retrieval_kg, query):
    # Get the answer from the chain
    res = retrieval_kg({"query": query})
    answer, docs = res['result'], res['source_documents']

    return query, answer, docs


def print_query_answer(query, answer):
    # Print the result
    print("\n\n> 질문:")
    print(query)
    print("\n> 대답:")
    print(answer)


def print_docs(docs):
    # Print the relevant sources used for the answer
    print("----------------------------------참조한 문서---------------------------")
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print("----------------------------------참조한 문서---------------------------")


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--model_type', default='koAlpaca', help='model to run on, select koAlpaca or openai')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE.')
def main(device_type, model_type, db_type, embedding_type):
    load_dotenv()

    llm = load_model(model_type, device_type=device_type)

    embeddings = Embedding(embed_type=embedding_type, device_type=device_type).embedding()

    # load the vectorstore
    db = DB(db_type, embeddings).load()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    retrieval_kg = make_kg(llm, retriever)
    while True:
        sentence = input("문장을 입력하세요: ")
        if sentence in ["exit", "종료"]:
            break
        query, sentence, docs = get_kg(retrieval_kg, sentence)
        answer = slice_stop_words(answer, ["Sentence :", "sentence:"])
        print_query_answer(query, answer)
        print_docs(docs)


if __name__ == "__main__":
    main()
