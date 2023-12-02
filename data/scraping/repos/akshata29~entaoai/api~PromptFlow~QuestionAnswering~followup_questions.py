from promptflow import tool
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from promptflow.connections import CustomConnection
import uuid
import json
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import QueryType
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,  
)
from azure.search.documents.models import Vector
from langchain.chains import LLMChain

def indexDocs(SearchService, SearchKey, indexName, docs):
    print("Total docs: " + str(len(docs)))
    searchClient = SearchClient(endpoint=f"https://{SearchService}.search.windows.net/",
                                    index_name=indexName,
                                    credential=AzureKeyCredential(SearchKey))
    i = 0
    batch = []
    for s in docs:
        batch.append(s)
        i += 1
        if i % 1000 == 0:
            results = searchClient.upload_documents(documents=batch)
            succeeded = sum([1 for r in results if r.succeeded])
            print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")
            batch = []

    if len(batch) > 0:
        results = searchClient.upload_documents(documents=batch)
        succeeded = sum([1 for r in results if r.succeeded])
        print(f"\tIndexed {len(results)} sections, {succeeded} succeeded")

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def generateFollowupQuestions(retrievedDocs: list, question: str, embeddedQuestion:object, promptTemplate:str, overrides:list, llm, modifiedAnswer:str, 
    existingAnswer:bool, jsonAnswer:list, indexType:str, indexNs:str, conn:CustomConnection) -> list:
  if existingAnswer:
    results = {}
    results["values"] = []
    results["values"].append({
                "recordId": 0,
                "data": jsonAnswer
                })
    return results
  else:
    kbData = []
    kbId = str(uuid.uuid4())
    overrideChain = overrides.get("chainType") or 'stuff'

    followupTemplate = """
    Generate three very brief questions that the user would likely ask next.
    Use double angle brackets to reference the questions, e.g. <What is Azure?>.
    Try not to repeat questions that have already been asked.  Don't include the context in the answer.

    Return the questions in the following format:
    <>
    <>
    <>
    
    ALWAYS return a "NEXT QUESTIONS" part in your answer.

    {context}
    """
    followupPrompt = PromptTemplate(template=followupTemplate, input_variables=["context"])
    followupChain = load_qa_chain(llm, chain_type='stuff', prompt=followupPrompt)
    

    if promptTemplate == '':
        template = """
            Given the following extracted parts of a long document and a question, create a final answer. 
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
            If the answer is not contained within the text below, say \"I don't know\".

            {summaries}
            Question: {question}
        """
    else:
        template = promptTemplate
    
    rawDocs=[]
    for doc in retrievedDocs:
        rawDocs.append(doc.page_content)

    qaPrompt = PromptTemplate(template=template, input_variables=["summaries", "question"])

    if overrideChain == "stuff" or overrideChain == "map_rerank" or overrideChain == "map_reduce":
        thoughtPrompt = qaPrompt.format(question=question, summaries=rawDocs)
    elif overrideChain == "refine":
        thoughtPrompt = qaPrompt.format(question=question, context_str=rawDocs)
    
    # Followup questions
    # followupAnswer = followupChain({"input_documents": retrievedDocs, "question": question}, return_only_outputs=True)
    # nextQuestions = followupAnswer['output_text'].replace("Answer: ", '').replace("Sources:", 'SOURCES:').replace("Next Questions:", 'NEXT QUESTIONS:').replace('NEXT QUESTIONS:', '').replace('NEXT QUESTIONS', '')
    llm_chain = LLMChain(prompt=followupPrompt, llm=llm)
    nextQuestions = llm_chain.predict(context=rawDocs)
    print("Next questions: " + str(nextQuestions))

    sources = ''                
    if (modifiedAnswer.find("I don't know") >= 0):
        sources = ''
        nextQuestions = ''
    else:
        sources = sources + "\n" + retrievedDocs[0].metadata['source']

    outputFinalAnswer = {"data_points": rawDocs, "answer": modifiedAnswer, 
            "thoughts": f"<br><br>Prompt:<br>" + thoughtPrompt.replace('\n', '<br>'),
                "sources": sources, "nextQuestions": nextQuestions, "error": ""}
    
    try:
        kbData.append({
            "id": kbId,
            "question": question,
            "indexType": indexType,
            "indexName": indexNs,
            "vectorQuestion": embeddedQuestion,
            "answer": json.dumps(outputFinalAnswer),
        })

        indexDocs(conn.SearchService, conn.SearchKey, conn.KbIndexName, kbData)
    except Exception as e:
        print("Error in KB Indexing: " + str(e))
        pass

    results = {}
    results["values"] = []
    results["values"].append({
                "recordId": 0,
                "data": outputFinalAnswer
                })
    return results