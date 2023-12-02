from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.docstore.document import Document
from dotenv import load_dotenv

class AnalysisExecutor:


    def __init__(self, criteria_set, vector_store):

        self.criteria_set = criteria_set
        self.vector_store = vector_store
        self.retrieval_chain = self.get_retrieval_chain(vector_store)
        self.answer_list, self.cost = self.get_and_store_all_llm_responses_and_source_docs(criteria_set, self.retrieval_chain)


    def get_retrieval_chain(self, vector_store):
        load_dotenv()
        llm = OpenAI(temperature=0.0)  # initialize LLM model
        #         turbo_llm = ChatOpenAI(temperature= 0.0, model_name='gpt-3.5-turbo')
        retrieval_chain = RetrievalQA.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            # memory = memory,
            return_source_documents=True)
        return retrieval_chain


    def get_llm_response_with_sources(self, retrieval_chain, prompt):
        response = retrieval_chain({"query": prompt})
        return response


    # source documents must be list containing <class 'langchain.schema.Document'> objects
    def combine_doc_strings(self, source_documents):
        if source_documents is None:
            pass
            # raise Exception("No source document returned!")
        else:
            source_str = ""
            for d in source_documents:
                if "page" in d.metadata:
                    source_str += f"{d.page_content} (source: {d.metadata['source']}, pg. {d.metadata['page']}) "
                else:
                    source_str += f"{d.page_content} (source: {d.metadata['source']}) "
            return source_str


    def get_and_store_llm_response_and_source_docs(self, crit_dict, retrieval_chain):
        result = self.get_llm_response_with_sources(retrieval_chain, crit_dict["prompt"])
        crit_dict["response"] = result["result"]
        crit_dict["source"] = self.combine_doc_strings(result["source_documents"])


    # takes criteria set dict and langchain retrieval chain as arguments
    # returns list which is a version of the original criteria list, expanded to include LLM responses and retrieved source docs
    # also returns cost
    def get_and_store_all_llm_responses_and_source_docs(self, criteria_set, retrieval_chain):
        criteria_and_response_set = criteria_set
        with get_openai_callback() as cost:   # this gets the token cost from the OpenAI API
            for c in criteria_and_response_set:
                if "subcriteria" in c:
                    for s in c["subcriteria"]:
                        if s["prompt"]:
                            self.get_and_store_llm_response_and_source_docs(s, retrieval_chain)
                        else:
                            raise Exception(f"Missing prompt for criterion: {c['name']}, subcriterion {s['name']}")
                elif c["prompt"]:
                    self.get_and_store_llm_response_and_source_docs(c, retrieval_chain)
                else:
                    raise Exception(f"Missing prompt for criterion: {c['name']}")
        return criteria_and_response_set, cost


# ae = AnalysisExecutor(None, None)
# ret_chain = ae.get_retrieval_chain(None)
# print(type(ret_chain))


# doc1 = Document(page_content="hello", metadata={"source": "source_a"})
# doc2 = Document(page_content="world", metadata={"source": "source_b"})
# docs = [doc1, doc2]
# source_string = ae.combine_doc_strings(docs)
# print(source_string)
# self.assertEquals(source_string, "hello (source: source_a) world (source: source_b)")

# from acc_checker.text_prep.create_chromaDB import get_chroma_db
# db = get_chroma_db()
#
# load_dotenv()
#
# llm = OpenAI(temperature=0.0)  # initialize LLM model
#
# prompt = "Does 1 + 1 equal 4? Answer 'Yes' or 'No'"
# criteria_set = [{"name": "crit a", "subcriteria": [{"name": "crit a.a", "prompt": prompt}, {"name": "crit a.b", "prompt": prompt}]},
#                              {"name": "crit b", "prompt": "this is bollocks"}]
#
# ae = AnalysisExecutor(criteria_set, db)

