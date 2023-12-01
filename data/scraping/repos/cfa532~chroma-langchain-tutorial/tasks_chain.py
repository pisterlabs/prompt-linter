import langchain
from config import CHROMA_CLIENT, MAX_TOKENS
from langchain import OpenAI
from langchain.chains import RetrievalQA, LLMChain, SimpleSequentialChain, LLMCheckerChain, AnalyzeDocumentChain, RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.memory import SimpleMemory

embedding_function = OpenAIEmbeddings()
llm = OpenAI(temperature=0, max_tokens=MAX_TOKENS)

# create retriever for law files
db_law = Chroma(client=CHROMA_CLIENT, collection_name="law-docs", embedding_function=embedding_function)
# retriever for case related files
db_case = Chroma(client=CHROMA_CLIENT, collection_name="lawyer-tao", embedding_function=embedding_function)
# merger multiple retrievers
retriever_all = MergerRetriever(retrievers=[db_law.as_retriever(), db_case.as_retriever()])

# chains
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="refine", retriever=retriever_all)
checker_chain = LLMCheckerChain.from_llm(llm=llm, verbose=True, output_key="result2")
translation_chain = LLMChain.from_string(llm=llm, template="把下面的内容翻译成中文。\n\n{content}")

# overall_chain = SimpleSequentialChain(
#     chains=[qa_chain, checker_chain, translation_chain],
#     # memory=SimpleMemory(memories={"language":"中文",}),
#     # input_variables=["content", "query"],
#     # output_variables=["content"],
#     verbose=True, )

# qa = LLMChain(llm=llm, prompt=PromptTemplate.from_template("{query}"))
query = "2018年9月19日，答辩人与原告就案涉店铺承租主体变更达成一致并签订《店铺租赁合同》主体变更协议，各方同意将案涉店铺的承租方由答辩人变更为阿家公司。因主体变更协议签订当时，阿家公司尚未注册成立，故双方在协议中进一步明确了在阿家公司取得营业执照之前，答辩人仍需作为共同承租人与阿家公司承担共同连带责任，但自阿家公司取得营业执照之日起，则原合同承租方的权利义务全部由阿家公司独自承担。而阿家公司已经于2018年10月12日取得营业执照，故自2018年10月12日之后的债务应该由阿家公司独自承担，于答辩人无关。\n\n"
query += "根据案件内容，列出5条提纲如何回复。以Json格式输出结果"

# langchain.debug = True
result = qa_chain(query)
# langchain.debug = False
print(result)

# Get human input to confirm the list
