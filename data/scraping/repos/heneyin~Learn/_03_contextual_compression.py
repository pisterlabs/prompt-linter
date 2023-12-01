"""
https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/contextual_compression/
上下文压缩
retrieval 的一项挑战是，当数据引入系统时，你不知道文档系统将会遇到哪些查询。

这意味着最相关的信息可能被隐藏在大量不相关文本的文档中。

通过传递完整的文件信息可能会导致更昂贵的 LLM 交互 和更差的响应。

上下文压缩旨在解决这个问题。
这个想法很简单：您可以使用给定查询的上下文来压缩它们，以便只返回相关信息，而不是立即按原样返回检索到的文档。
这里的“压缩”既指压缩单个文档的内容，也指批量过滤文档。

要使用上下文压缩检索器，您需要：
1. 基础检索器
2. 一个文档压缩器

上下文压缩检索器将查询传递给基础检索器，获取初始文档并将它们传递给文档压缩器。
文档压缩器获取文档列表并通过减少文档内容或完全删除文档来缩短它。
"""
from langchain.chat_models import ChatOpenAI

import env

# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS

documents = TextLoader('../../texts/maodun.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
retriever = FAISS.from_documents(texts, OpenAIEmbeddings()).as_retriever()

# 初始查找，找到许多非相关的内容
docs = retriever.get_relevant_documents("介绍一下 山海经")
pretty_print_docs(docs)

"""
使用 ContextualCompressionRetriever 包装初始检索器，
我们将添加一个 LLMChainExtractor，它将迭代最初返回的文档，并从每个文档中仅提取与查询相关的内容。
"""

from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

chat = ChatOpenAI(temperature=1, model_name="gpt-3.5-turbo-16k-0613")

# llm = OpenAI(max_tokens=8000,
#              model_name="gpt-3.5-turbo-16k-0613")

compressor = LLMChainExtractor.from_llm(chat)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

"""
实际的 prompt 为：
Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question. If none of the context is relevant return NO_OUTPUT. 

Remember, *DO NOT* edit the extracted parts of the context.

> Question: 介绍一下 山海经
> Context:
>>>
[26] 见列宁《黑格尔〈逻辑学〉一书摘要》。新的译文是：“辩证法是一种学说，它研究对立面怎样才能够同一，是怎样（怎样成为）同一的——在什么条件下它们是相互转化而同一的，——为什么人的头脑不应该把这些对立面看作僵死的、凝固的东西，而应该看作活生生的、有条件的、活动的、彼此转化的东西。”（《列宁全集》第55卷，人民出版社1990年版，第90页）

[27] 《山海经》是一部中国古代地理著作，其中记载了不少远古的神话传说。夸父是《山海经海外北经》上记载的一个神人。据说：“夸父与日逐走。入日，渴欲得饮，饮于河渭。河渭不足，北饮大泽。未至，道渴而死。弃其杖，化为邓林。”

[28] 羿是中国古代传说中的英雄，“射日”是关于他善射的著名故事。据西汉淮南王刘安（公元前二世纪人）及其门客所著《淮南子》一书说：“尧之时，十日并出，焦禾稼，杀草木，而民无所食。猰豸、凿齿、九婴、大风、封狶、修蛇，皆为民害。尧乃使羿……上射十日而下杀猰豸。……万民皆喜。”东汉著作家王逸（公元二世纪人）关于屈原诗篇《天问》的注释说：“淮南言，尧时十日并出，草木焦枯。尧命羿仰射十日，中其九日……留其一日。”

[29] 《西游记》是明代作家吴承恩着的一部神话小说。孙悟空是书中的主角。他是一个神猴，有七十二变的法术，能够随意变成各式各样的鸟兽虫鱼草木器物或者人形。

[30] 《聊斋志异》是清代文学家蒲松龄着的短篇小说集，大部分是叙述神仙狐鬼的故事。

[31] 见马克思《〈政治经济学批判〉导言》（《马克思恩格斯选集》第2卷，人民出版社1972年版，第113页）。

[32] 见马克思《〈政治经济学批判〉导言》（《马克思恩格斯选集》第2卷，人民出版社1972年版，第114页）。

[33] 见列宁《谈谈辩证法问题》。新的译文是：“对立面的统一（一致、同一、均势）是有条件的、暂时的、易逝的、相对的。相互排斥的对立面的斗争是绝对的，正如发展、运动是绝对的一样。”（《列宁全集》第55卷，人民出版社1990年版，第306页）

[34] 见东汉著名史学家班固（三二——九二）所著《汉书艺文志》，原文是：“诸子十家，其可观者，九家而已。皆起于王道既微，诸侯力政，时君世主，好恶殊方。是以九家之术，蜂出并作，各引一端，崇其所善，以此驰说，取合诸侯。其言虽殊，辟犹水火，相灭亦相生也。仁之与义，敬之与和，相反而皆相成也。”
>>>
Extracted relevant parts:
"""
compressed_docs = compression_retriever.get_relevant_documents("介绍一下 山海经")
print("---------- 压缩之后 ！效果最好！--------------")
pretty_print_docs(compressed_docs)


"""
更多压缩器：
1. filters LLMChainFilter:决定要过滤掉哪些最初检索到的文档以及要返回哪些文档，而无需操作文档内容。
"""

from langchain.retrievers.document_compressors import LLMChainFilter

_filter = LLMChainFilter.from_llm(chat)
compression_retriever = ContextualCompressionRetriever(base_compressor=_filter, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("介绍一下 山海经")
print("---------- filters 压缩之后 --------------")
pretty_print_docs(compressed_docs)

"""
EmbeddingsFilter:  不使用 LLM。
EmbeddingsFilter 通过向量文档和查询并仅返回那些与查询具有足够相似向量的文档来提供更便宜和更快的选项
"""

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings = OpenAIEmbeddings()
embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
compression_retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("介绍一下 山海经")
print("---------- EmbeddingsFilter 压缩之后 --------------")
pretty_print_docs(compressed_docs)


"""
将压缩器和文档转换器串在一起

使用 DocumentCompressorPipeline 我们还可以轻松地按顺序组合多个压缩器。
除了压缩器之外，我们还可以将 BaseDocumentTransformers 添加到管道中，它不执行任何上下文压缩，而只是对一组文档执行一些转换。
例如，TextSplitters 可以用作文档转换器，将文档分割成更小的部分，而 EmbeddingsRedundantFilter 可以用于根据文档之间嵌入的相似性来过滤掉冗余文档。
"""

# 下面我们创建一个压缩器管道，首先将文档分割成更小的块，然后删除冗余文档，然后根据与查询的相关性进行过滤。

"""
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=". ")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)

# 连在一起
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=retriever)

compressed_docs = compression_retriever.get_relevant_documents("What did the president say about Ketanji Jackson Brown")
pretty_print_docs(compressed_docs)

"""
