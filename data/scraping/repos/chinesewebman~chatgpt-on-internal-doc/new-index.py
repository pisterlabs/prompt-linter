# for gpt-index 0.5.8 and above
import os
import re
from langchain.chat_models import ChatOpenAI
from gpt_index import (
    GPTSimpleVectorIndex, 
    Document,
    MockLLMPredictor, 
    PromptHelper,
    LLMPredictor,
    MockEmbedding, 
    SimpleDirectoryReader,
    ServiceContext,
)
from gpt_index.langchain_helpers.text_splitter import SentenceSplitter
from gpt_index.node_parser import SimpleNodeParser
from gpt_index.embeddings.openai import OpenAIEmbedding

# 专有名词的字典，每个词一行
def load_buddha_dict(dict_path='./buddha-dict.txt'):
    with open(dict_path, 'r', encoding='utf-8') as f:
        words = f.read().splitlines()
    # 根据长度对词语列表进行排序
    words.sort(key=len, reverse=True)
    return words

def add_space_around_words(text, words):
    for word in words:
        # 使用正则表达式进行全词匹配，并在词语前后加英文空格
        pattern = r'\b{}\b'.format(re.escape(word))
        text = re.sub(pattern, f' {word} ', text)
    return text
def merge_consecutive_spaces(text):
    return re.sub(r'\s+', ' ', text)

# 文档预分词处理，在一些标点符号和专有名词前后加了英文空格，为文本块切割做准备。
def refine_doc(directory_path, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    refined_documents = []

    add_space_after = '，。、：”？！；》】）'
    add_space_before = '“《【（'
    buddha_dict = load_buddha_dict()

    for i, doc in enumerate(documents):
        if isinstance(doc, Document):
            text = doc.text
        for char in add_space_after:
            text = text.replace(char, char + ' ')

        for char in add_space_before:
            text = text.replace(char, ' ' + char)

        text = add_space_around_words(text, buddha_dict)
        text = merge_consecutive_spaces(text)
        doc.text = text
        refined_documents.append(doc)

        with open(os.path.join(output_dir, f'output{i+1}.txt'), 'w', encoding='utf-8') as f:
            f.write(doc.text)

    return refined_documents

# 把directory_path目录中的所有文件创建成一个语义向量索引库文件，目前的测试表明，该类型的向量库在达到　1G　容量时依然能快速查询，因为工作时该文件完全被读入内存。相比之下，占用时间的是llm的调用，而不是向量库的读取。
def construct_index(directory_path):
    print("读取"+str(directory_path)+"目录里的所有文件（不包括子目录）...")
    documents = refine_doc(directory_path)

    # 一次调用llm请求中所有内容（包括prompt、提问、回答等合在一起）的最大token数，取决于llm，对gpt-3.5-turbo来说，是4096
    max_input_size = 4096
    
    # 设置回答最多可以用多少token，不能设太大，因为要给传过去的上下文信息留额度
    num_outputs = 2000
    
    # 文本块之间可重叠多少token。这个数值如果过大，会导致文本块切分程序陷入死循环
    max_chunk_overlap = 5
    
    # chunk size limit　是文本块的最大token值，向量基于文本块而计算和存储，查询时也以文本块为单位做匹配，这里设为600，不仅因为简答应该足够，而且查询时也可以指定返回多个匹配的文本块，增加最佳匹配的覆盖率
    chunk_size_limit = 600
    
    # first, we count the tokens
    llm_predictor = MockLLMPredictor(max_tokens=num_outputs)
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    # MockLLMPredictor 和 MockEmbedding 的意思是只计算token数，不真做问答调用，这样可以快速计算出token数，从而计算出预计的花费。
    embed_model = MockEmbedding(embed_dim=1536)
    
    # 分段符号，默认是"/n/n/n"，这里改成"###",作为切分文本块的标志，制作文本时，在###之间安放同主题的文本段落，以备接下来逐文本块制作成语义向量索引。
    paragraph_separator="###"
    # 备用分段符默认是"/n",就不改了

    # 句内分词符号，默认只有英文标点符号和中文句号
    secondary_chunking_regex="[^,.;，。、：”？！；》】“《【（]+[,.;，。、：”？！；》】“《【（]?" 
    # 默认的备选分词符号是英文空格，这里就不改了

    # Chunk_size 默认值为4000，过大了，容易引起多次调用llm做refined_response，这里改小，因此chunk_overlap（文本块之间可重叠的部分）也改小   
    sentence_splitter = SentenceSplitter(chunk_size=chunk_size_limit, chunk_overlap=max_chunk_overlap, paragraph_separator=paragraph_separator, secondary_chunking_regex=secondary_chunking_regex) 
    node_parser = SimpleNodeParser(text_splitter=sentence_splitter) 
    service_context = ServiceContext.from_defaults(node_parser=node_parser, llm_predictor=llm_predictor, prompt_helper=prompt_helper, embed_model=embed_model, chunk_size_limit=chunk_size_limit)
    # 调用创建索引的语句来获得计算的token数
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)
    # get number of tokens used
    embedding_token_usage = embed_model.last_token_usage
    token_price = 0.0004  # cost of embedding per 1000 tokens (0.0004 USD)，默认调用openai text-embedding-ada-002 的价格
    price_per_token = token_price / 1000
    total_price = round(price_per_token * embedding_token_usage,3)
    print("建索引所需Token数：", embedding_token_usage, "，预计花费：", total_price, "美元")
    ask_user()
    # 正式开始创建索引，如果内容比较多，这里建议用挂上付费方式48小时之后的用户账号生成的API_KEY来做，这样比较块，而且不容易失败。
    print("chunk_size_limit:", chunk_size_limit)
    # 为提高效率，一批次多做一些文本块的向量化，这里的值仅供参考，在chunk_size_limit默认为4000时，这个值默认为10，所以这里就按倍数计算了
    embed_batch_size=round(4000/chunk_size_limit*10)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    service_context = ServiceContext.from_defaults(node_parser=node_parser, llm_predictor=llm_predictor, prompt_helper=prompt_helper, chunk_size_limit=chunk_size_limit, embed_model=OpenAIEmbedding(embed_batch_size=embed_batch_size))
    
    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )
    index.save_to_disk('index.json')
    # 这里没有考虑存盘失败的情况，略
    print("索引文件已存盘")
    return
def ask_user():
    user_input = input("是否继续？(y/n)")

    if user_input.lower() == "y":
    # 用户想要继续执行程序
        pass
    else:
    # 用户不想继续执行程序
        print("那么就不再继续执行，再见！")
        exit()
    return
construct_index('input')