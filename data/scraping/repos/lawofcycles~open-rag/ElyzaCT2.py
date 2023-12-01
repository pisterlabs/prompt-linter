import time
import torch
from typing import Optional, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import CTranslate2
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import Generation, LLMResult


# data source
PERSIST_DIR = "./resource/211122_amlcft_guidelines.pdf"

loader = UnstructuredFileLoader(PERSIST_DIR)
documents = loader.load()
print(f"number of docs: {len(documents)}")
print("--------------------------------------------------")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=20,
)

splitted_texts = text_splitter.split_documents(documents)
print(f"チャンクの総数：{len(splitted_texts)}")
print(f"チャンクされた文章の確認（20番目にチャンクされたデータ）：\n{splitted_texts[20]}")

# embed model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

db = FAISS.from_documents(splitted_texts, embeddings)

question = "リスクベースのアプローチとはなんですか。"

start = time.time()
# 質問に対して、データベース中の類似度上位3件を抽出。質問の文章はこの関数でベクトル化され利用される
docs = db.similarity_search(question, k=3)
elapsed_time = time.time() - start
print(f"処理時間[s]: {elapsed_time:.2f}")
for i in range(len(docs)):
    print(docs[i])

# setup LLM
MODEL_NAME = "elyza/ELYZA-japanese-Llama-2-7b-fast-instruct"

## ELYZA LLama2 + Ctranslate2 (7B)
class ElyzaCT2LLM(CTranslate2):
    generator_params = {
        "max_length": 256,
        "sampling_topk": 20,
        "sampling_temperature": 0.7,
        "include_prompt_in_result": False,
    }

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        encoded_prompts = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
        tokenized_prompts = [
            self.tokenizer.convert_ids_to_tokens(encoded_prompt)
            for encoded_prompt in encoded_prompts
        ]

        # 指定したパラメータで文書生成を制御
        results = self.client.generate_batch(tokenized_prompts, **self.generator_params)

        sequences = [result.sequences_ids[0] for result in results]
        decoded_sequences = [self.tokenizer.decode(seq) for seq in sequences]

        generations = []
        for text in decoded_sequences:
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)

llm_ct2 = ElyzaCT2LLM(
    model_path="ct2_model",
    tokenizer_name=MODEL_NAME,
    device_map="auto",
    device_index=[0],
    compute_type="int8",
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問に簡潔に正確に答えてください。"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

chain = load_qa_chain(llm_ct2, chain_type="stuff", prompt=rag_prompt_custom)

start = time.time()
inputs = {"input_documents": docs, "question": question}
output = chain.run(inputs)
elapsed_time = time.time() - start
print("RAGあり")
print(f"処理時間[s]: {elapsed_time:.2f}")
print(f"出力内容：\n{output}")
print(f"トークン数: {llm_ct2.get_num_tokens(output)}")

###################################################
# メモリの解放

del model, tokenizer, pipe, llm, chain
torch.cuda.empty_cache()

