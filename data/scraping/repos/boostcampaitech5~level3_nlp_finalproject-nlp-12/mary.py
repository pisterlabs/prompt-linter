from dotenv import load_dotenv
from pymongo.database import Database

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema.document import Document

from config.settings import ModelSettings

from typing import Optional, Dict, Any, List
from pathlib import Path
import time

class CustomVectorStoreRetrieverMemory(VectorStoreRetrieverMemory):

    metadata: Optional[Dict[str, Any]] = None,

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {k: v for k, v in inputs.items() if k != self.memory_key}
        texts = [
            f"{k}: {v}"
            for k, v in list(filtered_inputs.items()) + list(outputs.items())
        ]
        page_content = "\n".join(texts)
        return [Document(page_content=page_content, metadata=self.metadata)]
    

def build_pre_filter(user_id: str, timestamp: float) -> dict:
    return {
        'compound': {
                    'filter': {
                        'text': {
                            'path': 'user_id',
                            'query': user_id
                            }
                    },
                    'should': {
                        'near': {
                            'origin': timestamp,
                            'path': 'timestamp',
                            'pivot': 10000000
                            }
                    }
        }
    }

def build_post_filter_pipeline(use_sort: bool = True) -> Optional[List]:
    return [{'$sort': {'timestamp': 1}}] if use_sort else None


class Mary:
    def __init__(self, settings:ModelSettings, mongodb_client:Database):

        # 모델 관련 설정
        model_full_path = str(Path(settings.MODEL_PATH).joinpath(settings.MODEL_NAME))

        # 모델 로드 및 초기화 코드
        load_dotenv()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_full_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device=f"cuda", non_blocking=True)

        tokenizer = AutoTokenizer.from_pretrained(model_full_path)
        model.eval()
        model.config.use_cache = True
        
        pipe = pipeline(
            'text-generation',
            model = model,
            tokenizer = tokenizer,
            device=0,
            min_new_tokens=10,
            max_new_tokens=128,
            early_stopping=True,
            do_sample=True,
            eos_token_id=2,
            repetition_penalty=1.05,
            temperature=0.9,
            top_k=20,
            top_p=0.95,
        )

        self.local_llm = HuggingFacePipeline(pipeline=pipe)

        embedding_fn = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY
        )

        collection = mongodb_client[settings.VECTOR_INDEX_COLLECTION]
        self.vectorstore = MongoDBAtlasVectorSearch(
            collection, embedding_fn
        )

        template = """이전 대화와 현재 대화의 명령어를 참고하여 상황에 공감하고 친절한 응답을 생성해주세요. 응답 마지막에는 지금까지의 내용과 관련된 질문을 해주세요.\n\n[이전 대화]\n{history}\n\n[현재 대화]\n### 명령어:\n{### 명령어}\n\n### 응답:\n"""

        self.prompt = PromptTemplate(
            input_variables=["history", "### 명령어"], template=template
        )


    def get_response(self, question:str, user_id:str) -> str:
        input_dict = {'### 명령어': question}
        timestamp = float(time.time())

        res = ConversationChain(
            llm=self.local_llm,
            prompt=self.prompt,
            memory=CustomVectorStoreRetrieverMemory(
                retriever=self.vectorstore.as_retriever(search_kwargs={
                    'k':2,
                    'pre_filter': build_pre_filter(user_id, timestamp),
                    'post_filter_pipeline': build_post_filter_pipeline()
                }),
                metadata={'user_id': user_id, 'timestamp': timestamp},
            ),
            input_key='### 명령어',
            output_key='### 응답',
            verbose=True
        ).predict(**input_dict)

        return res