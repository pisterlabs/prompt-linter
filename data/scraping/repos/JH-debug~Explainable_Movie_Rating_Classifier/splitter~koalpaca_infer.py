import os
import torch
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, T5Tokenizer, T5ForConditionalGeneration, GPT2TokenizerFast

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class MovieExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """Prompt template for the movie explainer task."""
    @validator("input_variables")
    def validate_input_variables(cls, v):
        """Validate that the input variables are correct."""
        if len(v) != 1 or "subtitles" not in v:
            raise ValueError("subtitles must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Generate the prompt to be sent to the language model
        prompt = "대사:" + kwargs["subtitles"] +\
        """\n"유해성검토기준": [{"주제": "해당 연령층의 정서와 가치관, 인격형성 등에 끼칠 영향 또는 그 이해와 수용정도"},""" +\
        """{"선정성": "신체 노출과 애무, 정사장면 등 성적 행위의 표현 정도"}, {"폭력성": "신체 분위, 도구 등을 이용한 물리적 폭력과 성폭력, 이로 인해 발생한 상해, 유혈, 신체훼손, 고통 등의 빈도와 표현정도"},""" +\
        """{"대사": "욕설, 비속어, 저속어 등의 빈도와 표현 정도"},""" +\
        """{"공포": "긴장감과 불안감, 그 자극과 위협으로 인한 정신적 충격 유발정도"},""" +\
        """{"약물": "소재나 수단으로 다루어진 약물 등의 표현 정도"},""" +\
        """{"모방위험": "살인, 마약, 자살, 학교 폭력, 따돌림, 청소년 비행과 무기류 사용, 범죄기술 등에 대한 모방심리를 고무, 자극하는 정도"}]}""" +\
        "\n\n대사의 유해성을 검토하고자 합니다. 이를 위해서는 다음 단계를 따르세요. " +\
        "\nStep 1. 유해성검토기준에 대한 각 항목의 심각도를 결정하여 낮음, 중간, 높음으로 분류하세요. 유해성이 없으면 없음이라고 하세요." +\
        "\nStep 2. 전체 분류를 기반으로 각 유해성검토기준 별로 정리하시오, " +\
        "\nStep 3. Step 1, 2의 검토 결과를 종합하여 설명을 작성하시오. Let\'s think step by step."
        # "\n\n주제:" +\
        # "\n선정성:" +\
        # "\n폭력성:" +\
        # "\n대사:" +\
        # "\n공포:" +\
        # "\n약물:" +\
        # "\n모방위험:" +\
        # "\nexplanation:"
        return prompt

    def _prompt_type(self):
        return "movie-explainer"

prompt = MovieExplainerPromptTemplate(input_variables=["subtitles"])

model_path = "beomi/KoAlpaca-Polyglot-12.8B"
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             low_cpu_mem_usage=True,
                                             device_map='auto',
                                             load_in_8bit=True)
model.tie_weights()
model.eval()

pipe = pipeline(
    "text-generation", model=model, tokenizer=model_path, max_new_tokens=512, model_kwargs={"temperature": 0, "low_cpu_mem_usage": True}
)
hf = HuggingFacePipeline(pipeline=pipe)
llm_chain = LLMChain(prompt=prompt, llm=hf)

subtitles = """아무리 세상이 좆같아도, 
'형이 동생 등에 칼을 꽂겄소', 
'상구야', 
'그래서 내가 그만두라고 했잖아', 
'원한을 품고 살면', 
'반드시 자신에게 칼이 돌아온다는 걸', 
'잊었냐', 
'에헤 대화의 본질이 빗나가는구먼', 
'내가 지금 나 배신한 거 갖고 이러겄소', 
'원하는 게 뭐야',
'근데', 
'주은혜는 꼭 죽여야만 했소', 
'내가 안 죽였어', 
'문일석이도 매수했소', 
'나한테 비자금 파일 넘긴 적 없다고', 
'난 모르는 일이야', 
'마지막으로 물읍시다', 
'희대의 사기꾼', 
'살인 청부와 성폭행을 일삼는 파렴치한', 
'형님 아이디어요', 
'뭐 할라고', 
'이것이 갖고 싶은', 
'이 개새끼가', 
'내가 그렇게 좆같이 보여', 
'응 여기가 어디라고 응', 
'이 깡패 새끼', 
'이 씨팔놈이', 
'씨팔', 
'씨팔', 
'장필우가 그런 거야', 
'장필우가 다 시킨 거야', 
'주은혜를 죽이고', 
'널', 
'성폭행', 
'살인범으로 만든 것도', 
'장필우가 시킨 거라고', 
'똥은 어느 쪽 손으로 닦아', 
'남은 손은 똥이나 닦으쇼', 
'글 같은 거 쓰지 말고', 
'네 형사과 김정호입니다', 
'네', 
'네 위치가 어떻게 되시죠', 
'저기요', 
'자수하러 왔는데요', 
'주은혜는 꼭 죽여야만 했소', 
'내가 안 죽였어', 
'문일석이도 매수했소', 
'나한테 비자금 파일 넘긴 적 없다고', 
'난 모르는 일이야', 
'장필우가 그런 거야'
"""
print("Prompt: ")
print(prompt.format(subtitles=subtitles))

print("\nPrediction: ")
print(llm_chain.run(subtitles))