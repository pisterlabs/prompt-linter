import os

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationKGMemory
from get_size_reco import GetSizeReco

def gpt(input, template):
    llm = OpenAI(temperature=0.9)

    prompt = PromptTemplate(
        input_variables=["history", "input"], 
        template=template
    )

    memory = ConversationKGMemory(llm=llm)

    conversation_with_kg = ConversationChain(
        llm=llm,
        verbose=True,
        prompt=prompt,
        memory=memory
    )

    return conversation_with_kg.predict(input=input).strip()

def SizeReco(apikey, id):
    os.environ['OPENAI_API_KEY'] = apikey
    url = 'https://www.musinsa.com/app/goods/' + id
    size_reco = GetSizeReco(url)

    if size_reco:
        size_reco_template = """You are the helpful agent that recommends height and weight per size options from input size contents. Calculate average range of heights and weights per sizes and sex. If there is no data in one size option, don't print height and weight but "데이터 없음".
    The result should be like this: Example1)[S size] (여성)155~160cm/45~50kg, (남성)데이터 없음\n[M size] (여성)160~165cm/50~60kg, (남성)170~175cm/64~77kg\n[L size] (여성)165~170cm/60~70kg, (남성)175~180cm/77~100kg Example2)[FREE size] (여성)160~168cm/48~63kg, (남성)172~185cm/62~84kg
    Relevant information:{history} size contents:{input} result:"""
        sizeReco = "구매자 통계를 기반으로 사이즈를 추천드리겠습니다.\n" + gpt(str(size_reco), size_reco_template)
    else:
        sizeReco = "아직 구매자가 없어 추천이 어렵습니다." # 추후 사이즈표 제시

    return {"size_reco": sizeReco}
# size = ['여자 164cm/68kg 적당함 M Size', '여자 164cm/51kg 적당함 M Size', '여자 159cm/47kg 적당함 M Size', '여자 165cm/62kg 큼 M Size', '남자 174cm/77kg 적당함 M Size', '여자 167cm/63kg 적당함 M Size', '여자 167cm/49kg 적당함 S Size', '여자 166cm/54kg 적당함 M Size', '여자 164cm/50kg 큼 M Size', '여자 160cm/50kg 큼 M Size', '여자 160cm/50kg 적당함 M Size', '여자 170cm/60kg 적당함 M Size', '여자 165cm/60kg 적당함 M Size', '여자 170cm/56kg 적당함 M Size', '여자 160cm/48kg 적당함 M Size', '여자 163cm/52kg 큼 M Size', '여자 163cm/77kg 적당함 L Size', '여자 169cm/60kg 적당함 M Size', '여자 173cm/60kg 적당함 L Size', '여자 160cm/45kg 적당함 S Size', '여자 170cm/47kg 적당함 M Size', '여자 155cm/55kg 적당함 L Size', '여자 163cm/63kg 큼 L Size', '여자 165cm/55kg 적당함 M Size', '여자 162cm/78kg 작음 M Size', '여자 160cm/50kg 큼 S Size', '여자 155cm/47kg 적당함 S Size', '남자 176cm/66kg 적당함 L Size', '남자 173cm/64kg 적당함 L Size', '남자 171cm/64kg 적당함 M Size']
# print(SizeReco(apikey, size))