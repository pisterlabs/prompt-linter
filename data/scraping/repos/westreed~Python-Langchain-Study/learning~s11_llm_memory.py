from langchain import PromptTemplate, LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from key import APIKEY

KEY = APIKEY()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat = ChatOpenAI(openai_api_key=KEY.openai_api_key, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

speech_list = [
    "당신은 면접관입니다. 지금부터, 회사인재를 뽑기 위한 면접관으로서 말투와 사고를 가져야 합니다. 개발자 직군과 관련해서 면접질문을 3개 생성해주세요.",
    "아까 생성했던 질문 3개 중 1개를 뽑아서 적어주세요.",
]

prompt = PromptTemplate(input_variables=["chat_history", "text"], template="{chat_history}\n{text}")
chain = LLMChain(llm=chat, memory=memory, prompt=prompt)

for speech in speech_list:
    chain.run(text=speech)
    print()

history = memory.load_memory_variables({})
print(f"History : {history}")

# 1. "개발자로서의 역량을 평가하기 위해 어떤 개발 프로젝트를 진행해 본 경험이 있나요? 해당 프로젝트에서 어떤 기술적 도전과정을 겪었으며, 어떻게 해결했나요?"
# 2. "개발자로서 협업이 필요한 상황에서 어떤 방식으로 팀원들과 의사소통하고 협업하는지 알려주세요. 특히, 어려움을 겪을 때 어떻게 대처하나요?"
# 3. "새로운 기술이나 개발 트렌드에 대한 관심과 학습 태도는 어떠신가요? 현재 어떤 기술적 이슈에 관심을 가지고 있으며, 이를 학습하거나 적용하기 위해 어떤 노력을 기울이고 계신지 알려주세요."

# "전 세계적으로 가장 많이 팔리는 음식은 무엇인가요?"

# History : {'history': [HumanMessage(content='당신은 면접관입니다. 지금부터, 회사인재를 뽑기 위한 면접관으로서 말투와 사고를 가져야 합니다. 개발자 직군과 관련해서 면접질문을 3개 생성해주세요.', additional_kwargs={}, example=False), AIMessage(content='1. "개발자로서의 역량을 평가하기 위해 어떤 개발 프로젝트를 진행해 본 경험이 있나요? 해당 프로젝트에서 어떤 기술적 도전과정을 겪었으며, 어떻게 해결했나요?"\n2. "개발자로서 협업이 필요한 상황에서 어떤 방식으로 팀원들과 의사소통하고 협업하는지 알려주세요. 특히, 어려움을 겪을 때 어떻게 대처하나요?"\n3. "새로운 기술이나 개발 트렌드에 대한 관심과 학습 태도는 어떠신가요? 현재 어떤 기술적 이슈에 관심을 가지고 있으며, 이를 학습하거나 적용하기 위해 어떤 노력을 기울이고 계신지 알려주세요."', additional_kwargs={}, example=False), HumanMessage(content='아까 생성했던 질문 3개 중 1개를 뽑아서 적어주세요.', additional_kwargs={}, example=False), AIMessage(content='"전 세계적으로 가장 많이 팔리는 음식은 무엇인가요?"', additional_kwargs={}, example=False)]}


# prompt1 = PromptTemplate(
#     input_variables=["count"],
#     template="""
#     당신은 면접관입니다. 지금부터, 회사인재를 뽑기 위한 면접관으로서 말투와 사고를 가져야 합니다.
#     개발자 직군과 관련해서 면접질문을 {count}개 생성해주세요.
#     """
# )
# prompt2 = PromptTemplate(
#     input_variables=["chain1"],
#     template="{chain1}\n아까 생성했던 질문 3개 중 1개를 뽑아서 적어주세요."
# )
#
# chain1 = LLMChain(llm=chat, memory=memory, prompt=prompt1, output_key="chain1")
# chain2 = LLMChain(llm=chat, memory=memory, prompt=prompt2)
#
# overall_chain = SequentialChain(
#     chains=[chain1, chain2],
#     input_variables=["count"],
#     output_variables=["chain1"],
#     verbose=True
# )
#
# overall = overall_chain({"count":"3"})