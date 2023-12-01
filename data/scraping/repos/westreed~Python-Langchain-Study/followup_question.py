from langchain import PromptTemplate, LLMChain
from custom_callback_handler import CustomCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from key import APIKEY

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

KEY = APIKEY()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chat = ChatOpenAI(
    openai_api_key=KEY.openai_api_key,
    streaming=True,
    callbacks=[CustomCallbackHandler()]
)

evaluate_answer_system_message = SystemMessagePromptTemplate.from_template(
    """
    당신은 면접관입니다.
    면접관으로써 면접자의 답변내용을 분석하고 좋은 점과 아쉬운 점으로 나눠서
    평가를 진행해주세요. 이때, 아쉬운 점은 정말로 아쉬웠던 부분만 언급하세요.
    
    그리고, 답변 내용은 아래의 양식을 지켜서 작성해주세요. 양식 이외의 말은 하지 않습니다.
    ```
    좋은 점:
    - 좋은 점 내용
    
    아쉬운 점:
    - 아쉬운 점 내용
    ```
    """
)

followup_question_system_message = SystemMessagePromptTemplate.from_template(
    """
    당신은 면접관입니다.
    면접 질문과 답변을 읽고, `Follow up Question`이 필요하다고 판단되면 아쉬운 점에서 물어보고 싶은 추가질문을 작성하고, 필요없다고 판단하면 `괜찮습니다.`라고 작성하세요. 그리고 다음 양식에 맞춰서 작성하세요.
    ```
    추가질문:
    추가질문 내용
    ```
    """
)

prompt = ChatPromptTemplate.from_messages([
    evaluate_answer_system_message,
    HumanMessagePromptTemplate.from_template(
        """
        {chat_history}
        {answer}
        """
    )
])
evaluate_chain = LLMChain(llm=chat, memory=memory, prompt=prompt)

prompt2 = ChatPromptTemplate.from_messages([
    followup_question_system_message,
    HumanMessagePromptTemplate.from_template(
        """
        {chat_history}
        {text}
        """
    )
])

followup_question_chain = LLMChain(llm=chat, memory=memory, prompt=prompt2)

# evaluate_chain.run("""
# 면접질문. 가장 자신 있는 컴퓨터 이론 하나를 설명해보세요.
# 면접답변. HTTP 통신은 데이터를 별도의 암호화 없이 통신합니다. 이때, 연락처나 비밀번호와 같은 중요정보를 제 3자가 중간에 탈취하여 조회하는 보안적인 문제가 있습니다. 이제 여기서 HTTP 통신에 암호화가 추가된 형태가 HTTPS 입니다.
# """)
evaluate_chain.run("""
면접질문. 본인은 리더형인가요, 팔로워형인가요?
면접답변. 저는 리더형에 가깝다고 생각합니다. 리더의 역할은 팀이 나아가야할 길을 헤매지 않고, 목표에 도달할 수 있게 이끌어주는 것이라고 생각합니다. 한 예로 졸업작품을 위해 팀을 이루게 되었는데, 협업개발을 해본 경험도 없었고 처음부터 끝까지 무언가를 만들어본 경험이 없었던 저희는 무엇부터 해야할지 모른채 방황했었습니다. 저는 인터넷조사를 통해 필요한 역할과 해당 역할에 적합한 팀원을 분배하였고, 각자 역할에 대해서 조사하여 팀내에서의 할일을 나누고 개발을 시작했습니다. 그 결과 학과 내에서 2등을 수상할 수 있었습니다.
""")
followup_question_chain.run("면접관의 판단에 따라 추가질문을 작성하세요.")

# Result
# 좋은 점:
# - 면접자는 HTTP와 HTTPS의 차이점을 알고 있으며, HTTPS가 중요 정보의 보안을 위해 암호화를 추가한 것을 설명하였습니다.
#
# 아쉬운 점:
# - 면접자의 답변은 전반적으로 부족한 내용을 담고 있습니다. HTTP 통신에 대한 이해가 부족하고, HTTPS에 대한 설명도 부족한 것으로 보입니다. 면접자는 좀 더 자세하게 설명할 수 있는 내용을 준비해야 할 것입니다.
# 추가질문:
# 면접자는 HTTP와 HTTPS의 차이점을 언급했지만, 좀 더 자세히 설명할 수 있는 내용이 있다면 알려주실 수 있을까요?


# 좋은 점:
# - 면접자는 리더형에 가깝다고 생각하고 있으며, 리더의 역할을 잘 이해하고 있다는 점이 좋습니다.
# - 면접자는 팀 프로젝트에서 리더로서 필요한 역할 분배와 팀원들의 역할에 대한 조사를 통해 팀을 이끌었고, 그 결과로 성과를 얻을 수 있었다는 경험을 언급하였습니다.
#
# 아쉬운 점:
# - 면접자는 리더형에 가깝다고 생각하고 있다고 말하였지만, 실제로 리더로서의 활동 경험에 대해서는 자세히 언급하지 않았습니다. 좀 더 구체적인 예시나 경험에 대해서 언급하면 더 좋았을 것 같습니다.
# - 면접자의 리더십 경험은 한 번의 팀 프로젝트를 통해 이루어진 것으로 보입니다. 다른 상황에서의 리더십 경험에 대해서도 언급할 수 있다면 더 좋았을 것입니다.
# 추가질문:
# 면접자님께서는 한 번의 팀 프로젝트를 통해 리더로서의 경험을 얻었다고 말씀하셨는데, 다른 상황에서도 리더십 경험이 있으신가요?
