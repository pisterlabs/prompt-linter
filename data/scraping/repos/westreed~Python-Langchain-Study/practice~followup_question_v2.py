from langchain import LLMChain
from custom_callback_handler import CustomCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from pprint import pprint
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
    You are an interviewer.
    As an interviewer, please analyze the interviewee's response and provide evaluations by dividing them into positive aspects and areas for improvement. When mentioning areas for improvement, please focus only on the truly disappointing aspects. Please write in Korean.

    Also, please adhere to the following format when providing the response. Do not include any other information beyond the given format.
    ```
    "좋은 점":
    - Content of positive aspects

    "아쉬운 점":
    - Content of areas for improvement
    ```
    """
)

prompt = ChatPromptTemplate(
    messages=[
        evaluate_answer_system_message,
        HumanMessagePromptTemplate.from_template(
            """
            {chat_history}
            {answer}
            """
        )],
    input_variables=["chat_history", "answer"],
)

evaluate_chain = LLMChain(llm=chat, memory=memory, prompt=prompt)

followup_question_system_message = SystemMessagePromptTemplate.from_template(
    """
    You are an interviewer.
    Please read the interview question and response. If you determine that a `Follow up Question` is necessary, write the additional question you would like to ask in the areas for improvement. If you determine that it is not necessary, write `It's okay`. Also, please adhere to the following format when providing your response. Please write in Korean.

    ```
    "심화질문":
    - Content of follow up question
    ```
    """
)

prompt2 = ChatPromptTemplate(
    messages=[
        followup_question_system_message,
        HumanMessagePromptTemplate.from_template(
            """
            {chat_history}
            {input_text}
            """
        )],
    input_variables=["chat_history", "input_text"]
)

followup_question_chain = LLMChain(llm=chat, memory=memory, prompt=prompt2)

output = evaluate_chain("""
면접질문. 본인은 리더형인가요, 팔로워형인가요?
면접답변. 저는 리더형에 가깝다고 생각합니다. 리더의 역할은 팀이 나아가야할 길을 헤매지 않고, 목표에 도달할 수 있게 이끌어주는 것이라고 생각합니다. 한 예로 졸업작품을 위해 팀을 이루게 되었는데, 협업개발을 해본 경험도 없었고 처음부터 끝까지 무언가를 만들어본 경험이 없었던 저희는 무엇부터 해야할지 모른채 방황했었습니다. 저는 인터넷조사를 통해 필요한 역할과 해당 역할에 적합한 팀원을 분배하였고, 각자 역할에 대해서 조사하여 팀내에서의 할일을 나누고 개발을 시작했습니다. 그 결과 학과 내에서 2등을 수상할 수 있었습니다.
""")

followup_question_chain("""
Please provide a Follow up question based on the interviewer's judgment. If there are no areas for improvement, you don't need to provide a Follow up question.
""")

pprint(memory.dict())
