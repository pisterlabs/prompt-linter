from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

whatismyhobby = '내가 좋아하는 취미는 뭐야'

llm = OpenAI()
result = llm.predict(whatismyhobby)
print(result) # 나의 취미는 음악감상, 영화감상, 스포츠, 독서 등이 있습니다.

chat_model = ChatOpenAI()
result = chat_model.predict(whatismyhobby)
print(result) # 저는 답변을 위해 사용자의 성향과 관심사를 알 수 없으므로 사용자가 좋아하는 취미를 알려주셔야 합니다. 어떤 취미에 대해 이야기해드릴까요? 예를 들어, 음악, 스포츠, 요리, 여행, 독서 등 다양한 취미 중 어떤 것에 관심이 있으신가요?