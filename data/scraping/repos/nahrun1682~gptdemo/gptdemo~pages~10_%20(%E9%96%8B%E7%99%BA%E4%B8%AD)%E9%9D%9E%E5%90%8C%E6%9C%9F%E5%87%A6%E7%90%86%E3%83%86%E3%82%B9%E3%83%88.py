# import asyncio
# from langchain.chat_models import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
# from langchain.memory import ConversationBufferMemory
# import streamlit as st
# import openai
# import os
# from dotenv import load_dotenv


# # .envファイルの読み込み
# load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
# openai.api_key = os.environ["OPENAI_API_KEY"]

# st.set_page_config(layout="wide")

# async def manual_run(panel, llm, prompt_template, memory, question):
#     message = prompt_template.format_prompt(
#         human_input=question, 
#         chat_history=memory.load_memory_variables({})["chat_history"]
#     )
#     with panel:
#         with st.chat_message("user"):
#             st.markdown(question)
#         with st.chat_message("assistant"):
#             container = st.empty()

#     response = ""
#     async for chunk in llm.astream(message):
#         response += chunk.content
#         container.markdown(response)

#     memory.save_context({"input": question}, {"output": response})
#     return response

# async def logic(panel,ChatMessage):
#     llm = ChatOpenAI(model="gpt-3.5-turbo-0613", temperature=0.2, max_tokens=512, streaming=True)
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         HumanMessagePromptTemplate.from_template("{human_input}")
#     ])
#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     # if "messages" not in st.session_state:
#     #     st.session_state["messages"] = [ChatMessage(role="assistant", content="なんでも聞いてね(非同期処理版)")]

#     # for msg in st.session_state.messages:
#     #     st.chat_message(msg.role).write(msg.content)
        
#     if prompt_input := st.chat_input():
#         # st.session_state.messages.append({
#         #     "role": "user",
#         #     "content": prompt_input
#         # })
#         st.session_state.messages.append(ChatMessage(role="user", content=prompt))
#         ans = await manual_run(panel, llm, prompt, memory, prompt_input)
#         st.session_state.messages.append(ChatMessage(role="assistant", content=ans))
#         print(f'asyns ans:{ans}')

# def main():
#     st.title("Simple Chat with GPT")
    
#     if "messages" not in st.session_state:
#         st.session_state["messages"] = [ChatMessage(role="assistant", content="なんでも聞いてね")]

#     for msg in st.session_state.messages:
#         st.chat_message(msg.role).write(msg.content)
        
#     asyncio.run(logic(st.container(),ChatMessage))

# if __name__ == "__main__":
#     main()