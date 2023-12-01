import streamlit as st
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import tool
import json


@tool("Document search")
def document_search(query: str) -> str:
	# this is the prompt to the tool itself
	"Use this function first to search for documents pertaining to the query before going into the internet"
	docs = st.session_state.vs.similarity_search(query)
	docs = docs[0].page_content
	json_string = json.dumps(docs, ensure_ascii=False, indent=4)
	return json_string

# combine vector store and internet search
def smart_agent():
	
	msgs = StreamlitChatMessageHistory()
	memory = ConversationBufferMemory(
		chat_memory=msgs,
		return_messages=True,
		memory_key="chat_history",
		output_key="output",
	)
	if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
		msgs.clear()
		msgs.add_ai_message("How can I help you?")
		st.session_state.steps = {}

	avatars = {"human": "user", "ai": "assistant"}
	for idx, msg in enumerate(msgs.messages):
		with st.chat_message(avatars[msg.type]):
			# Render intermediate steps if any were saved
			for step in st.session_state.steps.get(str(idx), []):
				if step[0].tool == "_Exception":
					continue
				with st.status(
					f"**{step[0].tool}**: {step[0].tool_input}", state="complete"
				):
					st.write(step[0].log)
					st.write(step[1])
			st.write(msg.content)

	if prompt := st.chat_input(placeholder="Enter a query on the Internet"):
		st.chat_message("user").write(prompt)

		llm = ChatOpenAI(
			model_name=st.secrets["default_model"], openai_api_key=st.session_state.api_key, streaming=True
		)
		if st.session_state.vs:
			tools = [document_search, DuckDuckGoSearchRun(name="Internet Search")]
		else:
			tools = [DuckDuckGoSearchRun(name="Internet Search")]
		chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)
		executor = AgentExecutor.from_agent_and_tools(
			agent=chat_agent,
			tools=tools,
			memory=memory,
			return_intermediate_steps=True,
			handle_parsing_errors=True,
		)
		with st.chat_message("assistant"):
			st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
			response = executor(prompt, callbacks=[st_cb])
			st.write(response["output"])
			st.session_state.steps[str(len(msgs.messages) - 1)] = response[
				"intermediate_steps"
			]