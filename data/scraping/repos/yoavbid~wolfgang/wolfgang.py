from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from utils import get_default_logger
from similarity import load_similarity_index

class Wolfgang():
    def __init__(self, prompt_path, index_path, model = 'gpt-3.5-turbo', temperature=0.4, 
                 num_contexts=8, state_json={}, logger=None):
        if logger is None:
            self.logger = get_default_logger()
    
        self.store = load_similarity_index(index_path)
        
        with open(prompt_path, "r") as f:
            promptTemplate = f.read()

            system_message_prompt = SystemMessagePromptTemplate.from_template(
                template=promptTemplate, input_variables=["history", "context", "recent_level"])
            
        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

        chat = ChatOpenAI(temperature=temperature, model=model)

        self.chain = LLMChain(llm=chat, prompt=chat_prompt)
        
        self.num_contexts = num_contexts
        
        self.recent_level = state_json.get("recent_level", "")
        self.history = state_json.get("history", [])
        
    def ask_question(self, question):
        try:
            history_for_context = self.history[-5:]
        except IndexError:
            history_for_context = self.history
            
        search_text = question + ' history: ' + ' '.join(history_for_context)
        if self.recent_level is not None:
            search_text += ' context: ' + self.recent_level
        
        contexts = []
        contexts_no_index = []
        if self.num_contexts > 0:
            docs = self.store.similarity_search(search_text, k=self.num_contexts)
            for i, doc in enumerate(docs):
                contexts.append(f"Context {i}:\n{doc.page_content}")
                contexts_no_index.append(doc.page_content)
        
        answer = self.chain.run(input=question, context="\n\n".join(contexts), 
                                recent_level=self.recent_level, history=self.history)
        
        self.history.append(f"Human: {question}")
        self.history.append(f"Bot: {answer}")
        
        return answer, contexts_no_index
    
    def get_state_json(self):
        return {
            'history': self.history,
            'recent_level': self.recent_level
        }