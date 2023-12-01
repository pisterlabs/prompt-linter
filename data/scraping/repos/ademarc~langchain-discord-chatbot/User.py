from langchain.chains import ConversationChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from config import get_openai_key, setup_logging
from memory import get_user_memory, save_user_memory
from langchain.callbacks import get_openai_callback

# Set up environment variables
OPENAI_API_KEY = get_openai_key()

# Set up logging
logger = setup_logging()

# Initialize LLMs
chat1 = ChatOpenAI(temperature=1)
chat0 = ChatOpenAI(temperature=0)

class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memory = None

    def ask_question(self, question):
        # Set up prompt template with system message for conversation chain
        chain_prompt = PromptTemplate(
            input_variables=['chat_history', 'input'],
            output_parser=None,
            partial_variables={},
            template='You are πGPT, an AI chatbot that assists humans.\n\nPrevious messaages in your conversation with human:\n{chat_history}\n\n(input is from Human, output is from AI)\n\nHuman: {input}\nAI:',
            template_format='f-string',
            validate_template=True)
        
        with get_openai_callback() as cb:
            # Initialize conversation chain with memory
            self.memory = get_user_memory(self.user_id)
            conversation = ConversationChain(llm=chat1, memory=self.memory, verbose=False, prompt=chain_prompt)                    
            result = conversation.run(input=question)
            save_user_memory(self.user_id, self.memory)
            logger.info(f'Generated response for user {self.user_id} with {cb.total_tokens} tokens')
        return result
