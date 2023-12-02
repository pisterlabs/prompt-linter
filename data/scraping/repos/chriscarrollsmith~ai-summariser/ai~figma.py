import os

from dotenv import load_dotenv
from langchain.document_loaders.figma import FigmaFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()

OPEN_API_KEY = os.getenv('OPENAI_API_KEY')

def figma_to_code():
    figma_loader = FigmaFileLoader(
        os.getenv('FIGMA_ACCESS_TOKEN'),
        '211-1478',
        'nVZYVuthGsKpVBgiXS6q7Z'
    )

    index = VectorstoreIndexCreator().from_loaders([figma_loader])
    figma_doc_retriever = index.vectorstore.as_retriever()

    return figma_doc_retriever

def generate_code(figma_doc_retriever, human_input):
    system_prompt_template = """You are an expert coder. Use the provided design context to create idomatic HTML/CSS code as possible based on the user request.
    Everything must be inline in one file and your response must be directly renderable by the browser.
    Figma file nodes and metadata: {context}"""

    human_prompt_template = "Code the {text}. Ensure it's mobile responsive"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_prompt_template)
    # delete the gpt-4 model_name to use the default gpt-3.5 turbo for faster results
    gpt_4 = ChatOpenAI(temperature=.02, model_name='gpt-4', openai_api_key=OPEN_API_KEY)
    
    relevant_nodes = figma_doc_retriever.get_relevant_documents(human_input)
    conversation = [system_message_prompt, human_message_prompt]
    chat_prompt = ChatPromptTemplate.from_messages(conversation)
    response = gpt_4(chat_prompt.format_prompt( 
        context=relevant_nodes, 
        text=human_input).to_messages())
    return response

if __name__ == "__main__":
    figma_doc_retriever = figma_to_code()
    result = generate_code(figma_doc_retriever, 'full page')
    # print(figma_doc_retriever.get_relevant_documents('full page'))

    with open('test.html', 'w') as file:
        file.write(str(result))