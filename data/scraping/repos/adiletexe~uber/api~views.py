from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# location of the pdf file/files.
doc_reader = PdfReader('content/agrimotrainpdf.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(doc_reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,  # striding over the text
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

template = """You are a professional agriculture analyser.

Given the following extracted parts of a examples of analysis from the document and an input, create a final answer. You should give very specific analysis, even if it is wrong. 

{context}

{chat_history}
Human: {human_input}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "human_input"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")

user_conversations = {}

class ChatbotView(APIView):
    def post(self, request):
        user_email = request.data.get('email')
        user_input = request.data.get('user_input')
        if not user_email or not user_input:
            return Response({'error': 'email and user_input parameters are required'}, status=status.HTTP_400_BAD_REQUEST)

        # Get or create a conversation for the user
        if user_email not in user_conversations:
            user_conversations[user_email] = load_qa_chain(
                OpenAI(temperature=0), chain_type="stuff", memory=memory, prompt=prompt
            )
        print(user_conversations)
        conversation = user_conversations[user_email]

        docs = docsearch.similarity_search(user_input)

        # Get AI response
        ai_response = conversation({"input_documents": docs, "human_input": user_input}, return_only_outputs=True)

        return Response({'response': ai_response}, status=status.HTTP_200_OK)

# load_dotenv()
#
# if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
#     print("No api key")
#     exit(1)
# else:
#     print("my api key:", os.getenv("OPENAI_API_KEY"))
#
# llm = OpenAI(model_name="text-davinci-003")
# user_conversations = {}  # Dictionary to store conversations for each user
#
# class ChatbotView(APIView):
#     def post(self, request):
#         user_email = request.data.get('email')
#         user_input = request.data.get('user_input')
#         if not user_email or not user_input:
#             return Response({'error': 'email and user_input parameters are required'}, status=status.HTTP_400_BAD_REQUEST)
#
#         # Get or create a conversation for the user
#         if user_email not in user_conversations:
#             user_conversations[user_email] = ConversationChain(
#                 llm=llm,
#                 memory=ConversationBufferMemory(),
#                 verbose=True,
#             )
#
#         conversation = user_conversations[user_email]
#
#         # Get AI response
#         ai_response = conversation.predict(input=user_input)
#
#         return Response({'response': ai_response}, status=status.HTTP_200_OK)
