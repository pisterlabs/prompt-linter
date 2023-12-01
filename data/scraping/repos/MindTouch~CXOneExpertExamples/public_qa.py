# ask a question using chatgpt of a public page
from langchain.document_loaders import WebBaseLoader
from langchain.indexes import VectorstoreIndexCreator

# Document loader. A page in which to ask questions about
loader = WebBaseLoader("https://success.mindtouch.com/Integrations/Touchpoints/1-Intro_to_Touchpoints")

# Index that wraps above steps
index = VectorstoreIndexCreator().from_loaders([loader])

# Question-answering
question = "How to get started with Touchpoints?"
answer =  index.query(question)
print(answer)
