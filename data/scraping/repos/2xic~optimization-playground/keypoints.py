import textract
from dotenv import load_dotenv
load_dotenv()

from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversation.memory import ConversationSummaryMemory
import json

class KeyIdeas:
    def __init__(self) -> None:
        template = """
        You are given some machine learning papers, you should read them CAREFULLY and give out the key ideas.
 
        You will interlay create a summary since I give you parts of the document in batches.

        Create the summary with bullet points!

        Previous summary: {history}

        New text batch: {text}

        New summary: 
        """

        prompt = PromptTemplate(
            input_variables=['text', 'history'], 
            template=template
        )
        llm = OpenAI(temperature=0)
        self.chatgpt_chain = LLMChain(
            llm=llm, 
            prompt=prompt, 
            verbose=True, 
            memory=ConversationSummaryMemory(llm=llm),
        )

    def summary(self, text):
        output = self.chatgpt_chain.predict(
            text=text
        )
        return output
    
if __name__ == "__main__":
    text =  textract.process('./1801.06146.pdf')
    batch_size = 3700
    model = KeyIdeas()
    for i in range(0, len(text), batch_size):
        batch = text[i:i+batch_size]
        if not len(batch):
            break
        response = model.summary(
            batch.decode('utf-8').replace("\n", " ").strip()
        )
        print(response)
    
