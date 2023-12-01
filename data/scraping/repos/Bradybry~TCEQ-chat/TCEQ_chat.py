global rule_path, preamble, model_params

import gradio as gr
from langchain.schema import AIMessage
import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config import ANTHROPIC_API_KEY, OPENAI_API_KEY, preamble, model_params
from expert import LanguageExpert

rule_path = './rule_pkls/217.pkl'

class ChatbotApp:
    def __init__(self):
        self.history = []
        self.embedding_model = "text-embedding-ada-002"
        self.df = pd.read_pickle(rule_path)
        self.chat = LanguageExpert(preamble=preamble, model_params=model_params)

    def get_embedding(self, message):
        from openai.embeddings_utils import get_embedding
        user_input = np.array(get_embedding(message, engine=self.embedding_model))
        user_input = user_input.reshape(1, -1)
        return user_input

    def predict(self, user_message):
        messages = []  # Initialize with your system message
        for message in self.history:
            if message is None:
                break
            ai_message = AIMessage(content=message[1])
            human_message = HumanMessage(content=message[0])
            messages.extend([human_message, ai_message])

        user_message = HumanMessage(content=user_message)
        user_input = self.hyde(user_message.content)
        self.df["cos_dist"] = cdist(np.stack(self.df.embeddings), user_input, metric="cosine")
        self.df.sort_values("cos_dist", inplace=True)

        res = 'Use the following relevant rules to answer the user\'s question:\n\n'
        for row in self.df.head(10).index:
            res += f"Rule: {row}\nText: {self.df.loc[row, 'rule']}\n---\n"
        context_message = AIMessage(content=res)

        messages.extend([context_message, user_message])

        response = self.chat(messages).content.strip()  # You need to define the 'chat' function or use the appropriate model
        self.history.append([user_message.content, response])
        return self.history
    def hyde(self, user_message):
        hyde_message = f"""You will be given a sentence.
If the sentence is a question, convert it to a plausible answer. 
If the sentence does not contain a question, 
just repeat the sentence as is without adding anything to it.

Examples:
- What is the minimum pressure that public water systems must maintain? --> All public water systems must be operated to provide a minimum pressure of 35 psi throughout the distribution system under normal operating conditions, and at least 20 psi during emergencies such as firefighting.
- Are there any specific requirements for service connections with booster pumps? --> Sservice connections that require booster pumps taking suction from the public water system lines must be equipped with automatic pressure cutoff devices so that the pumping units become inoperative at a suction pressure of less than 20 psi.
- Public water systems must be operated efficiently. --> Public water systems must be operated efficiently.
- What are the flushing velocity requirements for a force main? --> A minimum flushing velocity of 5.0 feet per second or greater must occur in a force main at least twice daily.

Sentence:
- {user_message} --> """
        message = HumanMessage(content=hyde_message)
        out = self.chat.chat([message]).content.strip()
        print(out)
        embedding = self.get_embedding(out)
        return embedding
    def run(self):      
        with gr.Blocks() as demo: 

            # creates a new Chatbot instance and assigns it to the variable chatbot.
            chatbot = gr.Chatbot() 

            # creates a new Row component, which is a container for other components.
            with gr.Row(): 
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
            txt.submit(self.predict, txt, chatbot) 
            txt.submit(None, None, txt, _js="() => {''}")
            with gr.Row():
                save_chat_submit = gr.Button(value="Save Chat History").style(full_width=False)
                save_chat_submit.click(self.save_history)
                reset_chat = gr.Button(value="Reset Chat").style(full_width=False)
                reset_chat.click(self.restart)

        demo.launch() 

    def restart(self):
        self.__init__()
    def save_history(self):
        '''This methods takes the conversation history and saves in a pickled file with information about the time and date of the conversation.
        '''        
        import pickle
        import datetime
        now = datetime.datetime.now()
        filename = 'conversation_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self.history, f)
        print('Saved conversation to ' + filename)
def main():
    chatbot_app = ChatbotApp()
    chatbot_app.run()
if __name__ == '__main__':
    main()   