from kanao.modules.pdf_module import process_pdf
from kanao.modules.word_module import process_word_doc
from kanao.modules.txt_module import process_txt
from kanao.modules.url_module import process_url
from kanao.modules.get_api_key import get_api_key
from kanao.modules.write_api_key import write_api_key
import openai
import os


chat_history = []

class Kanao:
    def __init__(self,api_key=None):
        self._chain = None
        if api_key is None:
            raise ValueError("Please provide the OpenAI API key")
        else:
            write_api_key(api_key)
            os.environ['OPENAI_API_KEY'] = api_key

    def _call_openai_api(self, prompt):
        openai.api_key = get_api_key()
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    
    def train_on_pdf(self, pdf_file):
        self._chain = process_pdf(pdf_file)

    def train_on_word(self, word_file):
        self._chain = process_word_doc(word_file)

    def train_on_url(self, url):
        self._chain = process_url(url)

    def train_on_txt(self, txt_file):
       self._chain = process_txt(txt_file)

    def _store_chat_history(self, question, answer):
        global chat_history
        chat_history += [(question, answer)]

    def _get_chat_history(self):
        global chat_history
        return chat_history



    def generate_response(self, prompt):
        if self._chain is None:
            # Call ChatGPT directly if no model is trained
            answer = self._call_openai_api(prompt)
            # Store the conversation history
            self._store_chat_history(prompt, answer)
            return answer

        # Retrieve the chat history
        history = self._get_chat_history()

        # Include the chat history in the inputs
        inputs = {"question": prompt, 'chat_history': history}

        result = self._chain(inputs, return_only_outputs=True)
        answer = result["answer"]

        # Store the conversation history
        self._store_chat_history(prompt, answer)

        return answer

