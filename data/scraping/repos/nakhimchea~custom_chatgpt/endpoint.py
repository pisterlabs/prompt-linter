import re

from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, load_index_from_storage, StorageContext
from langchain.chat_models import ChatOpenAI
from firestore_service import *


class Endpoint:
    def __init__(self):
        storage_context = StorageContext.from_defaults(persist_dir='./storage')
        self.brain = load_index_from_storage(storage_context)

    @staticmethod
    def construct_index(directory_path) -> GPTVectorStoreIndex:
        max_input_size = 2048
        num_outputs = 512
        chunk_overlap_ratio = 0.2
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, chunk_overlap_ratio, chunk_size_limit=chunk_size_limit)

        # noinspection PyTypeChecker
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.4666, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

        documents = SimpleDirectoryReader(directory_path).load_data()

        brain = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        brain.storage_context.persist()

        return brain

    def chatbot(self, input_text) -> str:
        response = self.brain.as_query_engine().query(input_text).response.strip()
        # response1 = brain.as_query_engine().query('Account: ' + input_text).response.strip()
        # if 'account' not in input_text:
        #     response1.replace(' account', '')

        # print('Res1: ', response) #->Open on Development
        # print('Res2: ', response1)

        number = re.findall(r'\d+', response)
        # number1 = re.findall(r'\d+', response1)

        answers = []

        try:
            string = ''
            if ('hello' in response.lower()) or ('hi' in response.lower()):
                string = '0 '
            else:
                for i in range(len(number)):
                    string += '{0} '.format(int(number[i]))
            if string != '':
                answers.append(string)

        except:
            print('Cannot extract number from Number')

        # try:
        #     string = ''
        #     for i in range(len(number1)):
        #         string += '{0} '.format(int(number1[i]))
        #     if string != '':
        #         answers.append(string)
        # except:
        #     print('Cannot extract number from Number1')

        if len(answers) == 0:
            FireStore.failed_faq(input_text)

        return answers[-1] if len(answers) != 0 else '69'
