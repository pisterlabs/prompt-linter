import json
import os
from src.config import Config
from src.utils.aws_s3 import AWSService
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.chains import create_extraction_chain, LLMChain, SimpleSequentialChain
from langchain.chains.summarize import load_summarize_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai

# Set the OpenAI API key from the environment
os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY


class DataExtractor:
    def __init__(self, api_key=Config.OPENAI_API_KEY):
        self.llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo",
                              openai_api_key=api_key, max_tokens=3000)

    def extract_from_bank_statement(self, data):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "address": {"type": "string"},
                "opening_balance": {"type": "integer"},
                "closing_balance": {"type": "integer"},
                "income/salary_total": {"type": "integer"},
                "Outgoings/Expenses_total": {"type": "integer"}
            }
        }
        chain = create_extraction_chain(schema, self.llm)
        return chain.run(data)

    def query_documents(self, loaders, phrases):
        outputs = []
        prompt_template = "Extract relevant information about the following phrases: {}"
        prompt = prompt_template.format(', '.join(phrases))

        for loader in loaders:
            index = VectorstoreIndexCreator().from_loaders([loader])
            outputs.append(index.query(prompt))

        return outputs

    def get_query_from_url(self, urls, phrases):
        return self.query_documents([WebBaseLoader(url) for url in urls], phrases)

    def get_query_from_pdfs(self, file_details, phrases):
        outputs = []

        for detail in file_details:
            AWSService().download_file(detail['folder_id'], detail['doc_name'])
            split_name = detail['doc_name'].split('/')[-1]
            file_path = f'/opt/src/documents/{split_name}'
            loader = PyPDFLoader(file_path)
            loader.load()
            outputs.append(self.query_documents([loader], phrases))
            os.remove(file_path) if os.path.exists(file_path) else None

        return outputs

    def summarize_data_extract(self, output):
        return self.perform_openai_completion(output, "summarize the outputs and return in a structured format like JSON")

    def custom_template_data_extract(self, web_scraped_text, phrases):
        return self.perform_openai_completion(web_scraped_text, f"extract relevant information about the following phrases {', '.join(phrases)} in a structured format like JSON")

    def perform_openai_completion(self, text, instruction):
        get_max_tokens = 4097 - len(text)
        prompt = f"Given the text: '{text}'\n\n{instruction}"
        openai.api_key = Config.OPENAI_API_KEY
        response = openai.Completion.create(engine="text-davinci-003", prompt=prompt,
                                            temperature=0.2, max_tokens=get_max_tokens)
        return json.loads(response.choices[0].text.strip())

    def reduce_summarize_pdf_data(self, data):
        chain = load_summarize_chain(self.llm, chain_type='map_reduce')
        return chain.run(data)

    def chunk_data(self, data):
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=1)
        return splitter.split_text(data)


class TemplateFormatter:
    def format_from_json(self, json_data):
        template_parts = [
            "Please extract the following information:",
            f" Focus on a {json_data.get('selectedCuisine')} inspired meal." if json_data.get('selectedCuisine') != 'Pot Luck' else '',
            " Please include a calorie breakdown" if json_data.get('selectedCalorie') == 'Yes' else '',
            f" Please make sure the recipe is {', '.join(json_data.get('selectedDietry'))}" if json_data.get('selectedDietry') else ''
        ]
        return ChatPromptTemplate.from_template(''.join(template_parts))

    def format_second_template(self, json_data):
        created_template = '''
        Make sure this recipe is in the following format:
        - Ingredients that will expire soon and are in the recipe have been specified
        - Ingredients that are in the pantry (other items) and are in the recipe have been specified
        - Any ingredients not owned yet are displayed in a shopping list
        - Full cooking instructions have been supplied
        ''' + (' A full calorie breakdown has been included' if json_data.get('selectedCalorie') == 'Yes' else '')
        return ChatPromptTemplate.from_template(created_template)


class ChatResponder:
    def __init__(self, temperature=0.0):
        self.chat = OpenAI(temperature=temperature)

    def get_response(self, prompt, prompt2):
        chain1 = LLMChain(llm=self.chat, prompt=prompt)
        chain2 = LLMChain(llm=self.chat, prompt=prompt2)
        seq_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
        return seq_chain.run('can you give me a recipe')
