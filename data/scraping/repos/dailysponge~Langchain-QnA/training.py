from llama_index import LLMPredictor, PromptHelper, ServiceContext, Document
from util.index_setting import create_index
from util.generate_nodes import convert_url_to_text_chunks, generate_nodes_from_document
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Define LLM
llm_predictor = LLMPredictor(llm=ChatOpenAI(
    temperature=0, max_tokens=500, model_name="gpt-3.5-turbo"))

max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# insert urls in training_data
training_data = []

node_list = []

# Convert urls into nodes via Document injection
for url in training_data:
    url_content = convert_url_to_text_chunks(url)
    document = Document(url_content, doc_id=url)
    nodes = generate_nodes_from_document([document])
    node_list += [node for node in nodes]

# create index
index = create_index(service_context, node_list)
