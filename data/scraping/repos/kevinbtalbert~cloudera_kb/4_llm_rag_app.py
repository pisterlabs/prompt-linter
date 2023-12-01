import os
import gradio

from milvus import default_server
from pymilvus import connections, Collection
import utils.vector_db_utils as vector_db
import utils.model_embedding_utils as model_embedding
import openai
import configparser

config = configparser.ConfigParser()
config.read('cloudera_kb_config.conf')

openai.api_key = config['DEFAULT']['open_ai_api_key']

# Custom CSS
custom_css = f"""
        gradio-app {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/cloudera_kb/main/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        #custom-logo {{
            text-align: center;
        }}
        .dark {{
            background-image: url('https://raw.githubusercontent.com/kevinbtalbert/cloudera_kb/main/app_assets/cldr_bg.jpg') !important;
            background-size: cover  !important;
            background-position: center center  !important;
            background-repeat: no-repeat  !important;
            background-attachment: fixed  !important;
        }}
        .gr-interface {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        .gradio-header {{
            background-color: rgba(0, 0, 0, 0.5);
        }}
        .gradio-input-box, .gradio-output-box {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        h1 {{
            color: white; 
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: large; !important;
        }}
"""
def main():
    # Configure gradio QA app 
    print("Configuring gradio app")
    demo = gradio.Interface(fn=get_responses,
                            title="ClouderaGPT",
                            description="This AI-powered assistant has been trained over Cloudera's public documentation as its knowledge base. The information it provides as a response is context driven by what is available at docs.cloudera.com. It uses OpenAI's technology to evaluate context and form an accurate response from the semantic search. It is fine tuned for questions driven around the Cloudera product stack and as such may have limited knowledge outside of this domain.",
                            inputs=[gradio.Radio(['gpt-3.5-turbo', 'gpt-4'], label="Select GPT Engine", value="gpt-3.5-turbo"), gradio.Textbox(label="Question", placeholder="")],
                            outputs=[gradio.Textbox(label="Response")],
                            allow_flagging="never",
                            css=custom_css)


    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT')))
    print("Gradio app ready")

# Helper function for generating responses for the QA app
def get_responses(engine, question):
    if engine is "" or question is "" or engine is None or question is None:
        return "No question and/or engine selected."
    
    # Load Milvus Vector DB collection
    vector_db_collection = Collection('cloudera_ml_docs')
    vector_db_collection.load()
    
    # Phase 1: Get nearest knowledge base chunk for a user question from a vector db
    context_chunk = get_nearest_chunk_from_vectordb(vector_db_collection, question)
    vector_db_collection.release()
    
    # Phase 3a: Perform text generation with LLM model using found kb context chunk
    response = get_llm_response(question, context_chunk, engine)

    return response

# Get embeddings for a user question and query Milvus vector DB for nearest knowledge base chunk
def get_nearest_chunk_from_vectordb(vector_db_collection, question):
    # Generate embedding for user question
    question_embedding =  model_embedding.get_embeddings(question)
    
    # Define search attributes for Milvus vector DB
    vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # Execute search and get nearest vector, outputting the relativefilepath
    nearest_vectors = vector_db_collection.search(
        data=[question_embedding], # The data you are querying on
        anns_field="embedding", # Column in collection to search on
        param=vector_db_search_params,
        limit=2, # limit results to 1
        expr=None, 
        output_fields=['relativefilepath'], # The fields you want to retrieve from the search result.
        consistency_level="Strong"
    )
    # print(nearest_vectors)
    # # Print the file path of the kb chunk
    # print(nearest_vectors[0].ids[0])
    
    # Return text of the nearest knowledgebase chunk
    response = ""
    for f in nearest_vectors[0]:
        response += str(load_context_chunk_from_data(f.id))
    
    return response
  
# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()
      
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response(question, context, engine):
    question = """Answer this question based on given context. If the context is insufficient to answer the question, rely on your own knowledge base. If your own knowledge base is insufficient, let me know that Cloudera has not trained you on this data yet.""" + question
    
    response = openai.ChatCompletion.create(
        model=engine, # The deployment name you chose when you deployed the GPT-35-Turbo or GPT-4 model.
        messages=[
            {"role": "system", "content": str(context)},
            {"role": "user", "content": str(question)}
            ]
    )
    
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    main()
