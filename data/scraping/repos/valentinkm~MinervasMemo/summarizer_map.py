# Summarizer_map.py
# Libraries
import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langfuse.callback import CallbackHandler
from langchain.callbacks import get_openai_callback
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI

# Modules
from map_reduce_prompts_minimal import map_prompt_template, combine_prompt_template, map_prompt_template2, combine_prompt_template2, bullet_prompt, all_in_one_prompt
from tokenizer import encoding_gpt3, encoding_gpt4
from splitter import r_splitter

# Initialization variables set to None
llm = None
llm_final = None
llm_all_in_one = None
summary_chain1 = None
summary_chain2 = None
bullet_chain = None
all_in_one_chain = None
handler = None

def initialize_summarizer():
    global llm, llm_final, summary_chain1, summary_chain2, bullet_chain, handler, all_in_one_chain
    
    _=load_dotenv()
    all_in_one_model_name = os.getenv('OPENAI_GPT_ALL_IN_ONE')
    final_model_name = os.getenv('OPENAI_GPT_FINAL')
    mapreduce_model_name = os.getenv('OPENAI_GPT_MAPREDUCE')

    print("Initializing summarizer...")
    print("all in one prompt model: ", all_in_one_model_name)
    print("model for final prompt:", final_model_name)
    print("map-reduce model:", mapreduce_model_name)

    PUBLIC_KEY = os.getenv('PUBLIC_KEY')
    SECRET_KEY = os.getenv('SECRET_KEY')
    
    handler = CallbackHandler(PUBLIC_KEY, SECRET_KEY)
    
    # get model name from environment variable:
    
    llm = ChatOpenAI(temperature=0.7, model_name =mapreduce_model_name)
    
    llm_final = ChatOpenAI(temperature=0.7, model_name =final_model_name)

    llm_all_in_one = ChatOpenAI(temperature=0.7, model_name =all_in_one_model_name)

    
    summary_chain1 = load_summarize_chain (
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
        verbose=False,
        token_max=4000
    )

    summary_chain2 = load_summarize_chain (
    llm=llm,
    chain_type='map_reduce',
    map_prompt=map_prompt_template2,
    combine_prompt=combine_prompt_template2,
    verbose=False,
    token_max=4000
    )
    bullet_chain = LLMChain(llm=llm_final, prompt=bullet_prompt, output_key="bullet-summary")
    all_in_one_chain = LLMChain(llm=llm_all_in_one, prompt=all_in_one_prompt, output_key="all-in-one-summary")

def generate_summary_map(docs, token_count_transcript):
    global llm, llm_final, summary_chain1, summary_chain2, bullet_chain, handler, all_in_one_chain

    if llm is None:
        initialize_summarizer()
    print("Summarizer initialized.")

    # Initialize the token_info dictionaries with default values
    token_info_map1 = {'Total Tokens': 0, 'Prompt Tokens': 0, 'Completion Tokens': 0, 'Total Cost (USD)': 0.0}
    token_info_map2 = {'Total Tokens': 0, 'Prompt Tokens': 0, 'Completion Tokens': 0, 'Total Cost (USD)': 0.0}
    token_info_final = {'Total Tokens': 0, 'Prompt Tokens': 0, 'Completion Tokens': 0, 'Total Cost (USD)': 0.0}
    aggregated_token_info = {'Total Tokens': 0, 'Total Cost (USD)': 0.0} 

    if token_count_transcript < 4000:
        print("Token count is less than 4000. Running single-prompt summary...")
        with get_openai_callback() as cb:
            result = all_in_one_chain(docs)
            final_summary = result['all-in-one-summary']

            token_info_final['Total Tokens'] = cb.total_tokens
            token_info_final['Prompt Tokens'] = cb.prompt_tokens
            token_info_final['Completion Tokens'] = cb.completion_tokens
            token_info_final['Total Cost (USD)'] = cb.total_cost

            # Update aggregated token info
            aggregated_token_info['Total Tokens'] += cb.total_tokens
            aggregated_token_info['Total Cost (USD)'] += cb.total_cost
    
    else:
        # Run first summary chain
        print("Running map reduce chain...")
        with get_openai_callback() as cb:
            first_summary = summary_chain1.run(docs, callbacks=[handler])

            token_info_map1['Total Tokens'] = cb.total_tokens
            token_info_map1['Prompt Tokens'] = cb.prompt_tokens
            token_info_map1['Completion Tokens'] = cb.completion_tokens
            token_info_map1['Total Cost (USD)'] = cb.total_cost

            # Update aggregated token info
            aggregated_token_info['Total Tokens'] += cb.total_tokens
            aggregated_token_info['Total Cost (USD)'] += cb.total_cost

        token_count_first_summary = len(encoding_gpt4.encode(first_summary))
        print(f"First summary chain complete. Used {token_info_map1['Total Tokens']} tokens. Returned {token_count_first_summary} tokens.")

        # Check if second chain needs to be run
        if token_count_first_summary > 4000:
            print("Running the second summary chain due to token limit breach...")
            first_summary_docs = r_splitter.create_documents([first_summary])
            with get_openai_callback() as cb:
                second_summary = summary_chain2.run(first_summary_docs, callbacks=[handler])

                token_info_map2['Total Tokens'] = cb.total_tokens
                token_info_map2['Prompt Tokens'] = cb.prompt_tokens
                token_info_map2['Completion Tokens'] = cb.completion_tokens
                token_info_map2['Total Cost (USD)'] = cb.total_cost
                
                # Update aggregated token info
                aggregated_token_info['Total Tokens'] += cb.total_tokens
                aggregated_token_info['Total Cost (USD)'] += cb.total_cost

            print(f"Second summary chain complete. Used {token_info_map2['Total Tokens']} tokens.")

            final_summary = second_summary
        else:
            final_summary = first_summary
        print("Generating final summary...")
        with get_openai_callback() as cb:
            final_summary = bullet_chain.run(summary = final_summary,callbacks=[handler])
            
            token_info_final['Total Tokens'] = cb.total_tokens
            token_info_final['Prompt Tokens'] = cb.prompt_tokens
            token_info_final['Completion Tokens'] = cb.completion_tokens
            token_info_final['Total Cost (USD)'] = cb.total_cost

            # Update aggregated token info
            aggregated_token_info['Total Tokens'] += cb.total_tokens
            aggregated_token_info['Total Cost (USD)'] += cb.total_cost

        print(f"Bullet summary complete. Used {token_info_final['Total Tokens']} tokens.")
        print(f"Total tokens used: {token_info_map1['Total Tokens'] + token_info_map2['Total Tokens'] + token_info_final['Total Tokens']}")
        print(f"Total estimated cost: {token_info_map1['Total Cost (USD)'] + token_info_map2['Total Cost (USD)'] + token_info_final['Total Cost (USD)']}$")

    return final_summary, aggregated_token_info