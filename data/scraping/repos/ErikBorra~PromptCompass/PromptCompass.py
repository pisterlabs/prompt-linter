import json
import streamlit as st
import os
import time
import gc
import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain.chains import LLMChain  # import LangChain libraries
from langchain.llms import OpenAI  # import OpenAI model
from langchain.chat_models import ChatOpenAI  # import OpenAI chat model
from langchain.callbacks import get_openai_callback  # import OpenAI callbacks
from langchain.prompts import PromptTemplate  # import PromptTemplate
from langchain.llms import HuggingFacePipeline  # import HuggingFacePipeline
import torch  # import torch
# pip install git+https://github.com/huggingface/transformers
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


def main():
    load_dotenv(".env")

    pipe = None
    open_ai_key = None
    uploaded_file = None

    # import css tasks and prompts
    with open('prompts.json') as f:
        promptlib = json.load(f)

    hide_default_format = """
       <style>
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    st.markdown(hide_default_format, unsafe_allow_html=True)

    # title
    st.title("Prompt Compass")
    st.subheader(
        "A Tool for Navigating Prompts for Computational Social Science and Digital Humanities")
    # Add Link to your repo
    st.markdown(
        '''
        [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/ErikBorra/PromptCompass)
        [![DOI](https://zenodo.org/badge/649855474.svg)](https://zenodo.org/badge/latestdoi/649855474)
        ''', unsafe_allow_html=True)
    # load available models
    model_with_names = [
        model for model in promptlib['models'] if model['name']]

    # create input area for model selection
    input_values = {}

    input_values['model'] = st.selectbox('Select a model', model_with_names,
                                         format_func=lambda x: x['name'])

    # If there is no previous state, set the default model as the first model
    if not st.session_state.get('previous_model'):
        st.session_state['previous_model'] = model_with_names[0]['name']

    st.caption(f"Model info: [{input_values['model']['name']}]({input_values['model']['resource']})" + (
        f". {input_values['model']['comment']}" if 'comment' in input_values['model'] else ""))

    # ask for open ai key if no key is set in .env
    if input_values['model']['resource'] in ["https://platform.openai.com/docs/models/gpt-3-5", "https://platform.openai.com/docs/models/gpt-4"]:
        # Load the OpenAI API key from the environment variable
        if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
            open_ai_key = st.text_input("Open AI API Key", "")
        else:
            open_ai_key = os.getenv("OPENAI_API_KEY")

    temperature = 0.001
    top_p = -1
    max_new_tokens = -1
    with st.expander("Advanced settings"):
        if input_values['model']['resource'] not in ["https://platform.openai.com/docs/models/gpt-3-5", "https://platform.openai.com/docs/models/gpt-4"]:
            st.markdown(
                """
            **Set Maximum Length**: Determines the maximum number of tokens of the **generated** text. A token is approximately four characters word, although this depends on the model.
            A value of -1 means the parameter will not be specified.
            """
            )
            max_new_tokens = st.number_input(
                'Maximum Length', value=256, min_value=-1, step=1)
            st.markdown(
                """
            **Set do_sample**: This controls how the model generates text. If do_sample=True, the model will use a probabilistic approach to generate text, where the likelihood of each word being chosen depends on its predicted probability. Use the below parameters to further control its behavior. If do_sample=False, the model will use a deterministic approach and always choose the most likely next word. 
            """
            )
            do_sample = st.radio(
                'Set do_sample',
                ('False', 'True')
            )
        st.markdown(
            """
        **Temperature**: Controls the randomness in the model's responses.
        Lower values (closer to 0.0) make the output more deterministic, while higher values (closer to 2.0) make it more diverse.
        A value of -1 means the parameter will not be specified.
        """
        )
        temperature = st.number_input(
            'Set Temperature', min_value=-1.0, max_value=2.0, value=0.001, format="%.3f")

        st.markdown(
            """
        **Top P**: Also known as "nucleus sampling", is an alternative to temperature that can also be used to control the randomness of the model's responses.
        It essentially trims the less likely options in the model's distribution of possible responses. Possible values lie between 0.0 and 1.0. 
        A value of -1 means the parameter will not be specified.
        """
        )
        top_p = st.number_input('Set Top-P', min_value=-
                                1.0, max_value=1.0, value=-1.0)

    # Check for correct values
    allgood = True
    # set model kwargs
    model_kwargs = {}

    if input_values['model']['resource'] not in ["https://platform.openai.com/docs/models/gpt-3-5", "https://platform.openai.com/docs/models/gpt-4"]:
        # check if max_new_tokens is at least 1 or -1
        if not (max_new_tokens > 0 or max_new_tokens == -1):
            st.error(
                'Error: Max Tokens must be at least 1. Choose -1 if you want to use the default model value.')
            max_new_tokens = -1
            allgood = False
        if max_new_tokens > 0:
            model_kwargs['max_new_tokens'] = max_new_tokens

        if do_sample not in ['True', 'False']:
            st.error(
                'Error: do_Sample must be True or False')
            do_sample = False
            allgood = False
        do_sample = True if do_sample == 'True' else False
        if do_sample in [True, False]:
            model_kwargs['do_sample'] = do_sample

    if not (0 <= temperature <= 2 or temperature == -1):
        st.error(
            "Temperature value must be between 0 and 2. Choose -1 if you want to use the default model value.")
        temperature = -1
        allgood = False
    if 0 <= temperature <= 2:
        model_kwargs['temperature'] = temperature

    if not (0 <= top_p <= 1 or top_p == -1):
        st.error(
            "Top P value must be between 0 and 1. Choose -1 if you want to use the default model value.")
        top_p = -1
        allgood = False
    if 0 <= top_p <= 1:
        model_kwargs['top_p'] = top_p

    # create input area for task selection
    tasks_with_names = [task for task in promptlib['tasks'] if task['name']]
    task = st.selectbox('Select a task', tasks_with_names,
                        format_func=lambda x: x['name'] + " - " + x['authors'])

    # Create input areas for prompts and user input
    if task:

        # concatenate all strings from prompt array
        prompt = '\n'.join(task['prompt'])

        # create input area for prompt
        input_values['prompt'] = st.text_area(
            "Inspect, and possibly modify, the prompt by ["+task['authors']+"]("+task['paper']+")", prompt, height=200)

        # allow the user to select the input type
        input_type = st.radio("Choose input type:",
                              ('Text input', 'Upload a CSV'), horizontal=True)

        if input_type == 'Text input':
            # create input area for user input
            input_values['user'] = st.text_area(
                "Input to be analyzed with the prompt (one thing per line):",
                "this user is happy\none user is just a user\nthe other user is a lier")
            # if the user's input is not a list (e.g. a string), then split it by newlines
            if isinstance(input_values['user'], str):
                input_values['user'] = input_values['user'].split('\n')
            data = pd.DataFrame(input_values['user'], columns=['user_input'])
        else:
            # upload CSV
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                # convert the uploaded file to a dataframe
                data = pd.read_csv(uploaded_file)

                # ask user to select a column
                column_to_extract = st.selectbox(
                    'Choose a column to apply the prompt on:', data.columns)

                # process the selected column from the dataframe
                input_values['user'] = data[column_to_extract].tolist()

        # Determine the output file name
        filename = uploaded_file.name if uploaded_file else 'output.csv'
        base_filename, file_extension = os.path.splitext(filename)
        output_filename = f"{base_filename}_promptcompass{file_extension}"

    # Submit button
    submit_button = st.button('Submit')

    st.write('---')  # Add a horizontal line

    # Process form submission
    if submit_button and allgood:
        if 'user' not in input_values or input_values['user'] is None:
            st.error("No user input provided")

        else:
            with st.spinner(text="In progress..."):

                try:

                    start_time = time.time()
                    st.write("Start time: " +
                             time.strftime("%H:%M:%S", time.localtime()))

                    if input_values['prompt'] and input_values['user']:

                        # create prompt template
                        # add location of user input to prompt
                        if task['location_of_input'] == 'before':
                            template = "{user_input}" + \
                                "\n\n" + input_values['prompt']
                        elif task['location_of_input'] == 'after':
                            template = input_values['prompt'] + \
                                "\n\n" + "{user_input}"
                        else:
                            template = input_values['prompt']

                        # make sure users don't forget the user input variable
                        if "{user_input}" not in template:
                            template = template + "\n\n{user_input}"

                        # fill prompt template
                        prompt_template = PromptTemplate(
                            input_variables=["user_input"], template=template)

                        # loop over user values in prompt
                        for key, user_input in enumerate(input_values['user']):

                            num_tokens = None

                            user_input = str(user_input).strip()
                            if user_input == "" or user_input == "nan":
                                continue

                            # set up and run the model
                            model_id = input_values['model']['name']
                            if model_id in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001', 'gpt-4', 'gpt-3.5-turbo-instruct', 'babbage-002', 'davinci-002']:
                                if open_ai_key is None or open_ai_key == "":
                                    st.error(
                                        "Please provide an Open AI API Key")
                                    exit(1)
                                with get_openai_callback() as cb:
                                    if model_id in ['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-4']:
                                        llm = ChatOpenAI(
                                            model=model_id, openai_api_key=open_ai_key, **model_kwargs)
                                    else:
                                        llm = OpenAI(
                                            model=model_id, openai_api_key=open_ai_key, **model_kwargs)

                                    llm_chain = LLMChain(
                                        llm=llm, prompt=prompt_template)

                                    output = llm_chain.run(user_input)

                                    encoding = tiktoken.encoding_for_model(
                                        model_id)
                                    num_tokens = len(encoding.encode(
                                        prompt_template.format(user_input=user_input)))

                                    st.success("Input:  " + user_input + "  \n\n " +
                                               "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
                                               "Output: " + output)
                                    st.text(cb)
                            elif model_id in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf']:
                                if pipe == None:
                                    with st.status('Loading model %s' % model_id) as status:
                                        # to use the llama-2 models,
                                        # you first need to get access to the llama-2 models via e.g. https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
                                        # once accepted, get a hugging face auth token https://huggingface.co/settings/tokens
                                        # and then run `huggingface-cli login` on the command line, filling in the generated token
                                        if model_id in ['meta-llama/Llama-2-7b-chat-hf', 'meta-llama/Llama-2-13b-chat-hf']:
                                            tokenizer = AutoTokenizer.from_pretrained(
                                                model_id, use_auth_token=True)
                                        else:
                                            tokenizer = AutoTokenizer.from_pretrained(
                                                model_id)

                                        if model_id == "meta-llama/Llama-2-13b-chat-hf":
                                            pipe = pipeline(
                                                "text-generation",
                                                model=model_id,
                                                tokenizer=tokenizer,
                                                # torch_dtype="auto",
                                                trust_remote_code=True,
                                                device_map="auto",
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,
                                                **model_kwargs
                                            )
                                        else:
                                            pipe = pipeline(
                                                "text-generation",
                                                model=model_id,
                                                tokenizer=tokenizer,
                                                torch_dtype="auto",
                                                trust_remote_code=True,
                                                device_map="auto",
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,
                                                **model_kwargs
                                            )

                                        local_llm = HuggingFacePipeline(
                                            pipeline=pipe)

                                    status.update(
                                        label='Model %s loaded' % model_id, state="complete")

                                llm_chain = LLMChain(
                                    llm=local_llm, prompt=prompt_template)

                                output = llm_chain.run(user_input)

                                num_tokens = len(tokenizer.tokenize(
                                    prompt_template.format(user_input=user_input)))

                                st.success("Input:  " + user_input + "  \n\n " +
                                           "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
                                           "Output: " + output)
                            elif model_id in ['google/flan-t5-large', 'google/flan-t5-xl', 'tiiuae/falcon-7b-instruct', 'tiiuae/falcon-40b-instruct', 'databricks/dolly-v2-3b', 'databricks/dolly-v2-7b']:
                                if pipe is None:
                                    with st.status('Loading model %s' % model_id) as status:
                                        tokenizer = AutoTokenizer.from_pretrained(
                                            model_id)

                                        if model_id in ['google/flan-t5-large', 'google/flan-t5-xl']:
                                            model = AutoModelForSeq2SeqLM.from_pretrained(
                                                model_id, load_in_8bit=False, device_map='auto')
                                            pipe = pipeline(
                                                "text2text-generation",
                                                model=model_id,
                                                tokenizer=tokenizer,
                                                torch_dtype="auto",
                                                trust_remote_code=True,
                                                device_map="auto",
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,
                                                **model_kwargs
                                            )
                                        # elif model_id in ['tiiuae/falcon-7b-instruct', 'tiiuae/falcon-40b-instruct']:
                                        else:
                                            pipe = pipeline(
                                                "text-generation",
                                                model=model_id,
                                                tokenizer=tokenizer,
                                                torch_dtype="auto",
                                                trust_remote_code=True,
                                                device_map="auto",
                                                num_return_sequences=1,
                                                eos_token_id=tokenizer.eos_token_id,
                                                **model_kwargs
                                            )

                                        local_llm = HuggingFacePipeline(
                                            pipeline=pipe)
                                        status.update(
                                            label='Model %s loaded' % model_id, state="complete")

                                llm_chain = LLMChain(
                                    llm=local_llm, prompt=prompt_template)

                                output = llm_chain.run(user_input)
                                num_tokens = len(tokenizer.tokenize(
                                    prompt_template.format(user_input=user_input)))
                                st.success("Input:  " + user_input + "  \n\n " +
                                           "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
                                           "Output: " + output)
                            elif model_id == "mosaicml/mpt-7b-instruct":
                                if pipe is None:
                                    with st.status('Loading model %s' % model_id) as status:

                                        model = AutoModelForCausalLM.from_pretrained(
                                            model_id,
                                            trust_remote_code=True,
                                            torch_dtype=torch.bfloat16,
                                            max_seq_len=2048,
                                            device_map="auto"
                                        )

                                        # MPT-7B model was trained using the EleutherAI/gpt-neox-20b tokenizer
                                        tokenizer = AutoTokenizer.from_pretrained(
                                            "EleutherAI/gpt-neox-20b")

                                        # mtp-7b is trained to add "<|endoftext|>" at the end of generations
                                        stop_token_ids = tokenizer.convert_tokens_to_ids(
                                            ["<|endoftext|>"])
                                        # define custom stopping criteria object

                                        class StopOnTokens(StoppingCriteria):
                                            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                                                for stop_id in stop_token_ids:
                                                    if input_ids[0][-1] == stop_id:
                                                        return True
                                                return False
                                        stopping_criteria = StoppingCriteriaList(
                                            [StopOnTokens()])

                                        pipe = pipeline(
                                            task='text-generation',
                                            model=model,
                                            tokenizer=tokenizer,
                                            torch_dtype="auto",
                                            device_map="auto",
                                            num_return_sequences=1,
                                            eos_token_id=tokenizer.eos_token_id,
                                            **model_kwargs,
                                            return_full_text=True,  # langchain expects the full text
                                            stopping_criteria=stopping_criteria,  # without this model will ramble
                                            repetition_penalty=1.1  # without this output begins repeating
                                        )

                                        local_llm = HuggingFacePipeline(
                                            pipeline=pipe)

                                    status.update(
                                        label='Model %s loaded' % model_id, state="complete")

                                llm_chain = LLMChain(
                                    llm=local_llm, prompt=prompt_template)

                                output = llm_chain.run(user_input)

                                num_tokens = len(tokenizer.tokenize(
                                    prompt_template.format(user_input=user_input)))

                                st.success("Input:  " + user_input + "  \n\n " +
                                           "Input tokens (incl. prompt): " + str(num_tokens) + "  \n\n " +
                                           "Output: " + output)
                            else:
                                st.error("Model %s not found" % model_id)
                                exit(1)

                            # add output to dataframe
                            data.loc[key, 'output'] = output
                            data.loc[key, 'llm'] = model_id
                            data.loc[key, 'prompt timestamp'] = time.strftime(
                                "%Y-%m-%d %H:%M:%S", time.localtime())
                            data.loc[key, 'prompt name'] = task['name']
                            data.loc[key, 'prompt authors'] = task['authors']
                            data.loc[key, '# input tokens'] = str(
                                int(num_tokens))
                            data.loc[key, 'prompt'] = template
                            if "temperature" in model_kwargs:
                                data.loc[key,
                                         'temperature'] = model_kwargs['temperature']
                            if "top_p" in model_kwargs:
                                data.loc[key, 'top_p'] = model_kwargs['top_p']
                            if "max_tokens" in model_kwargs:
                                data.loc[key,
                                         'max_tokens'] = int(model_kwargs['max_tokens'])
                            if "do_sample" in model_kwargs:
                                data.loc[key,
                                         'do_sample'] = int(model_kwargs['do_sample'])

                        # make output available as csv
                        csv = data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download CSV",
                            csv,
                            output_filename,
                            "text/csv",
                            key='download-csv'
                        )

                        # st.write(vars(pipe)) # print all variables set in the pipeline

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.write("End time: " +
                             time.strftime("%H:%M:%S", time.localtime()))
                    st.write("Elapsed time: " +
                             str(round(elapsed_time, 2)) + " seconds")

                except Exception as e:
                    st.error(e)

                finally:
                    # free up variables
                    if 'data' in locals() and data is not None:
                        del data
                    if 'pipe' in locals() and pipe is not None:
                        del pipe
                    if 'llm_chain' in locals() and llm_chain is not None:
                        del llm_chain
                    if 'llm' in locals() and llm is not None:
                        del llm
                    if 'local_llm' in locals() and local_llm is not None:
                        del local_llm
                    if 'model' in locals() and model is not None:
                        del model
                    if 'tokenizer' in locals() and tokenizer is not None:
                        del tokenizer
                    gc.collect()  # garbage collection
                    # empty cuda cache
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
