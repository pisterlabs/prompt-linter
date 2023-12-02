def initialize_chain(instructions, memory=None):
    from langchain import OpenAI, LLMChain, PromptTemplate
    from langchain.memory import ConversationBufferWindowMemory
    from dotenv import load_dotenv
    load_dotenv()
    if memory is None:
        memory = ConversationBufferWindowMemory()
        memory.ai_prefix = "Assistant"
    template = f"""
    Instructions: {instructions}
    {{{memory.memory_key}}}
    Human: {{human_input}}
    Assistant:"""
    prompt = PromptTemplate(
        input_variables=["history", "human_input"], template=template
    )
    chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(),
    )
    return chain


def initialize_meta_chain():
    from langchain import OpenAI, LLMChain, PromptTemplate
    from dotenv import load_dotenv
    load_dotenv()
    meta_template = """
    Assistant has just had the below interactions with a User. Assistant followed their "Instructions" closely. Your job is to critique the Assistant's performance and then revise the Instructions so that Assistant would quickly and correctly respond in the future.

    ####

    {chat_history}

    ####

    Please reflect on these interactions.

    You should first critique Assistant's performance. What could Assistant have done better? What should the Assistant remember about this user? Are there things this user always wants? Indicate this with "Critique: ...".

    You should next revise the Instructions so that Assistant would quickly and correctly respond in the future. Assistant's goal is to satisfy the user in as few interactions as possible. Assistant will only see the new Instructions, not the interaction history, so anything important must be summarized in the Instructions. Don't forget any important details in the current Instructions! Indicate the new Instructions by "Instructions: ...".
    """
    meta_prompt = PromptTemplate(
        input_variables=["chat_history"], template=meta_template
    )

    meta_chain = LLMChain(
        llm=OpenAI(temperature=0),
        prompt=meta_prompt,
        verbose=True,
    )
    return meta_chain


def get_chat_history(chain_memory):
    memory_key = chain_memory.memory_key
    chat_history = chain_memory.load_memory_variables(memory_key)[memory_key]
    return chat_history


def get_new_instructions(meta_output):
    delimiter = "Instructions: "
    new_instructions = meta_output[meta_output.find(delimiter) + len(delimiter) :]
    return new_instructions


def run_metaprompt(_task):
    _ans, _steps = "", ""
    from langchain.callbacks import get_openai_callback
    max_iters=3
    max_meta_iters=5
    failed_phrase = "task failed"
    success_phrase = "task succeeded"
    key_phrases = [success_phrase, failed_phrase]
    instructions = "None"
    with get_openai_callback() as cb:
        for i in range(max_meta_iters):
            _str = f"[Episode {i+1}/{max_meta_iters}]"
            _steps += f"{_str}\n"
            print(_str)
            _chain = initialize_chain(instructions, memory=None)
            _output = _chain.predict(human_input=_task)
            for j in range(max_iters):
                _str = f"(Step {j+1}/{max_iters})"
                _steps += f"{_str}\n"
                print(_str)
                _str = f"Assistant: {_output}"
                _steps += f"{_str}\n"
                print(_str)
                _str = f"Human: "
                _steps += f"{_str}\n"
                print(_str)
                human_input = input()
                _steps += f">>>{human_input}\n"
                if any(phrase in human_input.lower() for phrase in key_phrases):
                    break
                _output = _chain.predict(human_input=human_input)
            if success_phrase in human_input.lower():
                _str = f"You succeeded! Thanks for playing!"
                _steps += "="*40+"\n"+ f"{_str}\n"
                print(_str)
                _ans = _output.strip() +"\n\n"+ _str
                _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
                print(_token_cost)
                _steps = f"{_token_cost}\n\n" + _steps
                return [_ans, _steps]
            _meta_chain = initialize_meta_chain()
            _meta_output = _meta_chain.predict(chat_history=get_chat_history(_chain.memory))
            _str = f"Feedback: {_meta_output}"
            _steps += f"{_str}\n"
            print(_str)
            instructions = get_new_instructions(_meta_output)
            _str = f"New Instructions: {instructions}"
            _steps += f"{_str}\n"
            print(_str)
            _str = "\n" + "#" * 80 + "\n"
            _steps += f"{_str}\n"
            print(_str)
        _str = f"You failed! Thanks for playing!"
        _steps += "="*40+"\n"+ f"{_str}\n"
        print(_str)
        _ans = _output.strip() +"\n\n"+ _str
        _token_cost = f"Tokens: {cb.total_tokens} = (Prompt {cb.prompt_tokens} + Completion {cb.completion_tokens}) Cost: ${format(cb.total_cost, '.5f')}"
        print(_token_cost)
        _steps = f"{_token_cost}\n\n" + _steps
    return [_ans, _steps]


if __name__ == "__main__":
    
    _task = "Provide a systematic argument for why we should always eat pasta with olives."
    _re1 = run_metaprompt(_task)
    print(_re1)

