from langchain.llms import Ollama

model = "codellama:13b"
llama_url = "http://localhost:11434"
ollama = Ollama(base_url=llama_url, model=model)

def explain_diff(diff):
    content = diff["content"]
    systemPrompt = f"""    
    Only use the following information to answer the question. 
    - Do not use anything else
    - Do not use your own knowledge.
    - Do not use your own opinion.
    - Do not use anything that is not in the diff.
    - Don not use the character `"` or `'` in your answer.
    - Be as concise as possible.
    - Be as specific as possible.
    - Be as accurate as possible.
    Task: Write a git commit message for the following diff
    ```
    {content}
    ```
    """
    return ollama(systemPrompt)

