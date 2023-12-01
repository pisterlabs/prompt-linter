from langchain.llms import HuggingFaceHub

if __name__ == '__main__':
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.1, "max_length": 128})
    # https://huggingface.co/google/flan-t5-xxl

    prompt = "translate English to German: How old are you?"

    print(llm(prompt))
