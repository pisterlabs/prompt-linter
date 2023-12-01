from langchain.llms import OpenAI

def build_llm(stream_callback):
    return OpenAI(temperature=0, model_name='text-davinci-003',request_timeout=120,streaming=True,callbacks=[stream_callback])