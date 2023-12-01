from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from text2qa_prompts import prompt, refine_prompt
_ = load_dotenv()

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

chain = load_summarize_chain(llm, chain_type="refine")

chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)

# generate transcript
def generate_qa_transcript(docs, topic):
    result = chain({"input_documents":docs, "topic":topic}, return_only_outputs=False)
    intermediate_text = "\n".join(result[''])
    output_text = result['output_text']
    qa_transcript = f"{intermediate_text}\n\n{output_text}"
    return qa_transcript